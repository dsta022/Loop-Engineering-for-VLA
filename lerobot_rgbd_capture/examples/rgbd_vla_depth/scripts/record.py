#!/usr/bin/env python

"""Project-local recorder that stores RGB as video and depth as uint16 PNG images."""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import rerun as rr

# Async depth PNG writing - futures are tracked per episode
_depth_writer_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="depth_writer")
_depth_futures: list = []


def flush_depth_writes():
    """Block until all pending depth PNG writes are complete."""
    for f in _depth_futures:
        f.result()
    _depth_futures.clear()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import lerobot_policy_rgbd_vla_depth  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_policy_rgbd_vla_depth.data.depth_feature_contract import (
    build_depth_recording_manifest,
    force_depth_features_to_images,
    iter_depth_feature_keys,
)
from lerobot_policy_rgbd_vla_depth.data.depth_io import save_depth_sidecar_frame

DEPTH_MANIFEST_PATH = Path("meta") / "rgbd_vla_depth_recording.json"


def build_rgbd_dataset_features(
    cfg: RecordConfig,
    robot,
    teleop_action_processor,
    robot_observation_processor,
) -> dict[str, dict]:
    """Build dataset features with RGB streams as video and depth streams as PNG images."""
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    return force_depth_features_to_images(dataset_features)


def write_depth_manifest(dataset: LeRobotDataset) -> None:
    """Persist a small manifest describing the depth streams in the recorded dataset."""
    manifest = build_depth_recording_manifest(dataset.features)
    if not manifest["depth_streams"]:
        return
    manifest_path = Path(dataset.root) / DEPTH_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def persist_depth_sidecars(dataset: LeRobotDataset, observation_frame: dict[str, Any]) -> None:
    """Submit depth PNG writes to a background thread pool (non-blocking)."""
    writer = dataset.writer
    if writer is None:
        return

    episode_index_raw = writer.episode_buffer["episode_index"]
    if isinstance(episode_index_raw, np.ndarray):
        episode_index = int(episode_index_raw.item())
    else:
        episode_index = int(episode_index_raw)
    frame_index = int(writer.episode_buffer["size"])

    for depth_key in iter_depth_feature_keys(dataset.features):
        if depth_key not in observation_frame:
            continue
        # Copy the array so the main loop can reuse the buffer
        depth_copy = np.array(observation_frame[depth_key], copy=True)
        _depth_futures.append(
            _depth_writer_pool.submit(
                save_depth_sidecar_frame,
                dataset_root=dataset.root,
                feature_key=depth_key,
                episode_index=episode_index,
                frame_index=frame_index,
                depth_mm=depth_copy,
            )
        )


@safe_stop_image_writer
def record_loop_rgbd(
    robot,
    events: dict[str, Any],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
) -> None:
    """Project-local recording loop that also persists permanent depth PNG sidecars."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if isinstance(teleop, list):
        raise NotImplementedError(
            "Multi-teleop recording is not yet supported by examples/rgbd_vla_depth/scripts/record.py."
        )
    if control_time_s is None:
        raise ValueError("control_time_s must be provided.")

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)
        elif policy is None and isinstance(teleop, Teleoperator):
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        if policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        _sent_action = robot.send_action(robot_action_to_send)
        del _sent_action

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            persist_depth_sidecars(dataset, observation_frame)
            dataset.add_frame(frame)

        if display_data:
            rr.set_time("frame_index", sequence=(dataset.num_frames if dataset is not None else 0))
            rr.set_time("episode_time", duration=timestamp)
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = 1 / fps - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = build_rgbd_dataset_features(
        cfg=cfg,
        robot=robot,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        write_depth_manifest(dataset)

        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. RGB streams will be encoded after each episode; "
                "depth streams are still stored as uint16 PNG images."
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop_rgbd(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop_rgbd(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Wait for all async depth writes to finish before saving
                flush_depth_writes()

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        flush_depth_writes()
        _depth_writer_pool.shutdown(wait=True)

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main() -> None:
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
