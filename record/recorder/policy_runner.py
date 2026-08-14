"""Pluggable in-loop policy inference for feedback collection.

Teleoperated recording (rgb/rgbd) is 100% framework-free. Only ``feedback_record``
runs a policy in the loop, and it does so through the :class:`PolicyRunner`
interface below. The default :class:`LeRobotPolicyRunner` *lazily* imports lerobot
so nothing else in the recorder depends on it; supply your own ``PolicyRunner`` to
drive any other policy runtime with no lerobot dependency at all.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PolicyRunner(Protocol):
    """Minimal contract the feedback loop needs from a policy."""

    def reset(self) -> None:
        """Reset any recurrent/temporal state at the start of an episode."""

    def select_action(self, observation: dict[str, Any], *, task: str) -> dict[str, Any]:
        """Map a robot observation to an action dict ``{"<motor>.pos": value}``."""


class StubPolicyRunner:
    """Offline stand-in: holds the current joint positions.

    Useful for ``--mock`` dry-runs and as a template for custom runners. It needs
    no torch, no lerobot, and no checkpoint.
    """

    def __init__(self, motor_names: list[str]):
        self.motor_names = list(motor_names)

    def reset(self) -> None:
        return None

    def select_action(self, observation: dict[str, Any], *, task: str) -> dict[str, Any]:
        return {f"{m}.pos": float(observation.get(f"{m}.pos", 0.0)) for m in self.motor_names}


class LeRobotPolicyRunner:
    """Default adapter that runs a lerobot checkpoint (lazy lerobot import).

    Instantiated only when ``feedback_record`` is actually driven by a policy, so
    teleoperated recording never imports lerobot. Requires ``pip install lerobot``
    plus the policy's own extras; adjust or replace for a custom policy runtime.
    """

    def __init__(
        self,
        policy_path: str,
        motor_names: list[str],
        features: dict[str, dict[str, Any]],
        *,
        device: str = "cuda",
    ):
        self.motor_names = list(motor_names)
        self.features = features
        # -- lazy lerobot imports -------------------------------------------------
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
        from lerobot.utils.device_utils import get_safe_torch_device

        self._predict_action = predict_action
        self._get_device = get_safe_torch_device

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = device
        self._cfg = cfg
        self._policy = make_policy(cfg, ds_meta=None) if _accepts_meta(make_policy) else make_policy(cfg)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path
        )

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def select_action(self, observation: dict[str, Any], *, task: str) -> dict[str, Any]:
        from lerobot.datasets.feature_utils import build_dataset_frame
        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.constants import OBS_STR

        observation_frame = build_dataset_frame(self.features, observation, prefix=OBS_STR)
        action_values = self._predict_action(
            observation=observation_frame,
            policy=self._policy,
            device=self._get_device(self._cfg.device),
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            use_amp=getattr(self._cfg, "use_amp", False),
            task=task,
        )
        robot_action = make_robot_action(action_values, self.features)
        return {key: float(val) for key, val in robot_action.items() if key.endswith(".pos")}


def _accepts_meta(func) -> bool:
    import inspect

    try:
        return "ds_meta" in inspect.signature(func).parameters
    except (ValueError, TypeError):
        return False


def build_policy_runner(policy_cfg, motor_names, features, *, mock: bool = False) -> PolicyRunner:
    """Construct the default runner: Stub in mock mode, else the lerobot adapter."""
    if mock or policy_cfg is None or not policy_cfg.path:
        return StubPolicyRunner(motor_names)
    return LeRobotPolicyRunner(policy_cfg.path, motor_names, features, device=policy_cfg.device)
