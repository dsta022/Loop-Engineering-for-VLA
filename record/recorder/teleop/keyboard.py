"""Keyboard event listener for recording control (pynput, optional).

Mirrors lerobot's ``init_keyboard_listener`` events so operators keep the same
muscle memory:

* Right arrow -> end the current episode early (``exit_early``)
* Left arrow  -> re-record the current episode (``rerecord_episode``)
* Escape      -> stop the whole recording session (``stop_recording``)

Degrades safely: on a headless machine or without pynput, it returns an events
dict that never triggers, so time-based episode termination still works.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_headless() -> bool:
    if os.name == "nt":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def init_keyboard_listener() -> tuple[object | None, dict[str, bool]]:
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    if is_headless():
        logger.info("headless environment: keyboard shortcuts disabled (using time-based episodes)")
        return None, events

    try:
        from pynput import keyboard
    except Exception as exc:  # noqa: BLE001
        logger.warning("pynput unavailable (%s); keyboard shortcuts disabled", exc)
        return None, events

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow -> end episode")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow -> re-record episode")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape -> stop recording")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("keyboard handler error: %s", exc)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events
