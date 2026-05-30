#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Configuration class for Orbbec RGB-D cameras using pyorbbecsdk.

    `width`, `height`, and `fps` describe the RGB stream. Depth can use separate
    dimensions through `depth_width`, `depth_height`, and `depth_fps`.
    If `align_depth_to_color` is enabled, the camera reports depth observations
    in RGB image coordinates, so the depth feature shape matches RGB.
    """

    serial_number_or_name: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    depth_width: int | None = None
    depth_height: int | None = None
    depth_fps: int | None = None
    align_depth_to_color: bool = True

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        rgb_values = (self.fps, self.width, self.height)
        if any(v is not None for v in rgb_values) and any(v is None for v in rgb_values):
            raise ValueError(
                "For Orbbec RGB, `fps`, `width`, and `height` must either all be set or all be unset."
            )

        depth_values = (self.depth_fps, self.depth_width, self.depth_height)
        if any(v is not None for v in depth_values) and any(v is None for v in depth_values):
            raise ValueError(
                "For Orbbec depth, `depth_fps`, `depth_width`, and `depth_height` must either all be "
                "set or all be unset."
            )

        if self.use_depth and self.align_depth_to_color and (self.width is None or self.height is None):
            raise ValueError(
                "`align_depth_to_color=true` requires explicit RGB `width` and `height` so dataset "
                "depth feature shapes are known before the camera connects."
            )

    @property
    def output_depth_width(self) -> int | None:
        if self.align_depth_to_color:
            return self.width
        return self.depth_width or self.width

    @property
    def output_depth_height(self) -> int | None:
        if self.align_depth_to_color:
            return self.height
        return self.depth_height or self.height
