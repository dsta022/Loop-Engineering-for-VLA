#!/usr/bin/env python
"""Render uint16 depth PNGs into 8-bit turbo-colored previews.

Examples:
    # Single file
    python depth_visualize.py path/to/frame-000000.png

    # Whole directory (recursive)
    python depth_visualize.py ~/.cache/huggingface/lerobot/DerekLX/lerobot_derek_depth/depth_sidecar

    # Custom range + output dir
    python depth_visualize.py some_dir --vmin 350 --vmax 800 --out /tmp/depth_vis

    # Print value statistics only (no images written)
    python depth_visualize.py some_file.png --stats
"""

import argparse
import binascii
import struct
import zlib
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None

import sys

# Allow `python ./tools/depth_visualize.py` to run as a plain script: the repo-root
# packages must be importable without installing the project first.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from utils.media_utils import read_png_array


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB image, got dtype={rgb.dtype} shape={rgb.shape}")

    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(kind)
        crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    payload = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw)),
        chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def read_depth_png(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    arr = read_png_array(path)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr.astype(np.uint16, copy=False)


def fallback_colormap(norm: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [48, 18, 59],
            [49, 104, 142],
            [53, 183, 121],
            [253, 231, 37],
            [204, 71, 43],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(norm.astype(np.float32) / 255.0, 0.0, 1.0) * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    weight = (scaled - left)[..., None]
    rgb = stops[left] * (1.0 - weight) + stops[right] * weight
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, vmin: int, vmax: int) -> np.ndarray:
    """uint16 depth (mm) -> colored uint8 preview. Invalid pixels -> black."""
    invalid = (depth == 0) | (depth > vmax)
    clipped = np.clip(depth, vmin, vmax).astype(np.float32)
    norm = ((clipped - vmin) / max(vmax - vmin, 1) * 255).astype(np.uint8)
    if cv2 is not None:
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    else:
        colored = fallback_colormap(norm)
    colored[invalid] = 0
    return colored


def print_stats(path: Path, depth: np.ndarray) -> None:
    nz = depth[depth > 0]
    total = depth.size
    print(f"\n{path}")
    print(f"  dtype={depth.dtype} shape={depth.shape}")
    print(f"  raw min/max: {depth.min()} / {depth.max()}")
    if nz.size:
        print(f"  nonzero: {nz.size}/{total} ({100 * nz.size / total:.1f}%)")
        print(f"  nonzero min/median/max (mm): {nz.min()} / {int(np.median(nz))} / {nz.max()}")
        for p in (5, 25, 50, 75, 95):
            print(f"  p{p:02d}: {int(np.percentile(nz, p))} mm")


def process_file(src: Path, dst: Path | None, vmin: int, vmax: int, stats: bool) -> None:
    depth = read_depth_png(src)
    if depth is None:
        print(f"[skip] cannot read {src}")
        return
    if depth.dtype != np.uint16:
        print(f"[skip] {src} is not uint16 (got {depth.dtype})")
        return

    if stats:
        print_stats(src, depth)
        return

    colored = colorize_depth(depth, vmin, vmax)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(dst), colored)
    else:
        write_rgb_png(dst, colored)
    print(f"{src} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Depth PNG file or directory (searched recursively)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <input>_vis next to input)")
    parser.add_argument("--vmin", type=int, default=320, help="Min depth in mm for color scale (default 320)")
    parser.add_argument("--vmax", type=int, default=730, help="Max depth in mm for color scale (default 730)")
    parser.add_argument("--stats", action="store_true", help="Print value statistics only, do not write images")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Not found: {args.input}")
        return 1

    if args.input.is_file():
        files = [args.input]
        out_root = args.out or args.input.parent / f"{args.input.stem}_vis"
        out_root.mkdir(parents=True, exist_ok=True)
        dst = out_root / args.input.name if not args.stats else None
        process_file(args.input, dst, args.vmin, args.vmax, args.stats)
        return 0

    files = sorted(args.input.rglob("*.png"))
    if not files:
        print(f"No PNGs found under {args.input}")
        return 1

    out_root = args.out or args.input.parent / f"{args.input.name}_vis"
    print(f"Found {len(files)} files. vmin={args.vmin} vmax={args.vmax}")
    if not args.stats:
        print(f"Output -> {out_root}")

    for i, src in enumerate(files, 1):
        dst = None if args.stats else out_root / src.relative_to(args.input)
        process_file(src, dst, args.vmin, args.vmax, args.stats)
        if not args.stats and i % 100 == 0:
            print(f"  [{i}/{len(files)}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
