"""List connected cameras (opencv / realsense / orbbec), lerobot-free."""

from __future__ import annotations


def find_opencv(max_index: int = 10) -> list[dict]:
    import cv2

    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append({"index": index, "height": frame.shape[0], "width": frame.shape[1]})
        cap.release()
    return found


def find_realsense() -> list[dict]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
            }
        )
    return devices


def find_orbbec() -> list[dict]:
    try:
        import pyorbbecsdk as ob
    except Exception:
        import pyorbbecsdk2 as ob  # type: ignore

    ctx = ob.Context()
    devices = ctx.query_devices()
    count = devices.get_count() if hasattr(devices, "get_count") else len(devices)
    out = []
    for i in range(count):
        dev = devices.get_device_by_index(i) if hasattr(devices, "get_device_by_index") else devices[i]
        info = dev.get_device_info()
        out.append(
            {
                "name": str(getattr(info, "get_name", lambda: "")()),
                "serial_number": str(getattr(info, "get_serial_number", lambda: "")()),
            }
        )
    return out


def main(argv: list[str]) -> None:
    backend = argv[1] if len(argv) > 1 else "opencv"
    finders = {"opencv": find_opencv, "realsense": find_realsense, "orbbec": find_orbbec}
    if backend not in finders:
        raise SystemExit(f"usage: find_cameras.py [opencv|realsense|orbbec]; got {backend!r}")
    results = finders[backend]()
    if not results:
        print(f"no {backend} cameras found")
        return
    for item in results:
        print(item)
