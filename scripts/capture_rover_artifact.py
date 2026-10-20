"""Capture the section-16 camera artefacts from the rover: a plant frame and its height.

docs/rover_hardware_test_report.md section 3 requires two artefacts before the rover
camera can be used as evidence: one frame from the actual rover camera pointed at a
real plant, and the camera height above the ground MEASURED WITH A TAPE. This saves
the clean frame (/raw, without the status banner) and records the height and the
frame's own evidence beside it, so the artefact carries its provenance:

    python scripts/capture_rover_artifact.py --height-cm 24.5 --plant-distance-cm 35
        --note "tomato in a pot on the floor"

The heights are typed in from the tape, never estimated: the script refuses to run
without the ground height. A blank frame, or a stuck sensor's structureless stripes,
is refused too - that is a camera fault, not a plant. Writes results/rover_artifacts/<date_time>/ containing
plant_frame.jpg and artifact.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "rover")]

from rover_camera import RoverCamera  # noqa: E402
from spatial_policy import spatial_decision  # noqa: E402

#: Below this per-frame standard deviation a frame is uniform: no sensor data
#: (the same rule as rover/hardware_tests/test_components.py).
UNIFORM_FRAME_STD = 0.5


def capture(rover: str, height_cm: float, note: str, out_root: Path,
            plant_distance_cm: float | None = None) -> Path:
    if not height_cm > 0:
        raise SystemExit("--height-cm must be the tape-measured height, in cm, above 0")
    if plant_distance_cm is not None and not plant_distance_cm > 0:
        raise SystemExit("--plant-distance-cm, if given, must be a tape reading above 0")
    cam = RoverCamera(rover)
    ok, frame = cam.read()
    if not ok:
        raise SystemExit(f"no frame from {cam.base} - is the Pi up and a camera server "
                         f"running?")
    std = float(frame.std())
    if std < UNIFORM_FRAME_STD:
        raise SystemExit(f"frame is uniform (std {std:.3f}): a dead or capped camera, "
                         f"not a plant - nothing saved")
    sd = spatial_decision(frame)
    if sd["camera_fault"]:
        raise SystemExit("frame has no image along its rows - a stuck camera sensor "
                         "(seen 2026-10-21: power-cycle the Pi to clear it) - nothing saved")
    out = out_root / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "plant_frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = frame.shape[:2]
    meta = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera": f"rover:{cam.base}", "endpoint": cam.mode,
            "camera_height_above_ground_cm": height_cm,
            "lens_to_plant_cm": plant_distance_cm,
            "height_source": "tape measure (entered by the operator)",
            "note": note, "width": w, "height": h,
            "frame_std": round(std, 3), "mean_brightness": round(float(frame.mean()), 1),
            "tissue_frac": round(sd["tissue_frac"], 4),
            "tissue_zones": [round(z, 4) for z in sd["zones"]],
            "deferred_too_dark": sd["deferred"],
            "spatial_policy_would_spray": sd["fire"]}
    (out / "artifact.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rover", default="192.168.137.167:8080")
    ap.add_argument("--height-cm", type=float, required=True,
                    help="centre of the camera lens above the ground, measured with a tape")
    ap.add_argument("--plant-distance-cm", type=float,
                    help="optional: lens to the nearest leaf in view, measured with a tape")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "rover_artifacts")
    a = ap.parse_args()
    out = capture(a.rover, a.height_cm, a.note, a.out, a.plant_distance_cm)
    meta = json.loads((out / "artifact.json").read_text(encoding="utf-8"))
    print(f"saved {out / 'plant_frame.jpg'} ({meta['width']}x{meta['height']}, "
          f"from {meta['endpoint']})")
    print(f"lens {meta['camera_height_above_ground_cm']} cm above ground (tape); tissue "
          f"{meta['tissue_frac']:.0%} of the frame; spatial policy would spray: "
          f"{meta['spatial_policy_would_spray']}"
          + ("  [frame too dark to judge]" if meta["deferred_too_dark"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
