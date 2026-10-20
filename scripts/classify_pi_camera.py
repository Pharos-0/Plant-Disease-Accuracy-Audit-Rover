"""Run the disease classifier on the rover's live camera, from the laptop.

A read-out, not a controller. It pulls single frames from the Pi's camera server
(/snapshot) and runs the same MobileNetV3 classifier the thesis uses, printing the
prediction, confidence and P(diseased) for each frame, alongside the Pi's own
leaf-tissue fraction. It never drives the rover - the rover's automation is spatial
(spray where tissue is), by design, because the whole project finding is that this
classifier is not reliable enough under field shift to gate treatment.

The classifier is trained on tomato leaves (healthy / early blight / late blight),
so its output is only meaningful when the camera is pointed at a tomato leaf; on an
arbitrary scene it still emits one of the three classes, meaninglessly.

Frames are read banner-free (demo/rover_camera.py): the servers stamp a status
banner on /snapshot, which would otherwise reach the classifier. This is a read-out
only; for labelled, analysable frames use demo/live_sprayer.py --camera <rover URL>.

    python scripts/classify_pi_camera.py --pi 192.168.137.167:8080
    python scripts/classify_pi_camera.py --pi 192.168.137.167:8080 --interval 2 --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from live_sprayer import CLASSES, SprayPipeline  # noqa: E402
from rover_camera import RoverCamera  # noqa: E402


def _get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pi", default="192.168.137.167:8080",
                    help="host:port of the rover control/camera server")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between frames")
    ap.add_argument("--n", type=int, default=0, help="number of frames (0 = run until Ctrl-C)")
    a = ap.parse_args()
    cam = RoverCamera(a.pi)
    base = cam.base
    if not cam.isOpened():
        print(f"no frame from {base} - is the Pi on the hotspot and a camera server running?")
        return 1
    print(f"rover camera {base}, frames from {cam.mode}")

    print("loading the classifier (MobileNetV3, the deployable model)...")
    pipe = SprayPipeline(device="cpu")
    print(f"ready. classes: {CLASSES}. abstention tau={pipe.tau:.3f}")
    print("pointing the rover camera at a tomato leaf is what makes this meaningful.\n")
    print(f"{'time':8s} {'prediction':13s} {'conf':>5s} {'P(dis)':>6s} {'abstain':>7s} "
          f"{'tissue%':>7s}  probs")

    i = 0
    try:
        while a.n == 0 or i < a.n:
            i += 1
            ok, bgr = cam.read()
            if not ok:
                print("  frame fetch failed, retrying")
                time.sleep(a.interval)
                continue
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            d = pipe.decide(img)
            try:
                # the Pi's own tissue fraction, which only the control server reports
                tissue = json.loads(_get(f"{base}/telemetry").decode())["tissue_frac"] * 100
            except Exception:  # noqa: BLE001
                tissue = float("nan")
            probs = " ".join(f"{c[:5]}={p:.2f}" for c, p in zip(CLASSES, d.probs))
            print(f"{time.strftime('%H:%M:%S'):8s} {d.pred:13s} {d.msp:5.2f} "
                  f"{d.p_diseased:6.2f} {'YES' if d.abstain else 'no':>7s} {tissue:7.0f}  {probs}")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
