"""An unlabelled session on the rover: what the trained system says about a plant, and
what it estimates its own accuracy to be - with no human label anywhere.

This is the deployment question the thesis asks, put to the robot: given unlabelled
field images, can the software tell how well its classifier is doing? Each frame from
the rover's camera (banner-free /raw) goes through the deployed decision rule
(demo/live_sprayer.SprayPipeline: classifier, abstention, too-dark and camera-fault
deferral, both spray policies). Then each of the eight label-free accuracy estimators of
the thesis's standard set, fitted on the laboratory source-validation split exactly as in
the thesis, estimates the classifier's accuracy over the frames the system did judge.
(The ninth, GdScore, needs a fresh forward pass in train() mode - docs/gdscore_eligibility.md
section 4 - which a live sprayer does not make, so it is not run here.)

No estimate can be checked here - there are no labels, by design. The thesis measured
how far these estimators err under laboratory-to-field shift (best 0.1337 absolute;
none better than an oracle constant), and that is the yardstick for reading them.

    python scripts/unlabelled_rover_session.py capture --frames 60 --interval 1.0
    python scripts/unlabelled_rover_session.py estimate camera_sessions/<folder>
    python scripts/unlabelled_rover_session.py estimate --images "camera_sessions/*/frames/*.jpg"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "src")]

from estimators.base import SourceStats  # noqa: E402
from estimators.methods import ALL_ESTIMATORS, build  # noqa: E402
from inference.cache import CacheKey, load_features, load_labels  # noqa: E402
from live_sprayer import (BACKBONE, CLASSES, LABEL_SET, SEED, VERSION,  # noqa: E402
                          SprayPipeline)
from models.backbones import forward_features_and_logits  # noqa: E402
from rover_camera import RoverCamera  # noqa: E402

#: The thesis's measured yardstick for these estimators under severe shift.
THESIS_BEST_MAE, THESIS_ORACLE_CONSTANT_MAE = 0.1337, 0.0610


def _features_and_logits(pipe: SprayPipeline, rgb: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = pipe.tf(rgb).unsqueeze(0).to(pipe.device)
        feats, logits = forward_features_and_logits(pipe.model, x)
    return feats.float().cpu().numpy()[0], logits.float().cpu().numpy()[0]


def capture(rover: str, n: int, interval: float, out: Path) -> Path:
    pipe = SprayPipeline(device="cpu")
    cam = RoverCamera(rover)
    if not cam.isOpened():
        raise SystemExit(f"no frame from {cam.base}")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    saved = 0
    while saved < n:
        ok, bgr = cam.read()
        if not ok:
            time.sleep(0.5)
            continue
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        feats, lg = _features_and_logits(pipe, rgb)
        d = pipe.decide(rgb)
        saved += 1
        name = f"frames/unlabelled_{saved:04d}.jpg"
        cv2.imwrite(str(out / name), bgr)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "frame_file": name,
               "source": f"rover:{cam.base} ({cam.mode})", "truth": "",
               "model": f"{VERSION}/{BACKBONE}/seed{SEED}", "tau": pipe.tau,
               "logits": [float(v) for v in lg], "features_file": None,
               **{k: v for k, v in d.__dict__.items() if k != "nozzles"},
               "nozzles": list(d.nozzles), "spray": d.spray}
        np.save(out / "frames" / f"unlabelled_{saved:04d}_features.npy", feats)
        rec["features_file"] = f"frames/unlabelled_{saved:04d}_features.npy"
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{saved:3d}/{n}  pred {d.pred:13s} {d.msp:.2f}  "
              f"{'DEFERRED' if d.deferred else ('spray' if d.spray_spatial else 'no tissue')}",
              flush=True)
        time.sleep(interval)
    log.close()
    return out


def _source() -> SourceStats:
    k = CacheKey(VERSION, BACKBONE, SEED, "source_val", LABEL_SET)
    s = load_features(k)
    return SourceStats(logits=s["logits"], labels=load_labels(k), features=s["features"])


def _from_images(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    pipe = SprayPipeline(device="cpu")
    L, F, D = [], [], []
    for p in paths:
        rgb = Image.open(p).convert("RGB")
        f, lg = _features_and_logits(pipe, rgb)
        d = pipe.decide(rgb)
        L.append(lg), F.append(f)
        D.append({"pred": d.pred, "msp": d.msp, "abstain": d.abstain, "deferred": d.deferred,
                  "camera_fault": d.camera_fault, "spray": d.spray,
                  "spray_spatial": d.spray_spatial, "session_estimate": d.session_estimate})
    return np.array(L), np.array(F), D


def estimate(session: Path | None, images: str | None, out_json: Path | None) -> dict:
    if session is not None:
        recs = [json.loads(x) for x in (session / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        L = np.array([r["logits"] for r in recs])
        # The feature files sit beside the frames, which stay out of git; the eight
        # estimators run here need only the logits, which are in the log itself.
        fp = [session / r["features_file"] for r in recs]
        F = np.array([np.load(p) for p in fp]) if all(p.exists() for p in fp) else None
        D = recs
        what = str(session)
    else:
        paths = sorted(glob.glob(images))
        L, F, D = _from_images(paths)
        what = f"{len(paths)} images: {images}"
    judged = np.array([not d["deferred"] for d in D])
    src = _source()
    res = {"what": what, "frames": len(D), "judged": int(judged.sum()),
           "deferred_too_dark_or_fault": int((~judged).sum()),
           "predictions": dict(collections.Counter(d["pred"] for d in D)),
           "confident_healthy_calls": sum(d["pred"] == "healthy" and not d["abstain"] for d in D),
           "mean_confidence": round(float(np.mean([d["msp"] for d in D])), 3),
           "abstain_rate": round(float(np.mean([d["abstain"] for d in D])), 3),
           "spatial_sprays": sum(bool(d["spray_spatial"]) for d in D),
           "gated_sprays": sum(bool(d["spray"]) for d in D),
           "gated_equals_spatial": sum(bool(d["spray"]) == bool(d["spray_spatial"]) for d in D),
           "estimates": {}}
    if judged.sum() >= 2:
        for name in ALL_ESTIMATORS:
            try:
                est = build(name).fit(src)
                res["estimates"][name] = round(
                    est.estimate(L[judged], None if F is None else F[judged]), 3)
            except Exception as e:  # noqa: BLE001 - report, do not hide, a failing estimator
                res["estimates"][name] = f"failed: {type(e).__name__}: {e}"
    if out_json:
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def show(res: dict) -> None:
    print(f"\n{res['what']}")
    print(f"frames {res['frames']}, judged {res['judged']}, deferred {res['deferred_too_dark_or_fault']}")
    print(f"classifier calls: {res['predictions']}; confident 'healthy' calls: "
          f"{res['confident_healthy_calls']}; mean confidence {res['mean_confidence']}; "
          f"abstained {res['abstain_rate']:.0%}")
    print(f"spray: spatial-only {res['spatial_sprays']}, classifier-gated {res['gated_sprays']}; "
          f"same decision on {res['gated_equals_spatial']}/{res['frames']} frames")
    print("label-free estimates of the classifier's accuracy on these frames:")
    for k, v in res["estimates"].items():
        print(f"  {k:14s} {v}")
    print(f"(no label exists to check them; under laboratory-to-field shift the thesis measured "
          f"the best of these estimators wrong by {THESIS_BEST_MAE} on average, and none beat a "
          f"constant guess ({THESIS_ORACLE_CONSTANT_MAE}))")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--rover", default="192.168.137.167:8080")
    c.add_argument("--frames", type=int, default=60)
    c.add_argument("--interval", type=float, default=1.0)
    c.add_argument("--out", type=Path,
                   default=ROOT / "camera_sessions" / time.strftime("unlabelled_%Y%m%d_%H%M%S"))
    e = sub.add_parser("estimate")
    e.add_argument("session", type=Path, nargs="?")
    e.add_argument("--images")
    e.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "capture":
        out = capture(a.rover, a.frames, a.interval, a.out)
        res = estimate(out, None, out / "unlabelled_estimates.json")
    else:
        res = estimate(a.session, a.images, a.out)
    show(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
