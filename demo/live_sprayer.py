"""Live selective-sprayer demo: camera -> classifier -> reliability -> FSM -> decision.

This is the deployment demonstration of the software stack. It runs the SAME
components the rover would run - the deployable classifier, the per-image
abstention rule, the session-level reliability gate, and the real FSM
controller from `rover/sprayer_fsm.py` - on frames from a laptop camera, a
folder of photos, or a labelled manifest. No motor, relay or pump is driven:
the output is the decision the rover would have taken.

WHAT THIS VALIDATES, AND WHAT IT DOES NOT
-----------------------------------------
It shows live inference works end to end and lets the effect of lighting,
distance, angle, occlusion and background on predictions and decisions be
MEASURED - when frames are labelled with the true disease state. It does not
establish field accuracy, and it does not test any rover hardware.

DECISION RULE (declared before any camera frame was recorded)
-------------------------------------------------------------
Uncertainty may never REDUCE treatment. A visible leaf is left unsprayed only if
all three hold: the session gate is open, the classifier is confident on this
frame (max-softmax >= tau), and it predicts healthy. Otherwise the zone is
sprayed. tau is fixed from the source validation split - the smallest threshold
at which confident predictions there are >= 99% accurate - and never tuned on
camera data. The session gate is a dataset-level estimate over the last 25
frames; on one leaf held in front of a camera it is a demonstration of the
mechanism, not a meaningful accuracy estimate, because the frames are not
different plants.

MODES
-----
    python demo/live_sprayer.py --verify 300           # parity with cached predictions
    python demo/live_sprayer.py --camera 0             # interactive, laptop webcam
    python demo/live_sprayer.py --manifest frames.csv  # batch: path,truth,leaf_id,condition

Interactive keys: h / e / l set the true label (healthy / early / late blight),
1-6 set the condition, n starts a new leaf, SPACE saves a labelled frame, q quits.
Nothing is written to disk except on SPACE.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rover"))
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))

from data import paths                                             # noqa: E402
from data.mapping import LABEL_SETS                                # noqa: E402
from estimators.base import SourceStats                            # noqa: E402
from estimators.methods import build                               # noqa: E402
from inference.cache import CacheKey, load_features, load_labels   # noqa: E402
from models.backbones import create_model                          # noqa: E402
from sprayer_fsm import FusedReading, SprayerController, SprayPolicy  # noqa: E402
from training.dataset import eval_transform                        # noqa: E402

#: The deployable model: smallest of the four architectures, practical on a Pi.
VERSION, BACKBONE, SEED, LABEL_SET = "V9", "mobilenetv3_large", 0, "SHARED_3"
CLASSES = LABEL_SETS[LABEL_SET]
HEALTHY = CLASSES.index("healthy")
N_NOZZLES = 4
SESSION_LEN = 25
TAU_SAFE = 0.80
SOURCE_ACC_TARGET = 0.99
CONDITIONS = {"1": "normal", "2": "low_light", "3": "harsh_light",
              "4": "distance_angle", "5": "occlusion", "6": "background"}
TRUTH_KEYS = {"h": "healthy", "e": "early_blight", "l": "late_blight"}
HARDWARE_TAG = "NONE - decision only, no actuator driven"


#: Brightness gain the tissue test may apply; a frame needing more is deferred.
MAX_GAIN = 4.0
#: Median V (0-255) below which a frame is too dark to judge: 128 / MAX_GAIN.
MIN_EXPOSURE = 128.0 / MAX_GAIN


def foliage_mask(hsv: np.ndarray) -> np.ndarray:
    """Leaf tissue of any plant colour - green, yellow, orange-brown, red-brown.

    A green-only test was the first version, and the dry run showed it misses
    late blight, which turns leaves brown: the sicker the leaf, the less green,
    the less likely to be sprayed. Blue/purple hues and low-saturation pixels
    (grey, white, black backdrops) are excluded.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    # Exposure-invariant: judge tissue on a brightness-normalised V channel, so
    # a leaf in dim light is still a leaf. Frames needing more than MAX_GAIN are
    # deferred by the caller rather than amplified into noise.
    gain = min(MAX_GAIN, 128.0 / max(1.0, float(np.median(v))))
    v = np.minimum(255.0, v * gain)
    return ((h <= 95) | (h >= 170)) & (s >= 40) & (v >= 25)


@dataclass
class Decision:
    pred: str
    probs: list[float]
    p_diseased: float
    msp: float
    margin: float
    abstain: bool
    session_estimate: float | None
    gate_open: bool
    nozzles: tuple[bool, ...]
    fsm_state: str
    canopy: list[float]
    blank: bool
    deferred: bool
    spray_spatial: bool
    latency_ms: float

    @property
    def spray(self) -> bool:
        return any(self.nozzles)


class SprayPipeline:
    """The deployed software stack, minus actuation."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        ckpt = torch.load(paths.RAW_RESULTS / VERSION / f"{BACKBONE}_seed{SEED}.pt",
                          map_location="cpu", weights_only=False)
        bundle = create_model(BACKBONE, n_classes=len(CLASSES), pretrained=False)
        bundle.model.load_state_dict(ckpt["state_dict"])
        self.model = bundle.model.eval().to(device)
        self.tf = eval_transform()

        sk = CacheKey(VERSION, BACKBONE, SEED, "source_val", LABEL_SET)
        s = load_features(sk)
        y = load_labels(sk)
        self.estimator = build("atc_mc").fit(
            SourceStats(logits=s["logits"], labels=y, features=s["features"]))
        self.tau = self._abstention_threshold(s["logits"], y)
        self.session: collections.deque = collections.deque(maxlen=SESSION_LEN)
        self.ctl = SprayerController(policy=SprayPolicy.SELECTIVE, n_nozzles=N_NOZZLES)

    @staticmethod
    def _abstention_threshold(logits: np.ndarray, y: np.ndarray) -> float:
        """Smallest max-softmax threshold at which confident source-validation
        predictions are >= 99% accurate. Fixed here; never tuned on camera data."""
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        msp = p.max(1)
        ok = p.argmax(1) == y
        for t in np.sort(np.unique(msp)):
            keep = msp >= t
            if keep.sum() and ok[keep].mean() >= SOURCE_ACC_TARGET:
                return float(t)
        return 1.0

    @torch.no_grad()
    def logits(self, rgb: Image.Image) -> np.ndarray:
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        return self.model(x).float().cpu().numpy()[0]

    def decide(self, rgb: Image.Image) -> Decision:
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        blank = float(bgr.std()) < 0.5        # the dead-camera check
        lg = self.logits(rgb)
        z = lg - lg.max()
        p = np.exp(z) / np.exp(z).sum()
        ps = np.sort(p)
        msp, margin = float(ps[-1]), float(ps[-1] - ps[-2])
        p_dis = float(1.0 - p[HEALTHY])
        abstain = msp < self.tau

        self.session.append(lg)
        est = None
        gate = False
        if len(self.session) >= SESSION_LEN:
            est = float(self.estimator.estimate(np.asarray(self.session)))
            gate = est >= TAU_SAFE

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        deferred = blank or float(np.median(hsv[..., 2])) < MIN_EXPOSURE
        veg = foliage_mask(hsv)
        canopy = [float(b.mean()) for b in np.array_split(veg, N_NOZZLES, axis=1)]

        # Uncertainty never reduces treatment: an unsure frame is treated as
        # possibly diseased, so only a confident "healthy" can withhold spray.
        dis = 1.0 if abstain else p_dis
        reading = FusedReading(range_cm=150.0, range_age=0, row_offset=0.0, row_age=0,
                               canopy=tuple(canopy), disease_confidence=dis,
                               vision_age=99 if deferred else 0, tank_fraction=1.0,
                               battery_fraction=1.0, resource_age=0,
                               disease_by_zone=(dis,) * N_NOZZLES)
        if self.ctl.state.name == "IDLE":
            self.ctl.step(reading, gate_open=False)
        cmd = self.ctl.step(reading, gate_open=gate)
        # Second policy, logged beside the first: spray every zone with leaf
        # tissue, and never let a classifier call withhold it.
        spatial = (not deferred) and any(c >= 0.15 for c in canopy)
        return Decision(pred=CLASSES[int(p.argmax())], probs=[float(v) for v in p],
                        p_diseased=p_dis, msp=msp, margin=margin, abstain=abstain,
                        session_estimate=est, gate_open=gate, nozzles=cmd.nozzles,
                        fsm_state=cmd.state.name, canopy=canopy, blank=blank,
                        deferred=deferred, spray_spatial=spatial, latency_ms=(time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------
# overlay
# --------------------------------------------------------------------------
def draw(frame: np.ndarray, d: Decision, truth: str, cond: str, leaf: str,
         tau: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for k, on in enumerate(d.nozzles):
        x0, x1 = k * w // N_NOZZLES, (k + 1) * w // N_NOZZLES
        cv2.rectangle(out, (x0 + 3, h - 28), (x1 - 3, h - 6),
                      (40, 40, 220) if on else (120, 120, 120), -1)
    lines = [
        f"pred {d.pred}  conf {d.msp:.2f} (tau {tau:.2f}) {'ABSTAIN' if d.abstain else ''}",
        f"P(diseased) {d.p_diseased:.2f}   margin {d.margin:.2f}",
        f"session est {'-' if d.session_estimate is None else f'{d.session_estimate:.2f}'}"
        f"  gate {'OPEN' if d.gate_open else 'shut'}  FSM {d.fsm_state}",
        f"DECISION: {'SPRAY ' + str(sum(d.nozzles)) + '/4' if d.spray else 'no spray'}"
        f"{'  [TOO DARK - DEFERRED]' if d.deferred else ''}   {d.latency_ms:.0f} ms",
        f"leaf {leaf}  truth {truth or '?'}  condition {cond}   [h/e/l 1-6 n SPACE q]",
    ]
    cv2.rectangle(out, (0, 0), (w, 20 * len(lines) + 8), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
class SessionLog:
    def __init__(self, out_dir: Path, source: str) -> None:
        self.dir = out_dir
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.f = (self.dir / "log.jsonl").open("a", encoding="utf-8")
        self.source = source
        self.n = 0

    def write(self, d: Decision, *, truth: str, cond: str, leaf: str,
              frame_bgr: np.ndarray | None, src_path: str | None, tau: float) -> None:
        self.n += 1
        frame_file = None
        if frame_bgr is not None:
            frame_file = f"frames/{leaf}_{cond}_{self.n:04d}.jpg"
            cv2.imwrite(str(self.dir / frame_file), frame_bgr)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": self.source,
               "hardware": HARDWARE_TAG, "leaf_id": leaf, "condition": cond,
               "truth": truth, "frame_file": frame_file, "src_path": src_path,
               "model": f"{VERSION}/{BACKBONE}/seed{SEED}", "tau": tau,
               **{k: v for k, v in d.__dict__.items()}, "spray": d.spray,
               "correct": (truth == d.pred) if truth else None}
        rec["nozzles"] = list(rec["nozzles"])
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def run_verify(n: int) -> int:
    """Parity: the demo's preprocessing and model must reproduce the cached
    predictions the thesis results were computed from."""
    pipe = SprayPipeline(device="cpu")
    k = CacheKey(VERSION, BACKBONE, SEED, "source_test", LABEL_SET)
    cache = load_features(k)
    y = load_labels(k)
    idx = np.linspace(0, len(y) - 1, min(n, len(y))).astype(int)
    agree, diffs, lat, correct = 0, [], [], 0
    for i in idx:
        with Image.open(str(cache["paths"][i])) as im:
            rgb = im.convert("RGB")
        t0 = time.perf_counter()
        lg = pipe.logits(rgb)
        lat.append((time.perf_counter() - t0) * 1000)
        agree += int(lg.argmax() == cache["logits"][i].argmax())
        correct += int(lg.argmax() == y[i])
        diffs.append(float(np.abs(lg - cache["logits"][i]).max()))
    res = {"images": int(len(idx)), "argmax_agreement": agree / len(idx),
           "max_abs_logit_diff": max(diffs), "median_abs_logit_diff": float(np.median(diffs)),
           "accuracy_vs_truth": correct / len(idx), "tau": pipe.tau,
           "median_latency_ms_cpu_this_laptop": float(np.median(lat)),
           "note": "cache was computed on GPU in fp16; the demo runs fp32 on CPU"}
    out = ROOT / "results" / "camera_demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    ok = res["argmax_agreement"] >= 0.995
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def run_manifest(manifest: Path, out_dir: Path) -> int:
    pipe = SprayPipeline(device="cpu")
    log = SessionLog(out_dir, source=f"manifest:{manifest.name}")
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        with Image.open(r["path"]) as im:
            rgb = im.convert("RGB")
        d = pipe.decide(rgb)
        log.write(d, truth=r.get("truth", ""), cond=r.get("condition", "normal"),
                  leaf=r.get("leaf_id", "?"), frame_bgr=None, src_path=r["path"],
                  tau=pipe.tau)
    print(f"logged {log.n} frames to {out_dir / 'log.jsonl'}")
    return 0


def run_camera(index: int, out_dir: Path) -> int:
    pipe = SprayPipeline(device="cpu")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera {index} could not be opened")
        return 1
    log = SessionLog(out_dir, source=f"camera:{index}")
    truth, cond, leaf_n = "", "normal", 1
    print("keys: h/e/l truth, 1-6 condition, n new leaf, SPACE save, q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera returned no frame")
                break
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = pipe.decide(rgb)
            leaf = f"L{leaf_n:02d}"
            cv2.imshow("selective sprayer - decision only", draw(frame, d, truth, cond,
                                                                 leaf, pipe.tau))
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "q":
                break
            if key in TRUTH_KEYS:
                truth = TRUTH_KEYS[key]
            elif key in CONDITIONS:
                cond = CONDITIONS[key]
            elif key == "n":
                leaf_n += 1
            elif key == " ":
                log.write(d, truth=truth, cond=cond, leaf=leaf, frame_bgr=frame,
                          src_path=None, tau=pipe.tau)
                print(f"saved #{log.n}: {leaf} {cond} truth={truth or '?'} pred={d.pred} "
                      f"conf={d.msp:.2f} spray={d.spray}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"saved {log.n} labelled frames to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", type=int, metavar="N")
    g.add_argument("--camera", type=int, metavar="INDEX")
    g.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "camera_sessions" / time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    if a.verify is not None:
        return run_verify(a.verify)
    if a.manifest is not None:
        return run_manifest(a.manifest, a.out)
    return run_camera(a.camera, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
