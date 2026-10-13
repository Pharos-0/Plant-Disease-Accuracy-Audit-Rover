"""Open-field selective sprayer experiment - the pre-registered protocol.

Protocol: docs/openfield_protocol.md, committed 2026-10-13 11:28 BEFORE the data
existed, with deviations D1-D6 committed before any result. Nothing in this
file chooses a threshold; every constant is copied from the protocol.

    python scripts/openfield_sprayer.py groundtruth   # stage 1, all 938 images
    python scripts/openfield_sprayer.py features      # stage 2, the 193 images
    python scripts/openfield_sprayer.py detector      # stage 3
    python scripts/openfield_sprayer.py spray         # stage 4
    python scripts/openfield_sprayer.py all
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rover"))
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))

from data import paths                                             # noqa: E402

DATA = Path("C:/ds/groundbased_weed/recovered/TOMATO_1")
OUT = paths.RESULTS_DIR / "openfield"
SEED = 20261013

# ---- protocol constants (docs/openfield_protocol.md) -----------------------
GX, GY = 16, 12                        # sub-cell grid
LAYOUTS = {"4x3": (4, 3), "8x6": (8, 6), "16x12": (16, 12)}
PRIMARY = "4x3"
CELL_FRAC = 0.05                       # box covers >= 5% of the sub-cell, or centre inside
BLOCK = 20                             # split blocks
TRAIN_FRAC = 0.70
CANOPY_THRESHOLD = 0.15                # vision-only
CROP_THRESHOLD = 0.5                   # detector
TAU_SAFE, MIN_SESSION = 0.80, 25       # gate
N_PAIRINGS = 10
WORK = (3584, 2688)                    # landscape working size: each sub-cell 224 x 224
CROP = "LYPES"


# ===========================================================================
# stage 1: ground truth per sub-cell, from Pascal VOC
# ===========================================================================
def read_voc(x: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    r = ET.parse(x).getroot()
    w, h = int(r.find("size/width").text), int(r.find("size/height").text)
    boxes = []
    for o in r.iter("object"):
        b = o.find("bndbox")
        boxes.append((o.find("name").text,
                      *(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return w, h, boxes


def to_landscape(w, h, boxes):
    """D4: rotate portrait 90 deg counter-clockwise, as np.rot90(img, 1) does.
    A pixel at (row=y, col=x) moves to (row=w-1-x, col=y)."""
    if h <= w:
        return w, h, boxes
    out = [(n, y0, w - x1, y1, w - x0) for n, x0, y0, x1, y1 in boxes]
    return h, w, out


def grid_truth(w, h, boxes):
    """Per sub-cell: crop mask, non-crop-vegetation mask, and the tomato ids in it."""
    cw, ch = w / GX, h / GY
    crop = np.zeros((GY, GX), bool)
    other = np.zeros((GY, GX), bool)
    tom = [[[] for _ in range(GX)] for _ in range(GY)]
    tid = 0
    for name, x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, x0), min(float(w), x1)            # D5: clip
        y0, y1 = max(0.0, y0), min(float(h), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        is_crop = name == CROP
        for r in range(GY):
            for c in range(GX):
                ix = max(0.0, min(x1, (c + 1) * cw) - max(x0, c * cw))
                iy = max(0.0, min(y1, (r + 1) * ch) - max(y0, r * ch))
                inside = (c * cw <= cx < (c + 1) * cw) and (r * ch <= cy < (r + 1) * ch)
                if ix * iy >= CELL_FRAC * cw * ch or inside:
                    if is_crop:
                        crop[r, c] = True
                        tom[r][c].append(tid)
                    else:                                       # weeds and NR (D6)
                        other[r, c] = True
        if is_crop:
            tid += 1
    return crop, other, tom, tid


def stage_groundtruth() -> None:
    xmls = sorted((DATA / "Annotations").glob("*.xml"))
    have = {p.stem for p in (DATA / "Images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg")}
    rec = {}
    for x in xmls:
        w, h, boxes = to_landscape(*read_voc(x))
        crop, other, tom, n = grid_truth(w, h, boxes)
        rec[x.stem] = {"w": w, "h": h, "crop": crop.tolist(), "other": other.tolist(),
                       "tom": tom, "n_tomato": n, "has_pixels": x.stem in have}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "groundtruth.json").write_text(json.dumps(rec), encoding="utf-8")
    pix = sum(v["has_pixels"] for v in rec.values())
    crop_frac = np.mean([np.mean(v["crop"]) for v in rec.values()])
    print(f"ground truth: {len(rec)} images ({pix} with pixels); "
          f"crop sub-cells {crop_frac:.1%} of the field")


def split(stems: list[str]) -> dict[str, str]:
    """Pre-registered: blocks of 20 in filename order, 70/30, seed."""
    stems = sorted(stems)
    blocks = [stems[i:i + BLOCK] for i in range(0, len(stems), BLOCK)]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(blocks))
    n_tr = int(round(TRAIN_FRAC * len(blocks)))
    out = {}
    for rank, b in enumerate(order):
        for s in blocks[b]:
            out[s] = "train" if rank < n_tr else "test"
    return out


# ===========================================================================
# stage 2: pixel features for the images that exist
# ===========================================================================
def stage_features() -> None:
    import torch
    import timm
    from rover_relevance import _vegetation

    gt = json.loads((OUT / "groundtruth.json").read_text(encoding="utf-8"))
    stems = sorted(s for s, v in gt.items() if v["has_pixels"])
    img_by_stem = {p.stem: p for p in (DATA / "Images").glob("*")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    feats = np.zeros((len(stems), GY, GX, 2048), np.float16)
    veg = np.zeros((len(stems), GY, GX), np.float32)
    t0 = time.time()
    for i, s in enumerate(stems):
        img = cv2.imdecode(np.fromfile(str(img_by_stem[s]), np.uint8), cv2.IMREAD_COLOR)
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img, 1).copy()                      # D4, same as the boxes
        img = cv2.resize(img, WORK, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = _vegetation(hsv[..., 0], hsv[..., 1], hsv[..., 2])
        veg[i] = v.reshape(GY, 224, GX, 224).mean(axis=(1, 3))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tiles = rgb.reshape(GY, 224, GX, 224, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 224, 224, 3)
        x = torch.from_numpy(tiles).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad(), torch.autocast(dev, enabled=(dev == "cuda")):
            f = model((x - mean) / std)
        feats[i] = f.float().cpu().numpy().reshape(GY, GX, -1).astype(np.float16)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(stems)}  ({time.time() - t0:.0f} s)")
    np.savez_compressed(OUT / "features.npz", stems=np.array(stems), feats=feats, veg=veg)
    print(f"features: {len(stems)} images x {GY * GX} sub-cells")


# ===========================================================================
# stage 3: the tomato detector
# ===========================================================================
def stage_detector() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    gt = json.loads((OUT / "groundtruth.json").read_text(encoding="utf-8"))
    z = np.load(OUT / "features.npz")
    stems = list(z["stems"])
    sp = split(stems)
    X = z["feats"].reshape(len(stems), -1, 2048).astype(np.float32)
    Y = np.array([np.array(gt[s]["crop"]).reshape(-1) for s in stems])
    tr = np.array([sp[s] == "train" for s in stems])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    clf.fit(X[tr].reshape(-1, 2048), Y[tr].reshape(-1))
    prob = clf.predict_proba(X.reshape(-1, 2048))[:, 1].reshape(len(stems), GY, GX)
    te = ~tr
    yt, pt = Y[te].reshape(-1), prob[te].reshape(-1)
    rep = {"train_images": int(tr.sum()), "test_images": int(te.sum()),
           "test_subcells": int(len(yt)), "test_crop_subcells": int(yt.sum()),
           "precision": float(precision_score(yt, pt >= CROP_THRESHOLD)),
           "recall": float(recall_score(yt, pt >= CROP_THRESHOLD)),
           "average_precision": float(average_precision_score(yt, pt)),
           "prevalence": float(yt.mean())}
    np.savez_compressed(OUT / "detector.npz", stems=np.array(stems), prob=prob,
                        split=np.array([sp[s] for s in stems]))
    (OUT / "detector_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("TOMATO DETECTOR - held-out test sub-cells")
    print(f"  train {rep['train_images']} images, test {rep['test_images']} images, "
          f"{rep['test_subcells']} test sub-cells ({rep['prevalence']:.1%} tomato)")
    print(f"  precision {rep['precision']:.3f}  recall {rep['recall']:.3f}  "
          f"AP {rep['average_precision']:.3f}")


# ===========================================================================
# stage 4: spraying
# ===========================================================================
def aggregate(a: np.ndarray, cols: int, rows: int, how: str) -> np.ndarray:
    """Sub-cell grid (GY, GX) -> nozzle zones (rows, cols)."""
    by, bx = GY // rows, GX // cols
    r = a.reshape(rows, by, cols, bx)
    return r.max(axis=(1, 3)) if how == "max" else r.mean(axis=(1, 3))


def frozen_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["label_set"] == "SHARED_3" and c["version"] in ("V2", "V9"):
            out.append(c)
    return out


def stage_spray() -> None:
    from data.mapping import LABEL_SETS
    from estimators.base import SourceStats
    from estimators.methods import build
    from inference.cache import CacheKey, load_features, load_labels
    from sprayer_fsm import FusedReading, SprayerController, SprayPolicy

    gt = json.loads((OUT / "groundtruth.json").read_text(encoding="utf-8"))
    z = np.load(OUT / "features.npz")
    d = np.load(OUT / "detector.npz")
    stems = list(z["stems"])
    test = [i for i, s in enumerate(stems) if d["split"][i] == "test"]
    healthy = LABEL_SETS["SHARED_3"].index("healthy")
    rng = np.random.default_rng(SEED)

    # per-image, per-layout, per-policy counts, summed over every run
    keys = ("fired", "need", "hit", "fired_other", "fired_bare", "zones")
    acc = {(lay, pol, i): dict.fromkeys(keys, 0)
           for lay in LAYOUTS for pol in ("manual", "vision_only", "proposed", "ceiling")
           for i in test}

    for c in frozen_models():
        k = CacheKey(c["version"], c["backbone"], c["seed"], "tomatovillage", c["label_set"])
        f = load_features(k)
        y = load_labels(k)
        lg = f["logits"]
        e = np.exp(lg - lg.max(1, keepdims=True))
        p_dis = 1.0 - (e / e.sum(1, keepdims=True))[:, healthy]
        sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val", c["label_set"])
        s = load_features(sk)
        est = build("atc_mc").fit(SourceStats(logits=s["logits"], labels=load_labels(sk),
                                              features=s["features"]))
        for _rep in range(N_PAIRINGS):
            # One pairing per (run, image), shared by every layout, so layout
            # differences are purely geometric rather than partly re-drawn disease.
            picks = {i: rng.choice(len(y), max(1, gt[stems[i]]["n_tomato"]), replace=True)
                     for i in test}
            fp_dis = {i: rng.choice(p_dis, (GY, GX)) for i in test}
            for lay, (cols, rows) in LAYOUTS.items():
                ctl = {pol: SprayerController(policy=pp, n_nozzles=cols)
                       for pol, pp in (("vision_only", SprayPolicy.VISION_ONLY),
                                       ("proposed", SprayPolicy.SELECTIVE),
                                       ("ceiling", SprayPolicy.SELECTIVE))}
                session: list[np.ndarray] = []
                for i in test:
                    g = gt[stems[i]]
                    n_tom = g["n_tomato"]
                    pick = picks[i]
                    tom_dis_true = (y[pick] != healthy)[:n_tom]
                    tom_dis_pred = p_dis[pick][:n_tom]
                    for j in range(n_tom):
                        session.append(lg[pick[j]])
                    gate = (len(session) >= MIN_SESSION
                            and est.estimate(np.asarray(session)) >= TAU_SAFE)

                    crop_s = np.array(g["crop"])
                    other_s = np.array(g["other"])
                    need_s = np.zeros((GY, GX), bool)
                    pdis_s = fp_dis[i].copy()                  # false-positive cells
                    for r in range(GY):
                        for cc in range(GX):
                            ids = g["tom"][r][cc]
                            if ids:
                                need_s[r, cc] = bool(tom_dis_true[ids].any())
                                pdis_s[r, cc] = float(tom_dis_pred[ids].max())
                    zone = {
                        "crop": aggregate(crop_s, cols, rows, "max"),
                        "other": aggregate(other_s, cols, rows, "max"),
                        "need": aggregate(need_s, cols, rows, "max"),
                        "veg": aggregate(z["veg"][i], cols, rows, "mean"),
                        "prob": aggregate(d["prob"][i], cols, rows, "max"),
                        "pdis": aggregate(pdis_s, cols, rows, "max"),
                    }
                    for r in range(rows):
                        fired = {"manual": np.ones(cols, bool)}
                        base = dict(range_cm=150.0, range_age=0, row_offset=0.0, row_age=0,
                                    canopy=tuple(map(float, zone["veg"][r])),
                                    disease_confidence=0.0, vision_age=0,
                                    tank_fraction=1.0, battery_fraction=1.0, resource_age=0)
                        for pol, ct in ctl.items():
                            if ct.state.name == "IDLE":
                                ct.step(FusedReading(**base), gate_open=False)
                            if pol == "vision_only":
                                rd = FusedReading(**base)
                                g_open = False
                            elif pol == "proposed":
                                rd = FusedReading(**base,
                                                  crop_prob=tuple(map(float, zone["prob"][r])),
                                                  disease_by_zone=tuple(map(float, zone["pdis"][r])))
                                g_open = gate
                            else:   # ceiling: true crop, true disease, gate forced open
                                rd = FusedReading(**base,
                                                  crop_prob=tuple(map(float, zone["crop"][r])),
                                                  disease_by_zone=tuple(map(float, zone["need"][r])))
                                g_open = True
                            fired[pol] = np.array(ct.step(rd, gate_open=g_open).nozzles)
                        need = zone["need"][r].astype(bool)
                        crop = zone["crop"][r].astype(bool)
                        other = zone["other"][r].astype(bool) & ~crop
                        bare = ~crop & ~zone["other"][r].astype(bool)
                        for pol, fz in fired.items():
                            a = acc[(lay, pol, i)]
                            a["fired"] += int(fz.sum())
                            a["need"] += int(need.sum())
                            a["hit"] += int((fz & need).sum())
                            a["fired_other"] += int((fz & other).sum())
                            a["fired_bare"] += int((fz & bare).sum())
                            a["zones"] += cols

    rows_out = [{"layout": lay, "policy": pol, "image": stems[i], **v}
                for (lay, pol, i), v in acc.items()]
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "spray_counts.csv", index=False)

    boot = np.random.default_rng(SEED + 1)
    summ = []
    for (lay, pol), g in df.groupby(["layout", "policy"]):
        g = g.reset_index(drop=True)

        def metrics(gg):
            t = gg[["fired", "need", "hit", "fired_other", "fired_bare", "zones"]].sum()
            return (100 * t.fired / t.zones, 100 * t.hit / max(1, t.fired),
                    100 * t.hit / max(1, t.need), 100 * t.fired_other / max(1, t.fired),
                    100 * t.fired_bare / max(1, t.fired))

        point = metrics(g)
        bs = np.array([metrics(g.iloc[boot.integers(0, len(g), len(g))]) for _ in range(1000)])
        lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        names = ("use", "precision", "coverage", "offtarget_weeds", "offtarget_bare")
        rec = {"layout": lay, "policy": pol}
        for n_, p_, l_, h_ in zip(names, point, lo, hi):
            rec[n_], rec[n_ + "_lo"], rec[n_ + "_hi"] = p_, l_, h_
        summ.append(rec)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "spray_summary.csv", index=False)

    order = ["manual", "vision_only", "proposed", "ceiling"]
    for lay in LAYOUTS:
        tag = "  <- PRIMARY (pre-registered)" if lay == PRIMARY else ""
        print(f"\nLAYOUT {lay}{tag}")
        print(f"  {'policy':12s} {'use':>22s} {'precision':>22s} {'coverage':>22s} "
              f"{'on weeds':>9s} {'on soil':>8s}")
        for pol in order:
            r = s[(s.layout == lay) & (s.policy == pol)].iloc[0]
            print(f"  {pol:12s} {r.use:6.1f}% [{r.use_lo:5.1f},{r.use_hi:5.1f}] "
                  f"{r.precision:6.1f}% [{r.precision_lo:5.1f},{r.precision_hi:5.1f}] "
                  f"{r.coverage:6.1f}% [{r.coverage_lo:5.1f},{r.coverage_hi:5.1f}] "
                  f"{r.offtarget_weeds:8.1f}% {r.offtarget_bare:7.1f}%")


def stage_ceiling_bias() -> None:
    """D2: the ground-truth ceiling on the 193 vs all 938, to size the bias."""
    gt = json.loads((OUT / "groundtruth.json").read_text(encoding="utf-8"))
    prev = 0.866          # India diseased prevalence, Tomato-Village SHARED_3
    print("\nD2 - HOW UNREPRESENTATIVE IS THE 193? (ceiling = crop zones on diseased plants)")
    for lay, (cols, rows) in LAYOUTS.items():
        res = {}
        for label, keep in (("193 with pixels", lambda v: v["has_pixels"]),
                            ("all 938", lambda v: True)):
            fr = [aggregate(np.array(v["crop"]), cols, rows, "max").mean()
                  for v in gt.values() if keep(v)]
            res[label] = 100 * np.mean(fr)
        print(f"  {lay:6s} crop zones: 193 -> {res['193 with pixels']:5.1f}%   "
              f"938 -> {res['all 938']:5.1f}%   (expected ceiling use ~ x{prev})")
    (OUT / "ceiling_bias.json").write_text(json.dumps({
        lay: {lbl: float(100 * np.mean([aggregate(np.array(v["crop"]), c_, r_, "max").mean()
                                         for v in gt.values() if (v["has_pixels"] or lbl == "all")]))
              for lbl in ("193", "all")} for lay, (c_, r_) in LAYOUTS.items()}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("groundtruth", "all"):
        stage_groundtruth()
    if stage in ("features", "all"):
        stage_features()
    if stage in ("detector", "all"):
        stage_detector()
    if stage in ("spray", "all"):
        stage_spray()
        stage_ceiling_bias()
    if stage == "bias":
        stage_ceiling_bias()
    print(f"\ndone in {time.time() - t0:.0f} s")
