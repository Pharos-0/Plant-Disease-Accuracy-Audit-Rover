"""Title-and-abstract screening for the systematic search.

Applies the inclusion and exclusion criteria in `docs/litsearch_protocol.md`
sections 6-7 to every deduplicated record, and records **one decision with one
reason per record** so the screening is auditable rather than a black box.

This is a *triage*, not a verdict. It exists to reduce tens of thousands of
retrieved records to a set small enough to read. Everything it marks `include`
or `adjacent` is then read by hand; the excluded set is sampled to check the
rules are not discarding relevant work. Both steps are reported.

Separate from `scripts/litsearch.py` so retrieval can be re-run without
re-deciding anything, and so a rule change re-screens the same corpus.

    python scripts/litsearch_screen.py
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from utils.manifest import write_experiment_manifest

OUT = paths.RESULTS_DIR / "litsearch"


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return re.sub(r"[^a-z0-9]+", "", t)


def rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts), re.I)


# -- the task: estimating a fixed model's performance on UNLABELLED data ----
TASK = rx(
    r"unsupervised accuracy estimation", r"label[- ]free .{0,30}(estimation|evaluation)",
    r"automatic model evaluation", r"autoeval",
    r"accuracy (estimation|prediction) .{0,40}(unlabel|without label|distribution shift)",
    r"(estimat|predict)\w* .{0,40}\b(accuracy|performance|error rate)\b.{0,60}"
    r"(unlabel|without (any )?label|no label|target domain|unseen|out.of.distribution)",
    r"(unlabel\w+|without label\w*) .{0,40}(accuracy|performance) (estimation|prediction)",
    r"unsupervised (model )?(evaluation|risk estimation)",
    r"performance estimation .{0,30}(without|unlabel)",
    r"model performance (monitoring|estimation) .{0,40}unlabel",
)
#: named methods in this literature - a strong positive signal on their own
METHODS = rx(
    r"\bATC\b.{0,60}(confidence|accuracy)", r"average thresholded confidence",
    r"difference of confidences?\b", r"\bDoC\b.{0,40}(accuracy|shift)",
    r"confidence optimal transport", r"\bCOT\b.{0,40}(accuracy|estimation)",
    # "MaNo" is catastrophically polysemous - Mano River, Mano Dura, NFV MANO,
    # mano-thermo-sonication, Spanish "mano", grinding stones, a hand exoskeleton,
    # the MANO hand model. Bare \bMaNo\b pulled hundreds of them in, because a
    # named-method match deliberately overrides the noise filters (see below).
    # It must therefore carry its own context.
    r"\bMaNo\b.{0,80}(matrix norm|accuracy estimation|distribution shift)",
    r"(matrix norm|accuracy estimation|distribution shift).{0,80}\bMaNo\b",
    r"\bGdScore\b", r"agreement.on.the.line",
    r"projection norm", r"confidence and dispersity", r"nuclear norm.{0,40}accuracy",
    r"prediction.powered inference",
)

# -- exclusion: "accuracy" in an unrelated sense ---------------------------
WRONG_SENSE = rx(
    r"\b(gps|gnss|rtk|geolocation|positional|localizat\w+|localisat\w+|navigation)\b",
    r"\b(pose|depth|channel|state|frequency|attitude|orbit|motion|flow|pitch|"
    r"heart rate|blood pressure|dose|age|gaze) estimation\b",
    r"yield (estimation|prediction)", r"biomass estimation",
    r"\b(measurement|positioning|geometric|radiometric|spatial|thematic) accuracy\b",
    r"parameter estimation", r"density estimation", r"\bcrop yield\b",
)
# -- exclusion: adaptation / self-supervision rather than estimation -------
ADAPTATION = rx(
    r"domain adaptation", r"transfer learning", r"fine.?tun\w+", r"self.supervised",
    r"contrastive learning", r"pseudo.?label\w* .{0,30}train", r"data augmentation",
    r"generative adversarial", r"style transfer", r"test.time (training|adaptation)",
)
# -- exclusion: per-sample abstention rather than dataset-level estimate ---
SELECTIVE = rx(
    r"conformal predict\w+", r"selective (prediction|classification)",
    r"reject option", r"abstention", r"misclassification detection",
    r"failure prediction", r"per.(sample|image|instance) (confidence|uncertainty)",
    r"out.of.distribution detection", r"anomaly detection", r"novelty detection",
)
AGRI = rx(
    r"plant disease", r"crop disease", r"leaf disease", r"plantvillage",
    r"precision agricultur\w+", r"agricultur\w+", r"\bcrop\b", r"\bleaf\b",
    r"\bleaves\b", r"phenotyp\w+", r"horticultur\w+", r"\bweed\b", r"\bplant\b",
)
SHIFT = rx(
    r"distribution shift", r"domain shift", r"dataset shift", r"covariate shift",
    r"label shift", r"out.of.distribution", r"laboratory.to.field", r"lab.to.field",
    r"field condition", r"in the wild", r"real.world deployment",
)


def screen_one(title: str, abstract: str) -> tuple[str, str]:
    """Return (decision, reason). Decisions: include | adjacent | exclude.

    Crossref deposits abstracts only sparsely - about 73% of the retrieved
    corpus arrives with none. Dropping those unread would silently discard the
    majority of the search, so a record with no abstract is screened on its
    **title alone** and the weaker basis is recorded in the reason. Title-only
    screening has lower recall than title-and-abstract, which matters for a null
    claim, and that limitation is reported rather than hidden.
    """
    text = f"{title} . {abstract}"
    title_only = len(abstract or "") < 40
    if title_only:
        text = title or ""
        if len(text) < 8:
            return "exclude", "no_title_or_abstract_to_screen"
        task = bool(TASK.search(text)) or bool(METHODS.search(text))
        if not task:
            return "exclude", "not_performance_estimation_title_only"
        if AGRI.search(text):
            return "include", "task_and_agriculture_title_only"
        return "include", "task_general_ml_title_only"

    task_hit = bool(TASK.search(text))
    methods_hit = bool(METHODS.search(text))
    agri = bool(AGRI.search(text))

    if not (task_hit or methods_hit):
        if agri and SHIFT.search(text):
            return "adjacent", "agricultural_shift_no_estimation"
        return "exclude", "not_performance_estimation"

    # False-positive filters apply only when the *loose* TASK regex fired on its
    # own. A named method (METHODS) is the strongest positive signal there is, so
    # it is never overridden by them.
    #
    # This was a bug, found by the protocol's own excluded-set audit: the guards
    # read `and not TASK.search(text)`, which is True exactly when only METHODS
    # matched - so a named-estimator hit was *excluded* by the very filters meant
    # to remove noise. It threw out prediction-powered-inference papers
    # ("Prediction-Powered Semi-Supervised Learning with Online Power Tuning",
    # "Cross-Prediction-Powered Inference for Wireless Systems") because their
    # abstracts also contain phrases like "state estimation".
    if not methods_hit:
        if WRONG_SENSE.search(text):
            return "exclude", "wrong_sense_of_accuracy"
        if SELECTIVE.search(text):
            return "adjacent", "per_sample_selective_prediction"
        if ADAPTATION.search(text):
            return "adjacent", "adaptation_not_estimation"

    if agri:
        return "include", "task_and_agriculture"
    return "include", "task_general_ml"


def main() -> int:
    t0 = time.time()
    # Merge every retrieval stage present, so a staged search - one database
    # today, another when its daily quota resets - assembles into one corpus.
    raws = sorted(OUT.glob("raw_*.csv"))
    if not raws:
        print("no raw_*.csv found - run scripts/litsearch.py first")
        return 2
    raw = pd.concat([pd.read_csv(f) for f in raws], ignore_index=True)
    print(f"retrieval stages merged: {[f.name for f in raws]}")
    print(f"records identified     : {len(raw):,}")

    raw["norm_title"] = raw.title.fillna("").map(norm_title)
    raw = raw[raw.norm_title.str.len() > 8].copy()
    raw["doi"] = raw.doi.fillna("").astype(str).str.lower()
    raw["dedup_key"] = raw.doi.where(raw.doi.str.len() > 0, raw.norm_title)
    # keep the richest copy of each work: prefer one that carries an abstract
    raw["_has_abs"] = (raw.abstract.fillna("").str.len() > 40).astype(int)
    df = (raw.sort_values(["_has_abs", "cited_by"], ascending=False)
             .groupby("dedup_key", as_index=False).first())
    hits = (raw.groupby("dedup_key")
               .agg(n_queries=("found_by_query", "nunique"),
                    searches=("found_by_search", lambda s: ",".join(sorted(set(s)))),
                    databases=("database", lambda s: ",".join(sorted(set(s))))))
    df = df.merge(hits, on="dedup_key", how="left").drop(columns=["_has_abs"])
    df.to_csv(OUT / "deduplicated.csv", index=False)
    df = df.fillna({"title": "", "abstract": "", "venue": ""})
    print(f"unique works           : {len(df):,}\n", flush=True)

    dec = df.apply(lambda r: screen_one(r.title, r.abstract), axis=1)
    df["decision"] = [d for d, _ in dec]
    df["reason"] = [r for _, r in dec]
    df.to_csv(OUT / "screened.csv", index=False)

    print("=== decisions ===")
    print(df.decision.value_counts().to_string())
    print("\n=== reasons ===")
    print(df.reason.value_counts().to_string())

    inc = df[df.decision == "include"].copy()
    adj = df[df.decision == "adjacent"].copy()
    inc.sort_values(["year"], ascending=False).to_csv(OUT / "included.csv", index=False)
    adj.sort_values(["year"], ascending=False).to_csv(OUT / "adjacent.csv", index=False)

    # the decision-critical subset: task AND agriculture
    key = inc[inc.reason.isin(["task_and_agriculture",
                               "task_and_agriculture_title_only"])]
    key.sort_values("year", ascending=False).to_csv(OUT / "included_agricultural.csv",
                                                    index=False)
    print(f"\n=== Q2 decision set: label-free performance estimation IN agriculture ===")
    print(f"{len(key)} record(s)")
    for _, r in key.sort_values("year", ascending=False).iterrows():
        print(f"  {r.year} | {str(r.reason)[:31]:<31} | {str(r.title)[:80]}")

    prisma = pd.DataFrame([
        {"stage": "identified (all databases, with duplicates)", "n": int(len(raw))},
        {"stage": "unique works after deduplication", "n": int(len(df))},
        {"stage": "screened on title and abstract", "n": int(len(df))},
        {"stage": "excluded", "n": int((df.decision == "exclude").sum())},
        {"stage": "adjacent (kept for Related Work)", "n": int(len(adj))},
        {"stage": "included", "n": int(len(inc))},
        {"stage": "included AND agricultural (Q2 decision set)", "n": int(len(key))},
    ])
    prisma.to_csv(OUT / "prisma_counts.csv", index=False)
    print("\n=== PRISMA-style flow ===")
    print(prisma.to_string(index=False))

    write_experiment_manifest(
        "litsearch_screen",
        params={"protocol": "docs/litsearch_protocol.md sections 6-8",
                "screening": "title and abstract only; no full texts read",
                "note": "automated triage, then manual reading of include+adjacent"},
        inputs={"corpus": "results/litsearch/deduplicated.csv"},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"n_screened": int(len(df)), "n_included": int(len(inc)),
                 "n_adjacent": int(len(adj)), "n_agricultural": int(len(key))},
        runtime_seconds=time.time() - t0,
        notes="Triage only. The include and adjacent sets are read by hand and the "
              "exclude set is sampled to check the rules.",
        out_dir=OUT,
        filename="manifest_screen.json",
    )
    print(f"\ndone in {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
