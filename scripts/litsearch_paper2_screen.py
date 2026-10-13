"""Record the manual screening of the Paper 2 search (protocol §7, steps 2-3).

The include/exclude DECISION for each of the 946 first-pass records was made by
reading its title and abstract (or the title alone where no abstract exists).
Those decisions are written out below as id lists, keyed by the record's row in
results/litsearch_paper2/first_pass.csv. For EXCLUDED records only, the specific
exclusion code is then assigned by a keyword rule (see `exclusion_code`) and is
an approximation of the reason, not a separate judgement; the decision to
exclude is the manual one.

    python scripts/litsearch_paper2_screen.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "litsearch_paper2"

# ---------------------------------------------------------------- manual includes
# I1-disease: an image-based disease/pest detector decides whether or where to treat.
I1_DISEASE = [2, 24, 26, 38, 41, 47, 74, 76, 77, 79, 92, 93, 96, 111, 112, 116, 138, 160,
              163, 169, 189, 194, 195, 275, 284, 288, 297, 355, 356, 357, 361, 445, 456,
              467, 473, 487, 490, 495, 504, 509, 510, 511, 514, 515, 545, 546, 547, 568,
              588, 591, 611, 612, 669, 672, 681, 683, 685, 686, 691, 692, 693, 694, 695,
              696, 700, 701, 705, 731, 735, 743, 747, 785, 792, 799, 816, 822, 829, 847,
              857, 898, 902, 907, 928]
# I1-target: a detector decides where to spray weeds, crop or canopy (not disease).
I1_TARGET = [7, 19, 46, 52, 63, 71, 102, 140, 141, 162, 164, 216, 220, 222, 224, 228, 246,
             270, 271, 295, 300, 324, 330, 333, 335, 339, 341, 348, 351, 358, 364, 380,
             381, 383, 386, 395, 405, 407, 411, 412, 415, 468, 472, 480, 482, 485, 486,
             518, 519, 524, 529, 530, 535, 536, 603, 605, 616, 618, 622, 641, 643, 644,
             658, 676, 678, 679, 680, 703, 707, 733, 734, 740, 758, 762, 764, 765, 766,
             769, 795, 798, 800, 802, 814, 832, 833, 843, 844, 848, 853, 859, 861, 862,
             866, 903, 917, 939]
# I1-context: reviews and decision-support/recommendation systems for treatment.
I1_CONTEXT = [4, 50, 65, 68, 97, 99, 100, 101, 211, 264, 350, 368, 378, 461, 531, 555, 581,
              711, 746, 791, 817, 834, 919, 923]
# I2: classifier uncertainty, abstention or domain shift tied to a treatment decision.
I2 = [129, 137, 320, 801, 813, 846]
# I3: safety assurance, verification, supervision or simulation of agricultural robots/sprayers.
I3 = [10, 11, 106, 136, 303, 318, 671, 697, 698, 756, 794, 807, 894]

#: Read further (protocol §7.3): what was read is recorded.
CLOSEST = {
    137: "full text (arXiv 2501.07185)",
    857: "full text (repository copy, edepot.wur.nl/685888)",
    129: "abstract",
    163: "abstract (conference version of Oberti et al. 2016)",
    898: "abstract",
    816: "abstract",
    588: "abstract (highlights)",
    546: "abstract",
    731: "abstract",
    383: "abstract",
    764: "abstract",
    228: "abstract",
    407: "abstract",
    733: "abstract",
    697: "abstract",
    698: "abstract",
}

NONAG = re.compile(r"malaria|triatoma|chagas|thermal spray|coating|painting|paint|tumor|"
                   r"cancer|seizure|retinopathy|patient|smoking|nicotine|lidocaine|dental|"
                   r"wound|steroid|construction|fireproof|coal|mining|mosquito|transgender|"
                   r"neck pain|wall|turbine|icing|disinfekt", re.I)
PHYSICS = re.compile(r"nozzle|drift|deposit|droplet|atomi|boom suspension|spray quality|"
                     r"calibration training|inspection of agricultural sprayers|chassis|"
                     r"spray pattern|residue|runoff|dissipation", re.I)
NONSCHOLARLY = re.compile(r"^(thermal spraying|electrostatic hand-held)|first report of|"
                          r"project reports|handbook", re.I)


def exclusion_code(title: str, abstract: str) -> str:
    text = f"{title} {abstract}"
    if NONSCHOLARLY.search(title or ""):
        return "E4"
    if NONAG.search(text):
        return "E3"
    if PHYSICS.search(title or ""):
        return "E2"
    return "E1"


def main() -> int:
    c = pd.read_csv(D / "first_pass.csv").reset_index().rename(columns={"index": "rid"})
    groups = {**{i: "I1-disease" for i in I1_DISEASE}, **{i: "I1-target" for i in I1_TARGET},
              **{i: "I1-context" for i in I1_CONTEXT}, **{i: "I2" for i in I2},
              **{i: "I3" for i in I3}}
    dup = [i for i in groups if sum(i in g for g in (I1_DISEASE, I1_TARGET, I1_CONTEXT, I2, I3)) > 1]
    assert not dup, f"record in two include groups: {dup}"
    out = []
    for _, r in c.iterrows():
        inc = groups.get(r.rid)
        out.append({"rid": r.rid, "doi": r.doi, "title": r.title, "year": r.year,
                    "decision": "include" if inc else "exclude",
                    "code": inc or exclusion_code(str(r.title), str(r.abstract)),
                    "judged_on": ("title+abstract" if isinstance(r.abstract, str) and r.abstract
                                  else "title only"),
                    "closest": r.rid in CLOSEST, "read_further": CLOSEST.get(r.rid, "")})
    s = pd.DataFrame(out)
    s.to_csv(D / "screening_decisions.csv", index=False)
    print(s.code.value_counts().to_string())
    print(f"{(s.decision == 'include').sum()} included of {len(s)}; "
          f"{s.closest.sum()} read further")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
