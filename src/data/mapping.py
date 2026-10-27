"""Canonical class mapping across the three tomato domains.

Every decision here is fixed by ``docs/dataset_class_mapping.md`` (verified
2026-09-06) and must not be widened later. The rule from that document:
never map two biologically different classes because their names look similar.
"""
from __future__ import annotations

from typing import Final

SHARED_3: Final[list[str]] = ["healthy", "early_blight", "late_blight"]

SHARED_8: Final[list[str]] = SHARED_3 + [
    "bacterial_spot",
    "leaf_mold",
    "septoria_leaf_spot",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
]

#: SHARED_8 minus mosaic_virus - the documented fallback for missing grouping
#: metadata, NOT a re-mapping of classes.
#:
#: PlantVillage ships no ``leaf_grouping/filtered_leafmaps`` CSV for
#: ``Tomato___Tomato_mosaic_virus``, so none of its 373 images carries a physical
#: leaf id and none can enter a leakage-free leaf-grouped split. Training
#: SHARED_8 on the grouped pool would leave the mosaic_virus head with **zero**
#: examples, and PlantDoc's 54 mosaic-virus images would then be guaranteed
#: errors - depressing arm B's accuracy for a reason that has nothing to do with
#: distribution shift.
#:
#: We checked whether leaf identity could be recovered from the filename
#: numbering instead ("four consecutive numbers = one leaf"). It does not hold:
#: the rule reproduces the published leafmap for 100% of healthy leaves and 76%
#: of early blight, but only 19% of late blight and 24% of leaf mold. Inferring
#: ToMV groups that way would be guesswork, so the class is excluded from the
#: grouped protocol and the exclusion is reported.
SHARED_7: Final[list[str]] = [c for c in SHARED_8 if c != "mosaic_virus"]

LABEL_SETS: Final[dict[str, list[str]]] = {
    "SHARED_3": SHARED_3,
    "SHARED_7": SHARED_7,
    "SHARED_8": SHARED_8,
}

# --------------------------------------------------------------------------
# Folder name -> canonical label. A folder absent from a map is deliberately
# excluded; see EXCLUSIONS for the reason.
# --------------------------------------------------------------------------

PLANTVILLAGE_MAP: Final[dict[str, str]] = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
}

PLANTDOC_MAP: Final[dict[str, str]] = {
    # PlantDoc's unqualified "Tomato leaf" is its healthy class.
    "Tomato leaf": "healthy",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato mold leaf": "leaf_mold",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
}

TOMATOVILLAGE_MAP: Final[dict[str, str]] = {
    "Healthy": "healthy",
    "Early_blight": "early_blight",
    "Late_blight": "late_blight",
}

#: Fourth target arm (2026-09-28). Folder names come from the raw folder of the
#: Mendeley deposit 10.17632/3mbnb82mxd.2, after excluding its augmented folder
#: and removing the 311 images byte-identical to PlantDoc. Only the three
#: SHARED_3 classes are mapped; the deposit's other three folders have no
#: counterpart in this label set and are deliberately absent.
PAKISTAN_MAP: Final[dict[str, str]] = {
    "Healthy": "healthy",
    "Early_blight": "early_blight",
    "Late_blight": "late_blight",
}

DOMAIN_MAPS: Final[dict[str, dict[str, str]]] = {
    "plantvillage": PLANTVILLAGE_MAP,
    "plantdoc": PLANTDOC_MAP,
    "tomatovillage": TOMATOVILLAGE_MAP,
    "pakistan": PAKISTAN_MAP,
}

# --------------------------------------------------------------------------
# Deliberate exclusions, with the reason recorded next to the decision.
# --------------------------------------------------------------------------

EXCLUSIONS: Final[dict[str, str]] = {
    "Tomato___Target_Spot": "no PlantDoc or Tomato-Village counterpart",
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "PlantDoc counterpart has n=2; cannot support any session"
    ),
    "Tomato two spotted spider mites leaf": "n=2; cannot support any session",
    "Leaf Miner": "Liriomyza dipteran leaf mining is not tetranychid mite stippling",
    "Spotted Wilt Virus": "TSWV is a tospovirus; TYLCV is a begomovirus - different biology",
    "Magnesium Deficiency": "abiotic nutrient disorder; PlantVillage has no abiotic class",
    "Nitrogen Deficiency": "abiotic nutrient disorder; PlantVillage has no abiotic class",
    "Pottassium Deficiency": "abiotic nutrient disorder; PlantVillage has no abiotic class",
}

# Mappings that look plausible on name similarity and are forbidden outright.
# ``assert_mapping_allowed`` is the executable form of that prohibition.
FORBIDDEN_PAIRS: Final[set[tuple[str, str]]] = {
    ("Spotted Wilt Virus", "yellow_leaf_curl_virus"),
    ("Leaf Miner", "spider_mites"),
    ("Magnesium Deficiency", "healthy"),
    ("Nitrogen Deficiency", "healthy"),
    ("Pottassium Deficiency", "healthy"),
}


class ForbiddenMappingError(ValueError):
    """Raised when code attempts one of the explicitly rejected class mappings."""


def assert_mapping_allowed(folder: str, canonical: str) -> None:
    """Fail loudly if `folder` -> `canonical` is a rejected mapping."""
    if (folder, canonical) in FORBIDDEN_PAIRS:
        raise ForbiddenMappingError(
            f"{folder!r} -> {canonical!r} is explicitly rejected in "
            "docs/dataset_class_mapping.md section 3.2. Do not widen the overlap."
        )


def canonical_label(domain: str, folder: str) -> str | None:
    """Canonical label for a raw folder name, or None if deliberately excluded."""
    try:
        mapping = DOMAIN_MAPS[domain]
    except KeyError:
        raise KeyError(f"unknown domain {domain!r}; expected one of {sorted(DOMAIN_MAPS)}") from None
    label = mapping.get(folder)
    if label is not None:
        assert_mapping_allowed(folder, label)
    return label


def label_index(label_set: str) -> dict[str, int]:
    """Stable label -> integer index for a named label set."""
    return {name: i for i, name in enumerate(LABEL_SETS[label_set])}
