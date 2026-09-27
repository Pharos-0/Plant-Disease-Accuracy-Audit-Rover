"""Measurable rover-view relevance features, and the rubric that scores them.

The score answers exactly one question: **how useful is this real image as a
proxy for what a low-mounted ground-rover camera might observe in a tomato
crop?** It is never a claim that a rover captured the image. Nothing here infers
provenance.

Every feature is computed from pixels alone, with no model and no learned
component, so the rubric is inspectable and cannot drift.

THE GOVERNING QUESTION
----------------------
Does the scene continue past the subject, or does the subject sit on a backdrop?

A rover travelling a crop row sees a plant *embedded in its surroundings*:
foliage, soil, straw and neighbouring plants run to the edge of the frame. A
specimen photograph does not: whatever is at the frame border is a backdrop.
So the border band is where the answer lives, and `border_scene_frac` - the
fraction of border pixels that are vegetation- or earth-coloured - is the
primary feature.

WHAT AN EARLIER VERSION GOT WRONG, AND WHY IT IS RECORDED HERE
--------------------------------------------------------------
The first version tested for a backdrop by intensity uniformity, assuming a
studio background is smooth. Calibration against PlantVillage failed, and
inspection showed why: PlantVillage leaves sit on a *coarse speckled grey*
surface. Its border intensity spread is high and its Canny edge density is the
highest of the three local datasets - the detector fires on background grit.
Smoothness is the wrong test. Grey, white and black backdrops share something
else: they are achromatic. `border_achromatic_frac` captures that, and it
correctly groups PlantVillage's grey stone with Tomato-Village's pale cloth.

The same calibration appeared to fail on PlantDoc. It had not: PlantDoc is
web-scraped and genuinely mixed - field shots with straw and soil alongside
potted seedlings against black cloth. The expectation was wrong, not the
measurement. The rubric is therefore calibrated on *separation and ordering*,
not on forcing one label per source.

A third feature was dropped outright. `green_component_count` was meant to count
leaves, but contiguous canopy foliage forms ONE connected component, exactly as
an isolated leaf does. It measured nothing and is no longer used in scoring. It
is still recorded, so the claim can be re-checked.

FEATURES (all in [0, 1] unless noted)
-------------------------------------
  border_scene_frac        Border pixels that are vegetation- or earth-coloured.
                           High => the scene runs to the frame edge. PRIMARY.
  border_achromatic_frac   Border pixels with near-zero saturation.
                           High => grey/white/black backdrop.
  border_hue_entropy       Entropy of the border hue histogram, normalised.
  border_texture           Normalised intensity spread in the border band.
                           Reported, not used as backdrop evidence - see above.
  edge_density             Canny edge pixels over the frame. Confounded by
                           background grit, so supporting evidence only.
  green_fraction           Vegetation-hued, reasonably saturated pixels.
  soil_fraction            Earth-hued pixels anywhere in the frame.
  largest_green_blob       Largest vegetation component / frame area.
                           Near 1 => one leaf fills the frame.
  green_component_count    Recorded only. Not used in scoring.
  half_detail_scale        Morphological granulometry: the smallest opening
                           radius capturing half the image's residual detail,
                           normalised. SMALL => fine repeated texture => a wide
                           or distant view. LARGE => few coarse structures => a
                           close-up. Consulted only where a scene is present.
  sharpness                log-scaled variance of the Laplacian.
  brightness, contrast     Mean and spread of intensity, normalised.

WHY DISTANCE IS MEASURED SEPARATELY FROM SCENE
----------------------------------------------
`border_scene_frac` answers "is the subject embedded in surroundings?", which a
macro shot inside a canopy satisfies as fully as a rover driving an inter-row.
The pilot made this concrete: handheld close-ups outscored genuinely
rover-captured wide views. Embedded-in-a-scene is necessary for rover-relevance
but not sufficient, so scene and distance are two axes, and the top score
requires both.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

#: Width of the border band as a fraction of the shorter side.
BORDER_FRAC = 0.10

#: OpenCV hue is 0-179. Vegetation: yellow-green through cyan-green.
GREEN_HUE = (25, 95)
GREEN_MIN_SAT = 40
GREEN_MIN_VAL = 25

#: Earth tones: red-orange-brown at moderate saturation. Also catches dry straw.
SOIL_HUE = (5, 25)
SOIL_SAT = (30, 200)
SOIL_VAL = (35, 210)

#: Below this saturation a pixel carries no hue: grey, white or black.
ACHROMATIC_SAT = 45

#: A vegetation component must cover this fraction of the frame to be counted.
MIN_BLOB_FRAC = 0.005

#: Canonical working long side. Images are resized to this in BOTH directions,
#: never only downwards: granulometry radii are in pixels, so a fixed frame size
#: is what makes a radius mean the same fraction of the scene in every image.
WORK_LONG_SIDE = 512

#: Structuring-element radii probed by the granulometry, in pixels of the
#: canonical frame.
GRANULOMETRY_RADII = (1, 2, 3, 5, 8, 12, 18, 26, 36, 48)


@dataclass(frozen=True)
class Features:
    border_scene_frac: float
    border_achromatic_frac: float
    border_hue_entropy: float
    border_texture: float
    edge_density: float
    green_fraction: float
    soil_fraction: float
    largest_green_blob: float
    green_component_count: int
    half_detail_scale: float
    sharpness: float
    brightness: float
    contrast: float
    width: int
    height: int

    def as_dict(self) -> dict:
        return asdict(self)


def _standardise(bgr: np.ndarray) -> np.ndarray:
    """Resize to the canonical long side, up or down.

    Downscaling only would leave the granulometry resolution-dependent: a
    48-pixel structuring element covers most of a 256px image and a corner of a
    4000px one. Resizing in both directions makes a radius mean the same
    fraction of the frame everywhere.
    """
    h, w = bgr.shape[:2]
    s = WORK_LONG_SIDE / max(h, w)
    if s == 1.0:
        return bgr
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=interp)


def _half_detail_scale(gray: np.ndarray) -> float:
    """Characteristic structure size, as a fraction of the largest radius probed.

    r(k) = mean |g - opening(g, k)| is the detail removed by an opening of
    radius k, and rises monotonically with k. The smallest k reaching half of
    r(k_max) is the scale at which most of the image's structure lives.
    """
    g = gray.astype(np.float32)
    residual = []
    for k in GRANULOMETRY_RADII:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        residual.append(float(np.abs(g - cv2.morphologyEx(g, cv2.MORPH_OPEN, se)).mean()))
    residual = np.asarray(residual)
    if residual[-1] <= 1e-6:                 # a flat image has no characteristic scale
        return 1.0
    idx = int(np.argmax(residual / residual[-1] >= 0.5))
    return GRANULOMETRY_RADII[idx] / GRANULOMETRY_RADII[-1]


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    b = max(2, int(round(BORDER_FRAC * min(h, w))))
    m = np.zeros((h, w), dtype=bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def _normalised_entropy(values: np.ndarray, bins: int, value_range) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) / np.log2(bins)


def _vegetation(hue, sat, val):
    return ((hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) &
            (sat >= GREEN_MIN_SAT) & (val >= GREEN_MIN_VAL))


def _earth(hue, sat, val):
    return ((hue >= SOIL_HUE[0]) & (hue <= SOIL_HUE[1]) &
            (sat >= SOIL_SAT[0]) & (sat <= SOIL_SAT[1]) &
            (val >= SOIL_VAL[0]) & (val <= SOIL_VAL[1]))


def extract(bgr: np.ndarray) -> Features:
    """Compute the feature vector for one BGR image."""
    orig_h, orig_w = bgr.shape[:2]
    img = _standardise(bgr)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green = _vegetation(hue, sat, val)
    soil = _earth(hue, sat, val)

    band = _border_mask((h, w))
    border_scene_frac = float((green | soil)[band].mean())
    border_achromatic_frac = float((sat[band] < ACHROMATIC_SAT).mean())
    border_hue_entropy = _normalised_entropy(hue[band], bins=18, value_range=(0, 180))
    border_texture = float(np.clip(gray[band].astype(np.float64).std() / 64.0, 0.0, 1.0))

    med = float(np.median(gray))
    lo, hi = max(0, int(0.66 * med)), min(255, int(1.33 * med))
    edge_density = float((cv2.Canny(gray, lo, hi) > 0).mean())

    frame_px = h * w
    mask = cv2.morphologyEx(green.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([], dtype=np.int64)
    kept = areas[areas >= MIN_BLOB_FRAC * frame_px]

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()

    return Features(
        border_scene_frac=round(border_scene_frac, 4),
        border_achromatic_frac=round(border_achromatic_frac, 4),
        border_hue_entropy=round(border_hue_entropy, 4),
        border_texture=round(border_texture, 4),
        edge_density=round(edge_density, 4),
        green_fraction=round(float(green.mean()), 4),
        soil_fraction=round(float(soil.mean()), 4),
        largest_green_blob=round(float(kept.max() / frame_px) if kept.size else 0.0, 4),
        green_component_count=int(min(kept.size, 50)),
        half_detail_scale=round(_half_detail_scale(gray), 4),
        sharpness=round(float(np.clip(np.log10(lap + 1.0) / 4.0, 0.0, 1.0)), 4),
        brightness=round(float(gray.mean() / 255.0), 4),
        contrast=round(float(gray.std() / 128.0), 4),
        width=orig_w,
        height=orig_h,
    )


# --------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------

#: Below this, the frame border carries almost no vegetation or earth: the
#: subject is on a backdrop and the scene does not continue.
SCENE_ABSENT = 0.12
#: Above this, the scene clearly runs to the frame edge.
SCENE_STRONG = 0.45
#: Between the two, partial context.
SCENE_PARTIAL = 0.25

#: A border this colourless is a grey, white or black backdrop.
BACKDROP_ACHROMATIC = 0.60

#: One vegetation blob covering this much of the frame is a single specimen.
DOMINANT_BLOB = 0.55

#: Supporting cues.
SOIL_PRESENT = 0.030
CLUTTER_EDGE_DENSITY = 0.055

#: Camera-distance bands, read off the measured separation between a
#: rover-captured source (0.125) and a handheld in-canopy source (0.482).
WIDE_MAX = 0.20        # fine repeated texture: a travelling-rover-like view
MEDIUM_MAX = 0.35      # still several plants or many leaves in frame
CLOSE_MAX = 0.60       # one plant or one leaf; beyond this, macro


def camera_distance_estimate(f: "Features") -> str:
    """Coarse viewpoint band from the structure scale.

    This is a RELATIVE band inferred from image structure, not a metric
    distance. No camera height, focal length or field of view is known for the
    target platform, and none is invented here.
    """
    h = f.half_detail_scale
    if h < WIDE_MAX:
        return "WIDE"
    if h < MEDIUM_MAX:
        return "MEDIUM"
    if h < CLOSE_MAX:
        return "CLOSE"
    return "MACRO"


def classify(f: Features) -> tuple[int, str]:
    """Return (rover_relevance 0-3, one-sentence reason).

    0 = not relevant, 1 = weakly, 2 = moderately, 3 = strongly rover-relevant.
    A proxy judgement about viewpoint and visual conditions only. It says
    nothing about how the image was actually captured.
    """
    scene = f.border_scene_frac
    backdrop = f.border_achromatic_frac >= BACKDROP_ACHROMATIC
    dominant = f.largest_green_blob >= DOMINANT_BLOB
    soil = f.soil_fraction >= SOIL_PRESENT
    cluttered = f.edge_density >= CLUTTER_EDGE_DENSITY

    if scene < SCENE_ABSENT:
        if backdrop:
            return 0, (f"subject on a backdrop: only {scene:.0%} of the frame "
                       f"border is vegetation or earth, and {f.border_achromatic_frac:.0%} "
                       "of it is colourless")
        return 0, (f"no surrounding scene: {scene:.0%} of the frame border is "
                   "vegetation or earth")

    if scene < SCENE_PARTIAL:
        if dominant:
            return 1, (f"one leaf fills {f.largest_green_blob:.0%} of the frame and the "
                       f"border is mostly background ({scene:.0%} scene): close-up")
        return 1, f"only {scene:.0%} of the border is scene: marginal context"

    band = camera_distance_estimate(f)

    if scene < SCENE_STRONG:
        extra = sum((soil, cluttered))
        if extra >= 1:
            got = ", ".join(n for n, c in (("soil", soil), ("clutter", cluttered)) if c)
            return 2, (f"scene continues past the subject ({scene:.0%} of the border), "
                       f"with {got}; viewpoint {band}")
        return 2, (f"scene continues past the subject ({scene:.0%} of the border); "
                   f"viewpoint {band}")

    # A strong scene is necessary but not sufficient. A macro shot deep inside a
    # canopy has vegetation at every border and is still not what a rover sees.
    if band in ("CLOSE", "MACRO"):
        return 2, (f"surrounded by vegetation ({scene:.0%} of the border) but shot "
                   f"{band.lower()} up: structure scale {f.half_detail_scale:.2f} is "
                   "coarser than a travelling camera would give")

    if soil or cluttered:
        got = ", ".join(n for n, c in (("soil/straw visible", soil),
                                       ("foliage clutter", cluttered)) if c)
        return 3, (f"{band} view of a plant in its surroundings: {scene:.0%} of the "
                   f"frame border is vegetation or earth, with {got}")
    return 3, (f"{band} view of a plant in its surroundings: {scene:.0%} of the frame "
               "border is vegetation or earth")


def image_type_hint(f: Features, score: int) -> str:
    """A viewpoint hint, NOT a provenance claim.

    Field vs greenhouse, and real vs synthetic, are not decidable from pixels
    and are never guessed here: those come from the source dataset's own
    documentation, or stay UNKNOWN.
    """
    if score == 0:
        return "REAL_ISOLATED_LEAF_OR_STUDIO"
    if score >= 2:
        return "REAL_SCENE_ROVER_RELEVANT"
    return "UNKNOWN"
