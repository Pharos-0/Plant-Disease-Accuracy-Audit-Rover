"""Leaf-region proposal: frame -> illumination normalisation -> green mask -> crops.

This reuses the honest half of the delivered rover (`docs/rover_code_audit.md`
section 4): CLAHE on the LAB L-channel and a close-then-open morphology pass are
both sound. What is added is connected-component extraction, so the green mask
becomes a set of leaf crops a classifier can consume.

Two audited defects are fixed here and the fixes are flagged, not silent:

* **D1 - inverted gamma.** With the LUT convention ``out = (i/255)^(1/gamma) *
  255``, gamma > 1 brightens and gamma < 1 darkens. The delivered code applied
  ``gamma=2.0`` when the frame was already too bright and ``gamma=0.7`` when it
  was too dark, so both branches amplified the exposure problem. The constants
  are swapped here.
* **D10 - loose HSV bounds.** The inherited saturation floor of 30/255 admits
  grey and desaturated surfaces. The default is raised and, more importantly,
  exposed as a parameter instead of being buried as a literal.

The ``c > 1500`` spray-coverage threshold is **not** reimplemented here. It is
inherited, uncalibrated prototype logic (D9) and belongs to the *where to spray*
rule, not to leaf proposal.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ProposalConfig:
    # HSV gate. The inherited values were (25, 30, 30) - (95, 255, 255); the
    # saturation floor is raised to reject desaturated non-plant surfaces.
    hsv_lower: tuple[int, int, int] = (25, 60, 40)
    hsv_upper: tuple[int, int, int] = (95, 255, 255)
    #: reject components smaller than this many pixels
    min_component_area: int = 900
    #: pad each bounding box by this fraction of its size
    box_padding: float = 0.12
    #: crops are emitted square at this side length, ready for the classifier
    crop_size: int = 224
    #: at most this many crops per frame, largest first
    max_crops: int = 12
    bright_threshold: int = 170
    dark_threshold: int = 70


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma LUT. gamma > 1 brightens, gamma < 1 darkens.

    This is the convention the delivered code used, verified numerically in the
    audit: the LUT maps the 0..255 mean to 169.3 at gamma=2.0 and to 104.6 at
    gamma=0.7.
    """
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lut)


def adaptive_vision(frame: np.ndarray, cfg: ProposalConfig | None = None) -> np.ndarray:
    """CLAHE on the LAB L-channel, then an exposure-correcting gamma pass."""
    cfg = cfg or ProposalConfig()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    brightness = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
    if brightness > cfg.bright_threshold:
        out = adjust_gamma(out, 0.7)  # D1 fix: darken an over-exposed frame
    elif brightness < cfg.dark_threshold:
        out = adjust_gamma(out, 2.0)  # D1 fix: brighten an under-exposed frame
    return out


def detect_green(frame: np.ndarray, cfg: ProposalConfig | None = None) -> np.ndarray:
    """Binary vegetation mask. Close-then-open, as in the delivered code."""
    cfg = cfg or ProposalConfig()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


@dataclass
class LeafCrop:
    image: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int


def propose_leaf_crops(
    frame: np.ndarray, cfg: ProposalConfig | None = None
) -> list[LeafCrop]:
    """Frame -> illumination-normalised -> green mask -> components -> crops."""
    cfg = cfg or ProposalConfig()
    norm = adaptive_vision(frame, cfg)
    mask = detect_green(norm, cfg)

    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_img, w_img = frame.shape[:2]

    candidates = []
    for i in range(1, n_labels):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < cfg.min_component_area:
            continue
        pad_x, pad_y = int(w * cfg.box_padding), int(h * cfg.box_padding)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
        candidates.append(LeafCrop(image=crop, bbox=(x0, y0, x1 - x0, y1 - y0), area=int(area)))

    candidates.sort(key=lambda c: c.area, reverse=True)
    return candidates[: cfg.max_crops]


def green_coverage(mask: np.ndarray) -> float:
    """Fraction of the mask that is vegetation.

    Reported so the inherited ``c > 1500`` trigger can be *described* in the
    thesis in calibrated units (a fraction) rather than as a bare pixel count.
    It is not used as a spray decision here.
    """
    return float((mask > 0).mean())
