"""The V0 baseline must stay byte-identical, and the audit must stay true to it.

The rule: web_rover.py must remain byte-identical, because every rover
before/after number is measured against this exact file. The usual way it gets
corrupted on Windows is line-ending conversion, so `.gitattributes` marks it
`-text` and this test is the tripwire.

The second half re-verifies the defect list in `docs/rover_code_audit.md` against
the file itself. The audit was written in a previous session against an artefact
this machine did not have; now that it does, none of it has to be taken on trust.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "rover" / "original" / "web_rover.py"
EXPECTED_SHA256 = "8801697bd77185ba8a544586ed64233fa65d54a29c759ab22c9b6b856104d3ba"


def source() -> str:
    if not BASELINE.exists():
        pytest.skip(f"frozen baseline not present at {BASELINE}")
    return BASELINE.read_bytes().decode("utf-8")


def test_baseline_is_byte_identical() -> None:
    if not BASELINE.exists():
        pytest.skip("frozen baseline not present")
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, (
        "web_rover.py has changed. If this followed a git operation the cause is "
        "almost certainly CRLF conversion - check .gitattributes marks it -text."
    )


def test_baseline_has_no_crlf() -> None:
    if not BASELINE.exists():
        pytest.skip("frozen baseline not present")
    assert b"\r\n" not in BASELINE.read_bytes(), "line endings were converted"


def test_baseline_is_241_lines() -> None:
    assert len(source().splitlines()) == 241


# --------------------------------------------------------------------------
# the audit's claims, re-verified against the artefact
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,label",
    [
        (r"\bIR\b|infrared|line_sensor", "IR line sensors"),
        (r"battery|\bADC\b|voltage", "battery / ADC / voltage monitoring"),
        (r"tank", "pesticide tank sensor"),
        (r"\bFSM\b|state_machine", "finite state machine"),
        (r"return_to_home|homing", "return-to-home"),
        (r"threading|Thread\b", "threading"),
        (r"flask|http|server|socket", "web server"),
    ],
)
def test_subsystems_claimed_by_the_paper_are_absent(pattern: str, label: str) -> None:
    """Audit §2: the delivered paper claims subsystems with no code behind them."""
    hits = re.findall(pattern, source(), flags=re.IGNORECASE)
    assert not hits, f"{label} unexpectedly present: {hits}"


def test_d1_gamma_is_inverted_in_the_baseline() -> None:
    """Audit D1. Both branches amplify the exposure problem.

    With this LUT, gamma > 1 brightens. The baseline applies 2.0 when the frame
    is already too bright and 0.7 when it is already too dark. Our replacement in
    `src/rover/leaf_proposal.py` swaps them; this test pins the defect in the
    frozen file so the before/after comparison stays honest.
    """
    src = source()
    m = re.search(
        r"if brightness > 170:\s*\n\s*frame = adjust_gamma\(frame, ([\d.]+)\)"
        r"\s*\n\s*elif brightness < 70:\s*\n\s*frame = adjust_gamma\(frame, ([\d.]+)\)",
        src,
    )
    assert m, "the gamma branch structure has changed"
    too_bright_gamma, too_dark_gamma = float(m.group(1)), float(m.group(2))
    assert too_bright_gamma == 2.0 and too_dark_gamma == 0.7
    assert too_bright_gamma > 1.0, "D1: an over-exposed frame is brightened further"
    assert too_dark_gamma < 1.0, "D1: an under-exposed frame is darkened further"


def test_d2_three_stacked_fail_open_behaviours() -> None:
    """Audit D2. Sprayer armed at startup; sensor fails to None; None is falsy."""
    src = source()
    assert "SPRAY_ENABLED = True" in src, "sprayer is armed at startup"
    assert src.count("return None") == 2, "get_distance returns None on timeout"
    assert "if dist and dist < 20:" in src, (
        "None is falsy, so a failed sensor silently disables the obstacle stop"
    )


def test_d3_no_try_finally_around_the_main_loop() -> None:
    """Audit D3. Cleanup is unreachable on any exception; motors and relay stay live."""
    src = source()
    assert "finally:" not in src
    assert "try:" not in src


def test_d9_spray_threshold_is_an_uncalibrated_constant() -> None:
    """Audit D9. Inherited prototype logic, never an agronomic threshold."""
    assert "c > 1500" in source()


def test_no_logging_so_the_papers_table_i_is_unsupported() -> None:
    """Audit §2. The file writes nothing, so it cannot have produced any metric."""
    src = source()
    for token in ("logging", "csv", "json"):
        assert token not in src.lower(), f"unexpected {token} usage"
    assert "open(" not in src, "the baseline opens no file for writing"
