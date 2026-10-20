"""Tests for the rover deploy script that can run without the Pi.

The failure that matters most is installing a server that cannot start: the control
server imports spatial_policy, so a deploy that left it behind would crash-loop the
service on the robot. That is checked statically, from the servers' own imports.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_rover as dr  # noqa: E402


def test_every_deployed_file_and_unit_exists_in_the_repo():
    for f in dr.FILES:
        assert (ROOT / f).is_file(), f
    for parts in dr.UNITS.values():
        for p in parts:
            assert (ROOT / p).is_file(), p


def test_every_local_module_a_server_imports_is_deployed_with_it():
    local = {p.stem for p in (ROOT / "rover").glob("*.py")}
    deployed = {Path(f).stem for f in dr.FILES}
    for f in dr.FILES:
        tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        imported = {n.module.split(".")[0] for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module}
        imported |= {a.name.split(".")[0] for n in ast.walk(tree)
                     if isinstance(n, ast.Import) for a in n.names}
        missing = (imported & local) - deployed
        assert not missing, f"{f} imports {missing}, which would not be on the Pi"


def test_unit_comparison_ignores_systemctl_cat_headers_and_comments():
    live = """# /etc/systemd/system/rover-control.service
[Unit]
Description=Rover teleop control (camera + drive + spray) http :8080
After=network-online.target
Conflicts=rover-camstream.service

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/python3 /home/pi/pi_rover_control.py --port 8080 --camera 0
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/rover-control.service.d/workdir.conf
[Service]
WorkingDirectory=/home/pi
"""
    mine = [ln for p in dr.UNITS["rover-control.service"]
            for ln in dr._unit_lines((ROOT / p).read_text(encoding="utf-8"))]
    assert dr._unit_lines(live) == mine


def test_an_unreachable_pi_is_reported_not_deployed(capsys):
    assert dr.check("pi@rover.invalid") == 2        # .invalid never resolves
    assert "cannot reach" in capsys.readouterr().out
    assert dr.apply("pi@rover.invalid") == 2
