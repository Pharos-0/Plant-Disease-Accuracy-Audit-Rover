"""Deploy the rover's server code to the Pi: staged, checked there, reversible.

    python scripts/deploy_rover.py                 # check only: reach the Pi, show what differs
    python scripts/deploy_rover.py --apply         # back up, stage, check, install, restart, verify
    python scripts/deploy_rover.py --rollback ~/deploy_backup/20261020_081500

Three files make up the servers, and they must travel together:
pi_rover_control.py imports spatial_policy.py, so copying it alone would crash-loop
the control service. Files are first copied to a staging folder ON THE PI and
compiled and imported there with the Pi's own Python, OpenCV and Flask; only if
that succeeds are they installed. The previous versions are kept in a timestamped
backup on the Pi, and the rollback command is printed.

Nothing is actuated. Restarting a server leaves the rover disarmed with AUTO off
and the spray gate closed - its start-up state. The service that was running is
the one restarted; if none was running, none is started.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "pi@192.168.137.167"
PORT = 8080
#: Installed side by side in /home/pi; pi_rover_control imports spatial_policy.
FILES = ["rover/pi_rover_control.py", "rover/pi_camera_stream.py", "rover/spatial_policy.py"]
SERVICES = ["rover-control.service", "rover-camstream.service"]
#: The repo's record of each unit: the unit file, then its drop-ins.
UNITS = {
    "rover-control.service": ["rover/systemd/rover-control.service",
                              "rover/systemd/rover-control.service.d/workdir.conf"],
    "rover-camstream.service": ["rover/systemd/rover-camstream.service",
                                "rover/systemd/rover-camstream.service.d/conflict.conf"],
}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
JPEG_MAGIC = b"\xff\xd8"


def ssh(host: str, cmd: str, timeout: float = 90) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True,
                          timeout=timeout)


def scp(host: str, paths: list[Path], remote_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(["scp", *SSH_OPTS, *map(str, paths), f"{host}:{remote_dir}/"],
                          capture_output=True, text=True, timeout=120)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _unit_lines(text: str) -> list[str]:
    """A unit's meaningful lines: comments (incl. systemctl cat's path headers) and
    blank lines removed."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def reachable(host: str) -> bool:
    try:
        return ssh(host, "echo ok", timeout=20).stdout.strip() == "ok"
    except subprocess.TimeoutExpired:
        return False


def active_service(host: str) -> str | None:
    for s in SERVICES:
        if ssh(host, f"systemctl is-active {s}").stdout.strip() == "active":
            return s
    return None


def check(host: str) -> int:
    if not reachable(host):
        print(f"cannot reach {host} over SSH. Is the Pi powered, on the laptop hotspot, "
              f"and at that address? (ping it; the hotspot's device list shows its IP)")
        return 2
    print(f"reached {host}; running: {active_service(host) or 'no rover service'}")
    remote = dict(reversed(ln.split(None, 1)) for ln in ssh(
        host, "cd ~ && sha256sum " + " ".join(Path(f).name for f in FILES) + " 2>/dev/null"
    ).stdout.split("\n") if ln.strip())
    differs = 0
    for f in FILES:
        name, local = Path(f).name, sha256(ROOT / f)
        state = ("same" if remote.get(name) == local else
                 "MISSING on the Pi" if name not in remote else "differs")
        differs += state != "same"
        print(f"  {name:22s} {state}")
    for unit, parts in UNITS.items():
        live = _unit_lines(ssh(host, f"systemctl cat {unit} 2>/dev/null").stdout)
        mine = [ln for p in parts for ln in _unit_lines((ROOT / p).read_text(encoding="utf-8"))]
        print(f"  {unit:22s} {'matches the repo' if live == mine else 'DRIFT from rover/systemd/'}")
    print(f"{differs} file(s) to deploy." + (" Run with --apply to deploy." if differs else ""))
    return 0


def _wait_http(host: str, path: str, timeout: float = 25.0) -> bytes | None:
    ip = host.split("@")[-1]
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://{ip}:{PORT}{path}", timeout=4) as r:
                return r.read()
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return None


def verify(host: str, service: str | None) -> bool:
    if service is None:
        print("no service was running, so none was restarted; start one to verify.")
        return True
    ok = ssh(host, f"systemctl is-active {service}").stdout.strip() == "active"
    print(f"  {service}: {'active' if ok else 'NOT ACTIVE'}")
    raw = _wait_http(host, "/raw")
    jpeg = bool(raw) and raw[:2] == JPEG_MAGIC
    print("  /raw: " + (f"clean frame, {len(raw)} bytes" if jpeg else "NO FRAME"))
    ok &= jpeg
    if service == "rover-control.service":
        tel = _wait_http(host, "/telemetry") or b""
        has = b"spatial_fire" in tel
        print(f"  /telemetry: {'reports the spatial decision' if has else 'OLD FORMAT OR NONE'}")
        ok &= has
    log = ssh(host, f"journalctl -u {service} --since '-2 min' --no-pager 2>/dev/null").stdout
    if "Traceback" in log:
        print("  journal shows a Traceback:\n" + log[-1500:])
        ok = False
    return ok


def apply(host: str) -> int:
    if not reachable(host):
        return check(host)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup, stage = f"~/deploy_backup/{stamp}", "~/deploy_staging"
    names = " ".join(Path(f).name for f in FILES)
    service = active_service(host)
    print(f"running before deploy: {service or 'none'}")

    r = ssh(host, f"mkdir -p {backup} && rm -rf {stage} && mkdir -p {stage} && "
                  f"cd ~ && for f in {names}; do [ -f $f ] && cp -p $f {backup}/; done; ls {backup}")
    print(f"backed up to {backup}: {r.stdout.split() or 'nothing (fresh install)'}")
    r = scp(host, [ROOT / f for f in FILES], stage.replace("~", "/home/pi"))
    if r.returncode:
        print("copy to the Pi failed:\n" + r.stderr)
        return 3
    # Compile and import with the Pi's own interpreter and libraries. Importing the
    # servers is side-effect free: threads, the camera and GPIO start only in main().
    r = ssh(host, f"cd {stage} && python3 -m py_compile {names} && "
                  f"python3 -c 'import spatial_policy, pi_camera_stream, pi_rover_control' "
                  f"&& echo STAGE_OK")
    if "STAGE_OK" not in r.stdout:
        print("the new files do not compile or import ON THE PI - nothing was installed.\n"
              + r.stdout + r.stderr)
        return 4
    print("staged files compile and import on the Pi")
    ssh(host, f"cp -p {stage}/*.py ~/ && rm -rf {stage}")
    if service:
        ssh(host, f"sudo systemctl restart {service}")
        print(f"restarted {service} (it starts disarmed, AUTO off, spray gate closed)")
    good = verify(host, service)
    print(("DEPLOYED and verified." if good else "DEPLOYED but verification FAILED - roll back:")
          + f"\n  rollback: python scripts/deploy_rover.py --rollback {backup}")
    return 0 if good else 5


def rollback(host: str, backup: str) -> int:
    if not reachable(host):
        return check(host)
    service = active_service(host)
    r = ssh(host, f"ls {backup}/*.py >/dev/null 2>&1 && cp -p {backup}/*.py ~/ && echo RESTORED")
    if "RESTORED" not in r.stdout:
        print(f"no backup at {backup}:\n{r.stderr}")
        return 3
    print(f"restored files from {backup}")
    if service:
        ssh(host, f"sudo systemctl restart {service}")
    ok = ssh(host, f"systemctl is-active {service}").stdout.strip() == "active" if service else True
    print(f"{service or 'no service'}: {'active' if ok else 'NOT ACTIVE'}")
    return 0 if ok else 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=HOST)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--rollback", metavar="BACKUP_DIR")
    a = ap.parse_args()
    if a.rollback:
        return rollback(a.host, a.rollback)
    return apply(a.host) if a.apply else check(a.host)


if __name__ == "__main__":
    raise SystemExit(main())
