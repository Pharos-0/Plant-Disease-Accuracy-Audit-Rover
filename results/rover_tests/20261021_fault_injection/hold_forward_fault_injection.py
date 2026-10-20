"""Ultrasonic fault injection on the rover (hardware report section 1), driven from the laptop.

Holds FORWARD for up to DURATION s (a drive command every 0.15 s, as the page does)
while the operator switches the HC-SR04 off and on with its power switch. Whenever
the sensor has been lost for 2 s, it checks that BACKING OUT still works (1.5 s of
reverse), then goes back to asking for forward. Every tick's telemetry and, every
~1.5 s, the motor-driver pins IN1..IN4 read on the Pi (pinctrl) are logged, so the
record shows what the wheels were told to do, not only what was requested.

Preconditions (operator-confirmed this session): wheels off the ground; front clear.
Refuses to start if an obstacle (not a lost sensor) blocks forward. STOP at the end.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://192.168.137.167:8080"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
OUT = Path(__file__).with_name("hold_forward_log.jsonl")
PATTERN = {(1, 0, 1, 0): "forward", (0, 1, 0, 1): "back", (0, 0, 0, 0): "stopped"}
t0 = time.time()
lines = []


def tel():
    with urllib.request.urlopen(BASE + "/telemetry", timeout=3) as r:
        return json.loads(r.read())


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=3).read()


def log(rec):
    rec["t"] = round(time.time() - t0, 2)
    lines.append(rec)


def pin_sampler(stop):
    while not stop.is_set():
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                "pi@192.168.137.167", "pinctrl get 17,27,22,23"],
                               capture_output=True, text=True, timeout=10)
            # pinctrl lists pins in NUMBER order (17, 22, 23, 27), not the order asked:
            # read each line's pin number, then report IN1..IN4 = GPIO 17, 27, 22, 23.
            level = {int(ln.split(":")[0]): 1 if "| hi" in ln else 0
                     for ln in r.stdout.splitlines() if ":" in ln}
            lv = tuple(level.get(p) for p in (17, 27, 22, 23))
            log({"kind": "pins", "IN1_IN4": lv, "wheels": PATTERN.get(lv, f"other {lv}")})
        except Exception as e:  # noqa: BLE001
            log({"kind": "pins", "error": str(e)})
        stop.wait(1.0)


s = tel()
if s["blocked"] and not s["sensor_lost"]:
    raise SystemExit(f"front not clear ({s['distance_cm']:.0f} cm) - not moving")
post("/arm", {"armed": True})
stop = threading.Event()
threading.Thread(target=pin_sampler, args=(stop,), daemon=True).start()
lost_since, backed_out = None, False
try:
    while time.time() - t0 < DURATION:
        s = tel()
        now = time.time()
        if s["sensor_lost"]:
            lost_since = lost_since or now
        else:
            lost_since, backed_out = None, False
        if lost_since and now - lost_since > 2.0 and not backed_out:
            cmd = "back"
            if now - lost_since > 3.5:
                backed_out = True
        else:
            cmd = "forward"
        post("/drive", {"dir": cmd})
        log({"kind": "tick", "cmd": cmd, "armed": s["armed"], "sensor_lost": s["sensor_lost"],
             "blocked": s["blocked"], "dist": s["distance_cm"] and round(s["distance_cm"])})
        time.sleep(0.15)
finally:
    post("/stop")
    stop.set()
    time.sleep(1.2)
    log({"kind": "end", "armed_after_stop": tel()["armed"]})
    OUT.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    print(f"logged {len(lines)} records to {OUT}")
