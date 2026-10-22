"""Motor and relay load on the Pi's 5 V input. Runs ON the Pi, against rover-control.

Phases (about 30 s in all; wheels off the ground, operator-confirmed right before):
  rest 3 s -> forward 3 s -> rest 2 s -> back 3 s -> rest 2 s
  -> spray pulse (1 s) -> rest 3 s -> spray pulse -> rest 3 s
Drive commands go through the control server, so its dead-man (0.4 s), obstacle stop
and spray gate all stay in force; forward is simply refused if the front is not clear.

A sampler thread logs, at ~100 Hz: EXT5V_V (the Pi's 5 V input, 3 ms single-channel
PMIC read) and the motor-driver and relay pins (pinctrl), so each reading carries what
the wheels and relay were actually told. Every 0.25 s it also logs the full PMIC rail sum.
STOP and spray-gate off in `finally`, whatever happens. One JSON object per line on stdout.
"""
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
RELAY_ONLY = sys.argv[1:] == ["relay"]   # `python3 - relay`: skip the wheels, pulses only
RAIL = re.compile(r"^\s*(\S+)_([AV]) \w+\(\d+\)=([\d.]+)[AV]$")
t0 = time.time()
out, phase = [], {"name": "start"}


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=2).read()


def tel():
    with urllib.request.urlopen(BASE + "/telemetry", timeout=2) as r:
        return json.loads(r.read())


def ext5v():
    s = subprocess.run(["vcgencmd", "pmic_read_adc", "EXT5V_V"], capture_output=True, text=True).stdout
    return float(s.split("=")[-1].rstrip("V\n"))


def rail_w():
    a, v = {}, {}
    for line in subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout.splitlines():
        m = RAIL.match(line)
        if m:
            (a if m.group(2) == "A" else v)[m.group(1)] = float(m.group(3))
    return sum(v[k] * a[k] for k in a if k in v)


def pins():
    # pinctrl lists pins in NUMBER order; parse each line by its pin number.
    lv, relay_on = {}, False
    for ln in subprocess.run(["pinctrl", "get", "17,27,22,23,24"], capture_output=True, text=True).stdout.splitlines():
        n = int(ln.split(":")[0])
        lv[n] = 1 if "| hi" in ln else 0
        if n == 24:
            relay_on = " op " in ln and "| lo" in ln      # active-low: driven low = energised
    wheels = {(1, 0, 1, 0): "forward", (0, 1, 0, 1): "back", (0, 0, 0, 0): "stopped"}.get(
        tuple(lv.get(p, 0) for p in (17, 27, 22, 23)), "other")
    return wheels, relay_on


def sampler(stop):
    last_full = 0.0
    while not stop.is_set():
        now = time.time()
        rec = {"k": "s", "t": round(now - t0, 3), "phase": phase["name"], "ext5v": ext5v()}
        rec["wheels"], rec["relay"] = pins()
        if now - last_full >= 0.25:
            rec["rail_W"] = round(rail_w(), 3)
            last_full = now
        out.append(rec)
        time.sleep(0.005)


def hold(name, secs, cmd=None):
    phase["name"] = name
    end = time.time() + secs
    while time.time() < end:
        if cmd:
            post("/drive", {"dir": cmd})
        time.sleep(0.15)


def mark(**kw):
    out.append({"k": "e", "t": round(time.time() - t0, 3), **kw})


s = tel()
mark(event="telemetry_before", **{k: s[k] for k in ("armed", "auto", "blocked", "sensor_lost", "distance_cm")})
stop = threading.Event()
th = threading.Thread(target=sampler, args=(stop,), daemon=True)
th.start()
try:
    hold("rest0", 3)
    if RELAY_ONLY:
        mark(event="relay_only")
    else:
        post("/arm", {"armed": True})
        hold("forward", 3, "forward")
        mark(event="after_forward", blocked=tel()["blocked"])
        hold("rest1", 2)
        hold("back", 3, "back")
        hold("rest2", 2)
    post("/spray_enable", {"on": True})
    phase["name"] = "spray1"; post("/spray"); hold("spray1", 1.4)
    hold("rest3", 3)
    phase["name"] = "spray2"; post("/spray"); hold("spray2", 1.4)
    hold("rest4", 3)
finally:
    post("/stop")
    post("/spray_enable", {"on": False})
    time.sleep(0.5)
    stop.set()
    th.join(timeout=2)
    s = tel()
    mark(event="telemetry_after", **{k: s[k] for k in ("armed", "auto", "direction")})
    mark(event="pins_after", wheels_relay=pins())
    print("\n".join(json.dumps(r) for r in out))
