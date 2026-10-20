"""Sensor-loss fault injection on the robot, without touching the wiring.

The ultrasonic sensor shares its power switch with the motors and is soldered in, so
it cannot be failed alone by hand. Instead the deployed server's test hook is used:
while /tmp/rover_inject_sensor_loss exists, every ultrasonic reading is discarded,
exactly as if the sensor had died. Everything else is real - the control loop on the
Pi, the motor driver, the wheels (operator-confirmed off the ground, front clear).

Phases (about 13 s of motion): forward 3 s; INJECT; forward 3 s (must be refused);
reverse 2 s (must be allowed); forward 1.5 s (still refused); CLEAR; forward 3 s
(must resume); STOP. A logger on the Pi reads the motor-driver pins every 0.1 s, so
the record is the pin levels themselves, timed by the Pi's clock like the inject and
clear moments. The fault file is always removed at the end.
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
BASE, PI = "http://192.168.137.167:8080", "pi@192.168.137.167"
FAULT = "/tmp/rover_inject_sensor_loss"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", PI]
PATTERN = {(1, 0, 1, 0): "forward", (0, 1, 0, 1): "back", (0, 0, 0, 0): "stopped"}


def ssh(cmd):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=20).stdout.strip()


def tel():
    with urllib.request.urlopen(BASE + "/telemetry", timeout=3) as r:
        return json.loads(r.read())


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=3).read()


def hold(direction, seconds, phase, log):
    end = time.time() + seconds
    while time.time() < end:
        post("/drive", {"dir": direction})
        t = tel()
        log.append({"phase": phase, "asked": direction, "pi_time": None, "laptop": time.time(),
                    "sensor_lost": t["sensor_lost"], "blocked": t["blocked"],
                    "fault_injected": t.get("sensor_fault_injected")})
        time.sleep(0.15)


def main():
    ssh(f"rm -f {FAULT}")
    t0 = tel()
    if t0["blocked"]:
        raise SystemExit(f"front not clear or sensor already lost ({t0['distance_cm']}) - not moving")
    subprocess.run(["scp", "-o", "BatchMode=yes", str(HERE / "pin_logger.sh"),
                    f"{PI}:/tmp/pin_logger.sh"], capture_output=True, timeout=20)
    ssh("chmod +x /tmp/pin_logger.sh; nohup /tmp/pin_logger.sh 30 > /tmp/motor_pin_log.txt 2>&1 &")
    time.sleep(0.5)
    log, marks = [], {}
    try:
        post("/arm", {"armed": True})
        hold("forward", 3.0, "A sensor ok", log)
        marks["inject"] = float(ssh(f"touch {FAULT}; date +%s.%N"))
        hold("forward", 3.0, "B fault: forward asked", log)
        hold("back", 2.0, "C fault: reverse asked", log)
        hold("forward", 1.5, "D fault: forward asked", log)
        marks["clear"] = float(ssh(f"rm -f {FAULT}; date +%s.%N"))
        hold("forward", 3.0, "E cleared: forward asked", log)
    finally:
        post("/stop")
        ssh(f"rm -f {FAULT}")
        marks["stop"] = float(ssh("date +%s.%N"))
    time.sleep(1.0)
    pins = []
    for line in ssh("cat /tmp/motor_pin_log.txt").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        lv = dict(p.split("=") for p in parts[1:] if "=" in p)
        bits = tuple(1 if lv.get(str(n)) == "hi" else 0 for n in (17, 27, 22, 23))  # IN1..IN4
        pins.append({"pi_time": float(parts[0]), "IN1_IN4": bits,
                     "wheels": PATTERN.get(bits, f"other {bits}")})
    (HERE / "injected_sensor_fault_run.json").write_text(
        json.dumps({"marks_pi_time": marks, "telemetry": log, "pins": pins,
                    "armed_after_stop": tel()["armed"]}, indent=1), encoding="utf-8")
    # summary on the Pi's clock, relative to the injection
    ti, tc = marks["inject"], marks["clear"]
    prev = None
    print("pin state over time (seconds relative to fault injection, Pi clock):")
    for p in pins:
        if p["pi_time"] > marks["stop"] + 0.5:
            break
        if p["wheels"] != prev:
            rel = p["pi_time"] - ti
            tag = "  <- after CLEAR" if p["pi_time"] >= tc and (prev is None or True) and p["pi_time"] - tc < 5 else ""
            print(f"  {rel:+6.2f}s  wheels {p['wheels']}{tag}")
            prev = p["wheels"]
    lost = [x for x in log if x["sensor_lost"]]
    print(f"inject at 0.00s; clear at {tc - ti:+.2f}s; stop at {marks['stop'] - ti:+.2f}s")
    print(f"telemetry: sensor_lost reported on {len(lost)} of {len(log)} samples; "
          f"armed after stop = {tel()['armed']}")


if __name__ == "__main__":
    main()
