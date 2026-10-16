"""Phone/browser teleoperation server for the rover, served from the Pi.

Open the printed URL on a laptop or phone on the same network and you get the
live camera plus drive and spray controls. One process owns the camera and the
GPIO; Flask handlers only set a desired state that a single control loop applies.

SAFETY (this server drives real motors and the real spray relay)
----------------------------------------------------------------
* ARM toggle   - motor commands are ignored until you ARM; it auto-disarms after
                 DISARM_AFTER_S with no command.
* Dead-man     - every drive command is valid for only HOLD_S; the UI re-sends it
                 while a button is held, so releasing it - or losing the
                 connection - stops the rover within HOLD_S.
* Obstacle stop- a background read of the HC-SR04 forces motion off (and refuses
                 forward) when something is within OBSTACLE_STOP_CM.
* STOP         - cuts motors and disarms immediately.
* Spray        - the relay pulses only when the UI's "tank empty / pump
                 disconnected" box is ticked; it is a momentary test, SPRAY_S long.

Pins and the forward/reverse convention are those confirmed working by
rover/hardware_tests/test_components.py on 2026-10-16.

    python3 rover/pi_rover_control.py --port 8080
"""
from __future__ import annotations

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

# -- pins (BCM), from docs/rover_gpio_map.md, motor directions verified 2026-10-16
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 19
RELAY = 24
TRIG, ECHO = 5, 6
PWM_HZ = 1000

# -- safety constants
DUTY = 35                 # motor duty cycle while driving, %
HOLD_S = 0.4              # a drive command is valid this long (dead-man)
DISARM_AFTER_S = 60.0     # auto-disarm after this idle time
OBSTACLE_STOP_CM = 25.0   # forward is blocked within this distance
SPRAY_S = 1.0             # relay pulse length for the spray test
LOOP_HZ = 20

_state = {
    "armed": False, "direction": "stop", "active_until": 0.0,
    "distance_cm": None, "distance_at": 0.0, "blocked": False,
    "spray_request": False, "relay_until": 0.0, "last_cmd": 0.0,
}
DIST_STALE_S = 1.0        # a distance older than this is treated as unknown
_lock = threading.Lock()
_frame: bytes | None = None


# -------------------------------------------------------------- camera thread
def _camera_loop(index: int) -> None:
    global _frame
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened")
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        with _lock:
            armed, dist, blocked = _state["armed"], _state["distance_cm"], _state["blocked"]
        h, w = frame.shape[:2]
        banner = (f"{'ARMED' if armed else 'disarmed'}  "
                  f"dist {'-' if dist is None else f'{dist:.0f}cm'}"
                  f"{'  OBSTACLE' if blocked else ''}  {time.strftime('%H:%M:%S')}")
        cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(frame, banner, (8, 17), font, 0.5,
                    (0, 0, 255) if blocked else (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _lock:
                _frame = jpg.tobytes()
        time.sleep(0.03)


# -------------------------------------------------------------- ultrasonic
def _distance_cm(GPIO) -> float | None:
    GPIO.output(TRIG, False)
    time.sleep(0.0002)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    t0 = time.time()
    start = t0
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if start - t0 > 0.03:
            return None
    t1 = start
    while GPIO.input(ECHO) == 1:
        stop = time.time()
        if stop - t1 > 0.03:
            return None
    return (time.time() - t1) * 17150.0  # speed of sound /2, cm


def _ultrasonic_loop() -> None:
    """Read the HC-SR04 in its own thread, so its microsecond timing is not
    disturbed by the camera/control threads (that interference made every read
    time out to None when the read lived inside the control loop)."""
    import RPi.GPIO as GPIO
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    while True:
        d = _distance_cm(GPIO)
        if d is not None:
            with _lock:
                _state["distance_cm"] = d
                _state["distance_at"] = time.time()
        time.sleep(0.06)


# -------------------------------------------------------------- control loop
def _apply_motors(GPIO, direction: str) -> None:
    lf = direction in ("forward", "right")      # left side forward
    lr = direction in ("back", "left")          # left side reverse
    rf = direction in ("forward", "left")       # right side forward
    rr = direction in ("back", "right")         # right side reverse
    GPIO.output(IN1, lf); GPIO.output(IN2, lr)
    GPIO.output(IN3, rf); GPIO.output(IN4, rr)


def _control_loop() -> None:
    import RPi.GPIO as GPIO
    for p in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(p, GPIO.OUT)
    GPIO.setup(RELAY, GPIO.IN)                   # released = input (float trick)
    pwm_a, pwm_b = GPIO.PWM(ENA, PWM_HZ), GPIO.PWM(ENB, PWM_HZ)
    pwm_a.start(0); pwm_b.start(0)
    try:
        while True:
            now = time.time()
            with _lock:
                armed = _state["armed"]
                direction = _state["direction"]
                expired = now > _state["active_until"]
                idle = now - _state["last_cmd"] > DISARM_AFTER_S
                dist = _state["distance_cm"]
                fresh = now - _state["distance_at"] < DIST_STALE_S
                spray_req = _state["spray_request"]
                relay_until = _state["relay_until"]
                if idle and armed:
                    _state["armed"] = armed = False
                blocked = (fresh and dist is not None and dist < OBSTACLE_STOP_CM)
                _state["blocked"] = blocked
                if blocked and direction == "forward":
                    direction = "stop"
                if not armed or expired:
                    direction = "stop"
                # spray relay handling (all GPIO in this one thread)
                if spray_req and relay_until == 0.0:
                    _state["relay_until"] = now + SPRAY_S
                    _state["spray_request"] = False
                    relay_until = now + SPRAY_S

            # actuate motors
            if direction == "stop":
                pwm_a.ChangeDutyCycle(0); pwm_b.ChangeDutyCycle(0)
                _apply_motors(GPIO, "stop")
            else:
                _apply_motors(GPIO, direction)
                pwm_a.ChangeDutyCycle(DUTY); pwm_b.ChangeDutyCycle(DUTY)

            # actuate relay (active-low assert while within relay_until)
            if relay_until and now < relay_until:
                GPIO.setup(RELAY, GPIO.OUT); GPIO.output(RELAY, GPIO.LOW)
            elif relay_until:
                GPIO.setup(RELAY, GPIO.IN)
                with _lock:
                    _state["relay_until"] = 0.0

            time.sleep(1.0 / LOOP_HZ)
    finally:
        pwm_a.stop(); pwm_b.stop()
        for p in (IN1, IN2, IN3, IN4):
            GPIO.output(p, False)
        GPIO.setup(RELAY, GPIO.IN)


# -------------------------------------------------------------- web
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,user-scalable=no">
<title>Rover control</title><style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center;touch-action:none}
img{max-width:100%;height:auto}#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;max-width:360px;margin:8px auto}
button{font-size:20px;padding:16px;border:0;border-radius:10px;background:#2b4;color:#042;font-weight:700}
.dir{background:#39c;color:#021}.stop{background:#e33;color:#fff;grid-column:1/4}.sp{background:#fa0;color:#311}
#arm{background:#555;color:#ccc;width:95%;max-width:360px;margin:8px auto;display:block}
#arm.on{background:#2b4;color:#042}.row{max-width:360px;margin:6px auto;font-size:14px;color:#9b9}
label{font-size:14px}</style></head><body>
<img src="/stream" alt="rover camera"><div class=row id=tele>…</div>
<button id=arm onclick="arm()">ARM to enable driving</button>
<div id=grid>
<span></span><button class=dir data-d=forward>▲</button><span></span>
<button class=dir data-d=left>◀</button><button class=dir data-d=back>▼</button><button class=dir data-d=right>▶</button>
<button class=stop onclick="stopAll()">■ STOP</button></div>
<div class=row><label><input type=checkbox id=tank> tank empty / pump disconnected</label>
<button class=sp onclick="spray()">spray test</button></div>
<script>
let armed=false,held=null;
function arm(){armed=!armed;document.getElementById('arm').className=armed?'on':'';
 document.getElementById('arm').textContent=armed?'ARMED — tap to disarm':'ARM to enable driving';
 fetch('/arm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({armed})});}
function send(d){fetch('/drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:d})});}
function stopAll(){armed=false;document.getElementById('arm').className='';
 document.getElementById('arm').textContent='ARM to enable driving';
 clearInterval(held);held=null;fetch('/stop',{method:'POST'});}
function spray(){if(!document.getElementById('tank').checked){alert('Tick the tank-empty box first.');return;}
 fetch('/spray',{method:'POST'});}
document.querySelectorAll('.dir').forEach(b=>{
 const start=e=>{e.preventDefault();if(!armed)return;const d=b.dataset.d;send(d);clearInterval(held);held=setInterval(()=>send(d),150);};
 const end=e=>{e.preventDefault();clearInterval(held);held=null;fetch('/stop',{method:'POST'});};
 b.addEventListener('pointerdown',start);b.addEventListener('pointerup',end);b.addEventListener('pointerleave',end);});
setInterval(async()=>{try{const t=await(await fetch('/telemetry')).json();
 document.getElementById('tele').textContent=
  (t.armed?'ARMED':'disarmed')+' · '+(t.distance_cm==null?'dist —':'dist '+t.distance_cm.toFixed(0)+'cm')
  +(t.blocked?' · OBSTACLE':'')+' · '+t.direction;}catch(e){}},400);
</script></body></html>"""


@app.post("/arm")
def arm():
    on = bool(request.get_json(force=True).get("armed"))
    with _lock:
        _state["armed"] = on
        _state["last_cmd"] = time.time()
        if not on:
            _state["direction"] = "stop"
    return ("", 204)


@app.post("/drive")
def drive():
    d = request.get_json(force=True).get("dir", "stop")
    if d not in ("forward", "back", "left", "right"):
        d = "stop"
    now = time.time()
    with _lock:
        if _state["armed"]:
            _state["direction"] = d
            _state["active_until"] = now + HOLD_S
            _state["last_cmd"] = now
    return ("", 204)


@app.post("/stop")
def stop():
    with _lock:
        _state["direction"] = "stop"
        _state["active_until"] = 0.0
    return ("", 204)


@app.post("/spray")
def spray():
    with _lock:
        _state["spray_request"] = True
        _state["last_cmd"] = time.time()
    return ("", 204)


@app.get("/telemetry")
def telemetry():
    with _lock:
        fresh = time.time() - _state["distance_at"] < DIST_STALE_S
        return jsonify({"armed": _state["armed"], "direction": _state["direction"],
                        "blocked": _state["blocked"],
                        "distance_cm": _state["distance_cm"] if fresh else None})


@app.get("/")
def index():
    return PAGE


@app.get("/stream")
def stream():
    def gen():
        while True:
            with _lock:
                f = _frame
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    threading.Thread(target=_camera_loop, args=(a.camera,), daemon=True).start()
    threading.Thread(target=_ultrasonic_loop, daemon=True).start()
    threading.Thread(target=_control_loop, daemon=True).start()
    print(f"rover control at http://0.0.0.0:{a.port}/  (camera {a.camera})")
    app.run(host="0.0.0.0", port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
