"""Phone/browser teleoperation + spatial auto-spray for the rover, served from the Pi.

Open the printed URL on a laptop or phone on the same network: live camera, drive
and spray controls, and an AUTO button. One process owns the camera and the GPIO;
Flask handlers set a desired state that a single control loop applies.

AUTOMATION IS SPATIAL, BY DESIGN
--------------------------------
AUTO sprays the relay when *leaf tissue* is in view, not when a disease classifier
calls a plant diseased. That is the project's "spray where, not whether" rule: under
laboratory-to-field shift the classifier is unreliable, so letting it withhold or
trigger treatment is the unsafe design. The decision is rover/spatial_policy.py -
the analysed spatial-only policy itself, checked frame for frame against the demo
(tests/test_spatial_policy.py): any of 4 nozzle zones >= 15% tissue of any plant
colour, and a too-dark frame DEFERRED (no spray, flagged) rather than judged.
Running the disease classifier on the live feed is a separate, laptop-side
*readout* (scripts/classify_pi_camera.py) that never drives the relay.

SAFETY (this server drives real motors and the real spray relay)
----------------------------------------------------------------
* ARM         - motor commands ignored until armed; auto-disarms after idle.
* Dead-man    - each drive command is valid HOLD_S; the UI re-sends while held, so
                releasing it or losing the link stops the rover within HOLD_S.
* Obstacle    - the HC-SR04 runs in its own thread; forward is refused within
                OBSTACLE_STOP_CM. FAIL-SAFE: a reading older than DIST_STALE_S, or
                none at all (sensor off, unplugged or failed), counts as an
                obstacle, so forward is refused until the sensor reports again and
                the UI shows SENSOR LOST. Reverse and turns stay available so the
                operator can back out. (The delivered code failed open here -
                defect D2: a missing reading silently removed the obstacle stop.)
* Spray gate  - the relay fires (manual or AUTO) only when the UI's "tank empty /
                pump disconnected" box is ticked (a server-side flag, not just
                client-side), and AUTO auto-stops after AUTO_MAX_S.
* STOP        - cuts motors, disarms, turns AUTO off.

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

from spatial_policy import spatial_decision   # deploy beside this file

# -- pins (BCM); motor directions verified 2026-10-16
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 19
RELAY = 24
TRIG, ECHO = 5, 6
PWM_HZ = 1000

# -- safety / behaviour constants
DUTY = 35                 # motor duty while driving, %
HOLD_S = 0.4              # a drive command is valid this long (dead-man)
DISARM_AFTER_S = 60.0     # auto-disarm after this idle time (manual mode)
OBSTACLE_STOP_CM = 25.0   # forward blocked within this distance
SPRAY_S = 1.0             # relay pulse length
DIST_STALE_S = 1.0        # a distance older than this is "unknown"
LOOP_HZ = 20
# auto-spray (spatial): the decision itself is rover/spatial_policy.py
AUTO_SPRAY_COOLDOWN = 3.0
AUTO_MAX_S = 120.0        # AUTO turns itself off after this

_state = {
    "armed": False, "direction": "stop", "active_until": 0.0,
    "distance_cm": None, "distance_at": 0.0, "blocked": False, "sensor_lost": True,
    "spray_request": False, "relay_until": 0.0, "last_cmd": 0.0,
    "spray_enabled": False, "auto": False, "auto_until": 0.0,
    "tissue_frac": 0.0, "tissue_zones": [0.0] * 4, "deferred": False,
    "spatial_fire": False, "auto_last_spray": 0.0, "auto_fires": 0,
}
_lock = threading.Lock()
_frame: bytes | None = None   # annotated with the status banner, for the operator
_raw: bytes | None = None     # the same frame without it, for measurement (/raw)


def _camera_loop(index: int) -> None:
    global _frame, _raw
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened")
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        sd = spatial_decision(frame)      # judged on the clean frame
        tissue = sd["tissue_frac"]
        # Encode the clean frame before the banner is drawn onto it in place: the
        # banner is black with coloured text and must not reach a classifier or a
        # tissue measurement made off the Pi.
        ok_raw, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok_raw:
            with _lock:
                _raw = raw.tobytes()
        with _lock:
            _state["tissue_frac"] = tissue
            _state["tissue_zones"] = sd["zones"]
            _state["deferred"] = sd["deferred"]
            _state["spatial_fire"] = sd["fire"]
            armed, dist = _state["armed"], _state["distance_cm"]
            blocked, auto = _state["blocked"], _state["auto"]
            lost = _state["sensor_lost"]
        h, w = frame.shape[:2]
        banner = (f"{'AUTO' if auto else ('ARMED' if armed else 'disarmed')}  "
                  f"tissue {tissue * 100:3.0f}%"
                  f"{'  TOO DARK' if sd['deferred'] else ('  LEAF' if sd['fire'] else '')}  "
                  f"dist {'-' if lost or dist is None else f'{dist:.0f}cm'}"
                  f"{'  SENSOR LOST' if lost else ('  OBSTACLE' if blocked else '')}"
                  f"  {time.strftime('%H:%M:%S')}")
        cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(frame, banner, (8, 17), font, 0.5,
                    (0, 0, 255) if blocked else (0, 220, 0) if auto else (255, 255, 255),
                    1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _lock:
                _frame = jpg.tobytes()
        time.sleep(0.03)


# -------------------------------------------------------------- ultrasonic
def _distance_cm(GPIO) -> float | None:
    GPIO.output(TRIG, False); time.sleep(0.0002)
    GPIO.output(TRIG, True); time.sleep(0.00001)
    GPIO.output(TRIG, False)
    t0 = start = time.time()
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if start - t0 > 0.03:
            return None
    t1 = start
    while GPIO.input(ECHO) == 1:
        if time.time() - t1 > 0.03:
            return None
    return (time.time() - t1) * 17150.0


def _ultrasonic_loop() -> None:
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
    GPIO.output(IN1, direction in ("forward", "right"))
    GPIO.output(IN2, direction in ("back", "left"))
    GPIO.output(IN3, direction in ("forward", "left"))
    GPIO.output(IN4, direction in ("back", "right"))


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
                dist = _state["distance_cm"]
                fresh = now - _state["distance_at"] < DIST_STALE_S
                spray_enabled = _state["spray_enabled"]
                auto = _state["auto"] and now < _state["auto_until"]
                _state["auto"] = auto
                spatial_fire = _state["spatial_fire"]   # False when deferred (too dark)
                relay_until = _state["relay_until"]
                # idle auto-disarm (manual only; AUTO keeps itself alive)
                if not auto and armed and now - _state["last_cmd"] > DISARM_AFTER_S:
                    _state["armed"] = armed = False
                # Fail-safe: no fresh reading is treated as an obstacle, not as a
                # clear path - loss of evidence must never widen what the rover does.
                sensor_lost = not (fresh and dist is not None)
                blocked = sensor_lost or dist < OBSTACLE_STOP_CM
                _state["blocked"] = blocked
                _state["sensor_lost"] = sensor_lost
                if blocked and direction == "forward":
                    direction = "stop"
                if not armed or expired:
                    direction = "stop"
                # spatial auto-spray: fire when any nozzle zone holds leaf tissue
                if (auto and armed and spray_enabled and spatial_fire
                        and now - _state["auto_last_spray"] > AUTO_SPRAY_COOLDOWN):
                    _state["spray_request"] = True
                    _state["auto_last_spray"] = now
                    _state["auto_fires"] += 1
                # relay request -> timed pulse, but only if the tank-empty gate is set
                if _state["spray_request"]:
                    _state["spray_request"] = False
                    if spray_enabled and relay_until == 0.0:
                        _state["relay_until"] = relay_until = now + SPRAY_S

            if direction == "stop":
                pwm_a.ChangeDutyCycle(0); pwm_b.ChangeDutyCycle(0)
                _apply_motors(GPIO, "stop")
            else:
                _apply_motors(GPIO, direction)
                pwm_a.ChangeDutyCycle(DUTY); pwm_b.ChangeDutyCycle(DUTY)

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
#arm,#auto{width:95%;max-width:360px;margin:6px auto;display:block;background:#555;color:#ccc}
#arm.on{background:#2b4;color:#042}#auto.on{background:#0c6;color:#021}
.row{max-width:360px;margin:6px auto;font-size:14px;color:#9b9}label{font-size:14px}</style></head><body>
<img src="/stream" alt="rover camera"><div class=row id=tele>…</div>
<button id=arm onclick="arm()">ARM to enable driving</button>
<div id=grid>
<span></span><button class=dir data-d=forward>▲</button><span></span>
<button class=dir data-d=left>◀</button><button class=dir data-d=back>▼</button><button class=dir data-d=right>▶</button>
<button class=stop onclick="stopAll()">■ STOP</button></div>
<div class=row><label><input type=checkbox id=tank onchange="enable()"> tank empty / pump disconnected</label></div>
<button class=sp onclick="spray()" style="width:95%;max-width:360px">spray once</button>
<button id=auto onclick="toggleAuto()">AUTO: spray where leaf tissue is seen (off)</button>
<div class=row>Automation is spatial: it sprays where it sees a plant, not from a disease call.</div>
<script>
let armed=false,auto=false,held=null;
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):null});}
function arm(){armed=!armed;const e=document.getElementById('arm');e.className=armed?'on':'';
 e.textContent=armed?'ARMED — tap to disarm':'ARM to enable driving';post('/arm',{armed});}
function enable(){post('/spray_enable',{on:document.getElementById('tank').checked});}
function spray(){if(!document.getElementById('tank').checked){alert('Tick the tank-empty box first.');return;}post('/spray');}
function toggleAuto(){if(!auto&&!document.getElementById('tank').checked){alert('Tick the tank-empty box before AUTO.');return;}
 auto=!auto;const e=document.getElementById('auto');e.className=auto?'on':'';
 e.textContent='AUTO: spray where leaf tissue is seen ('+(auto?'ON':'off')+')';post('/auto',{on:auto});
 if(auto){armed=true;document.getElementById('arm').className='on';document.getElementById('arm').textContent='ARMED — tap to disarm';}}
function stopAll(){armed=false;auto=false;
 document.getElementById('arm').className='';document.getElementById('arm').textContent='ARM to enable driving';
 document.getElementById('auto').className='';document.getElementById('auto').textContent='AUTO: spray where leaf tissue is seen (off)';
 clearInterval(held);held=null;post('/stop');}
function send(d){post('/drive',{dir:d});}
document.querySelectorAll('.dir').forEach(b=>{
 const start=e=>{e.preventDefault();if(!armed)return;const d=b.dataset.d;send(d);clearInterval(held);held=setInterval(()=>send(d),150);};
 const end=e=>{e.preventDefault();clearInterval(held);held=null;post('/stop');};
 b.addEventListener('pointerdown',start);b.addEventListener('pointerup',end);b.addEventListener('pointerleave',end);});
setInterval(async()=>{try{const t=await(await fetch('/telemetry')).json();
 document.getElementById('tele').textContent=
  (t.auto?'AUTO':(t.armed?'ARMED':'disarmed'))+' · tissue '+(t.tissue_frac*100).toFixed(0)+'%'
  +(t.deferred?' · TOO DARK':(t.spatial_fire?' · LEAF':''))
  +' · '+(t.distance_cm==null?'dist —':'dist '+t.distance_cm.toFixed(0)+'cm')
  +(t.sensor_lost?' · SENSOR LOST (forward refused)':(t.blocked?' · OBSTACLE':''))
  +' · fires '+t.auto_fires;}catch(e){}},400);
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
        _state["auto"] = False
        _state["armed"] = False
    return ("", 204)


@app.post("/spray_enable")
def spray_enable():
    on = bool(request.get_json(force=True).get("on"))
    with _lock:
        _state["spray_enabled"] = on
        if not on:
            _state["auto"] = False
    return ("", 204)


@app.post("/spray")
def spray():
    with _lock:
        if _state["spray_enabled"]:
            _state["spray_request"] = True
            _state["last_cmd"] = time.time()
    return ("", 204)


@app.post("/auto")
def auto():
    on = bool(request.get_json(force=True).get("on"))
    now = time.time()
    with _lock:
        if on and not _state["spray_enabled"]:
            return ("", 204)                    # refuse AUTO without the tank gate
        _state["auto"] = on
        _state["auto_until"] = now + AUTO_MAX_S if on else 0.0
        if on:
            _state["armed"] = True
            _state["last_cmd"] = now
    return ("", 204)


@app.get("/telemetry")
def telemetry():
    with _lock:
        fresh = time.time() - _state["distance_at"] < DIST_STALE_S
        return jsonify({"armed": _state["armed"], "auto": _state["auto"],
                        "direction": _state["direction"], "blocked": _state["blocked"],
                        "sensor_lost": _state["sensor_lost"],
                        "tissue_frac": _state["tissue_frac"], "auto_fires": _state["auto_fires"],
                        "tissue_zones": _state["tissue_zones"], "deferred": _state["deferred"],
                        "spatial_fire": _state["spatial_fire"],
                        "distance_cm": _state["distance_cm"] if fresh else None})


@app.get("/")
def index():
    return PAGE


@app.get("/snapshot")
def snapshot():
    with _lock:
        f = _frame
    if f is None:
        return ("no frame", 503)
    return Response(f, mimetype="image/jpeg")


@app.get("/raw")
def raw():
    """The latest frame without the status banner, at higher JPEG quality - what a
    laptop-side classifier or protocol capture should judge. Read-only."""
    with _lock:
        f = _raw
    if f is None:
        return ("no frame", 503)
    return Response(f, mimetype="image/jpeg")


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
