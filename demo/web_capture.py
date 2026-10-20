"""Labelled camera capture through a web browser - the protocol's capture without a window.

demo/live_sprayer.py --camera opens an OpenCV window and takes keys; the OpenCV in
this project's environment has no window support, so that mode cannot run here.
This serves the same capture as a page on http://127.0.0.1:<port>/, usable from the
laptop's browser (or a phone, with --web-host 0.0.0.0). Nothing about the
measurement changes: the SAME SprayPipeline decides every frame, continuously, as
the window mode does; the SAME overlay is drawn; and SAVE writes the SAME log
record through the SAME SessionLog as the window's SPACE key.

Two guards the window left to the operator are enforced: a frame cannot be saved
without a true label, and a camera-fault frame (a stuck sensor) cannot be saved.

    python demo/live_sprayer.py --camera 192.168.137.167:8080 --web 8090 \
        --diagnosis-source "visual only (nursery stock)"
"""
from __future__ import annotations

import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from PIL import Image

from live_sprayer import (CONDITIONS, TRUTH_KEYS, SessionLog, SprayPipeline, draw,
                          open_camera)

#: Mean brightness above which a frame tagged 'low light' earns a warning. The first
#: real pilot tagged frames of 69-142 median brightness 'low light'; the dry run's low
#: light was photographs dimmed to about a third.
LOW_LIGHT_WARN_V = 90.0


class Capture:
    """The capture state the window keeps in local variables, made shareable."""

    def __init__(self, cap, pipe: SprayPipeline, log: SessionLog, max_misses: int) -> None:
        self.cap, self.pipe, self.log, self.max_misses = cap, pipe, log, max_misses
        self.lock = threading.Lock()
        self.truth, self.cond, self.leaf_n, self.plant_n = "", "normal", 1, 1
        self.frame = self.d = self.jpg = None
        self.frames_seen = 0
        self.per_cell: collections.Counter = collections.Counter()
        self.leaf_truth: dict[str, str] = {}
        self.message = "set the true label, then SAVE"
        self.stop = threading.Event()
        self.camera_lost = False

    @property
    def plant(self) -> str:
        return f"P{self.plant_n:02d}"

    @property
    def leaf(self) -> str:
        return f"{self.plant}L{self.leaf_n:02d}"

    def run(self) -> None:
        """Grab and decide continuously, exactly as the window loop does."""
        misses = 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                misses += 1
                if misses >= self.max_misses:
                    with self.lock:
                        self.camera_lost = True
                        self.message = "camera returned no frames - check the rover"
                time.sleep(0.25)
                continue
            misses = 0
            d = self.pipe.decide(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            with self.lock:
                ann = draw(frame, d, self.truth, self.cond, self.leaf, self.pipe.tau, self.plant)
                self.frame, self.d, self.camera_lost = frame, d, False
                self.jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.frames_seen += 1

    def act(self, what: str, value: str = "") -> dict:
        with self.lock:
            if what == "truth" and value in TRUTH_KEYS.values():
                self.truth = value
            elif what == "cond" and value in CONDITIONS.values():
                self.cond = value
            elif what == "next_leaf":
                self.leaf_n += 1
            elif what == "next_plant":
                self.plant_n, self.leaf_n = self.plant_n + 1, 1
            elif what == "save":
                self._save()
            return self.state()

    def _save(self) -> None:
        earlier = self.leaf_truth.get(self.leaf)
        if self.d is None:
            self.message = "no frame yet"
        elif not self.truth:
            self.message = "NOT saved: set the true label first (h / e / l)"
        elif self.d.camera_fault:
            self.message = "NOT saved: camera fault (stuck sensor) - power-cycle the Pi"
        elif earlier and earlier != self.truth:
            # Seen in the first real pilot: the same leaf saved as two diagnoses.
            self.message = (f"NOT saved: {self.leaf} was saved as {earlier} - one leaf has one "
                            f"diagnosis. Set the label back, or press n for a new leaf.")
        else:
            self.leaf_truth[self.leaf] = self.truth
            self.log.write(self.d, truth=self.truth, cond=self.cond, leaf=self.leaf,
                           frame_bgr=self.frame, src_path=None, tau=self.pipe.tau,
                           plant=self.plant)
            self.per_cell[(self.leaf, self.cond)] += 1
            d = self.d
            self.message = (f"saved #{self.log.n}: {self.leaf} {self.cond} truth={self.truth} "
                            f"pred={d.pred} conf={d.msp:.2f} spray={d.spray} "
                            f"spatial={d.spray_spatial}{' DEFERRED' if d.deferred else ''}")
            print(self.message, flush=True)
            v = float(cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)[..., 2].mean())
            if self.cond == "low_light" and v > LOW_LIGHT_WARN_V:
                # A warning, not a refusal: the protocol defines low light by the room
                # (lights off, indirect light only), not by a number.
                self.message += (f"  - WARNING: this looks brightly lit for 'low light' "
                                 f"(mean brightness {v:.0f}/255). Lights off?")

    def state(self) -> dict:
        d = self.d
        return {"plant": self.plant, "leaf": self.leaf, "truth": self.truth, "cond": self.cond,
                "saved": self.log.n, "this_cell": self.per_cell[(self.leaf, self.cond)],
                "message": self.message, "camera_lost": self.camera_lost,
                "pred": d.pred if d else None, "conf": round(d.msp, 3) if d else None,
                "spray": d.spray if d else None, "spatial": d.spray_spatial if d else None,
                "deferred": d.deferred if d else None,
                "camera_fault": d.camera_fault if d else None, "out": str(self.log.dir)}


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Leaf capture</title><style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
#cam{display:block;width:100%;max-width:640px;aspect-ratio:4/3;margin:0 auto;background:#000}
.row{max-width:640px;margin:6px auto;display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
button{font-size:15px;padding:10px 12px;border:0;border-radius:8px;background:#333;color:#eee;
 user-select:none;-webkit-user-select:none}
button.on{background:#2b4;color:#031}#save{background:#39c;color:#021;font-size:20px;
 width:95%;max-width:640px;padding:16px;font-weight:700}#fin{background:#a33;color:#fff}
#st{max-width:640px;margin:6px auto;font-size:14px;color:#9b9}#msg{color:#fd6;font-size:14px}
</style></head><body>
<canvas id=cam width=640 height=480></canvas>
<div id=st>...</div><div id=msg></div>
<div class=row id=truth></div><div class=row id=cond></div>
<button id=save>SAVE frame (space)</button>
<div class=row><button data-a=next_leaf>n: next leaf</button>
<button data-a=next_plant>p: next plant</button><button id=fin>finish session</button></div>
<script>
const T={h:'healthy',e:'early_blight',l:'late_blight'};
const C={'1':'normal','2':'low_light','3':'harsh_light','4':'distance_angle','5':'occlusion','6':'background'};
const tb=document.getElementById('truth'),cb=document.getElementById('cond');
for(const[k,v]of Object.entries(T))tb.insertAdjacentHTML('beforeend',`<button data-a=truth data-v=${v}>${k}: ${v.replace('_',' ')}</button>`);
for(const[k,v]of Object.entries(C))cb.insertAdjacentHTML('beforeend',`<button data-a=cond data-v=${v}>${k}: ${v.replace('_',' ')}</button>`);
async function act(a,v){const r=await fetch('/act',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({what:a,value:v||''})});show(await r.json());}
function show(s){document.getElementById('st').textContent=`plant ${s.plant} - leaf ${s.leaf} - truth ${s.truth||'NOT SET'} - ${s.cond}`+
 ` - saved ${s.saved} (this leaf+condition: ${s.this_cell}/5)`+(s.pred?` - pred ${s.pred} ${s.conf}`:'')+
 (s.camera_fault?' - CAMERA FAULT':s.deferred?' - deferred (too dark)':'');
 document.getElementById('msg').textContent=s.message;
 document.querySelectorAll('[data-a=truth]').forEach(b=>b.className=b.dataset.v===s.truth?'on':'');
 document.querySelectorAll('[data-a=cond]').forEach(b=>b.className=b.dataset.v===s.cond?'on':'');}
// 'next plant' changes the resampling unit of the whole analysis: confirm it is meant.
function go(a,v){if(a==='next_plant'&&!confirm('Start a NEW plant? Only for a different '+
 'physical plant - another leaf of the same plant is "n: next leaf".'))return;act(a,v);}
document.querySelectorAll('[data-a]').forEach(b=>b.onclick=()=>go(b.dataset.a,b.dataset.v));
document.getElementById('save').onclick=()=>act('save');
document.getElementById('fin').onclick=async()=>{if(confirm('Finish the session?')){await fetch('/quit',{method:'POST'});
 document.getElementById('msg').textContent='session finished - you can close this page';}};
document.addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();act('save');}else if(T[k])act('truth',T[k]);else if(C[k])act('cond',C[k]);
 else if(k==='n')act('next_leaf');else if(k==='p')go('next_plant');});
const cam=document.getElementById('cam'),ctx=cam.getContext('2d'),buf=new Image();let tm=null,at=Date.now();
function next(ms){clearTimeout(tm);tm=setTimeout(()=>{at=Date.now();buf.src='/frame.jpg?t='+at;},ms);}
buf.onload=()=>{ctx.drawImage(buf,0,0,cam.width,cam.height);next(60);};buf.onerror=()=>next(800);next(0);
setInterval(()=>{if(Date.now()-at>3000)next(0);},1000);
setInterval(async()=>{try{show(await(await fetch('/state')).json());}catch(e){}},500);
</script></body></html>"""


def make_server(spec: str, out_dir: Path, diagnosis_source: str = "",
                host: str = "127.0.0.1", port: int = 8090):
    """(server, capture); call capture.run in a thread and server.serve_forever."""
    cap, source, max_misses = open_camera(spec)
    if not cap.isOpened():
        raise SystemExit(f"camera {spec} could not be opened")
    if getattr(cap, "mode", None):
        source = f"{source} ({cap.mode})"
    pipe = SprayPipeline(device="cpu")
    log = SessionLog(out_dir, source=f"{source} via web capture",
                     diagnosis_source=diagnosis_source)
    capture = Capture(cap, pipe, log, max_misses)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/frame.jpg":
                with capture.lock:
                    jpg = capture.jpg
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(503)
            if path == "/state":
                with capture.lock:
                    return self._send(200, json.dumps(capture.state()).encode())
            return self._send(404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/act":
                return self._send(200, json.dumps(
                    capture.act(body.get("what", ""), body.get("value", ""))).encode())
            if path == "/quit":
                capture.stop.set()
                threading.Thread(target=srv.shutdown, daemon=True).start()
                return self._send(204)
            return self._send(404)

    srv = ThreadingHTTPServer((host, port), Handler)
    return srv, capture


def run_web(spec: str, out_dir: Path, diagnosis_source: str = "",
            host: str = "127.0.0.1", port: int = 8090) -> int:
    srv, capture = make_server(spec, out_dir, diagnosis_source, host, port)
    threading.Thread(target=capture.run, daemon=True).start()
    print(f"capture page: http://{'127.0.0.1' if host == '0.0.0.0' else host}:"
          f"{srv.server_address[1]}/  (frames -> {out_dir})", flush=True)
    try:
        srv.serve_forever()
    finally:
        capture.stop.set()
        capture.cap.release()
    print(f"saved {capture.log.n} labelled frames to {out_dir}", flush=True)
    return 0
