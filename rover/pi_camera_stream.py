"""View-only rover camera stream, served from the Pi to any browser on the network.

A free, account-free replacement for the camera half of a VNC session: open the
printed URL on a laptop or phone (same Wi-Fi / hotspot as the Pi) and watch the
rover's USB camera live. Multiple viewers share one capture.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It serves the camera. It drives **no GPIO** - no motors, no relay, no pump - and
imports nothing that could. It is safe to leave running. Operating the rover
(drive / spray) is a separate, deliberately gated server and is not this file.

    python3 rover/pi_camera_stream.py            # 0.0.0.0:8080, /dev/video0
    python3 rover/pi_camera_stream.py --port 8080 --camera 0

Then browse to  http://<pi-ip>:8080/  from any device on the same network.
Routes: /  (page)   /stream  (MJPEG)   /snapshot  (one JPEG)   /healthz,
and /raw - one JPEG without the view-only banner, for measurement: the banner is
black with white text and must not reach a classifier or tissue measurement.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_lock = threading.Lock()
_latest: bytes | None = None       # annotated, for viewers
_latest_raw: bytes | None = None   # clean, for measurement (/raw)
_meta = {"frames": 0, "started": time.time(), "w": 0, "h": 0}


def _capture_loop(index: int) -> None:
    global _latest, _latest_raw
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened")
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        ok_raw, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok_raw:      # before the banner is drawn onto the frame in place
            with _lock:
                _latest_raw = raw.tobytes()
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(frame, "ROVER CAMERA - VIEW ONLY  " + time.strftime("%H:%M:%S"),
                    (8, 17), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        with _lock:
            _latest = jpg.tobytes()
            _meta.update(frames=_meta["frames"] + 1, w=w, h=h)
        time.sleep(0.03)  # ~30 fps cap


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Rover camera</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif;text-align:center}
#cam{display:block;width:100%;max-width:640px;aspect-ratio:4/3;margin:8px auto;background:#000}.b{padding:6px;font-size:14px;color:#9b9}</style>
</head><body><div class=b>Rover camera - view only. No control is exposed on this page.</div>
<canvas id=cam width=640 height=480></canvas>
<script>
// One fresh /snapshot at a time: self-heals after a restart or a Wi-Fi drop, which an
// <img> on the MJPEG /stream does not, and paces itself to the link.
const cam=document.getElementById('cam'),camCtx=cam.getContext('2d'),buf=new Image();
let camT=null,camAt=Date.now(),camFrames=0;
function nextFrame(ms){clearTimeout(camT);camT=setTimeout(()=>{camAt=Date.now();
 buf.src='/snapshot?t='+camAt;},ms);}
buf.onload=()=>{if(cam.width!==buf.naturalWidth||cam.height!==buf.naturalHeight){
 cam.width=buf.naturalWidth;cam.height=buf.naturalHeight;}
 camCtx.drawImage(buf,0,0);camFrames++;nextFrame(40);};
buf.onerror=()=>nextFrame(800);nextFrame(0);
setInterval(()=>{if(Date.now()-camAt>3000)nextFrame(0);},1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        # the page asks for /snapshot?t=<time> so browsers never cache a frame
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/healthz":
            with _lock:
                m = dict(_meta)
            body = (f"ok frames={m['frames']} size={m['w']}x{m['h']} "
                    f"uptime={int(time.time() - m['started'])}s").encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/snapshot", "/raw"):
            with _lock:
                frame = _latest_raw if path == "/raw" else _latest
            if frame is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _latest
                    if frame is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(frame)).encode()
                                         + b"\r\n\r\n" + frame + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    threading.Thread(target=_capture_loop, args=(a.camera,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"view-only rover camera at http://0.0.0.0:{a.port}/  (camera {a.camera})")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
