"""Sample the Pi 5 PMIC rails under three no-motion conditions. Runs on the Pi.

idle   : camera server capturing, no viewer
stream : one viewer pulling /stream (JPEG encoding per frame)
cpu4   : all four cores busy (worst-case compute draw)
No GPIO is touched.
"""
import json, multiprocessing as mp, re, subprocess, threading, time, urllib.request

PAT = re.compile(r"^\s*(\S+)_([AV]) \w+\(\d+\)=([\d.]+)[AV]$")


def read():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    a, v = {}, {}
    for line in out.splitlines():
        m = PAT.match(line)
        if m:
            (a if m.group(2) == "A" else v)[m.group(1)] = float(m.group(3))
    rails = {k: v[k] * a[k] for k in a if k in v}
    thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    return {"rail_W": sum(rails.values()), "core_W": rails.get("VDD_CORE", 0.0),
            "ext5v_V": v.get("EXT5V", float("nan")), "throttled": thr.split("=")[-1],
            "temp_C": temp}


def sample(label, n=20, dt=0.5):
    rows = []
    for _ in range(n):
        r = read(); r["cond"] = label; r["t"] = time.time(); rows.append(r)
        time.sleep(dt)
    return rows


def spin(stop):
    while not stop.is_set():
        pass


def pull_stream(stop):
    with urllib.request.urlopen("http://127.0.0.1:8080/stream", timeout=5) as r:
        while not stop.is_set():
            r.read(65536)


if __name__ == "__main__":
    rows = sample("idle")
    stop = threading.Event()
    t = threading.Thread(target=pull_stream, args=(stop,), daemon=True); t.start()
    time.sleep(2)
    rows += sample("stream")
    stop.set(); t.join(timeout=5)
    ev = mp.Event()
    procs = [mp.Process(target=spin, args=(ev,)) for _ in range(4)]
    for p in procs: p.start()
    time.sleep(2)
    rows += sample("cpu4")
    ev.set()
    for p in procs: p.join()
    time.sleep(3)
    rows += sample("idle_after", n=10)
    for r in rows:
        print(json.dumps(r))
