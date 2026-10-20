# Plant session runbook

One session closes the remaining hardware items and runs the first real camera session.
Steps are in the order they have to happen. Each says what to do, what you should see, and
what gets recorded. **You can also just tell Claude "step 3" and it will run that step.**

Run commands from the project folder (`D:\Final Project M.Tech`), in PowerShell or Git Bash.

> **What this session can and cannot show.** One shop-bought plant is a **pilot** (protocol
> §11): it cannot measure the paper's primary outcome (missed treatment on *diseased* plants),
> and no interval is computable from one plant. What it *can* show is fixed in advance in
> `docs/camera_experiment_protocol.md` §11.2 — false alarms on real healthy leaves, tissue
> detection on real foliage, confidence under shift, the §16 camera artefacts, and the rover's
> AUTO on a real plant. The analysis labels the session a PILOT by itself.

## Bring

| | Item | For |
|---|---|---|
| ☐ | the tomato plant (any leaves that look spotted or brown are worth photographing too) | steps 6–8 |
| ☐ | a **tape measure** | step 5 — camera height, never estimated |
| ☐ | something to lift the wheels off the floor (a box or two books) | steps 4, 8 |
| ☐ | a desk lamp or phone torch, a sheet of paper, a dark cloth | step 6 conditions |
| ☐ | *optional:* a cup of **water** (a drop of food colour helps) and a towel | step 9 — only if you decide to run the pump wet |
| ☐ | *optional:* a USB power meter | step 10 |

## 0. Power-on — relay boot state (§2 of the hardware report) · 2 min

**This must be the first thing**: it is about the moment the Pi powers up.

1. Tank empty / pump outlet disconnected. Turn the laptop hotspot on.
2. Look at the **relay module's LED** and power the rover on. Keep watching for ~60 s while it
   boots.
3. **Record verbatim:** did the relay LED light, or the relay click, at any moment during boot?

Expected: it never lights. (Released = pin floating as input; asserted = output low.)

## 1. Reach the Pi · 1 min

```bash
ping 192.168.137.167
```

No reply? The hotspot's device list (Windows: Settings → Network → Mobile hotspot) shows the
Pi's current address. If it changed, use the new one everywhere below
(`--host pi@NEW`, `--camera NEW:8080`, `--rover NEW:8080`, `--pi NEW:8080`).

## 2. See what will change · 1 min

```bash
.venv/Scripts/python.exe scripts/deploy_rover.py
```

Expected: three files `differs` (or `MISSING`, for `spatial_policy.py`), and the two systemd
units `matches the repo`. A `DRIFT` line just means the unit on the Pi differs from
`rover/systemd/` — tell Claude, nothing is broken.

## 3. Deploy the new servers · 2 min — needs your OK

What changes on the robot (all tested in software, `tests/test_rover_control_server.py`):

- **AUTO** sprays when *any* of 4 nozzle zones shows ≥ 15% leaf tissue, and **defers** (no
  spray, shows `TOO DARK`) on a frame too dark to judge. Before, it judged the whole frame and
  could miss a plant filling one zone, or spray in the dark.
- **Obstacle stop fails safe**: if the ultrasonic sensor stops reporting, forward is refused
  and the page shows `SENSOR LOST (forward refused)`. Reverse and turns still work.
  **So switch the ultrasonic sensor on, or forward will not drive — by design.**
- a clean `/raw` camera frame for measurement (no status banner).

```bash
.venv/Scripts/python.exe scripts/deploy_rover.py --apply
```

Expected: `staged files compile and import on the Pi`, then `DEPLOYED and verified.` It prints
a rollback line — keep it. If verification fails, run that rollback line.

## 4. Ultrasonic fault injection (§1 of the hardware report) · 5 min

Wheels **off the ground**. Start control mode and open the page:

```bash
ssh pi@192.168.137.167 "sudo systemctl start rover-control.service"
```

Open `http://192.168.137.167:8080/` on the laptop or phone. Sensor **switched on**.

1. Tap **ARM**, hold **▲** — the wheels turn forward.
2. While holding ▲, **switch the ultrasonic off with its power switch** (do not pull jumpers
   from a powered board).
3. **Record verbatim:** what the wheels do, and what the status line shows.
   Expected: wheels stop within about a second; `SENSOR LOST (forward refused)`.
4. Hold **▼** — expected: the wheels turn (you can always back out).
5. Switch the sensor back on, hold **▲** — expected: forward drives again.
6. Tap **■ STOP**.

## 5. Camera artefacts (§3 / §16) · 3 min

Put the plant in front of the rover, filling a good part of the camera view (check the page).
With the **tape**, measure (a) the **centre of the camera lens above the floor**, and (b) the
lens to the nearest leaf.

```bash
.venv/Scripts/python.exe scripts/capture_rover_artifact.py --height-cm <a> --plant-distance-cm <b> --note "tomato, pot on floor"
```

Expected: `saved results/rover_artifacts/<time>/plant_frame.jpg (640x480, from raw)` and
`spatial policy would spray: True`. It refuses a blank frame or a missing height.

## 6. Pilot capture through the rover camera (protocol §11) · 30–45 min

Switch to **view-only** mode so nothing can move or spray while your hands are at the plant:

```bash
ssh pi@192.168.137.167 "sudo systemctl start rover-camstream.service"
.venv/Scripts/python.exe demo/live_sprayer.py --camera 192.168.137.167:8080 --diagnosis-source "visual only (nursery stock)"
```

Check the first line says `rover camera: ... (raw)`. A window opens with the live decision
overlay; **click the window** so it receives the keys.

| key | does |
|---|---|
| `h` / `e` / `l` | true label: healthy / early blight / late blight — **only what you can honestly say** |
| `1` normal · `2` low light · `3` harsh light · `4` distance/angle · `5` occlusion · `6` background | condition |
| **SPACE** | save this frame (nothing is saved otherwise) |
| `n` | next leaf, same plant |
| `p` | next **plant** (only if you have more than one) |
| `q` | quit |

For **each leaf** (2–3 leaves): press `h`, then

1. `1` **normal** — room light, leaf about half the frame → SPACE ×5, nudging the leaf between shots.
2. `2` **low light** — room lights off, only indirect light → SPACE ×5. *(Priority condition.)*
3. if time: `3` torch close at an angle · `4` twice as far and tilted ~45° · `5` a third of the
   leaf covered with paper · `6` on dark cloth, then among soil/plants — SPACE ×5 each.
4. `n` for the next leaf.

Labels: a shop plant sold healthy is `h`, recorded as *visual only*. If a leaf looks spotted
or brown, do **not** guess a disease — leave it out, or photograph it and tell Claude; a guess
labelled as a diagnosis would be worse than no frame.

## 7. Analyse · 1 min

```bash
.venv/Scripts/python.exe scripts/analyse_camera_session.py camera_sessions/<the folder step 6 printed>
```

Expected: `PILOT - below the protocol's minimum…`, `PRIMARY OUTCOME NOT ESTIMABLE` (correct for
a healthy plant), and a **PILOT QUANTITIES** table: P2 false alarms, P3 tissue detection and
too-dark deferral, P4 confidence. Missed treatment shows `n/a` — *not measured*, not 0%.

Optional live read-out: `.venv/Scripts/python.exe scripts/classify_pi_camera.py --pi 192.168.137.167:8080 --n 10`

## 8. AUTO on the real plant (P6) · 5 min

Tank empty / pump disconnected (or step 9 first). Wheels off the ground.

```bash
ssh pi@192.168.137.167 "sudo systemctl start rover-control.service"
```

On the page: tick **tank empty / pump disconnected**, plant in view → the status line shows
`LEAF`. Tap **AUTO** → the relay clicks about every 3 s and `fires` counts up. Turn the room
lights off or cover the lens → `TOO DARK`, the clicking stops. Lights on → it resumes.
AUTO switches itself off after 2 minutes; tap it again to continue.
**Record** the `fires` count and what you saw. Tap **■ STOP**.

## 9. *Optional* — pump with water · 5 min — your decision

Only if you want the pump verified wet. **Water only — never pesticide.** Nozzle into a cup or
onto a towel, away from the Pi and the wiring. The page's checkbox says *tank empty / pump
disconnected*: for this test, ticking it is your deliberate bench-safety acknowledgement — it is
the gate that lets the relay fire. Tap **spray once** (one 1-second pulse). **Record:** did water
come out, and roughly how much per pulse (measure into the cup: that is a real dose-per-pulse
number).

## 10. *Optional* — power draw

With a USB power meter in line: read the watts idle, driving, and during a spray pulse.
Otherwise it stays *not measured*, which is fine — say so.

## 11. Finish

Tap **■ STOP**. Then switch back to the safe default and tell Claude what you recorded:

```bash
ssh pi@192.168.137.167 "sudo systemctl start rover-camstream.service"
```

Claude then writes the results into `docs/rover_hardware_test_report.md` (steps 0, 4, 5, 8–10)
and the pilot into the thesis and Paper 2 (steps 6–7), labelled as a one-plant pilot.

## If something goes wrong

| What you see | What it means / do |
|---|---|
| `cannot reach pi@…` | Pi off, not on the hotspot, or a new IP — see step 1 |
| `DEPLOYED but verification FAILED` | run the printed `--rollback` line; tell Claude |
| live_sprayer: `camera … could not be opened` | open `http://<ip>:8080/raw` in a browser; is a rover service running? |
| live_sprayer says `(snapshot-cropped)` | the new servers aren't deployed — still valid (banner cropped off), but do step 3 |
| forward does nothing, `SENSOR LOST` | ultrasonic switched off — switch it on (this is the fail-safe working) |
| wheels hum but don't move on the floor | power-bank current under load — lift the wheels, or use the battery |
| keys do nothing in live_sprayer | click on the video window first |
| `TOO DARK` in normal light | camera covered or very dim — that frame is deferred, by design |
