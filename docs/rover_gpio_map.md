# Rover GPIO Map

**Status: derived from source code, NOT measured.** Every pin below comes from the
static audit of the delivered `web_rover.py` recorded in `docs/rover_code_audit.md`.
None has been confirmed against the physical wiring, because the physical rover has
not been inspected in this session (§9 requires hands-on verification).

Treat this file as **a hypothesis about the wiring**, to be confirmed or corrected by
running `rover/hardware_tests/` on the actual Pi. A pin that has not been tested is
marked `UNVERIFIED` and must stay that way until an isolation test produces evidence.

Numbering convention: **BCM**, as used by `RPi.GPIO.setmode(GPIO.BCM)`.

---

## 1. Pin assignments

| BCM pin | Subsystem | Role | Direction | Verified? |
|---:|---|---|---|---|
| 17 | Motor driver | IN1 — left motor direction A | OUT | `UNVERIFIED` |
| 27 | Motor driver | IN2 — left motor direction B | OUT | `UNVERIFIED` |
| 22 | Motor driver | IN3 — right motor direction A | OUT | `UNVERIFIED` |
| 23 | Motor driver | IN4 — right motor direction B | OUT | `UNVERIFIED` |
| 18 | Motor driver | ENA — left enable, PWM @ 1 kHz, duty 20–100 | OUT (PWM) | `UNVERIFIED` |
| 19 | Motor driver | ENB — right enable, PWM @ 1 kHz, duty 20–100 | OUT (PWM) | `UNVERIFIED` |
| 24 | Relay | Pump on/off. Active-low module | OUT / IN | `UNVERIFIED` |
| 5 | HC-SR04 | TRIG | OUT | `UNVERIFIED` |
| 6 | HC-SR04 | ECHO | IN | `UNVERIFIED` |

Camera is not a GPIO device: the code opens `cv2.VideoCapture(0)` at 320×240, i.e. a
USB/V4L2 device, **not** the CSI ribbon camera. Which physical camera occupies index 0
is `UNVERIFIED`.

## 2. The relay float trick — preserve it

The delivered code releases the relay by reconfiguring pin 24 as an **input**
(`GPIO.setup(RELAY, GPIO.IN)`) and asserts it as `OUT` + `LOW`. The audit judged this
correct and deliberate: it keeps an active-low relay module de-asserted at boot, when
GPIOs float. Any rewrite must keep this behaviour — driving the pin `HIGH` instead
would energise the pump on some modules during boot.

Defect **D6** notes the delivered code calls `GPIO.setup()` on this pin every frame
rather than on state transitions. Fix that, but keep the float semantics.

## 3. Pins deliberately absent

The delivered paper claims subsystems that have **no pin assignment and no code**
(audit §2): IR line sensors, battery/ADC voltage monitoring, and a pesticide tank
sensor all appear **0 times** in the source. There is nothing to map. If hardware for
these is physically present on the chassis, it is unwired as far as the software is
concerned, and that must be stated rather than implied otherwise.

## 4. Safety notes carried from the audit

- **D2 — three stacked fail-open behaviours.** `SPRAY_ENABLED = True` at startup; the
  ultrasonic returns `None` on timeout; and `if dist and dist < 20` treats `None` as
  falsy, so a failed sensor **silently disables** the obstacle stop while the rover
  keeps driving and spraying. Any bench test must therefore include *unplugging the
  HC-SR04* and confirming the intended behaviour, not just that it works when healthy.
- **D3 — no `try/finally`.** Motors and the spray relay stay live on any exception.
  Fix before any unattended run.
- Test the relay and pump **with the tank empty or the pump outlet disconnected**
  until the wiring is confirmed.

## 5. How to verify

Run each script in `rover/hardware_tests/` on the Pi, one subsystem at a time, and
paste its output into `docs/rover_hardware_test_report.md`. Each prints a machine
-readable result line so the report can be assembled from evidence rather than memory.
