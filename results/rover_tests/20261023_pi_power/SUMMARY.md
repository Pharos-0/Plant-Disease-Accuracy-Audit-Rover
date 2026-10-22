# Raspberry Pi 5 power draw — 2026-10-23

**What was measured:** the Pi's own power, read from its power-management chip
(`vcgencmd pmic_read_adc`), on the rover, with nothing moving. No GPIO was touched (motor pins
and the relay pin unclaimed before and after; `pinctrl get` checked after the run). The
view-only camera server (`rover-camstream`) was running throughout, capturing continuously.
It was not stopped, because restarting it reopens the camera, and this camera can fail on a
reopen (`docs/rover_hardware_test_report.md`).

**What was not measured:** the motors, the pump, the USB camera's own draw (it is fed from
the Pi's USB 5 V, which the chip does not meter), and the battery or power-bank capacity. No
watt-hour or run-time figure follows from this.

## Method

`pmic_sample.py` ran on the Pi. Sampling was 20 samples per condition at 0.5 s intervals (10
for `idle_after`). For each sample it reads every rail that reports both a voltage and a
current, multiplies them, and sums the products ("rail power"). It also logs `EXT5V_V`
(the 5 V input at the Pi), `get_throttled` and the SoC temperature. Raw samples:
`pmic_samples.jsonl`.

| Condition | What was running |
|---|---|
| `idle` | camera server capturing, nobody viewing (Pi up < 1 min) |
| `stream` | one viewer pulling `/stream` on the Pi itself |
| `cpu4` | all four cores in busy loops — worst-case compute |
| `idle_after` | back to idle, 3 s after the load stopped |

## Results

| Condition | Rail power, median [min–max] | of which VDD_CORE | 5 V input, min / median | SoC temp max | Throttled |
|---|---|---|---|---|---|
| idle | 1.97 W [1.47–2.89] | 0.68 W | 5.021 / 5.044 V | 44.1 °C | 0x0 |
| stream | 1.67 W [1.41–2.15] | 0.46 W | 5.021 / 5.048 V | 44.1 °C | 0x0 |
| cpu4 | 4.78 W [3.85–5.30] | 3.52 W | 4.951 / 4.974 V | 57.9 °C | 0x0 |
| idle_after | 1.72 W [1.45–2.48] | 0.52 W | 5.014 / 5.041 V | 51.8 °C | 0x0 |

## Reading it

- **The Pi itself draws about 1.7–2 W** running the rover's camera server, and **about 5 W at
  full compute**. Viewing the stream adds nothing measurable, because the server already
  captures and encodes every frame whether or not anyone is watching. The first idle block
  started less than a minute after boot, which probably explains its slightly higher median.
- **The supply held:** the 5 V input stayed at or above 4.95 V under full load, with no
  under-voltage or throttling flag (`0x0` in every sample).
- **The rail sum is a lower bound on the Pi's input power, not the input power itself.** It
  leaves out the power chip's conversion losses, the DDR rails (they report 0 A on this board)
  and anything on USB, including the camera.
- For the whole rover, the motors (four gearmotors on two channels) and the pump will dominate.
  Measuring them needs a USB inline meter or a current sensor (e.g. INA219) on the battery
  lead. The motor-load sag on the Pi's 5 V input could be read the same way as here, but only
  if the motors share the Pi's supply, and only with the wheels off the ground.

---

# Motor, relay and pump load on the Pi's 5 V input — 2026-10-23

**Setup (operator-confirmed right before each run):** wheels in the air, front clear, hands
away. For both runs the pump was connected **with water**, nozzle into a cup. Run 1: the
operator reported "pump ran, water out". Run 2: confirmed "pump wet" just before firing. The
water volume was not measured. The rover was switched to `rover-control` for the runs and back
to view-only afterwards. Everything went through the control server, so its dead-man, obstacle
stop and spray gate stayed in force.

**Script:** `load_sag_run.py`, run on the Pi so that the readings and the commands share a
clock. It samples the Pi's 5 V input (`EXT5V_V`, a 3 ms single-channel read) at about 90 Hz.
Each sample carries the motor-driver pins IN1..IN4 and the relay pin as `pinctrl` read them,
plus the full rail sum every 0.25 s. Raw data: `load_sag_run1.jsonl` (wheels + 2 pulses) and
`load_sag_run2_relay.jsonl` (`python3 - relay`: 2 pulses, wheels not driven).

## What the pins show (what the hardware was told)

| Run | Wheels | Relay pulses on the pin |
|---|---|---|
| 1 | forward 3.0 s, then back 3.0 s; stopped ≈ 0.23 s after the last drive command (dead-man) | 1.008 s, 1.011 s |
| 2 | stopped throughout | 1.004 s, 1.010 s |

Both runs ended stopped, disarmed and with the relay released.

## The Pi's 5 V input

| Condition | Samples | Median | Min |
|---|---|---|---|
| run 1, wheels stopped and relay off | 1 292 | 5.026 V | 4.898 V |
| run 1, wheels forward | 297 | 5.032 V | 4.937 V |
| run 1, wheels back | 298 | 5.024 V | 4.908 V |
| run 1, relay on (pump with water) | 182 | 5.028 V | 4.919 V |
| run 2, relay off | 952 | 5.022 V | 4.899 V |
| run 2, relay on (pump with water) | 180 | 5.025 V | 4.923 V |

**Neither the wheels nor the relay and pump pull the Pi's 5 V input down measurably.** Every
median lies within 0.01 V of the at-rest median. The minimums are no lower under load than at
rest: the ~4.90 V dips appear equally with nothing running. The likely source is the Pi's own
load, which the 90 Hz sampling raises to about 2.5–2.7 W of rail power, against 1.7–2 W
without it.

**Reading:** the motor and pump current does not show up at the Pi's 5 V input. The wiring
was not traced; the operator has said the motors share a power switch with the ultrasonic
sensor. So either the motors and pump have their own supply, or a shared source is stiff
enough not to sag. Either way, at the duty used here (35 %) and in the air, a motor start or
a pump pulse did not brown out the Pi. That matters, because the safety stops run on the Pi.
A loaded drive on the floor was not tested. **This is not a measurement of the motors' or pump's own current**; that still
needs an inline meter or a current sensor on their supply. The water dose per pulse was not
measured either.

**Camera:** the camera survived the switch to `rover-control` (no fault in telemetry), but on
the switch back to view-only it came up with stuck stripes. The guard detected it
(`is_stuck_stripes` = True on `/raw`). As before, only a Pi power-cycle clears it.
