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
