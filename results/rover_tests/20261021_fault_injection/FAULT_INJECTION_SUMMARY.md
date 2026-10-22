# Ultrasonic fault injection on the robot — 2026-10-23: PASS

**Why injected in software.** The HC-SR04 shares its power switch with the motors (switching
it off stops the wheels by cutting motor power, which tests nothing) and is soldered and glued
in (cannot be unplugged alone). So the deployed server's test hook was used: while
`/tmp/rover_inject_sensor_loss` exists, every ultrasonic reading is discarded, exactly as if the
sensor had died. The control loop, the motor driver and the wheels are the real ones.

**Setup.** Wheels off the ground, front clear (42 cm), operator's hands away — confirmed by the
operator immediately before the run. Motor-driver pins read on the Pi every 0.1 s
(`pin_logger.sh`), on the same clock as the inject/clear moments. Record:
`injected_sensor_fault_run.json`.

| Seconds after the fault | Requested | Motor pins (IN1..IN4) | Verdict |
|---|---|---|---|
| before | forward | forward | normal |
| **+1.09** | forward | **stopped** | refused ~1 s after the sensor "died" (`DIST_STALE_S` = 1.0 s) |
| +3.21 | reverse | **back** | backing out still allowed |
| +5.23 | forward | **stopped** | still refused while the fault lasts |
| +7.01 cleared → **+7.15** | forward | **forward** | recovers 0.14 s after the sensor returns |
| +10.71 | STOP | stopped | disarmed after stop |

The fault file was removed at the end (checked). Incidentally, while the laptop paused ~1 s to
send the inject command, the wheels stopped by themselves — the dead-man (`HOLD_S` = 0.4 s);
the 0.1 s stops later in the forward phase are the same dead-man reacting to network jitter.
Both fail towards stopping.

**Not covered.** An *electrical* failure of the sensor (as opposed to its readings vanishing) is
not exercised; its effect on the control loop — no fresh reading — is the same, and the 21 Oct
run with the sensor switched off showed `sensor_lost` set within ~1 s on the hardware.
