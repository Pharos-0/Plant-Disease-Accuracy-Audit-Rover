# Rover systemd units

The services that run the rover's camera and control servers on the Pi
(`pi@192.168.137.167`, Raspberry Pi 5, Debian 13).

| Unit | Runs | Starts at boot | Drives GPIO |
|---|---|---|---|
| `rover-camstream.service` | `/home/pi/pi_camera_stream.py` - view-only camera | **yes** (enabled) | **no** |
| `rover-control.service` | `/home/pi/pi_rover_control.py` - drive, spray, AUTO | no - start it on purpose | **yes** |

Both listen on `:8080` and each `Conflicts=` the other, so starting one stops the
other; the camera is only ever opened by one process.

`rover-control.service.d/workdir.conf` sets `WorkingDirectory=/home/pi`: RPi.GPIO on
the Pi 5 (lgpio backend) writes a `.lgd-nfy` file into the working directory, and
without a writable one the service crash-looped on start (fixed 2026-10-16).

**Provenance.** These units were written directly on the Pi during the 2026-10-16
session and are reconstructed here from that session's record. `scripts/deploy_rover.py`
compares them with what the Pi actually runs (`systemctl cat`) and reports any drift,
so this copy is checked rather than assumed.

Installing them on a fresh Pi (copy the server files to `/home/pi/` first):

```bash
sudo cp rover-camstream.service rover-control.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/rover-camstream.service.d /etc/systemd/system/rover-control.service.d
sudo cp rover-camstream.service.d/conflict.conf /etc/systemd/system/rover-camstream.service.d/
sudo cp rover-control.service.d/workdir.conf /etc/systemd/system/rover-control.service.d/
sudo systemctl daemon-reload
sudo systemctl enable --now rover-camstream.service
```
