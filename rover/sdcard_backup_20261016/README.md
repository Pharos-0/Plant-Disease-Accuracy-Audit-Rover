# Pi SD card — access-recovery change, 2026-10-16

The rover's Raspberry Pi (Raspberry Pi OS, image of 2025-12-04) had SSH disabled and the
account password was unknown. With the owner's go-ahead, the boot partition was changed so
that on the next boot cloud-init creates a key-only maintenance account and enables SSH.

## What changed on the card's boot partition (`bootfs`)

| File | Change |
|---|---|
| `user-data` | replaced: adds user `rover` (SSH key only, no password, passwordless sudo, groups sudo/video + gpio/i2c/spi/dialout/plugdev if present); enables SSH; writes the existing account's username to `rover-original-user.txt` and a log to `rover-access-log.txt` |
| `meta-data` | `instance_id: rpios-image` → `rpios-image-rover-20261016`, so cloud-init applies `user-data` again |
| `ssh` | new empty file (Raspberry Pi's own switch for enabling SSH) |
| `user-data.bak`, `meta-data.bak` | copies of the originals |

Not touched: the existing account, its password and files, `network-config` (Wi-Fi), `cmdline.txt`,
`config.txt`.

The originals are also in this folder (`user-data.orig`, `meta-data.orig`); the SHA-256 of each
matched the card before the change.

## To undo

On the card: copy `user-data.bak` over `user-data`, `meta-data.bak` over `meta-data`, and delete
`ssh`. On the Pi: `sudo deluser --remove-home rover`.

The private key that grants access is `C:\Users\tjpha\.ssh\id_ed25519` on the laptop; anyone with
that file can log in as `rover` with full admin rights.
