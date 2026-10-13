"""Recover every complete member from a truncated ZIP archive.

WHY THIS EXISTS
---------------
DIGITAL.CSIC ends every download of GROUNDBASED_WEED at about 1.08 GB. Three
browsers produced three partial files cut within 0.5 MB of each other
(1,083,592,484 / 1,083,821,860 / 1,084,075,070 bytes), and resuming restarts
from zero, so a complete TOMATO_1.zip cannot be obtained by downloading.

A ZIP's central directory sits at the END of the file, so a truncated archive
will not open with zipfile. But the members themselves are written one after
another from the start, each behind its own local header carrying its name and,
when the general-purpose flag's bit 3 is clear, its sizes. So the archive can be
walked from offset 0 and every member that arrived in full extracted.

Integrity is not assumed. Every recovered member is checked against the CRC-32
in its local header; a member whose CRC does not match is reported and not
written. The walk stops at the first member whose data runs past the end of
the file - that member, and everything after it, did not arrive.

ZIP64 is handled: when a size field reads 0xFFFFFFFF the real sizes are read
from the 0x0001 extra field, as in this archive.

    python scripts/recover_truncated_zip.py <partial.zip> <out_dir>
"""
from __future__ import annotations

import json
import mmap
import struct
import sys
import zlib
from pathlib import Path

LOCAL = b"PK\x03\x04"
DESC = b"PK\x07\x08"
FULL = 0xFFFFFFFF


def zip64_sizes(extra: bytes, usize: int, csize: int) -> tuple[int, int]:
    i = 0
    while i + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[i:i + 4])
        body = extra[i + 4:i + 4 + hlen]
        if hid == 0x0001:
            j = 0
            if usize == FULL:
                usize = struct.unpack("<Q", body[j:j + 8])[0]
                j += 8
            if csize == FULL:
                csize = struct.unpack("<Q", body[j:j + 8])[0]
            break
        i += 4 + hlen
    return usize, csize


def find_descriptor(mm, start: int, total: int) -> tuple[int, int, int] | None:
    """For a STORED member with a data descriptor, return (data_end, crc,
    next_offset), or None if no valid descriptor exists before EOF."""
    pos = start
    while True:
        k = mm.find(DESC, pos, total)
        if k < 0:
            return None
        for zip64 in (True, False):
            n = 24 if zip64 else 16
            if k + n > total:
                continue
            crc = struct.unpack("<I", mm[k + 4:k + 8])[0]
            if zip64:
                csize = struct.unpack("<Q", mm[k + 8:k + 16])[0]
            else:
                csize = struct.unpack("<I", mm[k + 8:k + 12])[0]
            if csize == k - start and zlib.crc32(mm[start:k]) & 0xFFFFFFFF == crc:
                return k, crc, k + n
        pos = k + 1


def recover(src: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    rep = {"source": str(src), "source_bytes": total, "recovered": [], "bad_crc": [],
           "stopped_at": None, "reason": None}
    with src.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        off = 0
        while True:
            f.seek(off)
            hdr = f.read(30)
            if len(hdr) < 30:
                rep["reason"] = "end of file inside a local header"
                break
            if hdr[:4] != LOCAL:
                rep["reason"] = f"no local header at offset {off} (reached central directory or garbage)"
                break
            (_, flag, method, _, _, crc, csize, usize,
             nlen, xlen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
            name = f.read(nlen).decode("utf-8", "replace")
            extra = f.read(xlen)
            start = off + 30 + nlen + xlen
            if flag & 0x08:
                # Sizes follow the data. Locate the end without trusting the header.
                if method == 0:
                    found = find_descriptor(mm, start, total)
                    if found is None:
                        rep["stopped_at"] = name
                        rep["reason"] = (f"'{name}' has no complete data descriptor before "
                                         f"EOF at {total:,}: it did not arrive in full")
                        break
                    data_end, crc, nxt = found
                    data = mm[start:data_end]
                elif method == 8:
                    d = zlib.decompressobj(-15)
                    data = d.decompress(mm[start:min(total, start + (1 << 30))])
                    if not d.eof:
                        rep["stopped_at"] = name
                        rep["reason"] = f"'{name}' deflate stream truncated at EOF"
                        break
                    used = min(total, start + (1 << 30)) - start - len(d.unused_data)
                    k = start + used
                    sig = mm[k:k + 4] == DESC
                    base = k + (4 if sig else 0)
                    crc = struct.unpack("<I", mm[base:base + 4])[0]
                    nxt = base + 4 + (16 if mm[base + 20:base + 24] != LOCAL else 8)
                    nxt = mm.find(LOCAL, k, total)
                    if nxt < 0:
                        nxt = total
                else:
                    rep["bad_crc"].append(f"{name} (unsupported method {method})")
                    break
                if not name.endswith("/"):
                    if zlib.crc32(data) & 0xFFFFFFFF != crc:
                        rep["bad_crc"].append(name)
                    else:
                        dest = out / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        rep["recovered"].append(name)
                off = nxt
                continue
            usize, csize = zip64_sizes(extra, usize, csize)
            end = start + csize
            if end > total:
                rep["stopped_at"] = name
                rep["reason"] = (f"'{name}' needs bytes up to {end:,} but the file ends at "
                                 f"{total:,}: it and everything after it did not arrive")
                break
            if not name.endswith("/"):
                data = mm[start:end]
                if method == 8:
                    data = zlib.decompress(data, -15)
                elif method != 0:
                    rep["bad_crc"].append(f"{name} (unsupported method {method})")
                    off = end
                    continue
                if zlib.crc32(data) & 0xFFFFFFFF != crc:
                    rep["bad_crc"].append(name)
                else:
                    dest = out / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    rep["recovered"].append(name)
            off = end
    return rep


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    rep = recover(Path(sys.argv[1]), Path(sys.argv[2]))
    imgs = [n for n in rep["recovered"] if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    xmls = [n for n in rep["recovered"] if n.lower().endswith(".xml")]
    print(f"source: {rep['source_bytes']:,} bytes")
    print(f"recovered and CRC-verified: {len(rep['recovered'])} members "
          f"({len(imgs)} images, {len(xmls)} XML)")
    print(f"CRC failures: {len(rep['bad_crc'])}")
    print(f"stopped: {rep['reason']}")
    (Path(sys.argv[2]) / "_recovery_report.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    return 0 if not rep["bad_crc"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
