#!/usr/bin/env python3
"""
Verify a WeedScan recording session (run it after every field day, before
deleting anything from the laptop).

  python verify_session.py M:/Research/Data/.../Session_20261008_101500_plotA
  python verify_session.py <session> --decode     # full decode, checks every FFV1 slice CRC
  python verify_session.py <session> --checksum   # write SHA-256 manifest (for copies/backups)
  python verify_session.py <session> --check-copy # re-hash and compare to the manifest

Checks: required files present, frames.csv integrity, hardware timestamp gaps,
.mkv frame counts vs frames.csv, stream alignment, SVO2 frame count (if pyzed is
installed), FFV1 CRCs (--decode). Writes verify_report.json into the session.
Needs only Python + numpy-free stdlib + ffmpeg/ffprobe on PATH.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

MKV_FILES = ["RGB_left.mkv", "Depth.mkv", "Confidence.mkv"]


def ffprobe_packets(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets,width,height,pix_fmt,codec_name",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["streams"][0]
    except Exception:
        return {"error": r.stderr.strip()}


def full_decode(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-err_detect", "crccheck+bitstream", "-i", path,
                        "-f", "null", "-"], capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if l.strip()]
    return {"returncode": r.returncode, "errors": errs[:50], "n_errors": len(errs)}


def sha256(path, bufsize=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def svo_frames(path):
    try:
        import pyzed.sl as sl
    except Exception:
        return None
    ip = sl.InitParameters()
    ip.set_from_svo_file(path)
    ip.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
        return -1
    n = cam.get_svo_number_of_frames()
    cam.close()
    return n


def svo_timestamps(path):
    """All image timestamps stored in an SVO/SVO2 file (decodes every frame; no depth)."""
    import pyzed.sl as sl
    ip = sl.InitParameters()
    ip.set_from_svo_file(path)
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.svo_real_time_mode = False
    cam = sl.Camera()
    if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
        return None
    rt = sl.RuntimeParameters()
    rt.enable_depth = False
    n_total = cam.get_svo_number_of_frames()
    end_code = getattr(sl.ERROR_CODE, "END_OF_SVOFILE_REACHED", None)
    out = []
    failures = 0
    while True:
        err = cam.grab(rt)
        if err == sl.ERROR_CODE.SUCCESS:
            failures = 0
            out.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if n_total > 0 and len(out) >= n_total:
                break
        elif err == end_code or "".join(c for c in str(err).upper() if c.isalnum()) == "ENDOFSVOFILEREACHED":
            break
        else:
            failures += 1
            if failures > 100 or (n_total > 0 and cam.get_svo_position() >= n_total - 1):
                break  # never loop forever on an unexpected error code
    cam.close()
    return out


def match_timestamps(live_ts, svo_ts):
    """Match logged (live) timestamps to SVO2 playback timestamps.

    The SVO2 may store timestamps with a different precision or a constant offset from
    the live get_timestamp() values, so frames are matched to the nearest timestamp
    within 30 % of a frame period after removing the median offset.
    """
    a = sorted(int(x) for x in live_ts)
    b = sorted(int(x) for x in svo_ts)
    if not a or not b:
        return {"matched": 0, "missing_in_svo": len(a), "extra_in_svo": len(b)}
    diffs = [y - x for x, y in zip(a, a[1:])]
    period = sorted(diffs)[len(diffs) // 2] if diffs else 66_666_667
    tol = int(0.3 * period)
    k = min(len(a), len(b), 50)
    offset = sorted(b[i] - a[i] for i in range(k))[k // 2]   # robust constant offset
    i = j = matched = exact = 0
    resid = []
    missing = []
    while i < len(a) and j < len(b):
        d = b[j] - (a[i] + offset)
        if abs(d) <= tol:
            matched += 1
            exact += (b[j] == a[i])
            resid.append(abs(d))
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            missing.append(a[i])
            i += 1
    missing += a[i:]
    return {"matched": matched, "missing_in_svo": len(a) - matched, "extra_in_svo": len(b) - matched,
            "exact_timestamp_matches": exact, "offset_ns": offset,
            "max_residual_ms": (max(resid) / 1e6) if resid else None,
            "first_missing": missing[:10], "period_ms": period / 1e6}


def compare_svo_to_csv(svo_path, csv_ts):
    """-> dict with frames in file and logged frames missing from / extra in the SVO2."""
    ts = svo_timestamps(svo_path)
    if ts is None:
        return {"error": "cannot open SVO2"}
    out = {"svo_frames": len(ts), "csv_frames": len(csv_ts), "svo_first_ts": ts[:3],
           "csv_first_ts": sorted(csv_ts)[:3]}
    out.update(match_timestamps(csv_ts, ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--checksum", action="store_true")
    ap.add_argument("--check-copy", action="store_true")
    ap.add_argument("--svo-full", action="store_true",
                    help="decode the whole SVO2 and match every frames.csv timestamp (slow)")
    args = ap.parse_args()
    s = args.session
    rep = {"session": os.path.abspath(s), "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "problems": [], "warnings": []}
    P, W = rep["problems"].append, rep["warnings"].append

    for f in ["frames.csv", "session_metadata.json", "calibration.json", "events.log"]:
        if not os.path.exists(os.path.join(s, f)):
            P(f"missing {f}")
    meta = {}
    try:
        with open(os.path.join(s, "session_metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        rep["metadata_status"] = meta.get("status")
        if not str(meta.get("status", "")).startswith("COMPLETE"):
            W("session_metadata status is not COMPLETE (app crashed or was killed?)")
    except Exception as e:
        P(f"cannot read session_metadata.json: {e}")
    fps = (meta.get("camera") or {}).get("fps") or 15

    rows = []
    try:
        with open(os.path.join(s, "frames.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        P(f"cannot read frames.csv: {e}")
    rep["csv_rows"] = len(rows)
    ts = [int(r["image_timestamp_ns"]) for r in rows if r.get("image_timestamp_ns")]
    if ts:
        period = 1e9 / fps
        d = [b - a for a, b in zip(ts, ts[1:])]
        gaps = [(i + 1, x / 1e6) for i, x in enumerate(d) if x > 1.5 * period]
        nonmono = sum(1 for x in d if x <= 0)
        missing = sum(round(x / period) - 1 for x in d if x > 1.5 * period)
        rep["timestamps"] = {"duration_s": (ts[-1] - ts[0]) / 1e9, "gaps": len(gaps),
                             "estimated_missing_frames": missing, "non_monotonic": nonmono,
                             "max_interval_ms": max(d) / 1e6 if d else None,
                             "first_gaps": gaps[:20]}
        if missing:
            W(f"{missing} camera frames missing (timestamp gaps) - see first_gaps")
        if nonmono:
            P(f"{nonmono} non-increasing timestamps")
    mkv_rows = [r for r in rows if r.get("mkv_written") == "1"]
    rep["csv_mkv_written"] = len(mkv_rows)
    idxs = [int(r["mkv_frame_idx"]) for r in mkv_rows]
    if idxs != list(range(len(idxs))):
        P("mkv_frame_idx in frames.csv is not a contiguous 0..N-1 sequence")
    dropped = sum(1 for r in rows if r.get("mkv_written") == "0")
    if dropped:
        W(f"{dropped} frames not in the .mkv files (encoder backlog); they are in the SVO2 - "
          f"rebuild with extract_svo.py --mkv if you need them")
    svo_bad = sum(1 for r in rows if r.get("svo_frame_ok") == "0")
    if svo_bad:
        W(f"SDK status flagged {svo_bad} frames as not stored in SVO2 (advisory; the file "
          f"frame count below is authoritative, --svo-full matches every timestamp)")

    rep["mkv"] = {}
    counts = set()
    for f in MKV_FILES:
        p = os.path.join(s, f)
        if not os.path.exists(p):
            continue
        info = ffprobe_packets(p)
        n = int(info.get("nb_read_packets", -1))
        counts.add(n)
        entry = {"frames": n, "codec": info.get("codec_name"), "pix_fmt": info.get("pix_fmt"),
                 "size": [info.get("width"), info.get("height")], "bytes": os.path.getsize(p)}
        if n != len(mkv_rows):
            P(f"{f}: {n} frames in file but frames.csv says {len(mkv_rows)} written")
        if info.get("codec_name") != "ffv1":
            P(f"{f}: codec is {info.get('codec_name')}, expected ffv1")
        if args.decode:
            print(f"decoding {f} (CRC check)…")
            entry["decode"] = full_decode(p)
            if entry["decode"]["n_errors"] or entry["decode"]["returncode"]:
                P(f"{f}: decode errors: {entry['decode']['errors'][:3]}")
        rep["mkv"][f] = entry
    if len(counts) > 1:
        P(f".mkv streams have different frame counts: {sorted(counts)}")

    svo = os.path.join(s, "recording.svo2")
    if os.path.exists(svo):
        n = svo_frames(svo)
        rep["svo"] = {"bytes": os.path.getsize(svo), "frames": n}
        if n is None:
            W("pyzed not installed: SVO2 frame count not checked")
        elif n < 0:
            P("SVO2 file cannot be opened")
        elif n < len(rows):
            P(f"SVO2 has {n} frames but frames.csv logged {len(rows)}")
        if args.svo_full and n is not None and n >= 0:
            print("decoding SVO2 to match timestamps…")
            cmp_ = compare_svo_to_csv(svo, ts)
            rep["svo"]["timestamp_check"] = cmp_
            if cmp_.get("missing_in_svo"):
                P(f"{cmp_['missing_in_svo']} logged frames are not in the SVO2")
    elif (meta.get("capture") or {}).get("record_svo"):
        P("recording.svo2 missing although SVO recording was enabled")

    manifest = os.path.join(s, "checksums.sha256")
    if args.checksum or args.check_copy:
        files = sorted(f for f in os.listdir(s) if os.path.isfile(os.path.join(s, f))
                       and f not in ("checksums.sha256", "verify_report.json"))
        if args.check_copy:
            if not os.path.exists(manifest):
                P("no checksums.sha256 to compare against")
            else:
                bad = []
                with open(manifest, encoding="utf-8") as f:
                    for line in f:
                        hsh, name = line.rstrip("\n").split("  ", 1)
                        p = os.path.join(s, name)
                        print(f"hashing {name}…")
                        if not os.path.exists(p) or sha256(p) != hsh:
                            bad.append(name)
                rep["copy_check_failed"] = bad
                if bad:
                    P(f"checksum mismatch / missing: {bad}")
        else:
            with open(manifest, "w", encoding="utf-8") as f:
                for name in files:
                    print(f"hashing {name}…")
                    f.write(f"{sha256(os.path.join(s, name))}  {name}\n")
            rep["checksums_written"] = len(files)

    rep["ok"] = not rep["problems"]
    with open(os.path.join(s, "verify_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("ok", "problems", "warnings", "csv_rows")}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
