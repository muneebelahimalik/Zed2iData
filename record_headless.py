#!/usr/bin/env python3
"""
Headless WeedScan recorder: same data, same files and same safety checks as the GUI,
but driven from a terminal. Use it on a Jetson over SSH (from a laptop or phone) or
with no monitor attached.

  python3 record_headless.py --plot A3 --operator MM --height 1.0
  python3 record_headless.py --res HD1080 --fps 30 --exposure 4 --gain 60 --duration 300

While recording, type in the terminal:
  <text> + Enter   add a marker with that text (e.g. "start row 3")
  Enter            add a plain marker
  q + Enter        stop and save (Ctrl+C also stops and saves cleanly)
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Zed2i_video_datacollection as app  # noqa: E402
from Zed2i_video_datacollection import sl, CONFIG  # noqa: E402

RES_KEYS = {v[0]: k for k, v in app.RESOLUTIONS.items()}  # "HD2K" -> GUI label


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=CONFIG["base_directory"], help="field-day folder (sessions go inside)")
    ap.add_argument("--res", default=app.RESOLUTIONS[CONFIG["default_resolution"]][0],
                    choices=list(RES_KEYS))
    ap.add_argument("--fps", type=int, default=None, help="default: highest allowed for --res")
    ap.add_argument("--depth", default=CONFIG["default_depth_mode"],
                    help="live depth: NEURAL_LIGHT, NEURAL, NEURAL_PLUS, ULTRA, NONE")
    ap.add_argument("--no-tracking", action="store_true", help="disable pose/speed tracking")
    ap.add_argument("--svo-compression", default="LOSSLESS",
                    choices=["LOSSLESS", "H265_LOSSLESS", "H264_LOSSLESS"])
    ap.add_argument("--no-svo", action="store_true", help="do NOT write the SVO2 master record")
    ap.add_argument("--mkv", dest="mkv", action="store_true", default=CONFIG["default_record_mkv"],
                    help="also write live FFV1 RGB/Depth/Confidence .mkv")
    ap.add_argument("--no-mkv", dest="mkv", action="store_false")
    ap.add_argument("--exposure", type=int, help="manual exposure %% (0-100 of frame period)")
    ap.add_argument("--gain", type=int, help="manual gain 0-100 (with --exposure)")
    ap.add_argument("--lock-auto", type=float, default=0,
                    help="let auto exposure/WB settle N seconds, then lock them")
    ap.add_argument("--duration", type=float, default=0, help="stop automatically after N seconds")
    ap.add_argument("--plot", default="")
    ap.add_argument("--site", default="")
    ap.add_argument("--operator", default="")
    ap.add_argument("--crop-stage", default="")
    ap.add_argument("--travel-direction", default="image top = forward")
    ap.add_argument("--weather", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--height", type=float, default=CONFIG["default_camera_height_m"],
                    help="lens height above ground (m)")
    ap.add_argument("--plant-height", type=float, default=CONFIG["default_plant_height_m"])
    ap.add_argument("--target-speed", type=float, default=0.25)
    return ap.parse_args()


def main():
    a = parse_args()
    fps = a.fps or app.RESOLUTIONS[RES_KEYS[a.res]][1][0]
    if fps not in app.RESOLUTIONS[RES_KEYS[a.res]][1]:
        sys.exit(f"{a.res} supports fps {app.RESOLUTIONS[RES_KEYS[a.res]][1]}")
    depth = "NONE" if a.depth.upper().startswith("NONE") else a.depth.upper()
    if depth != "NONE" and not hasattr(sl.DEPTH_MODE, depth):
        sys.exit(f"depth mode {depth} not available in this ZED SDK")
    if a.no_svo and not a.mkv:
        sys.exit("nothing to record (--no-svo without --mkv)")
    if a.mkv and not app.check_ffmpeg_available():
        sys.exit("ffmpeg not found (needed for --mkv)")
    problem = app.storage_problem(a.out)
    if problem:
        sys.exit(problem)
    os.makedirs(a.out, exist_ok=True)
    free = app.get_free_space_bytes(a.out)
    if 0 <= free < CONFIG["min_free_space_gb"] * 1024 ** 3:
        sys.exit(f"only {free / 1024 ** 3:.1f} GB free in {a.out}")

    print(f"Opening ZED {a.res}@{fps}, live depth {depth} "
          f"(first NEURAL use on a new device may take several minutes)…")
    zed = sl.Camera()
    ok, msg, rt, open_params, tracking_ok, warnings = app.open_zed(
        zed, a.res, fps, depth, not a.no_tracking)
    if not ok:
        sys.exit(f"Cannot open camera: {msg}")
    for w in warnings:
        print(f"[WARN] {w}")
    cam_info = app.camera_info_dict(zed)
    events = queue.Queue()
    engine = app.CaptureEngine(zed, rt, open_params["width"], open_params["height"],
                               open_params["fps"], open_params["depth_enabled"], tracking_ok, events)
    engine.start()
    time.sleep(1.0)

    if a.exposure is not None:
        engine.submit("set_setting", "EXPOSURE", max(0, min(100, a.exposure)))
        if a.gain is not None:
            engine.submit("set_setting", "GAIN", max(0, min(100, a.gain)))
    elif a.lock_auto > 0:
        print(f"Letting auto exposure settle {a.lock_auto:.0f} s…")
        time.sleep(a.lock_auto)
        print(engine.submit("lock_current")[1])
    time.sleep(0.5)
    cs = engine.cam_settings
    print(f"Camera {cam_info['serial_number']}  exposure {cs.get('exposure_pct')}% "
          f"(≈{cs.get('exposure_ms_approx') or 0:.1f} ms)  gain {cs.get('gain')}  "
          f"WB {cs.get('wb_temp_k')} K  auto={cs.get('aec_agc')}")

    lc = cam_info["calibration_rectified"]["left_cam"]
    plan = None
    if cs.get("exposure_ms_approx"):
        plan = app.motion_plan(a.height, a.plant_height, lc["fy"], cam_info["resolution"][1],
                               open_params["fps"], cs["exposure_ms_approx"], 1.0, 20)
        print(f"Speed plan: max {plan['v_recommended_m_s']:.2f} m/s for ≤1 px blur "
              f"(GSD {plan['gsd_ground_mm_per_px']:.2f} mm/px at ground); target {a.target_speed:.2f} m/s")

    field = {"site": a.site, "plot_id": a.plot, "operator": a.operator, "crop_stage": a.crop_stage,
             "travel_direction": a.travel_direction, "weather": a.weather, "notes": a.notes,
             "camera_height_m": a.height, "plant_height_m": a.plant_height,
             "target_speed_m_s": a.target_speed, "travel_axis": "Image vertical (rows)",
             "recorded_with": "record_headless.py"}
    opts = {"record_svo": not a.no_svo, "svo_compression": a.svo_compression, "record_mkv": a.mkv,
            "depth_units_per_mm": CONFIG["depth_units_per_mm"],
            "confidence_scale": CONFIG["confidence_scale"], "ram_buffer_gb": CONFIG["ram_buffer_gb"]}
    session = app.RecordingSession(a.out, field, opts)
    ok, msg = app.start_session(zed, engine, session, open_params, speed_plan=plan)
    if not ok:
        engine.stop_event.set()
        engine.join(timeout=30)
        zed.close()
        sys.exit(f"Could not start recording: {msg}")
    app.keep_system_awake(True)
    app.play_wav_async(CONFIG["wave_start_path"])
    print(f"\n● RECORDING → {session.dir}\n  type text+Enter = marker, q+Enter or Ctrl+C = stop\n")

    stop = threading.Event()
    reason = {"why": "operator"}

    def on_signal(signum, frame):
        reason["why"] = f"signal {signum}"
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_signal)  # SSH connection dropped -> save cleanly

    def stdin_reader():
        while not stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":  # EOF (no terminal) -> just keep recording
                return
            text = line.strip()
            if text.lower() in ("q", "quit", "stop"):
                stop.set()
                return
            session.add_marker(max(0, session.frames - 1),
                               (engine.latest.get("fi") or {}).get("image_timestamp_ns"),
                               text or "marker")
            app.beep_async(1800, 60)
    threading.Thread(target=stdin_reader, daemon=True).start()

    t_start = time.monotonic()
    last_disk = 0
    free = app.get_free_space_bytes(a.out)
    while not stop.is_set():
        time.sleep(1.0)
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            print(f"\n[{kind.upper()}] {payload}")
            if kind == "camera_error":
                app.beep_async(400, 400)
        if time.monotonic() - last_disk > 5:
            last_disk = time.monotonic()
            free = app.get_free_space_bytes(a.out)
            if 0 <= free < CONFIG["autostop_free_space_gb"] * 1024 ** 3:
                session.log_event("ERROR", "free space below auto-stop threshold")
                reason["why"] = "disk almost full"
                break
        st = session.stats()
        fi = engine.latest.get("fi") or {}
        bl, bmax = session.backlog()
        spd = fi.get("speed_m_s")
        tilt = fi.get("tilt_from_nadir_deg")
        line = (f"\r{time.monotonic() - t_start:7.1f}s  frames {st['frames_grabbed']:6d}  "
                f"missing {st['estimated_missing_camera_frames']:3d}  "
                f"SVO ok/fail {st['svo_frames_ok']}/{st['svo_frames_failed']}  "
                + (f"mkv backlog {bl}/{bmax} drop {st['mkv_frames_dropped_backlog']}  " if session.writers else "")
                + (f"speed {spd:.2f} m/s  " if spd is not None else "")
                + (f"dist {st['distance_travelled_m']:.1f} m  " if st['distance_travelled_m'] else "")
                + (f"tilt {tilt:.0f}°  " if tilt is not None else "")
                + f"free {free / 1024 ** 3:.0f} GB   ")
        sys.stdout.write(line)
        sys.stdout.flush()
        if a.duration and time.monotonic() - t_start >= a.duration:
            reason["why"] = "duration reached"
            break

    print(f"\n\nStopping ({reason['why']}) – finalizing files, do not unplug…")
    ok, s2 = engine.submit("stop_session", timeout=30)
    session.end_iso = app.now_iso()
    session.log_event("INFO", f"recording stopped ({reason['why']}); finalizing files…")
    session.finalize()
    stats = app.finish_session_metadata(session)
    app.keep_system_awake(False)
    app.play_wav_async(CONFIG["wave_stop_path"])
    engine.stop_event.set()
    engine.join(timeout=30)
    zed.close()

    v = session.verification
    print(f"\n{'✔' if v.get('all_ok') else '⚠'} Saved {session.dir}")
    print(f"  frames {stats['frames_grabbed']}, {stats['duration_s']:.1f} s, "
          f"{stats['effective_fps'] or 0:.2f} fps effective")
    print(f"  missing camera frames {stats['estimated_missing_camera_frames']}, "
          f"SVO ok/failed {stats['svo_frames_ok']}/{stats['svo_frames_failed']}")
    for name, sv in v.get("streams", {}).items():
        print(f"  {name}: {sv['frames_in_file']} in file / {sv['frames_sent']} sent"
              f"{'  ERROR ' + str(sv['error']) if sv['failed'] else ''}")
    if stats["distance_travelled_m"]:
        print(f"  distance {stats['distance_travelled_m']:.2f} m, mean speed {stats['mean_speed_m_s'] or 0:.3f} m/s")
    time.sleep(0.5)  # let the stop sound start before exit
    sys.exit(0 if v.get("all_ok") else 2)


if __name__ == "__main__":
    main()
