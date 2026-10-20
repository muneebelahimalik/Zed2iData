#!/usr/bin/env python3
"""
Check that this machine (Jetson or PC) and the ZED 2i are ready for WeedScan recording,
and measure what it can sustain.

  python3 zed_doctor.py                       # checks only (camera must be plugged in)
  python3 zed_doctor.py --optimize            # build the NEURAL depth models for this GPU
                                              #   (first-time only, can take 5-20 min each)
  python3 zed_doctor.py --bench --out /media/<user>/SSD/ZED_Data
                                              # record test sessions and report real fps/drops

Every check prints OK / WARN / FAIL with the fix. Exit code 1 if anything FAILed.
"""

import argparse
import glob
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS = []
IS_WINDOWS = os.name == "nt"


def report(level, what, detail="", fix=""):
    RESULTS.append(level)
    mark = {"OK": "\033[32m OK \033[0m", "WARN": "\033[33mWARN\033[0m", "FAIL": "\033[31mFAIL\033[0m",
            "INFO": "INFO"}[level]
    print(f"[{mark}] {what}" + (f": {detail}" if detail else ""))
    if fix and level in ("WARN", "FAIL"):
        print(f"        fix: {fix}")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip("\x00\n ")
    except Exception:
        return None


def check_platform():
    print("\n== Platform ==")
    model = read("/proc/device-tree/model")
    l4t = read("/etc/nv_tegra_release")
    if l4t:
        report("OK", "NVIDIA Jetson", f"{model} | {l4t.splitlines()[0]}")
        if shutil.which("nvpmodel"):
            r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
            mode = " ".join(l for l in r.stdout.split("\n") if "NVPM ERROR" not in l).strip()
            mode = " ".join(mode.split())
            report("INFO", "power mode", mode or r.stderr.strip())
            if "MAXN" not in mode.upper():
                report("WARN", "power mode is not MAXN",
                       fix="sudo nvpmodel -m 0 && sudo jetson_clocks   (or ./jetson/setup_jetson.sh --max-perf); "
                           "check 'nvpmodel -q --verbose' for your module's max mode")
        # jetson_clocks pins min == max frequency (the governor name may stay 'schedutil')
        fmin = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq")
        fmax = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
        if fmin and fmax and fmin == fmax:
            report("OK", "CPU clocks locked", f"{int(fmax) / 1e6:.2f} GHz")
        elif fmin and fmax:
            report("WARN", f"CPU clocks not locked ({int(fmin) / 1e6:.2f}-{int(fmax) / 1e6:.2f} GHz)",
                   fix="sudo jetson_clocks   (run after every boot, or use setup_jetson.sh --max-perf)")
    else:
        report("INFO", "not a Jetson", f"{sys.platform} {model or ''}")
    report("INFO", "python", sys.version.split()[0] + f" ({sys.executable})")


def check_python():
    print("\n== Python packages ==")
    ok = True
    try:
        import pyzed.sl as sl
        report("OK", "pyzed (ZED SDK)", str(sl.Camera.get_sdk_version()))
    except Exception as e:
        ok = False
        report("FAIL", "pyzed (ZED SDK Python API)", str(e),
               "python3 /usr/local/zed/get_python_api.py   (with the same python3 you run the app with)")
    try:
        import numpy as np
        report("OK", "numpy", np.__version__)
    except Exception as e:
        ok = False
        report("FAIL", "numpy", str(e), "sudo apt install python3-numpy")
    for mod, fix in [("cv2", "sudo apt install python3-opencv  (JetPack usually ships it)"),
                     ("tkinter", "sudo apt install python3-tk"),
                     ("PIL.ImageTk", "sudo apt install python3-pil python3-pil.imagetk")]:
        try:
            m = importlib.import_module(mod)
            report("OK", mod, getattr(m, "__version__", ""))
        except Exception as e:
            level = "WARN" if mod in ("tkinter", "PIL.ImageTk") else "FAIL"
            ok &= level != "FAIL"
            report(level, mod, f"{e} (only the GUI needs it; record_headless.py works without)"
                   if level == "WARN" else str(e), fix)
    return ok


def check_tools():
    print("\n== Tools ==")
    for tool, need, fix in [("ffmpeg", "live .mkv output and extraction", "sudo apt install ffmpeg"),
                            ("ffprobe", "verification", "sudo apt install ffmpeg")]:
        if shutil.which(tool):
            report("OK", tool, shutil.which(tool))
        else:
            report("WARN", f"{tool} missing ({need})", fix=fix)
    if any(shutil.which(p) for p in ("paplay", "aplay", "ffplay")):
        report("OK", "sound player", next(p for p in ("paplay", "aplay", "ffplay") if shutil.which(p)))
    else:
        report("WARN", "no sound player (no start/stop/pace beeps)", fix="sudo apt install alsa-utils")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        report("OK", "display available", "GUI can run")
    else:
        report("INFO", "no display in this shell", "use record_headless.py, or run the GUI on the Jetson desktop")


def check_usb():
    print("\n== ZED 2i USB connection ==")
    devs = []
    for dev in glob.glob("/sys/bus/usb/devices/*"):
        if read(os.path.join(dev, "idVendor")) != "2b03":
            continue
        try:
            sp = int(read(os.path.join(dev, "speed")) or 0)
        except ValueError:
            sp = 0
        devs.append((sp, dev, read(os.path.join(dev, "product")) or "?",
                     read(os.path.join(dev, "idProduct")), read(os.path.join(dev, "power/control"))))
    if not devs:
        report("FAIL", "no ZED device on USB", fix="plug the ZED 2i into a USB 3 port; check `lsusb | grep 2b03`")
        return
    # The ZED 2i appears as two devices: the video camera (must be USB 3, >= 5000 Mb/s)
    # and the sensor/HID module (USB 2 is normal for it).
    devs.sort(reverse=True)
    for sp, dev, prod, pid, ctrl in devs:
        report("INFO", f"{prod} [2b03:{pid}] at {os.path.basename(dev)}", f"{sp} Mb/s, power/control={ctrl}")
    sp, dev, prod, pid, ctrl = devs[0]
    if sp >= 5000:
        report("OK", "camera on USB 3", f"{sp} Mb/s")
    else:
        report("FAIL", "camera is NOT on USB 3", f"best link {sp} Mb/s -> HD2K will fail or drop frames",
               "use the original ZED cable in a USB 3 port, no passive hub/extension, cable fully inserted "
               "(try flipping the USB-C end)")
    if any(c and c != "on" for *_, c in devs):
        report("WARN", "USB autosuspend enabled for the ZED",
               fix="sudo ./jetson/setup_jetson.sh   (installs a udev rule that keeps the ZED powered)")


def check_camera(sl):
    print("\n== ZED SDK camera detection ==")
    try:
        devs = sl.Camera.get_device_list()
    except Exception as e:
        report("FAIL", "get_device_list", str(e))
        return None
    if not devs:
        report("FAIL", "SDK sees no camera",
               fix="replug the camera; close ZED Explorer / ZED Depth Viewer; run ZED_Diagnostic")
        return None
    for d in devs:
        report("OK", "camera", f"{getattr(d, 'camera_model', '?')} SN {getattr(d, 'serial_number', '?')} "
                               f"state {getattr(d, 'camera_state', '?')}")
    serial = getattr(devs[0], "serial_number", None)
    conf = [p for p in glob.glob(f"/usr/local/zed/settings/SN{serial}.conf")]
    if conf:
        report("OK", "factory calibration file", conf[0])
    else:
        report("WARN", "factory calibration file not cached yet",
               fix="it is downloaded on first open (needs internet once) - open the camera while online "
                   "before going to the field")
    return devs


def check_disk(out):
    print("\n== Storage ==")
    import Zed2i_video_datacollection as app
    problem = app.storage_problem(out)
    if problem:
        report("FAIL", "data drive not mounted", problem,
               "mount the SSD (and add it to /etc/fstab, see JETSON.md)")
        return None
    os.makedirs(out, exist_ok=True)
    mp = app.mount_point_of(out)
    if mp != "/" and not IS_WINDOWS:
        in_fstab = False
        try:
            with open("/etc/fstab", encoding="utf-8") as f:
                in_fstab = any(len(l.split()) > 1 and l.split()[1] == mp
                               for l in f if not l.lstrip().startswith("#"))
        except Exception:
            pass
        if in_fstab:
            report("OK", "data drive mounted at boot", f"{mp} is in /etc/fstab")
        else:
            report("WARN", f"{mp} is not in /etc/fstab (won't be mounted after a reboot)",
                   fix="sudo ./jetson/setup_jetson.sh --data-dir <dir> --fstab   (see JETSON.md)")
    free = app.get_free_space_bytes(out)
    try:
        dev = subprocess.run(["df", "-hT", out], capture_output=True, text=True).stdout.splitlines()[-1]
    except Exception:
        dev = ""
    report("INFO", "target", f"{out}  ({dev})")
    if free >= 0:
        lvl = "OK" if free > 200 * 1024 ** 3 else ("WARN" if free > 50 * 1024 ** 3 else "FAIL")
        report(lvl, "free space", f"{free / 1024 ** 3:.0f} GB "
               f"(~{free / 1024 ** 3 / 400 * 60:.0f} min of HD2K SVO2 lossless at ~120 MB/s)",
               "record to an NVMe SSD / large external USB 3 SSD")
    if "mmcblk" in dev:
        report("WARN", "target looks like the eMMC/SD boot device",
               fix="record to an NVMe or external SSD (pass --out /media/<user>/<SSD>/ZED_Data)")
    mb = app.disk_write_benchmark(out, 1024)
    lvl = "OK" if mb >= 250 else ("WARN" if mb >= 130 else "FAIL")
    report(lvl, "sustained write speed", f"{mb:.0f} MB/s (SVO2 lossless HD2K needs ~120, + live .mkv ~150 more)",
           "use a faster SSD or a lower resolution")
    return mb


def optimize_models(sl):
    print("\n== Optimizing NEURAL depth models for this GPU (one-time) ==")
    for mode in ("NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS"):
        if not hasattr(sl.DEPTH_MODE, mode):
            continue
        print(f"  {mode}: opening camera (may take 5-20 min the first time)…", flush=True)
        t = time.time()
        ip = sl.InitParameters()
        ip.depth_mode = getattr(sl.DEPTH_MODE, mode)
        ip.camera_resolution = sl.RESOLUTION.HD720
        cam = sl.Camera()
        err = cam.open(ip)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.grab(sl.RuntimeParameters())
            cam.close()
            report("OK", f"{mode} ready", f"{time.time() - t:.0f} s")
        else:
            report("FAIL", f"{mode}", str(err))


def _cpu_times():
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]  # total, idle+iowait
    except Exception:
        return None


def bench(sl, out, configs, seconds, keep=False):
    import csv as _csv
    import queue as _q
    import Zed2i_video_datacollection as app
    from verify_session import compare_svo_to_csv
    print("\n== Recording benchmark (real camera, real disk, SVO2 verified frame by frame) ==")
    problem = app.storage_problem(out)
    if problem:
        report("FAIL", "benchmark skipped", problem)
        return
    os.makedirs(out, exist_ok=True)
    bench_dir = tempfile.mkdtemp(prefix="zed_bench_", dir=out)
    results = []
    for res, fps, depth, svo_mode, mkv in configs:
        label = f"{res}@{fps} depth={depth} svo={svo_mode}" + (" +mkv" if mkv else "")
        print(f"\n  >>> {label}: recording {seconds:.0f} s…", flush=True)
        zed = sl.Camera()
        ok, msg, rt, op, trk, _ = app.open_zed(zed, res, fps, depth, depth != "NONE")
        if not ok:
            report("FAIL", label, msg)
            continue
        eng = app.CaptureEngine(zed, rt, op["width"], op["height"], op["fps"], op["depth_enabled"],
                                trk, _q.Queue())
        eng.start()
        time.sleep(3)
        opts = {"record_svo": True, "svo_compression": svo_mode, "record_mkv": mkv,
                "depth_units_per_mm": 10, "confidence_scale": 100, "ram_buffer_gb": 1.0}
        s = app.RecordingSession(bench_dir, {"plot_id": "bench_" + label.replace(" ", "_")}, opts)
        ok, msg = app.start_session(zed, eng, s, op)
        if not ok:
            report("FAIL", label, msg)
            eng.stop_event.set()
            eng.join(timeout=30)
            zed.close()
            continue
        c0 = _cpu_times()
        time.sleep(seconds)
        c1 = _cpu_times()
        eng.submit("stop_session", timeout=30)
        eng.stop_event.set()
        eng.join(timeout=30)
        zed.close()
        s.end_iso = app.now_iso()
        s.finalize()
        st = app.finish_session_metadata(s)
        cpu = (100.0 * (1 - (c1[1] - c0[1]) / max(1, c1[0] - c0[0]))) if c0 and c1 else None
        with open(s.csv_path, newline="") as f:
            csv_ts = [int(r["image_timestamp_ns"]) for r in _csv.DictReader(f) if r["image_timestamp_ns"]]
        print("      verifying SVO2 frame by frame…", flush=True)
        cmp_ = compare_svo_to_csv(s.svo_path, csv_ts)
        cam_missing = st["estimated_missing_camera_frames"]
        svo_missing = cmp_.get("missing_in_svo", -1)
        lossless = cam_missing == 0 and svo_missing == 0
        comp = st.get("svo_compression_ms_mean")
        report("OK" if lossless else "WARN", label,
               f"{st['effective_fps'] or 0:.2f} fps | camera frames missing {cam_missing} | "
               f"SVO2 holds {cmp_.get('svo_frames')} of {cmp_.get('csv_frames')} logged "
               f"(missing {svo_missing}) | repeated {st['duplicate_frames_skipped']} | "
               f"CPU {cpu or 0:.0f}% | SVO compress "
               + (f"{comp:.0f}/{st.get('svo_compression_ms_max') or 0:.0f} ms mean/max" if comp else "n/a")
               + (f" | mkv dropped {st['mkv_frames_dropped_backlog']}" if mkv else ""))
        results.append((label, lossless and not st["mkv_frames_dropped_backlog"]))
    if keep:
        print(f"\n  benchmark sessions kept in {bench_dir}")
    else:
        shutil.rmtree(bench_dir, ignore_errors=True)
    best = [l for l, g in results if g]
    print("\n  Loss-free on this machine:" if best else "\n  No tested setting was loss-free.")
    for l in best:
        print(f"    - {l}")


def parse_configs(text, sl):
    """'HD2K:15:NEURAL_LIGHT:LOSSLESS[:mkv],HD1080:15:NONE:LOSSLESS' -> list of tuples."""
    out = []
    for item in text.split(","):
        p = item.strip().split(":")
        out.append((p[0].upper(), int(p[1]), p[2].upper(), p[3].upper(), len(p) > 4 and p[4] == "mkv"))
    return out


def main():
    import Zed2i_video_datacollection as app
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=app.CONFIG["base_directory"], help="where recordings will be saved")
    ap.add_argument("--optimize", action="store_true", help="pre-build NEURAL models (one-time)")
    ap.add_argument("--bench", action="store_true", help="run recording benchmarks")
    ap.add_argument("--seconds", type=float, default=60, help="per benchmark setting")
    ap.add_argument("--configs", help="custom list, e.g. HD2K:15:NEURAL_LIGHT:LOSSLESS,HD1080:15:NONE:LOSSLESS"
                                      " (append :mkv to also write live .mkv)")
    ap.add_argument("--keep", action="store_true", help="keep the benchmark recordings")
    ap.add_argument("--quick", action="store_true", help="skip the 1 GB disk test")
    a = ap.parse_args()

    check_platform()
    py_ok = check_python()
    check_tools()
    check_usb()
    sl = None
    if py_ok:
        import pyzed.sl as sl
        check_camera(sl)
    if not a.quick:
        check_disk(a.out)
    if sl and a.optimize:
        optimize_models(sl)
    if sl and a.bench:
        light = "NEURAL_LIGHT" if hasattr(sl.DEPTH_MODE, "NEURAL_LIGHT") else "NEURAL"
        if a.configs:
            configs = parse_configs(a.configs, sl)
        else:
            configs = [("HD2K", 15, light, "LOSSLESS", False),
                       ("HD2K", 15, "NONE", "LOSSLESS", False),
                       ("HD1080", 15, light, "LOSSLESS", False),
                       ("HD2K", 15, light, "H265_LOSSLESS", False)]
        bench(sl, a.out, list(dict.fromkeys(configs)), a.seconds, a.keep)

    n_fail = RESULTS.count("FAIL")
    n_warn = RESULTS.count("WARN")
    print(f"\nSummary: {n_fail} FAIL, {n_warn} WARN")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
