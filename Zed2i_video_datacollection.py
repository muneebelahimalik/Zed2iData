#!/usr/bin/env python3
"""
WeedScan: ZED 2i field data collection for Vidalia onion plots.

Data model (per recording session folder)
-----------------------------------------
recording.svo2        MASTER RECORD. Native ZED container, LOSSLESS compression.
                      Holds the raw left+right stereo images, IMU / magnetometer /
                      barometer streams and the hardware timestamps of every frame.
                      Depth, confidence, point clouds and poses can be recomputed
                      from it at any time (e.g. with NEURAL_PLUS) using extract_svo.py.
RGB_left.mkv          Rectified left image, FFV1 lossless (bgr0, bit-exact).
Depth.mkv             Depth aligned to RGB_left, FFV1 lossless gray16le,
                      1 unit = 1/DEPTH_UNITS_PER_MM mm (default 0.1 mm).
                      0 = no measurement (NaN), 65534 = too close (-inf),
                      65535 = too far (+inf) or beyond the encodable range.
Confidence.mkv        ZED confidence map, FFV1 gray16le, value = confidence * 100
                      (0..10000, HIGHER = LESS reliable), 65535 = NaN.
frames.csv            One row per grabbed frame: hardware timestamp, drop flags,
                      camera exposure/gain/WB, 6-DoF pose, speed, IMU, tilt.
markers.csv           Operator markers ("start row 3", "weed patch"...).
events.log            Every warning / error / drop with timestamps.
session_metadata.json Field + operator info, camera / SDK / host info,
                      all capture parameters, final statistics, verification.
calibration.json      Rectified + raw (factory) calibration, IMU extrinsics.
calibration_params.txt Human-readable calibration (legacy format).
SN<serial>.conf       Copy of the factory calibration file, if found.
ffmpeg_*.log          Encoder logs.
README.txt            How to read all of the above.

The .mkv files are written live as a convenience copy. If the PC cannot keep up,
frames are dropped from the .mkv files only (flagged in frames.csv); the SVO2 file
still has them, and extract_svo.py --mkv rebuilds the .mkv set offline without drops.
"""

from __future__ import annotations

import collections
import csv
import datetime
import glob
import json
import math
import multiprocessing
import os
import platform
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import pyzed.sl as sl

APP_NAME = "WeedScan ZED 2i Data Collection"
APP_VERSION = "2.1.0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = os.name == "nt"
IS_JETSON = os.path.exists("/etc/nv_tegra_release") or "tegra" in platform.release().lower()

###############################################################################
# Configuration
###############################################################################

# Per-machine settings (paths, defaults) can be overridden WITHOUT editing this file:
# put any of the keys below into config_local.json next to this script.
CONFIG = {
    "base_directory": ("M:/Research/Data/Vidalia_onions_visit2_day2" if IS_WINDOWS
                       else os.path.expanduser("~/ZED_Data/Vidalia_onions")),
    "wave_start_path": ("C:/Users/mm17889/Zed2iData/working.wav" if IS_WINDOWS
                        else os.path.join(SCRIPT_DIR, "working.wav")),
    "wave_stop_path": ("C:/Users/mm17889/Zed2iData/save.wav" if IS_WINDOWS
                       else os.path.join(SCRIPT_DIR, "save.wav")),

    "default_resolution": "HD2K (2208x1242)",
    # Jetson: the CPU cannot encode live FFV1 at HD2K and the GPU is shared with the
    # SVO pipeline, so record the lossless SVO2 only and use the light depth model
    # for preview / speed. Full-quality depth is recomputed offline from the SVO2.
    "default_depth_mode": "NEURAL_LIGHT" if IS_JETSON else "NEURAL",
    "default_record_mkv": not IS_JETSON,

    # Depth / confidence storage in the live .mkv files.
    "depth_units_per_mm": 10,      # 10 -> 0.1 mm per unit, max encodable 6553.3 mm
    "confidence_scale": 100,       # stored = round(confidence * 100)

    # Raw depth: keep every pixel the SDK produces; filter later with Confidence.mkv.
    "confidence_threshold": 100,
    "texture_confidence_threshold": 100,
    "enable_fill_mode": False,     # never hallucinate depth into holes
    "depth_stabilization": 0,      # 0 = independent per-frame depth (no temporal fusion)

    # Buffering between capture and the .mkv encoders (RAM, shared by all streams).
    "ram_buffer_gb": 3.0,

    # Disk safety.
    "min_free_space_gb": 20,       # refuse to start below this
    "autostop_free_space_gb": 5,   # stop + finalize cleanly below this

    # Preview / UI.
    "preview_max_width": 1400,
    "preview_interval_s": 0.10,
    # When the GUI is shown on another computer over SSH (X11 forwarding) every preview
    # image crosses the network: draw a smaller preview less often there.
    "remote_preview_max_width": 900,
    "remote_preview_interval_s": 0.4,
    "gui_poll_ms": 50,
    "camera_settings_poll_frames": 3,   # one setting per poll, round-robin (5 settings)
    "csv_flush_interval_s": 1.0,

    # Motion planning defaults (editable in the GUI).
    "default_camera_height_m": 1.00,
    "default_plant_height_m": 0.20,
    "default_blur_px": 1.0,
    "default_min_views": 20,
    "default_beep_spacing_m": 0.5,
}



def _load_local_config() -> None:
    path = os.path.join(SCRIPT_DIR, "config_local.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            local = json.load(f)
        unknown = [k for k in local if k not in CONFIG]
        CONFIG.update({k: v for k, v in local.items() if k in CONFIG})
        print(f"[INFO] loaded {path}" + (f" (ignored unknown keys {unknown})" if unknown else ""))
    except Exception as e:
        print(f"[WARN] could not read {path}: {e}")


_load_local_config()

# Allowed (resolution -> fps) combinations for the ZED 2i.
RESOLUTIONS = {
    "HD2K (2208x1242)": ("HD2K", [15]),
    "HD1080 (1920x1080)": ("HD1080", [30, 15]),
    "HD720 (1280x720)": ("HD720", [60, 30, 15]),
    "VGA (672x376)": ("VGA", [100, 60, 30, 15]),
}

DEPTH_MODES = ["NEURAL", "NEURAL_PLUS", "NEURAL_LIGHT", "ULTRA", "QUALITY",
               "NONE (SVO only, depth offline)"]

SVO_COMPRESSION = {
    "LOSSLESS (bit-exact, CPU)": "LOSSLESS",
    "H265_LOSSLESS (near-lossless, GPU)": "H265_LOSSLESS",
    "H264_LOSSLESS (near-lossless, GPU)": "H264_LOSSLESS",
}

DEPTH_CODE_NAN = 0
DEPTH_CODE_TOO_CLOSE = 65534
DEPTH_CODE_TOO_FAR = 65535
DEPTH_CODE_MAX_VALID = 65533
CONF_CODE_NAN = 65535

###############################################################################
# Small helpers
###############################################################################


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())
    return name.strip("_")[:40]


def safe(fn, default=None):
    """Evaluate fn() and return default on any exception (SDK-version tolerance)."""
    try:
        return fn()
    except Exception:
        return default


def err_is(err, name: str) -> bool:
    """True if a ZED ERROR_CODE equals ERROR_CODE.<name>.

    Never compare str(err): its text changed between SDK versions
    (SDK 4: 'END_OF_SVOFILE_REACHED', SDK 5: 'END OF SVO FILE REACHED').
    """
    code = getattr(sl.ERROR_CODE, name, None)
    if code is not None and err == code:
        return True
    norm = lambda x: "".join(ch for ch in str(x).upper() if ch.isalnum())
    return norm(err).endswith(norm(name))


def enum_name(value) -> str:
    s = str(value)
    return s.split(".")[-1] if "." in s else s


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def write_json_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mount_point_of(path: str) -> str:
    """Mount point of the filesystem that holds path (or would hold it once created)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    while not os.path.ismount(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def storage_problem(path: str):
    """Return an error message if path points at an external drive that is NOT mounted.

    Without this check a folder like /media/<user>/<SSD>/ZED_Data would silently be
    created on the (small) system disk when the SSD is unplugged or not mounted.
    """
    if IS_WINDOWS:
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        if drive and not os.path.exists(drive + "\\"):
            return f"Drive {drive} is not available. Connect / map it first."
        return None
    ap = os.path.abspath(path)
    if ap.startswith(("/media/", "/mnt/", "/run/media/")) and mount_point_of(ap) == "/":
        return (f"{path} is on an external drive that is NOT mounted (it would be written to "
                f"the system disk). Mount the SSD first (see JETSON.md: fstab entry).")
    return None


def get_free_space_bytes(path: str) -> int:
    try:
        probe = path
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return shutil.disk_usage(probe).free
    except Exception:
        return -1


def dir_size_bytes(path: str) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_file(follow_symlinks=False):
                    total += e.stat().st_size
    except Exception:
        pass
    return total


def svo_frame_count(path: str):
    """Number of frames stored in an SVO/SVO2 file, or None if it cannot be opened."""
    try:
        ip = sl.InitParameters()
        ip.set_from_svo_file(path)
        ip.depth_mode = sl.DEPTH_MODE.NONE
        safe(lambda: setattr(ip, "svo_real_time_mode", False))
        cam = sl.Camera()
        if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
            return None
        n = cam.get_svo_number_of_frames()
        cam.close()
        return int(n)
    except Exception:
        return None


def check_ffmpeg_available() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def ffmpeg_version() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        return r.stdout.splitlines()[0] if r.stdout else ""
    except Exception:
        return ""


def count_mkv_frames(path: str) -> int:
    """Packets in the video stream; for intra-only FFV1 this equals the frame count."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=600)
        return int(r.stdout.strip().split(",")[0])
    except Exception:
        return -1


def popen_hidden_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _linux_play(path: str) -> bool:
    """Play a WAV on Linux with whatever player exists (aplay ships with Jetson images)."""
    for player in (["paplay", path], ["aplay", "-q", path],
                   ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
        if shutil.which(player[0]):
            try:
                subprocess.run(player, timeout=10, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue
    return False


def play_wav_async(wav_path: str) -> None:
    def _play():
        if not wav_path or not os.path.exists(wav_path):
            # no sound file on this machine -> distinctive double beep instead
            beep_async(900, 120)
            return
        try:
            if IS_WINDOWS:
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            elif not _linux_play(wav_path):
                from pydub import AudioSegment
                from pydub.playback import play
                play(AudioSegment.from_wav(wav_path))
        except Exception as e:
            print(f"[WARN] Could not play {wav_path}: {e}")
    threading.Thread(target=_play, daemon=True).start()


_TONE_CACHE = {}


def _tone_wav(freq: int, dur_ms: int) -> str:
    """Write (once) a short sine-tone WAV to the temp dir and return its path."""
    key = (freq, dur_ms)
    if key not in _TONE_CACHE:
        import tempfile
        import wave
        rate = 22050
        t = np.arange(int(rate * dur_ms / 1000)) / rate
        ramp = np.minimum(1.0, np.minimum(t, t[::-1]) * 200)  # 5 ms fade, no clicks
        pcm = (np.sin(2 * np.pi * freq * t) * ramp * 0.6 * 32767).astype("<i2")
        path = os.path.join(tempfile.gettempdir(), f"weedscan_tone_{freq}_{dur_ms}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        _TONE_CACHE[key] = path
    return _TONE_CACHE[key]


def beep_async(freq: int = 1200, dur_ms: int = 80) -> None:
    def _beep():
        try:
            if IS_WINDOWS:
                import winsound
                winsound.Beep(freq, dur_ms)
            elif not _linux_play(_tone_wav(freq, dur_ms)):
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


_INHIBITOR = {"proc": None}


def keep_system_awake(enable: bool) -> None:
    """Stop the OS from sleeping / blanking the screen while recording."""
    if IS_WINDOWS:
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
            flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if enable else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except Exception:
            pass
        return
    # Linux / Jetson: hold a systemd inhibitor lock and stop X screen blanking.
    if enable and _INHIBITOR["proc"] is None and shutil.which("systemd-inhibit"):
        try:
            _INHIBITOR["proc"] = subprocess.Popen(
                ["systemd-inhibit", "--what=idle:sleep:handle-lid-switch", "--who=WeedScan",
                 "--why=ZED recording in progress", "--mode=block", "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    elif not enable and _INHIBITOR["proc"] is not None:
        try:
            _INHIBITOR["proc"].terminate()
        except Exception:
            pass
        _INHIBITOR["proc"] = None
    if os.environ.get("DISPLAY") and shutil.which("xset"):
        args = ["s", "off", "-dpms"] if enable else ["s", "on", "+dpms"]
        try:
            subprocess.run(["xset", *args], timeout=3, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass


def raise_process_priority() -> None:
    try:
        if IS_WINDOWS:
            import ctypes
            HIGH_PRIORITY_CLASS = 0x80
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),
                                                    HIGH_PRIORITY_CLASS)
        else:
            os.nice(-5)  # only works with CAP_SYS_NICE / root; harmless otherwise
    except Exception:
        pass


def disk_write_benchmark(directory: str, size_mb: int = 1024) -> float:
    """Sequential write speed in MB/s including fsync (what a long recording sees)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f".zed_disk_test_{os.getpid()}.bin")
    block = os.urandom(8 * 1024 * 1024)
    n = max(1, size_mb // 8)
    t0 = time.perf_counter()
    try:
        with open(path, "wb", buffering=0) as f:
            for _ in range(n):
                f.write(block)
            os.fsync(f.fileno())
        dt = time.perf_counter() - t0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return n * 8 / dt


###############################################################################
# Depth / confidence encoding (identical in extract_svo.py)
###############################################################################


def encode_depth_u16(depth_mm: np.ndarray, units_per_mm: float) -> np.ndarray:
    """float32 depth (mm, NaN/+inf/-inf codes) -> uint16 with documented sentinels."""
    q = np.rint(depth_mm.astype(np.float32, copy=False) * np.float32(units_per_mm))
    too_close = np.isneginf(q)
    np.nan_to_num(q, copy=False, nan=0.0, posinf=float(DEPTH_CODE_TOO_FAR), neginf=0.0)
    np.clip(q, 0.0, float(DEPTH_CODE_TOO_FAR), out=q)
    q[q > DEPTH_CODE_MAX_VALID] = DEPTH_CODE_TOO_FAR
    out = q.astype(np.uint16)
    out[too_close] = DEPTH_CODE_TOO_CLOSE
    return out


def encode_confidence_u16(conf: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(conf.astype(np.float32, copy=False) * np.float32(scale))
    np.nan_to_num(q, copy=False, nan=float(CONF_CODE_NAN), posinf=float(CONF_CODE_NAN),
                  neginf=0.0)
    np.clip(q, 0.0, float(CONF_CODE_NAN), out=q)
    return q.astype(np.uint16)


def decode_depth_u16(depth_u16: np.ndarray, units_per_mm: float) -> np.ndarray:
    d = depth_u16.astype(np.float32) / np.float32(units_per_mm)
    d[depth_u16 == DEPTH_CODE_NAN] = np.nan
    d[depth_u16 == DEPTH_CODE_TOO_CLOSE] = -np.inf
    d[depth_u16 == DEPTH_CODE_TOO_FAR] = np.inf
    return d


###############################################################################
# ZED SDK helpers (tolerant to SDK 4.x / 5.x API differences)
###############################################################################


def zed_get_setting(zed: sl.Camera, setting):
    r = safe(lambda: zed.get_camera_settings(setting))
    if isinstance(r, tuple):
        err, val = r[0], r[1]
        return val if err == sl.ERROR_CODE.SUCCESS else None
    return r


def cam_params_dict(c) -> dict:
    return {
        "fx": safe(lambda: c.fx), "fy": safe(lambda: c.fy),
        "cx": safe(lambda: c.cx), "cy": safe(lambda: c.cy),
        "disto": safe(lambda: list(np.asarray(c.disto, dtype=float))),
        "h_fov_deg": safe(lambda: c.h_fov), "v_fov_deg": safe(lambda: c.v_fov),
        "d_fov_deg": safe(lambda: c.d_fov),
        "image_size": safe(lambda: [c.image_size.width, c.image_size.height]),
        "focal_length_metric_mm": safe(lambda: c.focal_length_metric),
    }


def transform_dict(t) -> dict:
    return {
        "matrix_4x4": safe(lambda: np.asarray(t.m, dtype=float).tolist()),
        "translation": safe(lambda: list(t.get_translation().get())),
        "rotation_vector_rad": safe(lambda: list(t.get_rotation_vector())),
        "euler_angles_rad": safe(lambda: list(t.get_euler_angles(radian=True))),
    }


def calib_dict(cp) -> dict:
    return {
        "left_cam": cam_params_dict(cp.left_cam),
        "right_cam": cam_params_dict(cp.right_cam),
        "baseline": safe(lambda: cp.get_camera_baseline()),
        "stereo_transform": transform_dict(cp.stereo_transform),
    }


def camera_info_dict(zed: sl.Camera) -> dict:
    info = zed.get_camera_information()
    cfg = info.camera_configuration
    sens = safe(lambda: info.sensors_configuration)
    return {
        "serial_number": safe(lambda: int(info.serial_number)),
        "camera_model": safe(lambda: enum_name(info.camera_model)),
        "input_type": safe(lambda: enum_name(info.input_type)),
        "camera_firmware_version": safe(lambda: cfg.firmware_version),
        "sensors_firmware_version": safe(lambda: sens.firmware_version),
        "resolution": [cfg.resolution.width, cfg.resolution.height],
        "fps": cfg.fps,
        "units": "millimeter (translations, baseline)",
        "calibration_rectified": calib_dict(cfg.calibration_parameters),
        "calibration_raw_factory": safe(lambda: calib_dict(cfg.calibration_parameters_raw)),
        "camera_imu_transform": safe(lambda: transform_dict(sens.camera_imu_transform)),
    }


def write_legacy_calibration_txt(cam_info: dict, path: str) -> None:
    rect = cam_info["calibration_rectified"]
    lines = ["Rectified calibration (applies to RGB_left.mkv / Depth.mkv)", ""]
    for side in ("left_cam", "right_cam"):
        c = rect[side]
        lines += [f"{side}:",
                  f"  fx: {c['fx']}, fy: {c['fy']}",
                  f"  cx: {c['cx']}, cy: {c['cy']}",
                  f"  Distortion: {c['disto']}",
                  f"  HFOV: {c['h_fov_deg']}  VFOV: {c['v_fov_deg']}  DFOV: {c['d_fov_deg']}",
                  ""]
    lines += [f"Baseline (mm): {rect['baseline']}",
              f"Stereo translation (mm): {rect['stereo_transform']['translation']}",
              "",
              "Raw (unrectified, factory) calibration: see calibration.json",
              f"Serial: {cam_info['serial_number']}  Resolution: {cam_info['resolution']}"
              f"  FPS: {cam_info['fps']}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def copy_factory_calibration(serial, dest_dir: str) -> str | None:
    if serial is None:
        return None
    candidates = []
    for d in ("C:/ProgramData/Stereolabs/settings", "/usr/local/zed/settings",
              os.path.expanduser("~/.zed/settings")):
        candidates += glob.glob(os.path.join(d, f"SN{serial}.conf"))
    for c in candidates:
        try:
            shutil.copy2(c, dest_dir)
            return c
        except Exception:
            pass
    return None


def tilt_from_accel(acc) -> float | None:
    """Angle (deg) between the optical axis and vertical; 0 = looking straight down."""
    try:
        ax, ay, az = float(acc[0]), float(acc[1]), float(acc[2])
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            return None
        return math.degrees(math.acos(min(1.0, abs(az) / n)))
    except Exception:
        return None


###############################################################################
# Motion planning (speed recommendation)
###############################################################################


def motion_plan(camera_height_m, plant_height_m, f_px, n_px_along, fps, exposure_ms,
                blur_px, min_views) -> dict:
    """
    Nadir camera moving at speed v.
      GSD (mm/px) at the nearest surface = distance / f_px
      motion blur (px) = v * exposure / GSD          -> v_blur_max
      ground footprint along travel = H * n_px / f    -> views/point = footprint*fps/v
    """
    near_m = max(0.05, camera_height_m - plant_height_m)
    gsd_near_mm = near_m * 1000.0 / f_px
    gsd_ground_mm = camera_height_m * 1000.0 / f_px
    footprint_m = camera_height_m * n_px_along / f_px
    exp_s = max(1e-5, exposure_ms / 1000.0)
    v_blur = blur_px * (gsd_near_mm / 1000.0) / exp_s
    v_views = footprint_m * fps / max(1, min_views)
    v_rec = min(v_blur, v_views)
    return {
        "near_distance_m": near_m,
        "gsd_near_mm_per_px": gsd_near_mm,
        "gsd_ground_mm_per_px": gsd_ground_mm,
        "footprint_along_travel_m": footprint_m,
        "exposure_ms": exposure_ms,
        "v_max_blur_m_s": v_blur,
        "v_max_views_m_s": v_views,
        "v_recommended_m_s": v_rec,
        "views_per_point_at_rec": footprint_m * fps / v_rec if v_rec > 0 else float("inf"),
        "advance_per_frame_mm_at_rec": v_rec / fps * 1000.0,
    }


###############################################################################
# FFV1 stream writer (one thread + one ffmpeg process per stream)
###############################################################################


class FFV1StreamWriter(threading.Thread):
    def __init__(self, name, path, width, height, fps, in_pix_fmt, out_pix_fmt,
                 queue_len, log_path, encode_fn=None, metadata=None, session=None):
        super().__init__(daemon=False, name=f"writer-{name}")
        self.stream_name = name
        self.path = path
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.in_pix_fmt, self.out_pix_fmt = in_pix_fmt, out_pix_fmt
        self.q = queue.Queue(maxsize=max(2, int(queue_len)))
        self.log_path = log_path
        self.encode_fn = encode_fn
        self.metadata = metadata or {}
        self.session = session
        self.proc = None
        self.log_file = None
        self.frames_written = 0
        self.frames_lost_after_failure = 0
        self.failed = False
        self.error = None
        self.bytes_in = 0

    def start_process(self):
        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", self.in_pix_fmt,
               "-s", f"{self.width}x{self.height}", "-framerate", str(self.fps),
               "-i", "pipe:0",
               "-an", "-c:v", "ffv1", "-level", "3", "-g", "1",
               "-slices", "24", "-slicecrc", "1",
               "-pix_fmt", self.out_pix_fmt]
        for k, v in self.metadata.items():
            cmd += ["-metadata", f"{k}={v}"]
        cmd.append(self.path)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.log_file.write(" ".join(cmd) + "\n")
        self.log_file.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self.log_file, **popen_hidden_kwargs())

    def has_room(self) -> bool:
        return not self.q.full()

    def put(self, item) -> None:
        self.q.put_nowait(item)

    def finish(self) -> None:
        self.q.put(None)  # sentinel; blocking put is fine (consumer always drains)

    def _fail(self, msg):
        if not self.failed:
            self.failed = True
            self.error = msg
            if self.session:
                self.session.log_event("ERROR", f"[{self.stream_name}] {msg}")
            try:
                self.proc.kill()
            except Exception:
                pass

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.failed:
                self.frames_lost_after_failure += 1
                continue
            try:
                data = self.encode_fn(item) if self.encode_fn else item
                buf = memoryview(np.ascontiguousarray(data)).cast("B")
                self.proc.stdin.write(buf)
                self.bytes_in += buf.nbytes
                self.frames_written += 1
            except (BrokenPipeError, OSError, ValueError) as e:
                self._fail(f"ffmpeg pipe error after {self.frames_written} frames: {e}")
                self.frames_lost_after_failure += 1
        # finalize
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc:
                rc = self.proc.wait(timeout=600)
                if rc != 0 and not self.failed:
                    self._fail(f"ffmpeg exited with code {rc} (see {os.path.basename(self.log_path)})")
        except subprocess.TimeoutExpired:
            self._fail("ffmpeg did not finish within 600 s; killed")
        try:
            if self.log_file:
                self.log_file.close()
        except Exception:
            pass


###############################################################################
# Recording session (files, CSV, writers, statistics)
###############################################################################

FRAME_CSV_COLUMNS = [
    "rec_frame_idx", "grab_idx", "image_timestamp_ns", "system_time_ns", "system_time_iso",
    "dt_ms", "est_missing_before", "camera_dropped_total",
    "svo_frame_ok", "mkv_written", "mkv_frame_idx",
    "exposure_pct", "exposure_ms_approx", "gain", "aec_agc", "wb_temp_k", "wb_auto",
    "pose_state", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw", "pose_confidence",
    "speed_m_s", "distance_m",
    "imu_qx", "imu_qy", "imu_qz", "imu_qw",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "tilt_from_nadir_deg", "mag_heading_deg", "pressure_hpa", "rel_altitude_m", "imu_temp_c",
    "center_depth_mm",
]


class RecordingSession:
    def __init__(self, base_directory, field_info: dict, opts: dict):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = sanitize(field_info.get("plot_id", "") or "")
        name = f"Session_{stamp}" + (f"_{tag}" if tag else "")
        path = os.path.join(base_directory, name)
        i = 1
        while os.path.exists(path):
            path = os.path.join(base_directory, f"{name}_{i}")
            i += 1
        os.makedirs(path)
        self.dir = path
        self.field_info = field_info
        self.opts = opts
        self.lock = threading.Lock()
        self.events_path = os.path.join(path, "events.log")
        self.events_file = open(self.events_path, "a", encoding="utf-8")
        self.markers_path = os.path.join(path, "markers.csv")
        self.markers_file = open(self.markers_path, "w", newline="", encoding="utf-8")
        self.markers_writer = csv.writer(self.markers_file)
        self.markers_writer.writerow(["rec_frame_idx", "image_timestamp_ns", "system_time_iso", "text"])
        self.markers_file.flush()
        self.csv_path = os.path.join(path, "frames.csv")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(FRAME_CSV_COLUMNS)
        self.last_flush = time.monotonic()

        self.svo_path = os.path.join(path, "recording.svo2") if opts["record_svo"] else None
        self.writers: list[FFV1StreamWriter] = []
        self.metadata_path = os.path.join(path, "session_metadata.json")
        self.metadata = {}

        # statistics
        self.started_mono = None
        self.start_iso = None
        self.end_iso = None
        self.frames = 0
        self.mkv_enqueued = 0
        self.mkv_dropped = 0
        self.svo_ok = 0
        self.svo_failed = 0
        self.est_missing = 0
        self.grab_errors = collections.Counter()
        self.camera_dropped_at_start = None
        self.camera_dropped_now = None
        self.first_ts = None
        self.last_ts = None
        self.max_dt_ms = 0.0
        self.distance_m = 0.0
        self.svo_slow_warned = False
        self.duplicates = 0
        self.svo_comp_max_ms = 0.0
        self.svo_comp_sum_ms = 0.0
        self.svo_comp_n = 0
        self.finalized = False
        self.verification = {}

    # ---- logging -----------------------------------------------------------
    def log_event(self, level: str, msg: str) -> None:
        line = f"{now_iso()} [{level}] {msg}"
        print(line)
        with self.lock:
            try:
                if not self.events_file.closed:
                    self.events_file.write(line + "\n")
                    self.events_file.flush()
            except Exception:
                pass

    def add_marker(self, rec_frame_idx, ts_ns, text: str) -> None:
        with self.lock:
            if not self.markers_file.closed:
                self.markers_writer.writerow([rec_frame_idx, ts_ns, now_iso(), text])
                self.markers_file.flush()
        self.log_event("MARK", f"frame {rec_frame_idx}: {text}")

    # ---- setup -------------------------------------------------------------
    def setup_writers(self, width, height, fps, with_depth: bool) -> None:
        if not self.opts["record_mkv"]:
            return
        frame_bytes = width * height * 4 * (3 if with_depth else 1)
        qlen = int(self.opts["ram_buffer_gb"] * 1024 ** 3 / frame_bytes)
        qlen = max(8, min(qlen, 1000))
        self.queue_len = qlen
        common = dict(width=width, height=height, fps=fps, queue_len=qlen, session=self)
        self.writers.append(FFV1StreamWriter(
            "rgb", os.path.join(self.dir, "RGB_left.mkv"), in_pix_fmt="bgra", out_pix_fmt="bgr0",
            log_path=os.path.join(self.dir, "ffmpeg_rgb.log"),
            metadata={"title": "ZED rectified left image", "comment": "FFV1 lossless bgr0"},
            **common))
        if with_depth:
            upm = self.opts["depth_units_per_mm"]
            cs = self.opts["confidence_scale"]
            self.writers.append(FFV1StreamWriter(
                "depth", os.path.join(self.dir, "Depth.mkv"), in_pix_fmt="gray16le",
                out_pix_fmt="gray16le", log_path=os.path.join(self.dir, "ffmpeg_depth.log"),
                encode_fn=lambda d, upm=upm: encode_depth_u16(d, upm),
                metadata={"title": "ZED depth aligned to RGB_left",
                          "comment": f"uint16, 1 unit = {1.0 / upm} mm; 0=NaN, "
                                     f"65534=too close, 65535=too far"},
                **common))
            self.writers.append(FFV1StreamWriter(
                "confidence", os.path.join(self.dir, "Confidence.mkv"), in_pix_fmt="gray16le",
                out_pix_fmt="gray16le", log_path=os.path.join(self.dir, "ffmpeg_confidence.log"),
                encode_fn=lambda c, cs=cs: encode_confidence_u16(c, cs),
                metadata={"title": "ZED confidence map",
                          "comment": f"uint16 = confidence*{cs}; higher = less reliable; 65535=NaN"},
                **common))
        for w in self.writers:
            w.start_process()
            w.start()

    # ---- per frame (called from the capture thread) ------------------------
    def on_frame(self, fi: dict, bgra, depth, conf, period_ms: float) -> None:
        ts = fi["image_timestamp_ns"]
        dt_ms = ""
        missing = 0
        if self.last_ts is not None and ts is not None:
            dt = (ts - self.last_ts) / 1e6
            dt_ms = f"{dt:.3f}"
            self.max_dt_ms = max(self.max_dt_ms, dt)
            if dt > 1.5 * period_ms:
                missing = int(round(dt / period_ms)) - 1
                self.est_missing += missing
                self.log_event("WARN", f"timestamp gap {dt:.1f} ms before rec frame "
                                       f"{self.frames} (~{missing} camera frame(s) missing)")
            elif dt <= 0:
                self.log_event("WARN", f"non-increasing timestamp at rec frame {self.frames}")
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

        cur = fi.get("_svo_compress_cur_ms")
        if cur:
            self.svo_comp_max_ms = max(self.svo_comp_max_ms, cur)
            self.svo_comp_sum_ms += cur
            self.svo_comp_n += 1
        comp_ms = fi.get("_svo_compress_ms")
        if comp_ms and comp_ms > period_ms and not self.svo_slow_warned:
            self.svo_slow_warned = True
            self.log_event("WARN", f"SVO compression {comp_ms:.1f} ms/frame > frame period "
                                   f"{period_ms:.1f} ms: camera frames may be dropped. Use a lower "
                                   f"resolution or H265_LOSSLESS if drops appear.")
        svo_ok = fi.get("svo_frame_ok", "")
        if svo_ok == 1:
            self.svo_ok += 1
        elif svo_ok == 0:
            # The SDK status flag is advisory (it is commonly 0 for the very first frame);
            # finalize() counts the frames actually inside the SVO2 file.
            self.svo_failed += 1
            if self.svo_failed <= 20 or self.svo_failed % 100 == 0:
                self.log_event("WARN", f"SDK reported rec frame {self.frames} as not stored in SVO2 "
                                       f"(x{self.svo_failed}; checked against the file at stop)")

        mkv_written = ""
        mkv_idx = ""
        if self.writers:
            if all(w.has_room() for w in self.writers):
                items = [bgra, depth, conf]
                for w, it in zip(self.writers, items):
                    w.put(it)
                mkv_idx = self.mkv_enqueued
                self.mkv_enqueued += 1
                mkv_written = 1
            else:
                mkv_written = 0
                self.mkv_dropped += 1
                if self.mkv_dropped == 1 or self.mkv_dropped % 50 == 0:
                    self.log_event("WARN", f"MKV encoder backlog full: rec frame {self.frames} not "
                                           f"in .mkv (total {self.mkv_dropped}); "
                                           f"{'still in SVO2' if self.svo_path else 'LOST'}")

        self.camera_dropped_now = fi.get("camera_dropped_total")
        if self.camera_dropped_at_start is None:
            self.camera_dropped_at_start = self.camera_dropped_now
        self.distance_m = fi.get("distance_m") or self.distance_m

        row = dict(fi)
        row.update({"rec_frame_idx": self.frames, "dt_ms": dt_ms, "est_missing_before": missing,
                    "mkv_written": mkv_written, "mkv_frame_idx": mkv_idx})
        out = []
        for c in FRAME_CSV_COLUMNS:
            v = row.get(c, "")
            if isinstance(v, float):
                v = "" if not math.isfinite(v) else f"{v:.6g}"
            elif v is None:
                v = ""
            out.append(v)
        self.csv_writer.writerow(out)
        now = time.monotonic()
        if now - self.last_flush > CONFIG["csv_flush_interval_s"]:
            self.csv_file.flush()
            self.last_flush = now
        self.frames += 1

    def note_duplicate(self, ts) -> None:
        self.duplicates += 1
        if self.duplicates <= 5 or self.duplicates % 100 == 0:
            self.log_event("INFO", f"skipped a repeated camera frame (timestamp {ts}, "
                                   f"x{self.duplicates}); not logged or encoded twice")

    def note_grab_error(self, err_name: str) -> None:
        self.grab_errors[err_name] += 1
        if self.grab_errors[err_name] in (1, 10, 100) or self.grab_errors[err_name] % 1000 == 0:
            self.log_event("ERROR", f"grab() returned {err_name} "
                                    f"(x{self.grab_errors[err_name]} this session)")

    def backlog(self) -> tuple[int, int]:
        if not self.writers:
            return 0, 0
        return max(w.q.qsize() for w in self.writers), self.writers[0].q.maxsize

    # ---- finalize (called from a worker thread after capture stopped) -------
    def finalize(self) -> None:
        for w in self.writers:
            w.finish()
        for w in self.writers:
            w.join()
        try:
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            self.csv_file.close()
        except Exception:
            pass
        with self.lock:
            try:
                self.markers_file.close()
            except Exception:
                pass

        ver = {"checked_at": now_iso(), "streams": {}}
        ok = True
        for w in self.writers:
            n = count_mkv_frames(w.path)
            good = (n == w.frames_written) and not w.failed
            ok &= good
            ver["streams"][w.stream_name] = {
                "file": os.path.basename(w.path), "frames_sent": w.frames_written,
                "frames_in_file": n, "failed": w.failed, "error": w.error,
                "frames_lost_after_failure": w.frames_lost_after_failure,
                "size_bytes": safe(lambda: os.path.getsize(w.path)), "match": good}
        if self.writers:
            counts = {v["frames_in_file"] for v in ver["streams"].values()}
            ver["streams_aligned"] = len(counts) == 1
            ok &= ver["streams_aligned"]
        if self.svo_path:
            n_file = svo_frame_count(self.svo_path) if os.path.exists(self.svo_path) else -1
            ver["svo"] = {"file": os.path.basename(self.svo_path),
                          "exists": os.path.exists(self.svo_path),
                          "size_bytes": safe(lambda: os.path.getsize(self.svo_path)),
                          "frames_in_file": n_file, "frames_logged": self.frames,
                          "sdk_status_ok": self.svo_ok, "sdk_status_not_ok": self.svo_failed}
            if n_file is None:      # could not be checked (pyzed issue) -> rely on SDK status
                ok &= ver["svo"]["exists"] and self.svo_failed == 0
            else:
                ver["svo"]["missing_frames"] = self.frames - n_file
                ok &= n_file >= self.frames
                if n_file < self.frames:
                    self.log_event("ERROR", f"SVO2 holds {n_file} frames but {self.frames} were "
                                            f"logged ({self.frames - n_file} missing)")
        ver["all_ok"] = bool(ok)
        self.verification = ver
        self.finalized = True

    def stats(self) -> dict:
        duration = ((self.last_ts - self.first_ts) / 1e9) if (self.first_ts and self.last_ts) else 0
        cam_drop = None
        if self.camera_dropped_now is not None and self.camera_dropped_at_start is not None:
            cam_drop = self.camera_dropped_now - self.camera_dropped_at_start
            if cam_drop < 0:  # SDK counter was reset
                cam_drop = None
        return {
            "frames_grabbed": self.frames,
            "duration_s": duration,
            "effective_fps": (self.frames - 1) / duration if duration > 0 else None,
            "max_frame_interval_ms": self.max_dt_ms,
            "estimated_missing_camera_frames": self.est_missing,
            "camera_reported_dropped_frames": cam_drop,
            "duplicate_frames_skipped": self.duplicates,
            "capture_loop": getattr(self, "loop_diagnostics", None),
            "svo_compression_ms_mean": (self.svo_comp_sum_ms / self.svo_comp_n) if self.svo_comp_n else None,
            "svo_compression_ms_max": self.svo_comp_max_ms or None,
            "grab_errors": dict(self.grab_errors),
            "svo_frames_ok": self.svo_ok,
            "svo_frames_failed": self.svo_failed,
            "mkv_frames_enqueued": self.mkv_enqueued,
            "mkv_frames_dropped_backlog": self.mkv_dropped,
            "distance_travelled_m": self.distance_m,
            "mean_speed_m_s": self.distance_m / duration if duration > 0 else None,
        }

    def close_log(self) -> None:
        with self.lock:
            try:
                self.events_file.close()
            except Exception:
                pass


def session_readme(meta: dict) -> str:
    upm = meta["encoding"]["depth_units_per_mm"]
    cs = meta["encoding"]["confidence_scale"]
    return f"""WeedScan ZED 2i recording session
=================================
Created: {meta.get('start_time')}
Camera:  ZED serial {meta['camera'].get('serial_number')} at {meta['camera']['resolution']} @ {meta['camera']['fps']} fps
Live depth mode: {meta['capture']['depth_mode']}

FILES
-----
recording.svo2      MASTER RECORD (if enabled). Raw stereo pair + IMU + timestamps,
                    compression {meta['capture'].get('svo_compression')}. Open with ZED Explorer /
                    ZED Depth Viewer or extract_svo.py. Depth can be recomputed with any
                    depth mode (NEURAL_PLUS recommended for analysis).
RGB_left.mkv        Rectified LEFT image, FFV1 lossless, 8-bit BGR.
Depth.mkv           Depth (Z distance along optical axis) aligned pixel-for-pixel with
                    RGB_left.mkv. FFV1 lossless, 16-bit gray.
                      depth_mm = value / {upm}     (1 unit = {1.0 / upm} mm)
                      0     = no depth (occluded / no stereo match)
                      65534 = closer than minimum depth
                      65535 = farther than maximum depth or > {65533 / upm:.1f} mm
Confidence.mkv      ZED confidence, 16-bit gray. confidence = value / {cs} (0..100).
                    HIGHER = LESS reliable. 65535 = undefined. Depth was saved with
                    confidence_threshold=100 (nothing filtered); filter here, e.g. keep
                    confidence <= 50 for clean geometry.
frames.csv          One row per grabbed frame (see column list in session_metadata.json).
                    rec_frame_idx = row index; mkv_frame_idx = frame index inside the .mkv
                    files (blank if that frame was not written to .mkv);
                    image_timestamp_ns = ZED hardware timestamp -> use it to join .mkv
                    frames with SVO2 frames. Pose (tx..qw) is the left camera in the world
                    frame of positional tracking (mm, IMAGE coordinates: x right, y down,
                    z forward). tilt_from_nadir_deg: 0 = camera looking straight down.
markers.csv         Operator markers typed during recording.
events.log          All warnings/errors (timestamp gaps, encoder backlog, grab errors).
calibration.json    Rectified intrinsics (use these with RGB_left/Depth) + raw factory
                    calibration + camera-IMU transform. calibration_params.txt = summary.
session_metadata.json  Everything else: field info, parameters, statistics, verification.

EXTRACTING FRAMES
-----------------
RGB PNGs:    ffmpeg -i RGB_left.mkv -start_number 0 rgb/%06d.png
Depth PNGs:  ffmpeg -i Depth.mkv -start_number 0 -pix_fmt gray16le depth/%06d.png
Conf PNGs:   ffmpeg -i Confidence.mkv -start_number 0 -pix_fmt gray16le conf/%06d.png
(PNG number N = mkv_frame_idx N in frames.csv.)
Python:      depth_mm = png.astype(float32) / {upm}; set 0/65534/65535 to NaN.
From SVO2:   python extract_svo.py recording.svo2 --out extracted --depth-mode NEURAL_PLUS
3D point:    X = (u - cx) * Z / fx ; Y = (v - cy) * Z / fy ; Z = depth_mm
             (fx, fy, cx, cy from calibration.json -> calibration_rectified.left_cam)
"""


###############################################################################
# Capture engine: the ONLY thread that touches the camera after open()
###############################################################################


class CaptureEngine(threading.Thread):
    def __init__(self, zed, runtime_params, width, height, fps, depth_enabled,
                 tracking_enabled, ui_events: queue.Queue):
        super().__init__(daemon=False, name="capture")
        self.zed = zed
        self.rt = runtime_params
        self.width, self.height, self.fps = width, height, fps
        self.period_ms = 1000.0 / fps
        self.depth_enabled = depth_enabled
        self.tracking_enabled = tracking_enabled
        self.ui_events = ui_events
        self.stop_event = threading.Event()
        self.cmd_q = queue.Queue()

        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.conf_mat = sl.Mat()
        self.pose = sl.Pose()
        self.sensors = sl.SensorsData()

        self.session: RecordingSession | None = None
        self.grab_idx = 0
        self.consecutive_errors = 0
        self.cam_settings = {}
        self.pose_hist = collections.deque()
        self.prev_pos = None
        self.distance_m = 0.0

        self.latest_lock = threading.Lock()
        self.latest = {}
        self.last_preview = 0.0
        self.last_image_ts = None
        self.duplicate_frames = 0
        self.step_max = {}
        self.slow_steps = collections.Counter()
        self.prev_image_mat = sl.Mat()
        self.prev_depth_mat = sl.Mat()
        scale = min(1.0, (CONFIG["preview_max_width"] / 2) / max(1, width))
        self.preview_res = sl.Resolution(max(1, int(width * scale)), max(1, int(height * scale)))
        self.grab_times = collections.deque(maxlen=60)

    # ---- command interface (GUI thread) -------------------------------------
    def submit(self, name, *args, timeout=15.0):
        done = threading.Event()
        box = {}
        self.cmd_q.put((name, args, done, box))
        if not done.wait(timeout):
            return False, f"camera thread did not respond to '{name}' within {timeout:.0f} s"
        return box.get("result", (False, "no result"))

    def _ui(self, kind, payload):
        self.ui_events.put((kind, payload))

    def _process_commands(self):
        while True:
            try:
                name, args, done, box = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = getattr(self, f"_cmd_{name}")(*args)
            except Exception as e:
                box["result"] = (False, f"{type(e).__name__}: {e}")
            done.set()

    def _cmd_start_session(self, session: RecordingSession):
        if session.svo_path:
            mode_name = session.opts["svo_compression"]
            mode = getattr(sl.SVO_COMPRESSION_MODE, mode_name)
            params = sl.RecordingParameters(session.svo_path, mode)
            err = self.zed.enable_recording(params)
            if err != sl.ERROR_CODE.SUCCESS:
                return False, f"enable_recording({mode_name}) failed: {err}"
        self.distance_m = 0.0
        self.prev_pos = None
        self.session = session
        session.started_mono = time.monotonic()
        self.step_max = {}
        self.slow_steps = collections.Counter()
        return True, "ok"

    def _cmd_stop_session(self):
        s = self.session
        self.session = None
        if s is not None:
            s.loop_diagnostics = {
                "slow_step_counts": dict(self.slow_steps),
                "step_max_ms": {k: round(v * 1000, 1) for k, v in self.step_max.items()},
            }
        if s is not None and s.svo_path:
            try:
                self.zed.disable_recording()
            except Exception as e:
                s.log_event("ERROR", f"disable_recording failed: {e}")
        return True, s

    def _cmd_set_setting(self, setting_name, value):
        setting = getattr(sl.VIDEO_SETTINGS, setting_name)
        err = self.zed.set_camera_settings(setting, int(value))
        self._poll_camera_settings()
        if err is not None and err != sl.ERROR_CODE.SUCCESS:
            return False, f"set {setting_name}={value}: {err}"
        return True, "ok"

    def _cmd_lock_current(self):
        """Freeze the current auto exposure / gain / white balance as manual values."""
        self._poll_camera_settings()
        exp = self.cam_settings.get("exposure_pct")
        gain = self.cam_settings.get("gain")
        wb = self.cam_settings.get("wb_temp_k")
        if exp is None or gain is None:
            return False, "could not read current exposure/gain"
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(exp))
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(gain))
        if wb:
            wb = int(round(wb / 100.0) * 100)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
        self._poll_camera_settings()
        return True, f"locked exposure={exp}%, gain={gain}, WB={wb} K"

    def _cmd_reset_tracking(self):
        if self.tracking_enabled:
            self.zed.reset_positional_tracking(sl.Transform())
        self.pose_hist.clear()
        self.prev_pos = None
        return True, "ok"

    # ---- helpers ------------------------------------------------------------
    _SETTINGS = (("exposure_pct", "EXPOSURE"), ("gain", "GAIN"), ("aec_agc", "AEC_AGC"),
                 ("wb_temp_k", "WHITEBALANCE_TEMPERATURE"), ("wb_auto", "WHITEBALANCE_AUTO"))

    def _poll_camera_settings(self, only_next: bool = False):
        """Read camera settings over USB (~4 ms each on a Jetson).

        only_next=True reads ONE setting (round-robin) so the capture loop never spends
        ~20 ms in a burst of USB control transfers.
        """
        items = self._SETTINGS
        if only_next:
            self._settings_rr = (getattr(self, "_settings_rr", -1) + 1) % len(items)
            items = (items[self._settings_rr],)
        new = dict(self.cam_settings)
        for key, name in items:
            if hasattr(sl.VIDEO_SETTINGS, name):
                new[key] = zed_get_setting(self.zed, getattr(sl.VIDEO_SETTINGS, name))
        exp = new.get("exposure_pct")
        new["exposure_ms_approx"] = (exp / 100.0 * self.period_ms) if exp is not None else None
        self.cam_settings = new

    def _read_pose(self, fi, ts):
        state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
        fi["pose_state"] = enum_name(state)
        if state != sl.POSITIONAL_TRACKING_STATE.OK:
            return
        t = self.pose.get_translation(sl.Translation()).get()
        o = self.pose.get_orientation(sl.Orientation()).get()
        fi.update({"tx_mm": float(t[0]), "ty_mm": float(t[1]), "tz_mm": float(t[2]),
                   "qx": float(o[0]), "qy": float(o[1]), "qz": float(o[2]), "qw": float(o[3]),
                   "pose_confidence": safe(lambda: float(self.pose.pose_confidence))})
        pos = np.array([t[0], t[1], t[2]], dtype=float) / 1000.0
        if self.prev_pos is not None:
            step = float(np.linalg.norm(pos - self.prev_pos))
            if step < 1.0:  # ignore tracking jumps / relocalizations
                self.distance_m += step
        self.prev_pos = pos
        self.pose_hist.append((ts, pos))
        while self.pose_hist and ts - self.pose_hist[0][0] > 1.0e9:
            self.pose_hist.popleft()
        if len(self.pose_hist) >= 2:
            (t0, p0), (t1, p1) = self.pose_hist[0], self.pose_hist[-1]
            if t1 > t0:
                fi["speed_m_s"] = float(np.linalg.norm(p1 - p0) / ((t1 - t0) / 1e9))
        fi["distance_m"] = self.distance_m

    def _read_sensors(self, fi):
        err = self.zed.get_sensors_data(self.sensors, sl.TIME_REFERENCE.IMAGE)
        if err != sl.ERROR_CODE.SUCCESS:
            return
        imu = self.sensors.get_imu_data()
        q = safe(lambda: imu.get_pose().get_orientation().get())
        acc = safe(lambda: imu.get_linear_acceleration())
        gyr = safe(lambda: imu.get_angular_velocity())
        if q is not None:
            fi.update({"imu_qx": float(q[0]), "imu_qy": float(q[1]),
                       "imu_qz": float(q[2]), "imu_qw": float(q[3])})
        if acc is not None:
            fi.update({"acc_x": float(acc[0]), "acc_y": float(acc[1]), "acc_z": float(acc[2]),
                       "tilt_from_nadir_deg": tilt_from_accel(acc)})
        if gyr is not None:
            fi.update({"gyro_x": float(gyr[0]), "gyro_y": float(gyr[1]), "gyro_z": float(gyr[2])})
        fi["mag_heading_deg"] = safe(lambda: float(self.sensors.get_magnetometer_data().magnetic_heading))
        baro = safe(lambda: self.sensors.get_barometer_data())
        if baro is not None:
            fi["pressure_hpa"] = safe(lambda: float(baro.pressure))
            fi["rel_altitude_m"] = safe(lambda: float(baro.relative_altitude))
        fi["imu_temp_c"] = safe(lambda: float(
            self.sensors.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)))

    # ---- main loop ------------------------------------------------------------
    def run(self):
        raise_process_priority()
        self._poll_camera_settings()
        while not self.stop_event.is_set():
            self._process_commands()
            t_grab = time.perf_counter()
            err = self.zed.grab(self.rt)
            if err != sl.ERROR_CODE.SUCCESS:
                name = enum_name(err)
                self.consecutive_errors += 1
                if self.session:
                    self.session.note_grab_error(name)
                if self.consecutive_errors in (1, 15) or self.consecutive_errors % 150 == 0:
                    self._ui("camera_error", f"grab() -> {name} (x{self.consecutive_errors})")
                time.sleep(0.005)
                continue
            if self.consecutive_errors:
                self._ui("info", f"camera recovered after {self.consecutive_errors} failed grabs")
            self.consecutive_errors = 0
            t_after_grab = time.perf_counter()
            self.grab_times.append(t_after_grab - t_grab)
            steps = {"grab": t_after_grab - t_grab}

            ts = safe(lambda: self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if ts is not None and self.last_image_ts is not None and ts <= self.last_image_ts:
                # The SDK handed back an image it already delivered (it is not stored in the
                # SVO again either). Never log or encode it twice.
                self.duplicate_frames += 1
                if self.session:
                    self.session.note_duplicate(ts)
                continue
            self.last_image_ts = ts
            sys_ns = time.time_ns()
            if self.grab_idx % CONFIG["camera_settings_poll_frames"] == 0:
                t0 = time.perf_counter()
                self._poll_camera_settings(only_next=True)
                steps["camera_settings"] = time.perf_counter() - t0

            recording = self.session is not None
            want_preview = (time.monotonic() - self.last_preview) >= CONFIG["preview_interval_s"]
            need_depth = self.depth_enabled and (want_preview or (recording and self.session.writers))
            need_image = want_preview or (recording and self.session.writers)

            bgra = depth = conf = None
            full_res = recording and bool(self.session.writers)
            # Preview-only frames are downscaled by the SDK on the GPU: no full-resolution
            # copy to the CPU, which is the scarce resource on a Jetson.
            img_mat, dep_mat, res_args = ((self.image_mat, self.depth_mat, ()) if full_res else
                                          (self.prev_image_mat, self.prev_depth_mat,
                                           (sl.MEM.CPU, self.preview_res)))
            t0 = time.perf_counter()
            if need_image:
                self.zed.retrieve_image(img_mat, sl.VIEW.LEFT, *res_args)
                bgra = img_mat.get_data()
            if need_depth:
                self.zed.retrieve_measure(dep_mat, sl.MEASURE.DEPTH, *res_args)
                depth = dep_mat.get_data()
                if recording and self.session.writers:
                    self.zed.retrieve_measure(self.conf_mat, sl.MEASURE.CONFIDENCE)
                    conf = self.conf_mat.get_data()

            steps["retrieve"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            fi = {"grab_idx": self.grab_idx, "image_timestamp_ns": ts, "system_time_ns": sys_ns,
                  "system_time_iso": datetime.datetime.fromtimestamp(sys_ns / 1e9).isoformat(
                      timespec="milliseconds"),
                  "camera_dropped_total": safe(lambda: int(self.zed.get_frame_dropped_count()))}
            fi.update(self.cam_settings)
            if self.tracking_enabled:
                safe(lambda: self._read_pose(fi, ts))
            steps["pose"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            safe(lambda: self._read_sensors(fi))
            steps["sensors"] = time.perf_counter() - t0

            # Preview first: it fills fi["center_depth_mm"], which on_frame() writes to frames.csv
            t0 = time.perf_counter()
            if want_preview and bgra is not None:
                self.last_preview = time.monotonic()
                self._update_preview(bgra, depth, fi)
            else:
                with self.latest_lock:
                    self.latest["fi"] = fi
            steps["preview"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            if recording:
                s = self.session
                if s.svo_path:
                    rs = safe(lambda: self.zed.get_recording_status())
                    if rs is not None:
                        fi["svo_frame_ok"] = 1 if (rs.is_recording and rs.status) else 0
                        fi["_svo_compress_ms"] = safe(lambda: rs.average_compression_time)
                        fi["_svo_compress_cur_ms"] = safe(lambda: rs.current_compression_time)
                if s.writers:
                    # sl.Mat buffers are reused by the next grab -> copy for the writers
                    bgra_c = np.array(bgra, copy=True)
                    depth_c = np.array(depth, copy=True) if depth is not None else None
                    conf_c = np.array(conf, copy=True) if conf is not None else None
                else:
                    bgra_c = depth_c = conf_c = None
                s.on_frame(fi, bgra_c, depth_c, conf_c, self.period_ms)
            steps["record"] = time.perf_counter() - t0
            self._note_step_times(steps)
            self.grab_idx += 1

        # leaving: make sure SVO is closed properly
        if self.session is not None:
            self._cmd_stop_session()
        self._process_commands()

    def _note_step_times(self, steps: dict) -> None:
        """Keep per-step worst cases; log any step that could make the camera drop frames.

        grab() normally waits for the next frame, so it is only 'slow' beyond 1.5 periods;
        every other step should take a few ms.
        """
        period_s = self.period_ms / 1000.0
        for name, dt in steps.items():
            if dt > self.step_max.get(name, 0.0):
                self.step_max[name] = dt
            limit = 1.5 * period_s if name == "grab" else 0.5 * period_s
            if dt > limit:
                self.slow_steps[name] += 1
                if self.session is not None and self.slow_steps[name] <= 20:
                    self.session.log_event(
                        "WARN", f"slow step '{name}' took {dt * 1000:.0f} ms at grab {self.grab_idx} "
                                f"(steps: " + ", ".join(f"{k} {v * 1000:.0f}" for k, v in steps.items()) + " ms)")

    def _update_preview(self, bgra, depth, fi):
        h, w = bgra.shape[:2]
        scale = min(1.0, (CONFIG["preview_max_width"] / 2) / w)
        pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
        if (pw, ph) == (w, h):  # already downscaled on the GPU
            rgb_small = np.array(bgra, copy=True)
        else:
            rgb_small = cv2.resize(bgra, (pw, ph), interpolation=cv2.INTER_AREA)
        depth_small = None
        center_depth = None
        if depth is not None:
            if depth.shape[:2] == (ph, pw):
                depth_small = np.array(depth, copy=True)
            else:
                depth_small = cv2.resize(depth, (pw, ph), interpolation=cv2.INTER_NEAREST)
            cy0, cy1, cx0, cx1 = int(ph * 0.4), int(ph * 0.6), int(pw * 0.4), int(pw * 0.6)
            c = depth_small[cy0:cy1, cx0:cx1]
            c = c[np.isfinite(c)]
            if c.size:
                center_depth = float(np.median(c))
        fi["center_depth_mm"] = center_depth
        avg_grab = (sum(self.grab_times) / len(self.grab_times) * 1000) if self.grab_times else None
        with self.latest_lock:
            self.latest = {"rgb": rgb_small, "depth": depth_small, "fi": fi,
                           "grab_ms": avg_grab,
                           "current_fps": safe(lambda: self.zed.get_current_fps())}



###############################################################################
# Shared camera / session operations (used by the GUI and record_headless.py)
###############################################################################


def open_zed(zed, res_name: str, fps: int, depth_choice: str, tracking: bool):
    """Open the camera with the dataset's fixed parameters.

    Returns (ok, message, runtime_params, open_params, tracking_ok, warnings).
    """
    depth_enabled = not depth_choice.startswith("NONE")
    tracking = tracking and depth_enabled
    ip = sl.InitParameters()
    ip.camera_resolution = getattr(sl.RESOLUTION, res_name)
    ip.camera_fps = fps
    ip.depth_mode = getattr(sl.DEPTH_MODE, depth_choice) if depth_enabled else sl.DEPTH_MODE.NONE
    ip.coordinate_units = sl.UNIT.MILLIMETER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    # AUTO flip uses the IMU at open(); a downward-looking camera can come out
    # randomly flipped between sessions. Force a fixed orientation.
    safe(lambda: setattr(ip, "camera_image_flip", sl.FLIP_MODE.OFF))
    safe(lambda: setattr(ip, "depth_stabilization", int(CONFIG["depth_stabilization"])))
    ip.depth_minimum_distance = -1  # SDK default minimum (~300 mm for ZED 2i)

    err = zed.open(ip)
    if err != sl.ERROR_CODE.SUCCESS:
        hint = ""
        if err_is(err, "CAMERA_NOT_DETECTED") or err_is(err, "CAMERA_DETECTION_ISSUE"):
            hint = ("\nCheck the USB 3 cable/port (run zed_doctor.py) and that no other "
                    "program (ZED Explorer, Depth Viewer) is using the camera.")
        return False, f"{err}{hint}", None, None, False, []

    warnings = []
    cfg = zed.get_camera_information().camera_configuration
    if cfg.fps != fps or cfg.resolution.width <= 0:
        warnings.append(f"Camera opened at {cfg.resolution.width}x{cfg.resolution.height}"
                        f"@{cfg.fps} (requested {fps}).")

    tracking_ok = False
    if tracking:
        tp = sl.PositionalTrackingParameters()
        safe(lambda: setattr(tp, "enable_imu_fusion", True))
        safe(lambda: setattr(tp, "set_gravity_as_origin", True))
        terr = zed.enable_positional_tracking(tp)
        tracking_ok = terr == sl.ERROR_CODE.SUCCESS
        if not tracking_ok:
            warnings.append(f"Positional tracking unavailable ({terr}); speed will not be shown.")

    rt = sl.RuntimeParameters()
    rt.enable_depth = depth_enabled
    safe(lambda: setattr(rt, "confidence_threshold", CONFIG["confidence_threshold"]))
    safe(lambda: setattr(rt, "texture_confidence_threshold", CONFIG["texture_confidence_threshold"]))
    safe(lambda: setattr(rt, "enable_fill_mode", CONFIG["enable_fill_mode"]))
    safe(lambda: setattr(rt, "measure3D_reference_frame", sl.REFERENCE_FRAME.CAMERA))

    open_params = {
        "resolution": res_name, "fps": cfg.fps,
        "width": cfg.resolution.width, "height": cfg.resolution.height,
        "depth_mode": depth_choice if depth_enabled else "NONE",
        "depth_enabled": depth_enabled, "positional_tracking": tracking_ok,
        "init_parameters": {
            "coordinate_units": "MILLIMETER", "coordinate_system": "IMAGE",
            "camera_image_flip": safe(lambda: enum_name(ip.camera_image_flip)),
            "depth_stabilization": safe(lambda: ip.depth_stabilization),
            "depth_minimum_distance": safe(lambda: ip.depth_minimum_distance),
            "depth_maximum_distance": safe(lambda: ip.depth_maximum_distance),
            "enable_image_enhancement": safe(lambda: ip.enable_image_enhancement),
        },
        "runtime_parameters": {
            "enable_depth": depth_enabled,
            "confidence_threshold": safe(lambda: rt.confidence_threshold),
            "texture_confidence_threshold": safe(lambda: rt.texture_confidence_threshold),
            "enable_fill_mode": safe(lambda: rt.enable_fill_mode),
            "remove_saturated_areas": safe(lambda: rt.remove_saturated_areas),
        },
    }
    return True, "ok", rt, open_params, tracking_ok, warnings


def start_session(zed, engine, session, open_params: dict, speed_plan=None,
                  disk_bench_mb_s=None):
    """Write calibration + metadata, start encoders, then start capture. Returns (ok, msg)."""
    opts = session.opts
    field_info = session.field_info
    # Calibration + metadata BEFORE the first frame, so a crash still leaves them.
    cam_info = camera_info_dict(zed)
    write_json_atomic(os.path.join(session.dir, "calibration.json"), cam_info)
    write_legacy_calibration_txt(cam_info, os.path.join(session.dir, "calibration_params.txt"))
    conf_src = copy_factory_calibration(cam_info.get("serial_number"), session.dir)
    session.metadata = {
        "app": {"name": APP_NAME, "version": APP_VERSION, "script": os.path.abspath(__file__)},
        "start_time": now_iso(),
        "field": field_info,
        "camera": {k: cam_info[k] for k in ("serial_number", "camera_model", "camera_firmware_version",
                                            "sensors_firmware_version", "resolution", "fps")},
        "camera_settings_at_start": dict(engine.cam_settings),
        "capture": {**open_params, "svo_compression": opts["svo_compression"] if opts["record_svo"] else None,
                    "record_svo": opts["record_svo"], "record_mkv": opts["record_mkv"]},
        "encoding": {
            "rgb": "FFV1 level 3, intra-only (-g 1), 24 slices with CRC, bgr0 (lossless 8-bit BGR)",
            "depth": "FFV1 gray16le", "depth_units_per_mm": opts["depth_units_per_mm"],
            "depth_codes": {"no_data_nan": DEPTH_CODE_NAN, "too_close": DEPTH_CODE_TOO_CLOSE,
                            "too_far_or_out_of_range": DEPTH_CODE_TOO_FAR,
                            "max_valid": DEPTH_CODE_MAX_VALID},
            "confidence": "FFV1 gray16le", "confidence_scale": opts["confidence_scale"],
            "confidence_nan_code": CONF_CODE_NAN,
            "mkv_framerate_note": "Constant nominal rate; true timing is image_timestamp_ns in frames.csv",
        },
        "frames_csv_columns": FRAME_CSV_COLUMNS,
        "speed_plan": speed_plan,
        "factory_calibration_file": conf_src,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "jetson": jetson_info() if IS_JETSON else None,
                 "python": sys.version.split()[0], "numpy": np.__version__,
                 "opencv": cv2.__version__, "ffmpeg": ffmpeg_version(),
                 "zed_sdk": safe(lambda: str(sl.Camera.get_sdk_version())),
                 "disk_benchmark_mb_s": disk_bench_mb_s},
        "status": "RECORDING (if this persists the app did not stop cleanly; data up to the "
                  "crash is still usable - run verify_session.py)",
    }
    write_json_atomic(session.metadata_path, session.metadata)
    with open(os.path.join(session.dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(session_readme(session.metadata))

    try:
        session.setup_writers(engine.width, engine.height, engine.fps,
                              with_depth=open_params["depth_enabled"])
    except Exception as e:
        session.log_event("ERROR", f"cannot start ffmpeg: {e}")
        for w in session.writers:
            if w.is_alive():
                w.finish()
            elif w.proc is not None:
                safe(w.proc.kill)
        session.close_log()
        return False, f"cannot start encoders: {e}"

    ok, msg = engine.submit("start_session", session)
    if not ok:
        session.log_event("ERROR", f"start failed: {msg}")
        session.finalize()  # closes the (empty) encoders and files
        session.close_log()
        return False, msg
    session.start_iso = now_iso()
    session.log_event("INFO", f"recording started -> {session.dir}")
    return True, "ok"


def finish_session_metadata(session) -> dict:
    """After session.finalize(): write final statistics/verification and close the log."""
    stats = session.stats()
    session.metadata.update({"end_time": session.end_iso, "statistics": stats,
                             "verification": session.verification,
                             "status": "COMPLETE" if session.verification.get("all_ok") else
                             "COMPLETE_WITH_WARNINGS"})
    try:
        write_json_atomic(session.metadata_path, session.metadata)
    except Exception as e:
        print(f"[ERROR] cannot write metadata: {e}")
    session.log_event("INFO", f"finalized: {json.dumps(to_jsonable(stats))}")
    session.close_log()
    return stats


def jetson_info() -> dict:
    info = {}
    try:
        with open("/etc/nv_tegra_release", encoding="utf-8") as f:
            info["l4t"] = f.readline().strip()
    except Exception:
        pass
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            info["model"] = f.read().strip("\x00\n ")
    except Exception:
        pass
    if shutil.which("nvpmodel"):
        r = safe(lambda: subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                        timeout=5).stdout)
        if r:
            info["nvpmodel"] = " ".join(r.split())
    return info


###############################################################################
# Capture process: the camera, capture loop and recording run in their OWN process
# so the GUI (Tk redraws, dialogs, window moves) can never delay a grab().
###############################################################################

RES_SIZES = {"HD2K": (2208, 1242), "HD1080": (1920, 1080), "HD720": (1280, 720), "VGA": (672, 376)}


def _session_summary(s, stats) -> dict:
    return {"dir": s.dir, "has_svo": bool(s.svo_path), "has_writers": bool(s.writers),
            "stats": stats, "verification": s.verification}


def capture_server(conn, status_q, event_q, config: dict) -> None:
    """Entry point of the capture process (started with the 'spawn' method)."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the GUI process
    except Exception:
        pass
    CONFIG.update(config)
    # Status snapshots are disposable: never let unread ones block this process at exit.
    status_q.cancel_join_thread()
    raise_process_priority()
    zed = sl.Camera()
    ui = queue.Queue()
    st = {"engine": None, "session": None, "open_params": None, "finalizers": []}

    def finish(reason: str) -> None:
        s, eng = st["session"], st["engine"]
        st["session"] = None
        if s is None:
            return
        if eng is not None:
            eng.submit("stop_session", timeout=30)
        s.end_iso = now_iso()
        s.log_event("INFO", f"recording stopped ({reason}); finalizing files…")

        def run():
            try:
                s.finalize()
            except Exception as e:
                s.log_event("ERROR", f"finalize failed: {e}")
            stats = finish_session_metadata(s)
            event_q.put(("finalized", _session_summary(s, stats)))
        t = threading.Thread(target=run, name="finalize", daemon=False)
        t.start()
        st["finalizers"].append(t)

    def shutdown() -> None:
        finish("camera closed")
        for t in st["finalizers"]:
            t.join()
        eng = st["engine"]
        if eng is not None:
            eng.stop_event.set()
            eng.join(timeout=30)
        st["engine"] = None
        try:
            zed.close()
        except Exception:
            pass

    def handle(name, args):
        eng = st["engine"]
        if name == "open":
            res_name, fps, depth_choice, tracking = args
            ok, msg, rt, op, trk, warnings = open_zed(zed, res_name, fps, depth_choice, tracking)
            if not ok:
                return False, msg, None, False, [], None
            st["open_params"] = op
            st["engine"] = CaptureEngine(zed, rt, op["width"], op["height"], op["fps"],
                                         op["depth_enabled"], trk, ui)
            st["engine"].start()
            return True, "ok", op, trk, warnings, camera_info_dict(zed)
        if name == "submit":
            cmd, cargs = args
            return eng.submit(cmd, *cargs) if eng else (False, "camera not open")
        if name == "start_recording":
            base, field_info, opts, plan, bench = args
            if st["session"] is not None or eng is None:
                return False, "already recording or camera not open", None
            s = RecordingSession(base, field_info, opts)
            ok, msg = start_session(zed, eng, s, st["open_params"], speed_plan=plan,
                                    disk_bench_mb_s=bench)
            if not ok:
                return False, msg, s.dir
            st["session"] = s
            return True, "ok", s.dir
        if name == "stop_recording":
            finish(args[0])
            return True, "ok"
        if name == "marker":
            s = st["session"]
            if s is None or eng is None:
                return False, "not recording"
            with eng.latest_lock:
                fi = dict(eng.latest.get("fi") or {})
            s.add_marker(max(0, s.frames - 1), fi.get("image_timestamp_ns"), args[0])
            return True, "ok"
        if name == "log":
            if st["session"] is not None:
                st["session"].log_event(*args)
            return True, "ok"
        if name == "shutdown":
            shutdown()
            return True, "ok"
        return False, f"unknown command {name}"

    last_status = 0.0
    last_preview = None
    while True:
        try:
            if conn.poll(0.02):
                name, args = conn.recv()
                try:
                    result = handle(name, args)
                except Exception as e:
                    result = (False, f"{type(e).__name__}: {e}")
                conn.send(result)
                if name == "shutdown":
                    break
        except (EOFError, OSError, BrokenPipeError):
            # The GUI process is gone (crashed / killed): save what was recorded.
            print("[WARN] GUI process disappeared - stopping and saving the recording")
            shutdown()
            break
        while True:
            try:
                event_q.put(ui.get_nowait())
            except queue.Empty:
                break
        eng = st["engine"]
        now = time.monotonic()
        if eng is not None and now - last_status >= 0.1:
            last_status = now
            with eng.latest_lock:
                latest = dict(eng.latest)
            snap = {"fi": latest.get("fi"), "grab_ms": latest.get("grab_ms"),
                    "current_fps": latest.get("current_fps"), "cam_settings": dict(eng.cam_settings)}
            rgb = latest.get("rgb")
            if rgb is not None and rgb is not last_preview:
                last_preview = rgb
                snap["rgb"] = np.ascontiguousarray(rgb[:, :, :3])
                d = latest.get("depth")
                snap["depth"] = d.astype(np.float16) if d is not None else None
            s = st["session"]
            if s is not None:
                snap["session"] = {"stats": s.stats(), "backlog": s.backlog(), "frames": s.frames,
                                   "writers_failed": any(w.failed for w in s.writers)}
            try:
                status_q.put_nowait(snap)
            except queue.Full:
                pass
    for t in st["finalizers"]:
        t.join()


class RemoteEngine:
    """GUI-side handle of the capture process (same attributes the GUI used before)."""

    def __init__(self, ui_events: queue.Queue):
        ctx = multiprocessing.get_context("spawn")  # never fork a process that holds CUDA
        self._conn, child = ctx.Pipe()
        self._status = ctx.Queue(maxsize=4)
        self._events = ctx.Queue()
        self.proc = ctx.Process(target=capture_server, name="weedscan-capture",
                                args=(child, self._status, self._events, dict(CONFIG)))
        self.proc.start()
        self._lock = threading.Lock()
        self.ui_events = ui_events
        self.latest_lock = threading.Lock()
        self.latest = {}
        self.cam_settings = {}
        self.session_status = None
        self.width = self.height = self.fps = None
        self.depth_enabled = False

    def alive(self) -> bool:
        return self.proc.is_alive()

    def call(self, name, *args, timeout=60.0, tick=None):
        with self._lock:
            try:
                self._conn.send((name, args))
            except (BrokenPipeError, OSError):
                return (False, "capture process is not running")
            deadline = time.monotonic() + timeout
            while not self._conn.poll(0.05):
                if tick:
                    tick()
                if not self.proc.is_alive():
                    return (False, "capture process stopped unexpectedly")
                if time.monotonic() > deadline:
                    return (False, f"capture process did not answer '{name}' in {timeout:.0f} s")
            try:
                return self._conn.recv()
            except (EOFError, OSError) as e:
                return (False, f"lost connection to the capture process: {e}")

    def submit(self, name, *args, timeout=15.0):
        return self.call("submit", name, args, timeout=timeout)

    def pump(self) -> None:
        """Pull status snapshots and events from the capture process (GUI thread)."""
        while True:
            try:
                snap = self._status.get_nowait()
            except queue.Empty:
                break
            with self.latest_lock:
                if "rgb" in snap:
                    d = snap.get("depth")
                    self.latest["rgb"] = snap["rgb"]
                    self.latest["depth"] = d.astype(np.float32) if d is not None else None
                    self.latest["_shown"] = False
                for k in ("fi", "grab_ms", "current_fps"):
                    self.latest[k] = snap.get(k)
            self.cam_settings = snap.get("cam_settings") or self.cam_settings
            self.session_status = snap.get("session")
        while True:
            try:
                self.ui_events.put(self._events.get_nowait())
            except queue.Empty:
                break

    def shutdown(self, tick=None) -> None:
        if self.proc.is_alive():
            self.call("shutdown", timeout=3600, tick=tick)
        deadline = time.monotonic() + 30
        while self.proc.is_alive() and time.monotonic() < deadline:
            self.pump()            # keep draining so the child can flush and exit
            self.proc.join(timeout=0.1)
        if self.proc.is_alive():
            print("[WARN] capture process did not exit; terminating it")
            self.proc.terminate()
            self.proc.join(timeout=5)


class RemoteSession:
    """GUI-side view of the recording session running in the capture process."""

    def __init__(self, engine: RemoteEngine, directory: str):
        self.engine = engine
        self.dir = directory

    def _status(self):
        return self.engine.session_status or {}

    @property
    def frames(self) -> int:
        return self._status().get("frames", 0)

    @property
    def writers_failed(self) -> bool:
        return bool(self._status().get("writers_failed"))

    def stats(self) -> dict:
        st = self._status().get("stats")
        return st or {"duration_s": 0.0, "frames_grabbed": 0, "estimated_missing_camera_frames": 0,
                      "svo_frames_ok": 0, "svo_frames_failed": 0, "mkv_frames_dropped_backlog": 0}

    def backlog(self):
        return tuple(self._status().get("backlog") or (0, 0))

    def add_marker(self, text: str):
        return self.engine.call("marker", text, timeout=10)

    def log_event(self, level: str, msg: str):
        return self.engine.call("log", level, msg, timeout=10)

###############################################################################
# GUI
###############################################################################


class ZEDTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.camera_opened = False
        self.engine: RemoteEngine | None = None
        self.session: RemoteSession | None = None
        self.finalizing = False
        self.ui_events = queue.Queue()
        self.cam_info = None
        self.open_params = {}
        self.disk_bench_mb_s = None
        self.last_disk_check = 0.0
        self.last_size = (0.0, 0)
        self.write_rate_mb_s = None
        self.next_beep = None
        self.last_error_beep = 0.0

        self._style()
        self._build_gui()
        self.root.bind("<F9>", lambda e: self.add_marker())
        self.root.after(CONFIG["gui_poll_ms"], self._poll)

    # ------------------------------------------------------------------ layout
    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("AppBar.TFrame", background="#E9ECEF")
        style.configure("AppBar.TLabel", font=("Segoe UI", 16, "bold"), background="#E9ECEF")
        style.configure("Sub.TLabel", font=("Segoe UI", 13, "italic"), background="#E9ECEF")
        style.configure("TLabel", font=("Segoe UI", 10), background="white")
        style.configure("TFrame", background="white")
        style.configure("TLabelframe", background="white")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="white")
        style.configure("TCheckbutton", background="white")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), foreground="white",
                        background="#0078D7", padding=5)
        style.map("TButton", background=[("active", "#005A9E"), ("disabled", "#9AA5B1")])
        style.configure("Rec.TButton", background="#C62828")
        style.map("Rec.TButton", background=[("active", "#8E0000"), ("disabled", "#9AA5B1")])
        style.configure("WeedScan.TLabel", font=("Times New Roman", 18, "bold"),
                        foreground="#BA0C2F", background="#E9ECEF")
        style.configure("Big.TLabel", font=("Consolas" if IS_WINDOWS else "DejaVu Sans Mono", 10), background="white")
        style.configure("Speed.TLabel", font=("Segoe UI", 16, "bold"), background="white")
        self.root.configure(bg="white")

    def _build_gui(self):
        bar = ttk.Frame(self.root, padding=(10, 8, 10, 4), style="AppBar.TFrame")
        bar.pack(side=tk.TOP, fill=tk.X)
        c = ttk.Frame(bar, style="AppBar.TFrame")
        c.pack(anchor="center")
        ttk.Label(c, text="WeedScan", style="WeedScan.TLabel").pack(side=tk.LEFT)
        ttk.Label(c, text=": Data Collection System for Vidalia Onion Weeds",
                  style="AppBar.TLabel").pack(side=tk.LEFT, padx=5)
        ttk.Label(bar, text="Precision Crop Protection Lab", style="Sub.TLabel").pack(anchor="center")

        top = ttk.Frame(self.root, padding=5)
        top.pack(side=tk.TOP, fill=tk.X)
        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # --- camera
        cam = ttk.LabelFrame(left, text="1. Camera", padding=5)
        cam.pack(fill=tk.X)
        ttk.Label(cam, text="Resolution").grid(row=0, column=0, sticky="w")
        self.res_var = tk.StringVar(value=CONFIG["default_resolution"])
        self.res_menu = ttk.Combobox(cam, textvariable=self.res_var, values=list(RESOLUTIONS),
                                     width=18, state="readonly")
        self.res_menu.grid(row=0, column=1, sticky="w")
        self.res_menu.bind("<<ComboboxSelected>>", lambda e: self._update_fps_choices())
        ttk.Label(cam, text="FPS").grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.fps_var = tk.StringVar()
        self.fps_menu = ttk.Combobox(cam, textvariable=self.fps_var, width=5, state="readonly")
        self.fps_menu.grid(row=0, column=3, sticky="w")
        self._update_fps_choices()
        ttk.Label(cam, text="Live depth").grid(row=1, column=0, sticky="w")
        default_depth = CONFIG["default_depth_mode"]
        if not (default_depth.startswith("NONE") or hasattr(sl.DEPTH_MODE, default_depth)):
            default_depth = "NEURAL"  # e.g. NEURAL_LIGHT needs ZED SDK >= 4.2
        self.depth_var = tk.StringVar(value=default_depth)
        avail = [m for m in DEPTH_MODES if m.startswith("NONE") or hasattr(sl.DEPTH_MODE, m)]
        self.depth_menu = ttk.Combobox(cam, textvariable=self.depth_var, values=avail, width=18,
                                       state="readonly")
        self.depth_menu.grid(row=1, column=1, sticky="w")
        self.track_var = tk.BooleanVar(value=True)
        self.track_chk = ttk.Checkbutton(cam, text="Pose / speed tracking", variable=self.track_var)
        self.track_chk.grid(row=1, column=2, columnspan=2, sticky="w", padx=(6, 0))
        self.open_cam_btn = ttk.Button(cam, text="Open Camera", command=self.open_camera)
        self.open_cam_btn.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self.close_cam_btn = ttk.Button(cam, text="Close Camera", command=self.close_camera,
                                        state=tk.DISABLED)
        self.close_cam_btn.grid(row=2, column=2, columnspan=2, sticky="we", pady=(4, 0), padx=(6, 0))

        # --- exposure
        ex = ttk.LabelFrame(left, text="2. Exposure (short exposure = no motion blur)", padding=5)
        ex.pack(fill=tk.X, pady=(5, 0))
        self.auto_exp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ex, text="Auto exposure/gain", variable=self.auto_exp_var,
                        command=self._apply_auto_exposure).grid(row=0, column=0, columnspan=2, sticky="w")
        self.auto_wb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ex, text="Auto white balance", variable=self.auto_wb_var,
                        command=self._apply_auto_wb).grid(row=0, column=2, columnspan=2, sticky="w")
        ttk.Label(ex, text="Exposure %").grid(row=1, column=0, sticky="w")
        self.exp_var = tk.StringVar(value="5")
        ttk.Spinbox(ex, from_=1, to=100, textvariable=self.exp_var, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(ex, text="Gain").grid(row=1, column=2, sticky="w")
        self.gain_var = tk.StringVar(value="50")
        ttk.Spinbox(ex, from_=0, to=100, textvariable=self.gain_var, width=5).grid(row=1, column=3, sticky="w")
        ttk.Button(ex, text="Apply manual", command=self._apply_manual_exposure).grid(
            row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(ex, text="Lock current auto values", command=self._lock_current).grid(
            row=2, column=2, columnspan=2, sticky="we", pady=(4, 0), padx=(6, 0))

        # --- field info
        fld = ttk.LabelFrame(left, text="3. Field / session info (saved with data)", padding=5)
        fld.pack(fill=tk.X, pady=(5, 0))
        self.base_dir_var = tk.StringVar(value=CONFIG["base_directory"])
        ttk.Label(fld, text="Save to").grid(row=0, column=0, sticky="w")
        ttk.Entry(fld, textvariable=self.base_dir_var, width=34).grid(row=0, column=1, columnspan=2, sticky="we")
        ttk.Button(fld, text="…", width=3, command=self._browse).grid(row=0, column=3, sticky="w")
        self.field_vars = {}
        fields = [("site", "Site / field", ""), ("plot_id", "Plot / row ID", ""),
                  ("operator", "Operator", ""), ("crop_stage", "Crop stage / DAP", ""),
                  ("travel_direction", "Travel direction", "image top = forward"),
                  ("weather", "Weather / light", ""), ("notes", "Notes", "")]
        for i, (k, label, default) in enumerate(fields, start=1):
            ttk.Label(fld, text=label).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value=default)
            ttk.Entry(fld, textvariable=v, width=34).grid(row=i, column=1, columnspan=3, sticky="we")
            self.field_vars[k] = v

        # --- speed planning
        sp = ttk.LabelFrame(right, text="4. Speed planner (nadir camera)", padding=5)
        sp.pack(fill=tk.X)
        self.h_var = tk.StringVar(value=str(CONFIG["default_camera_height_m"]))
        self.plant_var = tk.StringVar(value=str(CONFIG["default_plant_height_m"]))
        self.blur_var = tk.StringVar(value=str(CONFIG["default_blur_px"]))
        self.views_var = tk.StringVar(value=str(CONFIG["default_min_views"]))
        self.axis_var = tk.StringVar(value="Image vertical (rows)")
        self.target_speed_var = tk.StringVar(value="0.25")
        self.beep_var = tk.BooleanVar(value=False)
        self.beep_spacing_var = tk.StringVar(value=str(CONFIG["default_beep_spacing_m"]))
        row = [("Lens height above ground (m)", self.h_var), ("Tallest plant (m)", self.plant_var),
               ("Max motion blur (px)", self.blur_var), ("Min views per point", self.views_var)]
        for i, (label, var) in enumerate(row):
            ttk.Label(sp, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky="w")
            ttk.Entry(sp, textvariable=var, width=7).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w")
        ttk.Label(sp, text="Travel along").grid(row=2, column=0, sticky="w")
        ttk.Combobox(sp, textvariable=self.axis_var, width=20, state="readonly",
                     values=["Image vertical (rows)", "Image horizontal (cols)"]).grid(row=2, column=1, sticky="w")
        ttk.Label(sp, text="Target speed (m/s)").grid(row=2, column=2, sticky="w")
        ttk.Entry(sp, textvariable=self.target_speed_var, width=7).grid(row=2, column=3, sticky="w")
        ttk.Button(sp, text="Use recommended", command=self._use_recommended).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(sp, text="Pace beep every", variable=self.beep_var).grid(row=3, column=1, sticky="e")
        ttk.Entry(sp, textvariable=self.beep_spacing_var, width=5).grid(row=3, column=2, sticky="w")
        ttk.Label(sp, text="m of travel").grid(row=3, column=3, sticky="w")
        self.plan_var = tk.StringVar(value="Open the camera to compute the speed plan.")
        ttk.Label(sp, textvariable=self.plan_var, style="Big.TLabel", justify=tk.LEFT,
                  wraplength=620).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.speed_var = tk.StringVar(value="Speed: –")
        self.speed_label = ttk.Label(sp, textvariable=self.speed_var, style="Speed.TLabel")
        self.speed_label.grid(row=5, column=0, columnspan=4, sticky="w")

        # --- recording
        rec = ttk.LabelFrame(right, text="5. Record", padding=5)
        rec.pack(fill=tk.X, pady=(5, 0))
        self.svo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rec, text="SVO2 master record", variable=self.svo_var).grid(row=0, column=0, sticky="w")
        self.svo_mode_var = tk.StringVar(value=list(SVO_COMPRESSION)[0])
        ttk.Combobox(rec, textvariable=self.svo_mode_var, values=list(SVO_COMPRESSION), width=32,
                     state="readonly").grid(row=0, column=1, columnspan=2, sticky="w")
        self.mkv_var = tk.BooleanVar(value=bool(CONFIG["default_record_mkv"]))
        ttk.Checkbutton(rec, text="Live FFV1 RGB + Depth + Confidence (.mkv)",
                        variable=self.mkv_var).grid(row=1, column=0, columnspan=3, sticky="w")
        self.start_btn = ttk.Button(rec, text="● Start Recording", style="Rec.TButton",
                                    command=self.start_recording, state=tk.DISABLED)
        self.start_btn.grid(row=2, column=0, sticky="we", pady=(4, 0))
        self.stop_btn = ttk.Button(rec, text="■ Stop Recording", command=self.stop_recording,
                                   state=tk.DISABLED)
        self.stop_btn.grid(row=2, column=1, sticky="we", pady=(4, 0), padx=5)
        ttk.Button(rec, text="Test disk speed", command=self._disk_test).grid(row=2, column=2, sticky="we", pady=(4, 0))
        self.marker_var = tk.StringVar()
        ttk.Entry(rec, textvariable=self.marker_var, width=36).grid(row=3, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(rec, text="Add marker (F9)", command=self.add_marker).grid(row=3, column=2, sticky="we", pady=(4, 0))

        # --- stats
        self.stats_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.stats_var, style="Big.TLabel", justify=tk.LEFT).pack(
            fill=tk.X, pady=(5, 0))

        # --- preview + log
        self.video_label = ttk.Label(self.root, background="black")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, background="white").pack(
            side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))

    def _update_fps_choices(self):
        fps_list = RESOLUTIONS[self.res_var.get()][1]
        self.fps_menu.config(values=[str(f) for f in fps_list])
        if self.fps_var.get() not in [str(f) for f in fps_list]:
            self.fps_var.set(str(fps_list[0]))

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.base_dir_var.get() or "/")
        if d:
            self.base_dir_var.set(d)

    def _set_status(self, msg):
        self.status_var.set(msg)
        print(f"[INFO] {msg}")

    # ------------------------------------------------------------- camera ops
    def open_camera(self):
        if self.camera_opened:
            return
        if self.mkv_var.get() and not check_ffmpeg_available():
            messagebox.showerror("FFmpeg not found",
                                 "FFmpeg is not on PATH. Install it, or untick the live .mkv option "
                                 "and record SVO2 only.")
            return
        res_name, _ = RESOLUTIONS[self.res_var.get()]
        fps = int(self.fps_var.get())
        depth_choice = self.depth_var.get()
        depth_enabled = not depth_choice.startswith("NONE")
        tracking = self.track_var.get() and depth_enabled

        self._set_status(f"Opening ZED {res_name}@{fps} depth={depth_choice} … (the FIRST use of a "
                         f"NEURAL model on a new device optimizes it and can take 5-20 min; run "
                         f"zed_doctor.py --optimize beforehand)")
        self.root.update_idletasks()
        # Fit the side-by-side preview on screen (controls take ~620 px of height).
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        rw, rh = RES_SIZES.get(res_name, (1920, 1080))
        CONFIG["preview_max_width"] = int(max(480, min(1400, sw - 40, (sh - 700) * 2 * rw / rh)))
        if display_is_remote():
            CONFIG["preview_max_width"] = min(CONFIG["preview_max_width"], CONFIG["remote_preview_max_width"])

        self.engine = RemoteEngine(self.ui_events)
        res = self.engine.call("open", res_name, fps, depth_choice, tracking, timeout=3600,
                               tick=self.root.update)
        ok, msg = res[0], res[1]
        if not ok:
            self.engine.shutdown()
            self.engine = None
            messagebox.showerror("ZED open error", f"Cannot open ZED camera:\n{msg}")
            self._set_status("Camera open failed")
            return
        _, _, self.open_params, tracking_ok, warnings, self.cam_info = res
        for w in warnings:
            messagebox.showwarning("Camera", w)
        op = self.open_params
        self.engine.width, self.engine.height, self.engine.fps = op["width"], op["height"], op["fps"]
        self.engine.depth_enabled = op["depth_enabled"]
        self.camera_opened = True
        for w in (self.res_menu, self.fps_menu, self.depth_menu, self.track_chk):
            w.config(state=tk.DISABLED)
        self.open_cam_btn.config(state=tk.DISABLED)
        self.close_cam_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self._set_status(f"Camera open: serial {self.cam_info['serial_number']} "
                         f"{op['width']}x{op['height']}@{op['fps']} (capture process {self.engine.proc.pid})")

    def close_camera(self):
        if not self.camera_opened:
            return
        if self.session is not None:
            self.stop_recording(block=True)
        self._wait_finalize()
        self._set_status("Closing camera…")
        self.engine.shutdown(tick=self.root.update)
        self.engine = None
        self.camera_opened = False
        for w in (self.res_menu, self.fps_menu, self.depth_menu):
            w.config(state="readonly")
        self.track_chk.config(state=tk.NORMAL)
        self.open_cam_btn.config(state=tk.NORMAL)
        self.close_cam_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(image="")
        self._set_status("Camera closed")

    # ------------------------------------------------------------ exposure ops
    def _engine_cmd(self, *args):
        if not self.engine:
            return
        ok, msg = self.engine.submit(*args)
        if not ok:
            messagebox.showwarning("Camera setting", str(msg))
        return ok, msg

    def _apply_auto_exposure(self):
        if self.engine:
            self._engine_cmd("set_setting", "AEC_AGC", 1 if self.auto_exp_var.get() else 0)
            if not self.auto_exp_var.get():
                self._apply_manual_exposure()

    def _apply_auto_wb(self):
        if self.engine:
            self._engine_cmd("set_setting", "WHITEBALANCE_AUTO", 1 if self.auto_wb_var.get() else 0)

    def _apply_manual_exposure(self):
        if not self.engine:
            return
        try:
            exp = int(float(self.exp_var.get()))
            gain = int(float(self.gain_var.get()))
        except ValueError:
            messagebox.showerror("Exposure", "Exposure and gain must be numbers 0–100.")
            return
        self.auto_exp_var.set(False)
        self._engine_cmd("set_setting", "EXPOSURE", max(0, min(100, exp)))
        self._engine_cmd("set_setting", "GAIN", max(0, min(100, gain)))

    def _lock_current(self):
        if not self.engine:
            return
        res = self._engine_cmd("lock_current")
        if res and res[0]:
            self.auto_exp_var.set(False)
            self.auto_wb_var.set(False)
            self._set_status(res[1])

    # ------------------------------------------------------------- speed plan
    def _plan(self):
        if not self.cam_info or not self.engine:
            return None
        try:
            h = float(self.h_var.get())
            ph = float(self.plant_var.get())
            blur = float(self.blur_var.get())
            views = int(float(self.views_var.get()))
        except ValueError:
            return None
        lc = self.cam_info["calibration_rectified"]["left_cam"]
        w, hgt = self.cam_info["resolution"]
        vertical = self.axis_var.get().startswith("Image vertical")
        f = lc["fy"] if vertical else lc["fx"]
        n = hgt if vertical else w
        exp_ms = self.engine.cam_settings.get("exposure_ms_approx")
        if not exp_ms:
            return None
        return motion_plan(h, ph, f, n, self.engine.fps, exp_ms, blur, views)

    def _use_recommended(self):
        p = self._plan()
        if p:
            self.target_speed_var.set(f"{p['v_recommended_m_s']:.2f}")

    # --------------------------------------------------------------- recording
    def _collect_field_info(self) -> dict:
        info = {k: v.get().strip() for k, v in self.field_vars.items()}
        for k, var in (("camera_height_m", self.h_var), ("plant_height_m", self.plant_var),
                       ("target_speed_m_s", self.target_speed_var)):
            info[k] = safe(lambda: float(var.get()), var.get())
        info["travel_axis"] = self.axis_var.get()
        return info

    def start_recording(self):
        if not self.camera_opened or self.session is not None or self.finalizing:
            return
        base = self.base_dir_var.get().strip()
        problem = storage_problem(base)
        if problem:
            messagebox.showerror("Save folder", problem)
            return
        try:
            os.makedirs(base, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save folder", f"Cannot create {base}:\n{e}")
            return
        free = get_free_space_bytes(base)
        if 0 <= free < CONFIG["min_free_space_gb"] * 1024 ** 3:
            messagebox.showerror("Low disk space",
                                 f"Only {free / 1024 ** 3:.1f} GB free on the target drive.\n"
                                 f"At least {CONFIG['min_free_space_gb']} GB is required.")
            return
        if not self.svo_var.get() and not self.mkv_var.get():
            messagebox.showerror("Nothing to record", "Enable SVO2 and/or the live .mkv output.")
            return
        if not self.svo_var.get() and not messagebox.askyesno(
                "No master record", "SVO2 is disabled. Frames dropped by the .mkv encoder will be "
                                    "LOST and depth cannot be recomputed later.\nContinue anyway?"):
            return
        if self.mkv_var.get() and not check_ffmpeg_available():
            messagebox.showerror("FFmpeg not found", "FFmpeg is required for the live .mkv output.")
            return
        if self.svo_var.get() and self.svo_mode_var.get().split()[0] not in ("LOSSLESS",):
            if not messagebox.askyesno("SVO compression",
                                       "H264/H265 'lossless' SVO modes are near-lossless (GPU, chroma "
                                       "subsampled), not bit-exact. Continue?"):
                return

        opts = {
            "record_svo": self.svo_var.get(),
            "svo_compression": SVO_COMPRESSION[self.svo_mode_var.get()],
            "record_mkv": self.mkv_var.get(),
            "depth_units_per_mm": CONFIG["depth_units_per_mm"],
            "confidence_scale": CONFIG["confidence_scale"],
            "ram_buffer_gb": CONFIG["ram_buffer_gb"],
        }
        field_info = self._collect_field_info()
        ok, msg, sdir = self.engine.call("start_recording", base, field_info, opts, self._plan(),
                                         self.disk_bench_mb_s, timeout=120, tick=self.root.update)
        if not ok:
            messagebox.showerror("Recording", f"Could not start recording:\n{msg}")
            return
        session = RemoteSession(self.engine, sdir)
        self.session = session
        self.last_size = (time.monotonic(), 0)
        self.write_rate_mb_s = None
        self.next_beep = None
        keep_system_awake(True)
        play_wav_async(CONFIG["wave_start_path"])
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status(f"RECORDING → {session.dir}")

    def stop_recording(self, block: bool = False, reason: str = "operator"):
        if self.session is None or not self.engine:
            return
        self.session = None
        self.stop_btn.config(state=tk.DISABLED)
        self.finalizing = True
        self._set_status("Finalizing – writing remaining frames, do not unplug the drive…")
        ok, msg = self.engine.call("stop_recording", reason, timeout=60, tick=self.root.update)
        if not ok:
            messagebox.showerror("Stop recording", str(msg))
        if block:
            self._wait_finalize()

    def _wait_finalize(self):
        """Keep the window responsive until the capture process reports 'finalized'."""
        if self.finalizing:
            self._set_status("Waiting for files to be finalized…")
        while self.finalizing and self.engine is not None and self.engine.alive():
            self.engine.pump()
            self._drain_ui_events()
            self.root.update()
            time.sleep(0.05)
        self._drain_ui_events()

    def _on_finalized(self, summary: dict):
        stats = summary["stats"]
        self.finalizing = False
        keep_system_awake(False)
        play_wav_async(CONFIG["wave_stop_path"])
        if self.camera_opened:
            self.start_btn.config(state=tk.NORMAL)

        v = summary["verification"]
        lines = [f"Saved: {summary['dir']}",
                 f"Frames: {stats['frames_grabbed']}  duration {stats['duration_s']:.1f} s  "
                 f"effective {stats['effective_fps'] or 0:.2f} fps",
                 f"Missing camera frames (timestamp gaps): {stats['estimated_missing_camera_frames']}"]
        if summary["has_svo"]:
            sv = v.get("svo", {})
            lines.append(f"SVO2 frames in file: {sv.get('frames_in_file')} of {stats['frames_grabbed']} logged"
                         + (f"  ({sv['missing_frames']} MISSING)" if sv.get("missing_frames") else ""))
        if summary["has_writers"]:
            lines.append(f".mkv frames: {stats['mkv_frames_enqueued']}  dropped (backlog): "
                         f"{stats['mkv_frames_dropped_backlog']}")
            for name, sv in v.get("streams", {}).items():
                lines.append(f"  {name}: {sv['frames_in_file']} in file / {sv['frames_sent']} sent"
                             f"{'  ERROR: ' + str(sv['error']) if sv['failed'] else ''}")
        if stats["distance_travelled_m"]:
            lines.append(f"Distance {stats['distance_travelled_m']:.2f} m, mean speed "
                         f"{stats['mean_speed_m_s'] or 0:.3f} m/s")
        msg = "\n".join(lines)
        self._set_status(("✔ " if v.get("all_ok") else "⚠ ") + f"Saved {summary['dir']}")
        if v.get("all_ok") and stats["estimated_missing_camera_frames"] == 0:
            messagebox.showinfo("Recording saved", msg)
        else:
            messagebox.showwarning("Recording saved with warnings", msg + "\n\nSee events.log.")

    def add_marker(self):
        if self.session is None or not self.engine:
            return
        text = self.marker_var.get().strip() or "marker"
        self.session.add_marker(text)
        self.marker_var.set("")
        beep_async(1800, 60)
        self._set_status(f"Marker added: {text}")

    def _disk_test(self):
        base = self.base_dir_var.get().strip()
        if self.session is not None:
            messagebox.showinfo("Disk test", "Not available while recording.")
            return
        self._set_status("Testing disk write speed (1 GB)…")
        self.root.update_idletasks()
        try:
            mb_s = disk_write_benchmark(base, 1024)
        except Exception as e:
            messagebox.showerror("Disk test", f"Failed: {e}")
            return
        self.disk_bench_mb_s = mb_s
        need = self._estimate_mb_s()
        msg = f"Sustained write: {mb_s:.0f} MB/s"
        if need:
            msg += f"\nEstimated need for current settings: ~{need:.0f} MB/s"
            if mb_s < need * 1.5:
                msg += "\n\n⚠ Drive is too slow / too close to the limit. Use a local NVMe SSD, " \
                       "lower the resolution, or record SVO2 only."
        messagebox.showinfo("Disk test", msg)
        self._set_status(msg.splitlines()[0])

    def _estimate_mb_s(self):
        if not self.engine:
            return None
        w, h, fps = self.engine.width, self.engine.height, self.engine.fps
        mb = 0.0
        if self.svo_var.get():
            mb += 2 * w * h * 3 * 0.5 * fps / 1e6     # stereo pair, ~50 % lossless ratio
        if self.mkv_var.get():
            mb += w * h * 3 * 0.55 * fps / 1e6        # RGB FFV1
            if self.engine.depth_enabled:
                mb += 2 * w * h * 2 * 0.5 * fps / 1e6  # depth + confidence
        return mb

    # ------------------------------------------------------------- GUI polling
    def _drain_ui_events(self):
        while True:
            try:
                kind, payload = self.ui_events.get_nowait()
            except queue.Empty:
                return
            if kind == "finalized":
                self._on_finalized(payload)
            elif kind == "camera_error":
                self._set_status(f"⚠ CAMERA: {payload}")
                if time.monotonic() - self.last_error_beep > 2:
                    self.last_error_beep = time.monotonic()
                    beep_async(400, 400)
            elif kind == "info":
                self._set_status(payload)

    def _poll(self):
        try:
            if self.engine is not None:
                self.engine.pump()
                if self.camera_opened and not self.engine.alive():
                    self._capture_process_died()
            self._drain_ui_events()
            if self.camera_opened and self.engine:
                self._refresh_preview_and_stats()
        except Exception as e:
            print(f"[WARN] GUI update error: {e}")
        self.root.after(CONFIG["gui_poll_ms"], self._poll)

    def _refresh_preview_and_stats(self):
        with self.engine.latest_lock:
            latest = dict(self.engine.latest)
        fi = latest.get("fi") or {}
        rgb, depth = latest.get("rgb"), latest.get("depth")
        rec = self.session is not None

        if rgb is not None and latest.get("_shown") is not True:
            bgr = rgb[:, :, :3]
            if depth is not None:
                finite = np.isfinite(depth)
                vis = np.zeros(depth.shape, np.uint8)
                if finite.any():
                    lo, hi = np.percentile(depth[finite], [2, 98])
                    hi = max(hi, lo + 1)
                    vis[finite] = (255 - np.clip((depth[finite] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                dcol = cv2.applyColorMap(vis, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
                dcol[~finite] = 0
                combined = np.hstack((bgr, dcol))
            else:
                combined = np.ascontiguousarray(bgr)
            # crosshair = centre region used for the height read-out
            hh, ww = combined.shape[:2]
            cw = bgr.shape[1]
            cv2.rectangle(combined, (int(cw * 0.4), int(hh * 0.4)), (int(cw * 0.6), int(hh * 0.6)),
                          (255, 255, 255), 1)
            txt = f"{'REC' if rec else 'PREVIEW'}  frame {fi.get('grab_idx')}"
            cv2.putText(combined, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if rec else (255, 255, 255), 2, cv2.LINE_AA)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=img)
            self.video_label.image = img
            with self.engine.latest_lock:
                self.engine.latest["_shown"] = True

        # --- speed plan + live speed
        plan = self._plan()
        if plan:
            v = plan["v_recommended_m_s"]
            self.plan_var.set(
                f"GSD {plan['gsd_ground_mm_per_px']:.2f} mm/px at ground, "
                f"{plan['gsd_near_mm_per_px']:.2f} at plant tops | footprint along travel "
                f"{plan['footprint_along_travel_m']:.2f} m | exposure ≈ {plan['exposure_ms']:.1f} ms\n"
                f"Max speed for ≤{self.blur_var.get()} px blur: {plan['v_max_blur_m_s']:.2f} m/s | "
                f"for ≥{self.views_var.get()} views/point: {plan['v_max_views_m_s']:.2f} m/s\n"
                f"RECOMMENDED ≤ {v:.2f} m/s  ({v * 3.6:.2f} km/h, {v * 3.281:.2f} ft/s, "
                f"{v * 2.237:.2f} mph) → {plan['advance_per_frame_mm_at_rec']:.0f} mm per frame, "
                f"{plan['views_per_point_at_rec']:.0f} views per ground point")
        speed = fi.get("speed_m_s")
        try:
            target = float(self.target_speed_var.get())
        except ValueError:
            target = None
        vmax = plan["v_recommended_m_s"] if plan else None
        if speed is not None:
            colour = "#2E7D32"
            if vmax and speed > vmax * 1.05:
                colour = "#C62828"
            elif target and abs(speed - target) > 0.25 * target:
                colour = "#EF6C00"
            blur_txt = ""
            if plan:
                # live motion blur at the plant tops: distance moved during one exposure / mm per px
                blur_px = speed * plan["exposure_ms"] / plan["gsd_near_mm_per_px"]
                blur_txt = f"   BLUR ≈ {blur_px:.1f} px"
                if blur_px > 2.0:
                    colour = "#C62828"
                    blur_txt += "  ⚠ too blurry: slow down / shorter exposure / more light"
            self.speed_var.set(f"Speed: {speed:.2f} m/s (target {target or 0:.2f}, max {vmax or 0:.2f})"
                               f"   distance {fi.get('distance_m', 0) or 0:.2f} m" + blur_txt)
            self.speed_label.configure(foreground=colour)
        else:
            self.speed_var.set("Speed: – (tracking off / initialising)")
            self.speed_label.configure(foreground="black")

        # --- pace beep
        now = time.monotonic()
        if rec and self.beep_var.get() and target:
            try:
                spacing = float(self.beep_spacing_var.get())
            except ValueError:
                spacing = 0
            if spacing > 0:
                if self.next_beep is None:
                    self.next_beep = now + spacing / target
                elif now >= self.next_beep:
                    beep_async()
                    self.next_beep += spacing / target
        else:
            self.next_beep = None

        # --- disk monitor
        s = self.session
        lines = []
        cs = self.engine.cam_settings
        tilt = fi.get("tilt_from_nadir_deg")
        cd = fi.get("center_depth_mm")
        lines.append(
            f"Camera fps {latest.get('current_fps') or 0:5.1f}  grab {latest.get('grab_ms') or 0:5.1f} ms | "
            f"exp {cs.get('exposure_pct')}% (≈{cs.get('exposure_ms_approx') or 0:.1f} ms) gain {cs.get('gain')} "
            f"WB {cs.get('wb_temp_k')}K {'AUTO' if cs.get('aec_agc') else 'MANUAL'}")
        lines.append(f"Tilt from nadir {tilt:.1f}°" if tilt is not None else "Tilt –")
        if cd:
            lines[-1] += f" | centre depth {cd / 1000:.3f} m"
            if cd < 350:
                lines[-1] += "  ⚠ near ZED 2i minimum depth (0.3 m)"
        base = self.base_dir_var.get().strip()
        if now - self.last_disk_check > 2.0:
            self.last_disk_check = now
            self._free = get_free_space_bytes(base)
            if s is not None:
                size = dir_size_bytes(s.dir)
                t0, s0 = self.last_size
                if now > t0 and s0:
                    self.write_rate_mb_s = (size - s0) / (now - t0) / 1e6
                self.last_size = (now, size)
                self._sess_size = size
                if 0 <= self._free < CONFIG["autostop_free_space_gb"] * 1024 ** 3:
                    s.log_event("ERROR", "free space below auto-stop threshold")
                    self.stop_recording(reason="disk almost full")
                    messagebox.showerror("Disk almost full", "Recording was stopped automatically "
                                                             "and the files were finalized.")
                    return
        free = getattr(self, "_free", -1)
        if free >= 0:
            line = f"Disk free {free / 1024 ** 3:.1f} GB"
            rate = self.write_rate_mb_s or self._estimate_mb_s()
            if rate:
                line += f" | writing ~{rate:.0f} MB/s → ~{free / 1e6 / rate / 60:.0f} min left"
            lines.append(line)
        if s is not None:
            st = s.stats()
            bl, bmax = s.backlog()
            lines.append(f"REC {st['duration_s']:6.1f} s  frames {st['frames_grabbed']}  "
                         f"size {getattr(self, '_sess_size', 0) / 1e9:.2f} GB  "
                         f"markers → F9")
            lines.append(f"Missing camera frames {st['estimated_missing_camera_frames']}  "
                         f"SVO status ok/not {st['svo_frames_ok']}/{st['svo_frames_failed']}  "
                         f".mkv backlog {bl}/{bmax} dropped {st['mkv_frames_dropped_backlog']}")
            if s.writers_failed:
                lines.append("⚠ an .mkv encoder FAILED – see events.log (SVO2 still recording)")
        elif self.finalizing:
            lines.append("Finalizing files…")
        self.stats_var.set("\n".join(lines))

    def _capture_process_died(self):
        code = self.engine.proc.exitcode
        self.engine = None
        self.session = None
        self.camera_opened = False
        self.finalizing = False
        for w in (self.res_menu, self.fps_menu, self.depth_menu):
            w.config(state="readonly")
        self.track_chk.config(state=tk.NORMAL)
        self.open_cam_btn.config(state=tk.NORMAL)
        for b in (self.close_cam_btn, self.start_btn, self.stop_btn):
            b.config(state=tk.DISABLED)
        keep_system_awake(False)
        self._set_status(f"⚠ capture process stopped (exit code {code})")
        messagebox.showerror("Camera", f"The capture process stopped unexpectedly (exit code {code}).\n"
                                       "Recorded data up to that point is on disk; run verify_session.py.\n"
                                       "Re-open the camera to continue.")

    # ----------------------------------------------------------------- closing
    def quit_now(self):
        """Ctrl+C in the terminal: stop + save any recording without asking, then exit."""
        print("\n[INFO] Ctrl+C - stopping, saving and closing…")
        self._quitting_without_prompt = True
        self.on_closing()

    def on_closing(self):
        try:
            if self.session is not None:
                if not getattr(self, "_quitting_without_prompt", False) and not messagebox.askyesno(
                        "Quit", "Recording in progress. Stop, save and quit?"):
                    return
                self.stop_recording(block=True)
            self._wait_finalize()
            if self.camera_opened:
                self.close_camera()
        finally:
            keep_system_awake(False)
            self.root.destroy()


def display_is_remote() -> bool:
    """True when the GUI is drawn on another machine (SSH X11 forwarding)."""
    disp = os.environ.get("DISPLAY", "")
    host = disp.split(":")[0]
    return bool(os.environ.get("SSH_CONNECTION")) and host not in ("", "unix")


NO_DISPLAY_HELP = """
No display available for the GUI (you are probably connected over SSH without X11 forwarding).

  * GUI on your laptop: connect with X11 forwarding, e.g. MobaXterm (SSH session, X11 on by
    default) or VS Code Remote-SSH with "ForwardX11 yes" + VcXsrv running on Windows.
    Then `echo $DISPLAY` must print something like localhost:10.0.  See JETSON.md.
  * Or record without a GUI:
      python3 record_headless.py --plot A1 --operator MM --height 0.855 --exposure 4 --gain 60
"""


def main():
    if not IS_WINDOWS and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print(NO_DISPLAY_HELP)
        sys.exit(2)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"Cannot open a window: {e}\n{NO_DISPLAY_HELP}")
        sys.exit(2)
    if display_is_remote():
        CONFIG["preview_interval_s"] = max(CONFIG["preview_interval_s"], CONFIG["remote_preview_interval_s"])
        print(f"[INFO] remote display {os.environ.get('DISPLAY')}: preview reduced to "
              f"~{1 / CONFIG['preview_interval_s']:.1f} fps, max {CONFIG['remote_preview_max_width']} px wide "
              f"(recording is not affected)")
    app = ZEDTkApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    # Ctrl+C in the terminal: Python runs this handler at the next GUI poll (<= 50 ms)
    signal.signal(signal.SIGINT, lambda *_: root.after(0, app.quit_now))
    root.mainloop()


if __name__ == "__main__":
    main()
