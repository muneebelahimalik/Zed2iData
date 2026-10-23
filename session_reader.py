#!/usr/bin/env python3
"""
Standalone reader for WeedScan ZED 2i recording sessions (see DATA_SPEC.md).

Needs only numpy + ffmpeg on PATH (no ZED SDK, no GUI). Typical use:

    from session_reader import Session
    s = Session("M:/Research/Data/.../Session_20261008_101500_plotA")
    for f in s.iter_frames(step=5):            # every 5th frame stored in the .mkv files
        bgr = f.bgr                             # (H, W, 3) uint8, rectified left image
        z = f.depth_mm                          # (H, W) float32 mm, NaN where invalid
        good = f.valid_mask(max_conf=50)        # finite depth AND confidence <= 50
        X, Y, Z = s.backproject(u, v, z[v, u])  # camera-frame mm
        Pw = s.camera_to_world(np.stack([X, Y, Z], -1), f.row)  # world-frame mm (if pose OK)

Do NOT read Depth.mkv / Confidence.mkv with cv2.VideoCapture: it converts to 8-bit BGR
and destroys the 16-bit values. This module decodes them through ffmpeg as gray16le.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

DEPTH_CODE_NAN = 0
DEPTH_CODE_TOO_CLOSE = 65534
DEPTH_CODE_TOO_FAR = 65535
CONF_CODE_NAN = 65535

_NUMERIC_INT = {"rec_frame_idx", "grab_idx", "image_timestamp_ns", "system_time_ns",
                "est_missing_before", "camera_dropped_total", "svo_frame_ok", "mkv_written",
                "mkv_frame_idx",
                # legacy (pre-v2) frames.csv columns
                "frame_idx", "timestamp_ns", "width", "height"}


def _parse_row(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if v is None or v == "":
            out[k] = None
        elif k in _NUMERIC_INT:
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def decode_depth(depth_u16: np.ndarray, units_per_mm: float = 10) -> np.ndarray:
    """uint16 depth -> float32 mm. 0 -> NaN, 65534 -> -inf (too close), 65535 -> +inf (too far)."""
    d = depth_u16.astype(np.float32) / np.float32(units_per_mm)
    d[depth_u16 == DEPTH_CODE_NAN] = np.nan
    d[depth_u16 == DEPTH_CODE_TOO_CLOSE] = -np.inf
    d[depth_u16 == DEPTH_CODE_TOO_FAR] = np.inf
    return d


def decode_confidence(conf_u16: np.ndarray, scale: float = 100) -> np.ndarray:
    """uint16 -> float32 ZED confidence 0..100 (HIGHER = LESS reliable), NaN where undefined."""
    c = conf_u16.astype(np.float32) / np.float32(scale)
    c[conf_u16 == CONF_CODE_NAN] = np.nan
    return c


def read_mkv(path: str, width: int, height: int, kind: str, start: int = 0):
    """Yield (mkv_frame_idx, array) from an FFV1 .mkv. kind: 'bgr' | 'gray16'."""
    if kind == "bgr":
        pix, shape, dtype = "bgr24", (height, width, 3), np.uint8
    else:
        pix, shape, dtype = "gray16le", (height, width), np.dtype("<u2")
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if start:
        # drop the first `start` frames without re-timing (option name differs by ffmpeg version)
        sync = ["-fps_mode", "passthrough"] if _ffmpeg_major() >= 5 else ["-vsync", "0"]
        cmd += ["-vf", f"select=gte(n\\,{start})", *sync]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i = start
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield i, np.frombuffer(buf, dtype).reshape(shape)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


_FFMPEG_MAJOR = []


def _ffmpeg_major() -> int:
    if not _FFMPEG_MAJOR:
        try:
            out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
            ver = out.split()[2]  # "ffmpeg version 4.2.7-0ubuntu0.1 ..." / "n6.1" / "7.0.2"
            _FFMPEG_MAJOR.append(int("".join(c for c in ver.split(".")[0] if c.isdigit()) or 0))
        except Exception:
            _FFMPEG_MAJOR.append(0)
    return _FFMPEG_MAJOR[0]


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Frame:
    row: dict                      # parsed frames.csv row
    bgr: np.ndarray | None         # (H, W, 3) uint8
    depth_mm: np.ndarray | None    # (H, W) float32, NaN / +-inf for invalid
    confidence: np.ndarray | None  # (H, W) float32 0..100, higher = less reliable

    def valid_mask(self, max_conf: float | None = 50) -> np.ndarray:
        m = np.isfinite(self.depth_mm)
        if max_conf is not None and self.confidence is not None:
            m &= self.confidence <= max_conf
        return m


def _parse_legacy_calibration(path: str) -> dict:
    """calibration_params.txt written by the pre-v2 recorder (rectified values, 2 decimals)."""
    import re
    txt = open(path, encoding="utf-8").read()
    left = txt.split("Right Camera")[0]
    num = lambda key, t: float(re.search(rf"{key}:\s*(-?[\d.]+)", t).group(1))
    return {"fx": num("fx", left), "fy": num("fy", left), "cx": num("cx", left), "cy": num("cy", left)}


class Session:
    """A recording session folder. Also reads LEGACY sessions made by the pre-v2 recorder
    (RGB_video.mkv / Depth_video.mkv / session_meta.txt); see DATA_SPEC.md section 13."""

    def __init__(self, path: str):
        self.path = path
        self.legacy = (not os.path.exists(os.path.join(path, "session_metadata.json"))
                       and os.path.exists(os.path.join(path, "session_meta.txt")))
        if self.legacy:
            self._init_legacy(path)
            return
        self.rgb_file, self.depth_file, self.conf_file = "RGB_left.mkv", "Depth.mkv", "Confidence.mkv"
        with open(os.path.join(path, "session_metadata.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(path, "calibration.json"), encoding="utf-8") as f:
            self.calib = json.load(f)
        with open(os.path.join(path, "frames.csv"), newline="", encoding="utf-8") as f:
            self.frames = [_parse_row(r) for r in csv.DictReader(f)]
        self.markers = []
        mp = os.path.join(path, "markers.csv")
        if os.path.exists(mp):
            with open(mp, newline="", encoding="utf-8") as f:
                self.markers = list(csv.DictReader(f))
        enc = self.meta.get("encoding", {})
        self.units_per_mm = enc.get("depth_units_per_mm", 10)
        self.conf_scale = enc.get("confidence_scale", 100)
        self.width, self.height = self.calib["resolution"]
        self.fps = self.calib["fps"]
        lc = self.calib["calibration_rectified"]["left_cam"]
        self.fx, self.fy, self.cx, self.cy = lc["fx"], lc["fy"], lc["cx"], lc["cy"]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])
        # rows that have a frame in the .mkv files, indexed by mkv_frame_idx
        self.mkv_rows = {r["mkv_frame_idx"]: r for r in self.frames if r.get("mkv_written") == 1}

    def _init_legacy(self, path: str) -> None:
        self.rgb_file, self.depth_file, self.conf_file = "RGB_video.mkv", "Depth_video.mkv", None
        meta = {}
        with open(os.path.join(path, "session_meta.txt"), encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        self.meta = {"legacy_session_meta": meta, "status": "LEGACY"}
        self.width, self.height = (int(x) for x in meta["resolution"].split("x"))
        self.fps = float(meta.get("fps", 15))
        lc = _parse_legacy_calibration(os.path.join(path, "calibration_params.txt"))
        self.calib = {"resolution": [self.width, self.height], "fps": self.fps,
                      "calibration_rectified": {"left_cam": lc}, "legacy": True}
        self.fx, self.fy, self.cx, self.cy = lc["fx"], lc["fy"], lc["cx"], lc["cy"]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])
        self.units_per_mm = 1          # legacy depth: whole millimetres (truncated), 0 = invalid
        self.conf_scale = 100
        self.frames = []
        with open(os.path.join(path, "frames.csv"), newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                row = _parse_row(r)
                # legacy rows are the frames written to the .mkv files, in order
                row.update({"rec_frame_idx": i, "mkv_written": 1, "mkv_frame_idx": i,
                            "image_timestamp_ns": row.get("timestamp_ns"), "grab_idx": row.get("frame_idx")})
                self.frames.append(row)
        self.markers = []
        self.mkv_rows = {r["mkv_frame_idx"]: r for r in self.frames}

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    def has(self, name: str) -> bool:
        return os.path.exists(self.file(name))

    def iter_frames(self, step: int = 1, start: int = 0, rgb=True, depth=True, confidence=True):
        """Yield Frame objects for .mkv frames start, start+step, ... (mkv_frame_idx)."""
        gens = {}
        if rgb and self.has(self.rgb_file):
            gens["rgb"] = read_mkv(self.file(self.rgb_file), self.width, self.height, "bgr", start)
        if depth and self.has(self.depth_file):
            gens["depth"] = read_mkv(self.file(self.depth_file), self.width, self.height, "gray16", start)
        if confidence and self.conf_file and self.has(self.conf_file):
            gens["conf"] = read_mkv(self.file(self.conf_file), self.width, self.height, "gray16", start)
        if not gens:
            return
        try:
            while True:
                got = {}
                for k, g in gens.items():
                    got[k] = next(g)
                idx = next(iter(got.values()))[0]
                if (idx - start) % step:
                    continue
                yield Frame(
                    row=self.mkv_rows.get(idx, {"mkv_frame_idx": idx}),
                    bgr=got["rgb"][1] if "rgb" in got else None,
                    depth_mm=decode_depth(got["depth"][1], self.units_per_mm) if "depth" in got else None,
                    confidence=decode_confidence(got["conf"][1], self.conf_scale) if "conf" in got else None,
                )
        except StopIteration:
            return
        finally:
            for g in gens.values():
                g.close()

    def backproject(self, u, v, z_mm):
        """Pixel (u, v) + depth Z (mm) -> camera-frame X, Y, Z in mm (x right, y down, z forward)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z_mm, dtype=np.float64)
        return (u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z

    def project(self, X, Y, Z):
        """Camera-frame mm -> pixel (u, v)."""
        return self.fx * np.asarray(X) / Z + self.cx, self.fy * np.asarray(Y) / Z + self.cy

    @staticmethod
    def pose(row: dict):
        """(R 3x3, t mm) of the left camera in the world frame, or None if tracking was not OK."""
        if row.get("pose_state") != "OK" or row.get("qw") is None:
            return None
        R = quat_to_rot(row["qx"], row["qy"], row["qz"], row["qw"])
        t = np.array([row["tx_mm"], row["ty_mm"], row["tz_mm"]], dtype=np.float64)
        return R, t

    def camera_to_world(self, pts_cam_mm: np.ndarray, row: dict):
        """(..., 3) camera-frame mm -> world-frame mm: P_w = R @ P_c + t. None if no pose."""
        p = self.pose(row)
        if p is None:
            return None
        R, t = p
        return pts_cam_mm @ R.T + t

    def world_to_camera(self, pts_world_mm: np.ndarray, row: dict):
        p = self.pose(row)
        if p is None:
            return None
        R, t = p
        return (pts_world_mm - t) @ R

    def row_by_timestamp(self, ts_ns: int, tol_ns: int = None):
        """frames.csv row nearest to an SVO2 timestamp (for joining extract_svo.py output).

        SVO2 playback timestamps can differ slightly from the live ones (precision), so the
        nearest row within 30 % of a frame period is returned (None if there is none).
        """
        if not hasattr(self, "_ts_index"):
            self._ts = np.array([r["image_timestamp_ns"] or 0 for r in self.frames], dtype=np.int64)
            self._ts_index = True
        if tol_ns is None:
            tol_ns = int(0.3 * 1e9 / self.fps)
        i = int(np.argmin(np.abs(self._ts - ts_ns)))
        return self.frames[i] if abs(int(self._ts[i]) - ts_ns) <= tol_ns else None


if __name__ == "__main__":
    import sys
    s = Session(sys.argv[1])
    print(f"{s.path}\n  {s.width}x{s.height}@{s.fps}  fx={s.fx:.2f} fy={s.fy:.2f} "
          f"cx={s.cx:.2f} cy={s.cy:.2f}\n  csv rows {len(s.frames)}, mkv frames {len(s.mkv_rows)}, "
          f"markers {len(s.markers)}, status {s.meta.get('status')}")
    for f in s.iter_frames(step=max(1, len(s.mkv_rows) // 3)):
        m = f.valid_mask()
        print(f"  mkv {f.row.get('mkv_frame_idx')}: valid depth {m.mean() * 100:.1f}%  "
              f"median Z {np.nanmedian(np.where(m, f.depth_mm, np.nan)):.1f} mm  "
              f"pose {f.row.get('pose_state')}")
