#!/usr/bin/env python3
"""
Extract frames, depth, confidence, poses (and optionally point clouds / FFV1 videos)
from a WeedScan recording.svo2 file. Depth is recomputed here, so you can use the
most accurate mode (NEURAL_PLUS) offline even if a lighter mode was used live.

Examples
--------
  # every frame, lossless PNGs, NEURAL_PLUS depth
  python extract_svo.py Session_x/recording.svo2 --out Session_x/extracted

  # one frame every 50 mm of travel (uses positional tracking), plus float depth + PLY
  python extract_svo.py rec.svo2 --out ex --every-mm 50 --float-depth --ply

  # motion-blurred recording: keep only the sharpest frame of every 15 (1 per second)
  python extract_svo.py rec.svo2 --out sharp --sharpest-per 15

  # rebuild the full RGB/Depth/Confidence .mkv set without any dropped frames
  python extract_svo.py rec.svo2 --out rebuilt --mkv --no-png

Outputs (in --out)
  left/000123.png           rectified left image (8-bit BGR, lossless PNG)
  right/000123.png          rectified right image (--right)
  depth/000123.png          16-bit depth, value/DEPTH_UNITS_PER_MM = mm, 0=NaN,
                            65534=too close, 65535=too far (same coding as Depth.mkv)
  depth_float/000123.npy    float32 depth in mm with NaN/+inf/-inf (--float-depth)
  confidence/000123.png     16-bit, value/100 = ZED confidence (higher = less reliable)
  pointcloud/000123.ply     XYZ (mm, camera frame) + RGB, valid points only (--ply)
  index.csv                 svo_frame, timestamp_ns, pose, file names
  calibration.json          rectified intrinsics used for these images
  extraction_params.json    exactly how this extraction was produced
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Zed2i_video_datacollection import (  # noqa: E402
    CONFIG, camera_info_dict, err_is, encode_confidence_u16, encode_depth_u16, to_jsonable, enum_name,
    safe, popen_hidden_kwargs)


def write_ply(path, xyz, bgr):
    n = xyz.shape[0]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = bgr[:, 2], bgr[:, 1], bgr[:, 0]
        f.write(rec.tobytes())


def score_sharpness(svo_path, start=0, end=-1):
    """{svo_frame: (sharpness, timestamp_ns)} for every frame. Fast: no depth, half resolution.

    Sharpness = variance of the Laplacian of the grey image: blurred frames have weak edges
    and score low. Only compare scores within one recording (it depends on the scene).
    """
    ip = sl.InitParameters()
    ip.set_from_svo_file(os.path.abspath(svo_path))
    ip.svo_real_time_mode = False
    ip.depth_mode = sl.DEPTH_MODE.NONE
    safe(lambda: setattr(ip, "camera_image_flip", sl.FLIP_MODE.OFF))
    cam = sl.Camera()
    if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Cannot open {svo_path} for sharpness scoring")
    res = cam.get_camera_information().camera_configuration.resolution
    half = sl.Resolution(res.width // 2, res.height // 2)
    n = cam.get_svo_number_of_frames()
    rt = sl.RuntimeParameters()
    rt.enable_depth = False
    mat = sl.Mat()
    if start > 0:
        cam.set_svo_position(start)
    out = {}
    t0 = time.time()
    while True:
        err = cam.grab(rt)
        if err != sl.ERROR_CODE.SUCCESS:
            if err_is(err, "END_OF_SVOFILE_REACHED") or len(out) >= n:
                break
            continue
        fno = cam.get_svo_position()
        if end >= 0 and fno > end:
            break
        cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, half)
        grey = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2GRAY)
        out[fno] = (float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                    cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        if len(out) % 500 == 0:
            print(f"  scored {len(out)}/{n} frames ({len(out) / (time.time() - t0):.0f}/s)")
        if len(out) >= n:
            break
    cam.close()
    return out


def ffv1_proc(path, w, h, fps, in_fmt, out_fmt):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
           "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0",
           "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "24", "-slicecrc", "1",
           "-pix_fmt", out_fmt, path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, **popen_hidden_kwargs())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-mode", default="NEURAL_PLUS",
                    help="NEURAL_PLUS (default, most accurate), NEURAL, NEURAL_LIGHT, ULTRA, NONE")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--every-mm", type=float, default=0,
                    help="keep one frame per this much camera travel (needs tracking)")
    ap.add_argument("--sharpest-per", type=int, default=0, metavar="N",
                    help="score every frame's sharpness and keep only the sharpest frame in each "
                         "block of N consecutive frames (e.g. 15 = sharpest frame per second)")
    ap.add_argument("--min-sharpness", type=float, default=0,
                    help="with --sharpest-per: skip a block whose best frame is below this score")
    ap.add_argument("--start", type=int, default=0, help="first SVO frame")
    ap.add_argument("--end", type=int, default=-1, help="last SVO frame (inclusive)")
    ap.add_argument("--right", action="store_true", help="also save the rectified right image")
    ap.add_argument("--unrectified", action="store_true",
                    help="also save raw (unrectified) left/right images")
    ap.add_argument("--float-depth", action="store_true", help="also save float32 .npy depth")
    ap.add_argument("--ply", action="store_true", help="also save coloured point clouds")
    ap.add_argument("--ply-max-conf", type=float, default=50,
                    help="only points with confidence <= this go into the PLY")
    ap.add_argument("--mkv", action="store_true", help="write RGB_left/Depth/Confidence .mkv (all kept frames)")
    ap.add_argument("--no-png", action="store_true", help="skip PNG output")
    ap.add_argument("--confidence-threshold", type=int, default=CONFIG["confidence_threshold"])
    ap.add_argument("--depth-stabilization", type=int, default=CONFIG["depth_stabilization"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    depth_on = args.depth_mode.upper() != "NONE"

    ip = sl.InitParameters()
    ip.set_from_svo_file(os.path.abspath(args.svo))
    ip.svo_real_time_mode = False
    ip.coordinate_units = sl.UNIT.MILLIMETER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    ip.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode.upper()) if depth_on else sl.DEPTH_MODE.NONE
    safe(lambda: setattr(ip, "camera_image_flip", sl.FLIP_MODE.OFF))
    safe(lambda: setattr(ip, "depth_stabilization", args.depth_stabilization))

    zed = sl.Camera()
    err = zed.open(ip)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Cannot open {args.svo}: {err}")

    tracking = False
    if depth_on and not args.sharpest_per:  # sharpest-per seeks frames; poses come from frames.csv
        tp = sl.PositionalTrackingParameters()
        safe(lambda: setattr(tp, "enable_imu_fusion", True))
        safe(lambda: setattr(tp, "set_gravity_as_origin", True))
        tracking = zed.enable_positional_tracking(tp) == sl.ERROR_CODE.SUCCESS
    if args.every_mm > 0 and not tracking:
        sys.exit("--every-mm needs positional tracking (and a depth mode other than NONE)")

    rt = sl.RuntimeParameters()
    rt.enable_depth = depth_on
    safe(lambda: setattr(rt, "confidence_threshold", args.confidence_threshold))
    safe(lambda: setattr(rt, "texture_confidence_threshold", 100))
    safe(lambda: setattr(rt, "enable_fill_mode", False))

    cam = camera_info_dict(zed)
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(cam), f, indent=2)
    w, h = cam["resolution"]
    fps = cam["fps"]
    n_total = safe(lambda: zed.get_svo_number_of_frames(), -1)
    upm = CONFIG["depth_units_per_mm"]
    cs = CONFIG["confidence_scale"]

    params = {"svo": os.path.abspath(args.svo), "args": vars(args), "svo_frames": n_total,
              "zed_sdk": safe(lambda: str(sl.Camera.get_sdk_version())),
              "depth_units_per_mm": upm, "confidence_scale": cs,
              "depth_codes": {"nan": 0, "too_close": 65534, "too_far": 65535},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(args.out, "extraction_params.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(params), f, indent=2)

    dirs = []
    if not args.no_png:
        dirs += ["left"] + (["right"] if args.right else []) + \
                (["left_raw", "right_raw"] if args.unrectified else [])
        if depth_on:
            dirs += ["depth", "confidence"]
    if args.float_depth and depth_on:
        dirs.append("depth_float")
    if args.ply and depth_on:
        dirs.append("pointcloud")
    for d in dirs:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    procs = {}
    if args.mkv:
        procs["rgb"] = ffv1_proc(os.path.join(args.out, "RGB_left.mkv"), w, h, fps, "bgra", "bgr0")
        if depth_on:
            procs["depth"] = ffv1_proc(os.path.join(args.out, "Depth.mkv"), w, h, fps, "gray16le", "gray16le")
            procs["conf"] = ffv1_proc(os.path.join(args.out, "Confidence.mkv"), w, h, fps, "gray16le", "gray16le")

    sharp = {}
    chosen = None
    if args.sharpest_per:
        sharp = score_sharpness(args.svo, args.start, args.end)
        with open(os.path.join(args.out, "sharpness.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["svo_frame", "timestamp_ns", "sharpness"])
            for fno in sorted(sharp):
                wr.writerow([fno, sharp[fno][1], f"{sharp[fno][0]:.2f}"])
        chosen = []
        frames_sorted = sorted(sharp)
        for i in range(0, len(frames_sorted), args.sharpest_per):
            block = frames_sorted[i:i + args.sharpest_per]
            best = max(block, key=lambda k: sharp[k][0])
            if sharp[best][0] >= args.min_sharpness:
                chosen.append(best)
        vals = sorted(v[0] for v in sharp.values())
        print(f"sharpness of {len(vals)} frames: min {vals[0]:.1f}, median {vals[len(vals) // 2]:.1f}, "
              f"max {vals[-1]:.1f}; keeping {len(chosen)} frames (sharpest of every {args.sharpest_per})")
        params["sharpness"] = {"frames_scored": len(vals), "median": vals[len(vals) // 2],
                               "kept": len(chosen), "metric": "variance of Laplacian, left image at 1/2 res"}

    left, right, depth, conf, xyz = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
    left_raw, right_raw = sl.Mat(), sl.Mat()
    pose = sl.Pose()
    if args.start > 0 and chosen is None:
        zed.set_svo_position(args.start)
    seek = list(chosen) if chosen is not None else None

    idx_f = open(os.path.join(args.out, "index.csv"), "w", newline="", encoding="utf-8")
    idx = csv.writer(idx_f)
    idx.writerow(["out_idx", "svo_frame", "timestamp_ns", "pose_state", "tx_mm", "ty_mm", "tz_mm",
                  "qx", "qy", "qz", "qw", "distance_mm", "file_stem", "sharpness"])
    last_kept_pos = None
    dist_mm = 0.0
    prev_pos = None
    out_idx = 0
    t0 = time.time()
    while True:
        if seek is not None:          # --sharpest-per: jump straight to each chosen frame
            if not seek:
                break
            target = seek.pop(0)
            zed.set_svo_position(target)
        err = zed.grab(rt)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] grab: {err}")
            if err_is(err, "END_OF_SVOFILE_REACHED"):
                break
            continue
        fno = zed.get_svo_position()
        if args.end >= 0 and fno > args.end:
            break
        ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        pstate, t, q = "", [None] * 3, [None] * 4
        pos = None
        if tracking:
            st = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            pstate = enum_name(st)
            if st == sl.POSITIONAL_TRACKING_STATE.OK:
                t = list(pose.get_translation(sl.Translation()).get())
                q = list(pose.get_orientation(sl.Orientation()).get())
                pos = np.array(t, dtype=float)
                if prev_pos is not None and np.linalg.norm(pos - prev_pos) < 1000:
                    dist_mm += float(np.linalg.norm(pos - prev_pos))
                prev_pos = pos

        keep = ((fno - args.start) % max(1, args.step) == 0)
        if seek is not None:
            keep = fno == target
            if not keep:
                print(f"[WARN] asked for frame {target}, SDK returned {fno}; skipped")
        if args.every_mm > 0:
            keep = pos is not None and (last_kept_pos is None or dist_mm - last_kept_pos >= args.every_mm)
            if keep:
                last_kept_pos = dist_mm
        if not keep:
            continue

        stem = f"{fno:06d}"
        zed.retrieve_image(left, sl.VIEW.LEFT)
        bgra = left.get_data()
        if depth_on:
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(conf, sl.MEASURE.CONFIDENCE)
            d = depth.get_data()
            c = conf.get_data()
            d16 = encode_depth_u16(d, upm)
            c16 = encode_confidence_u16(c, cs)

        if not args.no_png:
            cv2.imwrite(os.path.join(args.out, "left", stem + ".png"), bgra[:, :, :3],
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if args.right:
                zed.retrieve_image(right, sl.VIEW.RIGHT)
                cv2.imwrite(os.path.join(args.out, "right", stem + ".png"), right.get_data()[:, :, :3],
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if args.unrectified:
                zed.retrieve_image(left_raw, sl.VIEW.LEFT_UNRECTIFIED)
                zed.retrieve_image(right_raw, sl.VIEW.RIGHT_UNRECTIFIED)
                cv2.imwrite(os.path.join(args.out, "left_raw", stem + ".png"), left_raw.get_data()[:, :, :3])
                cv2.imwrite(os.path.join(args.out, "right_raw", stem + ".png"), right_raw.get_data()[:, :, :3])
            if depth_on:
                cv2.imwrite(os.path.join(args.out, "depth", stem + ".png"), d16)
                cv2.imwrite(os.path.join(args.out, "confidence", stem + ".png"), c16)
        if args.float_depth and depth_on:
            np.save(os.path.join(args.out, "depth_float", stem + ".npy"), np.array(d, dtype=np.float32))
        if args.ply and depth_on:
            zed.retrieve_measure(xyz, sl.MEASURE.XYZ)
            p = xyz.get_data()[:, :, :3].reshape(-1, 3)
            cc = c.reshape(-1)
            ok = np.isfinite(p).all(axis=1) & (cc <= args.ply_max_conf)
            write_ply(os.path.join(args.out, "pointcloud", stem + ".ply"), p[ok],
                      bgra[:, :, :3].reshape(-1, 3)[ok])
        if procs:
            procs["rgb"].stdin.write(np.ascontiguousarray(bgra).tobytes())
            if depth_on:
                procs["depth"].stdin.write(d16.tobytes())
                procs["conf"].stdin.write(c16.tobytes())

        idx.writerow([out_idx, fno, ts, pstate, *t, *q, f"{dist_mm:.1f}", stem,
                      f"{sharp[fno][0]:.2f}" if fno in sharp else ""])
        out_idx += 1
        if out_idx % 50 == 0:
            el = time.time() - t0
            print(f"  {fno + 1}/{n_total} frames read, {out_idx} kept ({out_idx / el:.1f} kept/s)")

    idx_f.close()
    for p in procs.values():
        p.stdin.close()
        p.wait()
    zed.close()
    params["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    params["frames_kept"] = out_idx
    with open(os.path.join(args.out, "extraction_params.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(params), f, indent=2)
    print(f"Done: {out_idx} frames -> {args.out}")


if __name__ == "__main__":
    main()
