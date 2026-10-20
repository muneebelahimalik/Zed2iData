# WeedScan ZED 2i dataset: data specification

This is a reference for anyone (human or agent) who processes these recordings for
frame extraction, weed detection, leaf-emergence-point localization or 3D mapping.
It describes exactly what the recorder (`Zed2i_video_datacollection.py`, app v2.0.0)
writes, how each file is encoded, and how to read it correctly.

**Quick start:** use `session_reader.py` (numpy + ffmpeg, no ZED SDK needed). It
already applies every decoding rule in this document.

---

## 1. Acquisition setup

| Item | Value |
|---|---|
| Camera | Stereolabs ZED 2i stereo camera (rolling shutter, built-in IMU, magnetometer and barometer). Serial number is in `calibration.json`. |
| Mounting | Nadir: the camera looks straight down at the crop row and is carried along the row. Measured lens height is in `session_metadata.json → field.camera_height_m`; the per-frame tilt is `tilt_from_nadir_deg` in `frames.csv`. |
| Crop | Vidalia onion plots. Targets: weeds, plus leaf emergence points of the onions. |
| Default mode | HD2K **2208 × 1242 @ 15 fps**. Other possible modes: HD1080 1920×1080 @ 15/30, HD720 1280×720 @ 15/30/60, VGA 672×376 @ 15–100. Always read the actual mode from `calibration.json → resolution, fps`. |
| Travel speed | Target about 0.25 m/s (about 17 mm of travel per frame at 15 fps, so each ground point appears in about 70 frames). The actual speed for every frame is `speed_m_s` in `frames.csv`. |
| Exposure | Usually manual or locked short exposure (about 2–5 ms) to limit motion blur. Per-frame exposure, gain and white balance are in `frames.csv`. |
| Live depth model | ZED `NEURAL` by default (see `session_metadata.json → capture.depth_mode`). The SVO2 file allows recomputing depth with `NEURAL_PLUS`. |
| Image orientation | Image flip is forced OFF, so every session has the same orientation. |
| Travel direction in image | Recorded by the operator in `field.travel_direction` and `field.travel_axis`. The default is that image top = forward, i.e. the ground scrolls from the top of the image to the bottom. Check it on the data: optical flow or the pose will show it. |

## 2. Session folder layout

One folder per recording, named `Session_YYYYmmdd_HHMMSS[_<plot_id>]`, inside a
field-day folder (for example `M:/Research/Data/Vidalia_onions_visit2_day2/`).

```
Session_20261008_101500_plotA/
├── recording.svo2          master record (raw stereo + sensors), may be absent if disabled
├── RGB_left.mkv            rectified left colour, FFV1 lossless
├── Depth.mkv               depth aligned to RGB_left, FFV1 lossless 16-bit
├── Confidence.mkv          ZED confidence map, FFV1 lossless 16-bit
├── frames.csv              one row per captured frame (timing, pose, IMU, camera settings, flags)
├── markers.csv             operator annotations typed during recording
├── calibration.json        rectified + raw factory calibration, IMU extrinsics
├── calibration_params.txt  human-readable summary of the above
├── SN<serial>.conf         copy of the factory calibration file (if found)
├── session_metadata.json   field info, all capture parameters, statistics, verification
├── events.log              warnings/errors with timestamps (gaps, drops, encoder issues)
├── ffmpeg_rgb.log / ffmpeg_depth.log / ffmpeg_confidence.log
├── README.txt              short per-session decoding guide
├── verify_report.json      written by verify_session.py (optional)
└── checksums.sha256        written by verify_session.py --checksum (optional)
```

If the operator disabled the live .mkv output, only the SVO2 file and the metadata
files exist. Use `extract_svo.py` (section 8) in that case.
**Sessions recorded on the Jetson have no live .mkv by default.** Its CPU cannot
encode them at HD2K. `session_metadata.json → capture.record_mkv` tells you which case
you have, and `host.jetson` identifies the recording device. Run
`extract_svo.py --mkv` (or the PNG export) on a PC to produce the same files.

## 3. Image data: `RGB_left.mkv`

* **Content:** the **rectified** left camera image (ZED `VIEW.LEFT`). Lens
  distortion is already removed, and the image is row-aligned with the right camera.
* **Codec:** FFV1 version 3 in Matroska. Every frame is a keyframe (`-g 1`), the
  image is split into 24 slices, and each slice carries a CRC. The pixel format is
  `bgr0`, i.e. 8-bit BGR with a padding byte. The images are **bit-exact**
  (lossless) relative to what the SDK produced.
* **Size:** the camera resolution, e.g. 2208×1242.
* **Frame index:** the k-th decoded frame is `mkv_frame_idx = k` (0-based). This is
  **not** necessarily `rec_frame_idx`; see section 6.
* **Container frame rate:** a constant nominal rate is written. **Do not use
  container timestamps**: real timing comes from `frames.csv → image_timestamp_ns`.
* **Decoding:**
  ```
  ffmpeg -i RGB_left.mkv -start_number 0 rgb/%06d.png
  ```
  In Python, use `session_reader.read_mkv(..., kind="bgr")`, or PyAV with
  `to_ndarray(format="bgr24")`. `cv2.VideoCapture` also works for the RGB file only.

## 4. Depth: `Depth.mkv`

* **Content:** ZED depth measure (`MEASURE.DEPTH`), pixel-aligned with
  `RGB_left.mkv`. Each value is the **Z distance along the optical axis**, not the
  Euclidean range to the point.
* **Codec:** FFV1 lossless, `gray16le` (single channel, unsigned 16-bit).
* **Encoding** (`depth_units_per_mm` is in `session_metadata.json → encoding`;
  default 10):

  | Stored uint16 | Meaning |
  |---|---|
  | `0` | no depth: occluded, no stereo match, or undefined (SDK NaN) |
  | `1 … 65533` | valid: `depth_mm = value / depth_units_per_mm` (0.1 mm steps, max 6553.3 mm) |
  | `65534` | closer than the minimum depth (SDK −inf). ZED 2i minimum is about 300 mm |
  | `65535` | farther than the maximum depth or beyond 6553.3 mm (SDK +inf) |

  Values are rounded to the nearest unit, so the quantisation error is ≤ 0.05 mm.
  That is negligible next to stereo noise.
* **Raw-ness:** depth was computed with `confidence_threshold = 100` and
  `texture_confidence_threshold = 100` (nothing rejected), `enable_fill_mode =
  False` (holes are **not** filled) and `depth_stabilization = 0` (no temporal
  smoothing; every frame is independent). Expect noisy pixels at leaf edges and on
  thin leaves. **You must filter using the confidence map yourself.**
* **⚠ Decoding:** never use `cv2.VideoCapture`; it converts to 8-bit BGR and
  destroys the depth values. Use one of these:
  ```
  ffmpeg -i Depth.mkv -start_number 0 -pix_fmt gray16le depth/%06d.png
  ```
  ```python
  d16 = cv2.imread("depth/000123.png", cv2.IMREAD_UNCHANGED)  # uint16
  depth_mm = d16.astype(np.float32) / 10
  depth_mm[(d16 == 0) | (d16 >= 65534)] = np.nan
  ```
  or `session_reader.decode_depth()`.

## 5. Confidence: `Confidence.mkv`

* ZED `MEASURE.CONFIDENCE`, pixel-aligned with the depth.
* FFV1 lossless `gray16le`. `confidence = value / confidence_scale` (default 100),
  giving 0 … 100. `65535` = undefined.
* **Higher means LESS reliable.** This is the ZED convention: a pixel passes
  threshold T when its confidence ≤ T.
* Suggested filters:

  | Purpose | Keep pixels with confidence ≤ |
  |---|---|
  | Clean geometry or ground plane | ≤ 30–50 |
  | General use | ≤ 70 |
  | Everything the SDK produced | ≤ 100 |

## 6. Per-frame log: `frames.csv`

There is one row per frame captured while recording, in capture order. The file is
UTF-8 with a header row. An empty cell means "not available" for that frame.

### 6.1 Indices and joining

| Column | Meaning |
|---|---|
| `rec_frame_idx` | 0-based row index in this recording |
| `grab_idx` | frame counter since the camera was opened (spans recordings) |
| `mkv_written` | `1` if this frame is in all three .mkv files, `0` if it was dropped from them because of encoder backlog (it is still in the SVO2) |
| `mkv_frame_idx` | index of this frame inside the .mkv files (blank when `mkv_written=0`). **Use this to map a decoded .mkv frame to its row.** The values are contiguous: 0 … N−1 |
| `svo_frame_ok` | `1` if the SVO2 stored this frame, `0` if the store failed, blank if SVO2 was disabled |
| `image_timestamp_ns` | ZED hardware image timestamp (UTC epoch, ns). **This is the universal join key**: the same value appears in SVO2 extractions (`extract_svo.py → index.csv timestamp_ns`) |

### 6.2 Timing and integrity

| Column | Meaning |
|---|---|
| `system_time_ns`, `system_time_iso` | host PC clock when the frame was retrieved |
| `dt_ms` | interval since the previous row's `image_timestamp_ns` |
| `est_missing_before` | estimated number of camera frames missing before this row (timestamp gap > 1.5 × frame period). Non-zero means a gap in the sequence |
| `camera_dropped_total` | SDK cumulative dropped-frame counter |

### 6.3 Camera settings

The values are polled about every 5 frames, so consecutive rows can repeat a value.

| Column | Meaning |
|---|---|
| `exposure_pct` | ZED exposure, 0–100 % of the frame period |
| `exposure_ms_approx` | `exposure_pct / 100 × 1000 / fps`. **This is approximate**; use it for blur estimates only |
| `gain` | 0–100 |
| `aec_agc` | 1 = auto exposure/gain on, 0 = manual or locked |
| `wb_temp_k`, `wb_auto` | white balance in kelvin, and whether auto white balance was on |

### 6.4 Pose: visual-inertial positional tracking

All pose columns are blank when tracking was disabled, or when `pose_state ≠ OK`.

| Column | Meaning |
|---|---|
| `pose_state` | `OK` means the values below are valid. Other values (`SEARCHING`, `OFF`, …) mean ignore them |
| `tx_mm, ty_mm, tz_mm` | position of the **left camera** in the world frame, in mm |
| `qx, qy, qz, qw` | orientation of the left camera in the world frame (unit quaternion, x,y,z,w order) |
| `pose_confidence` | 0–100, from the SDK |
| `speed_m_s` | 3D speed over a 1 s sliding window |
| `distance_m` | cumulative path length since recording start (single steps > 1 m, i.e. tracking jumps, are ignored) |

**Frames.** The SDK coordinate system is `IMAGE`: x right, y down, z forward (along
the optical axis), in mm. The world frame is fixed at the pose when the camera was
opened, aligned to gravity (`set_gravity_as_origin`). It is **not** a geographic
frame, and it is shared by all recordings made while the camera stayed open.

To transform a camera point to world coordinates:

`P_world = R(q) · P_cam + t`

(`session_reader.Session.camera_to_world`). The pose is visual-inertial odometry,
so expect drift of a few % of the distance travelled. For accurate mapping across
long rows, refine it (bundle adjustment, ground-plane constraint).

### 6.5 Inertial and environment

These come from the ZED sensor stack, sampled at the image timestamp, in the same
IMAGE coordinate convention.

| Column | Meaning |
|---|---|
| `imu_qx … imu_qw` | IMU-fused orientation quaternion |
| `acc_x, acc_y, acc_z` | linear acceleration in m/s² (includes gravity) |
| `gyro_x, gyro_y, gyro_z` | angular velocity in deg/s |
| `tilt_from_nadir_deg` | angle between the optical axis and vertical, from gravity. 0 = looking straight down. Gives the direction-agnostic magnitude only |
| `mag_heading_deg` | magnetometer heading, degrees. Uncalibrated, so treat it as coarse |
| `pressure_hpa`, `rel_altitude_m` | barometer |
| `imu_temp_c` | IMU temperature |

### 6.6 Live depth summary

| Column | Meaning |
|---|---|
| `center_depth_mm` | median valid depth in the central 20 %×20 % of the image. **Filled only on preview frames (about 10 Hz)**; blank otherwise. A rough canopy/ground distance; recompute it from Depth.mkv for analysis |

### 6.7 Recommended joins

```python
rows = {r["mkv_frame_idx"]: r for r in frames if r["mkv_written"] == 1}   # .mkv frame -> row
by_ts = {r["image_timestamp_ns"]: r for r in frames}                      # SVO2 frame -> row
```

`session_reader.Session.mkv_rows` and `.row_by_timestamp()` do exactly this.

## 7. Calibration: `calibration.json`

```
serial_number, camera_model, camera_firmware_version, sensors_firmware_version
resolution [W, H], fps
calibration_rectified        <- USE THIS with RGB_left / Depth / Confidence
  left_cam:  fx, fy, cx, cy (pixels), disto (all ~0 after rectification),
             h_fov_deg, v_fov_deg, d_fov_deg, image_size [W, H]
  right_cam: same
  baseline   (mm, about 120 for ZED 2i)
  stereo_transform: matrix_4x4, translation (mm), rotation_vector_rad
calibration_raw_factory      <- only for unrectified images (extract_svo.py --unrectified)
camera_imu_transform         <- IMU -> left camera transform
```

The intrinsics are for the recorded resolution, so you never need to rescale them
unless you resize the images yourself.

**Pinhole model.** For pixel (u, v) with depth Z in mm:

```
X = (u − cx) · Z / fx      Y = (v − cy) · Z / fy      Z = Z
```

(`Session.backproject` / `Session.project`). The ground sampling distance at
distance Z is `Z / fx` mm per pixel, about 0.94 mm/px at 1 m for HD2K
(fx ≈ 1060 for the 2.1 mm lens; read the real value from the file).

## 8. Master record: `recording.svo2`

* This is the native ZED SVO2 container with **LOSSLESS** compression
  (PNG/ZSTD, bit-exact) unless `session_metadata.json → capture.svo_compression`
  says `H264_LOSSLESS` or `H265_LOSSLESS`, which are near-lossless.
* It contains the **raw left and right images** plus the full IMU, magnetometer,
  barometer and temperature streams, all with hardware timestamps.
* It contains **no depth**: depth is recomputed whenever the file is opened. This
  lets you use a better model than the live one, get the right image, unrectified
  images, point clouds, or rebuild .mkv files that have frames missing.
* It requires the ZED SDK (with its `pyzed` Python API) and an NVIDIA GPU to compute
  depth.
* `extract_svo.py` writes into `--out`:

  | Output | Content |
  |---|---|
  | `left/NNNNNN.png` | rectified left image, 8-bit BGR PNG (lossless). `NNNNNN` = SVO frame number |
  | `right/` | rectified right image (`--right`) |
  | `left_raw/`, `right_raw/` | unrectified images (`--unrectified`) |
  | `depth/NNNNNN.png` | 16-bit depth, **same codes as Depth.mkv** (0 / 65534 / 65535, value/10 = mm) |
  | `depth_float/NNNNNN.npy` | float32 mm with NaN/±inf (`--float-depth`) |
  | `confidence/NNNNNN.png` | 16-bit, value/100 = confidence |
  | `pointcloud/NNNNNN.ply` | binary PLY: XYZ float32 (mm, camera frame) + RGB, confidence ≤ `--ply-max-conf` (`--ply`) |
  | `RGB_left.mkv`, `Depth.mkv`, `Confidence.mkv` | full, gap-free rebuild (`--mkv`) |
  | `index.csv` | `out_idx, svo_frame, timestamp_ns, pose_state, tx..qw, distance_mm, file_stem` |
  | `calibration.json`, `extraction_params.json` | intrinsics, and the exact arguments, depth mode and SDK version used |

  Useful options: `--depth-mode NEURAL_PLUS` (default), `--step N` (every Nth frame),
  `--every-mm D` (one frame per D mm of travel), `--start`, `--end`.
* **The tracking world frame in `index.csv` is not the same as in `frames.csv`.**
  Extraction re-runs tracking, so its origin is the first extracted frame. Join
  extractions to the recording by `timestamp_ns`, never by pose.

## 9. Other files

* **`markers.csv`:** `rec_frame_idx, image_timestamp_ns, system_time_iso, text`.
  These are free-text notes the operator typed during recording (for example
  "start row 3", "weed patch", "skip: person in view"). Use them to split or label
  sequences.
* **`session_metadata.json`:**
  * `field`: site, plot_id, operator, crop_stage, travel_direction, weather, notes, camera_height_m, plant_height_m, target_speed_m_s, travel_axis
  * `camera`: serial number, model, firmware, resolution, fps
  * `camera_settings_at_start`
  * `capture`: resolution, fps, depth_mode, positional_tracking, record_svo / record_mkv, svo_compression, `init_parameters`, `runtime_parameters`
  * `encoding`: depth_units_per_mm, depth_codes, confidence_scale, codec notes
  * `speed_plan`: the GSD, footprint, exposure and recommended speed computed at start
  * `host`: PC, OS, versions of Python, numpy, OpenCV, FFmpeg and the ZED SDK, disk benchmark
  * `statistics`: frames, duration, effective fps, missing camera frames, SVO2 OK/failed, mkv enqueued/dropped, distance, mean speed
  * `verification`: ffprobe frame count per stream, alignment, SVO2 existence
  * `status`: `COMPLETE`, `COMPLETE_WITH_WARNINGS`, or `RECORDING…` (the app did not stop cleanly; the data is still usable up to the crash)
* **`events.log`:** one line per event, `ISO-time [LEVEL] message`. LEVEL is one of
  INFO, WARN, ERROR or MARK.

## 10. Data-quality checklist before training or analysis

1. **Status:** `session_metadata.json → status` starts with `COMPLETE`, and
   `verification.all_ok` is true. Alternatively, run
   `python verify_session.py <session>`.
2. **Gaps:** look at `est_missing_before`. Gaps break the assumption of evenly spaced
   frames; use `image_timestamp_ns` for any time or velocity computation.
3. **.mkv drops:** rows with `mkv_written = 0` are missing from the .mkv files. If
   you need them, rebuild with `extract_svo.py --mkv`.
4. **Motion blur:** estimated blur in pixels is
   `speed_m_s × exposure_ms_approx/1000 / (Z/fx/1000)`. Reject or down-weight frames
   with more than about 1.5 px, or pick the sharpest frame per position, e.g. by
   variance of the Laplacian.
5. **Tilt:** prefer frames with `tilt_from_nadir_deg` < about 10°.
6. **Lighting:** if `aec_agc = 1` (auto), exposure and colour vary between frames;
   normalise colour if needed.
7. **Depth validity:** apply the confidence filter (section 5). Expect missing depth
   on thin leaves, specular soil and image borders (the left border of the left
   image has no stereo overlap).

## 11. Guidance for frame extraction, detection and localization

* **Frame sampling:** consecutive frames overlap by about 98 %. For a
  non-redundant detection dataset, sample by distance travelled rather than by frame
  count: about 0.3–0.5 × the footprint along travel (footprint ≈ `H × 1242 / fy` m,
  about 1.17 m at 1 m height on HD2K). Use `--every-mm 400`, or select rows by
  `distance_m`. Keep all frames for tracking-based or multi-view methods.
* **2D labels → 3D:** detections on `RGB_left` pixels map one-to-one to `Depth`
  pixels; no registration is needed.
* **Pixel → 3D:** take the median depth of the valid, low-confidence-value pixels in
  a small window (e.g. 5×5–15×15) at the point, then `backproject`.
* **Bounding-box depth:** use a robust statistic (median or low percentile)
  over the mask or box, after the confidence filter.
* **Leaf emergence points:** these sit at the base of the plant, near the
  soil/bulb. They are often partly occluded by leaves in a nadir view, so:
  * fit a local **ground plane** (RANSAC on low-confidence-value depth points
    around the plant) and measure plant points as height above that plane;
  * fuse observations from many frames (each point is seen about 70 times) in the
    world frame to reduce noise and fill occlusions.
* **Cross-frame localization and de-duplication:** transform each detection to world
  coordinates with the per-frame pose, then cluster (e.g. DBSCAN with a radius of
  about 20–50 mm) to get one entry per physical weed or plant. Because of odometry
  drift, prefer clustering within a sliding window, or add a ground-plane / row-line
  constraint.
* **Along-row coordinate:** project world positions onto the main direction of the
  camera path (PCA of `tx,ty,tz`) to get "metres along the row" for each detection.
  Operator markers give the row and plot boundaries.
* **Depth sign and units:** everything is in **mm** with +Z pointing away from the
  camera, toward the ground. Height above ground = `Z_ground − Z_point`.
* **Better depth:** for final localization, consider re-running depth from SVO2 with
  `NEURAL_PLUS`. Join the result to `frames.csv` by timestamp to reuse the pose,
  or use the extraction's own `index.csv` poses consistently within that extraction.

## 12. Reference code

```python
from session_reader import Session
import numpy as np

s = Session(r"M:\Research\Data\Vidalia_onions_visit2_day2\Session_20261008_101500_plotA")
print(s.width, s.height, s.fps, s.fx, s.fy, s.cx, s.cy)

for f in s.iter_frames(step=10):                 # f.row = frames.csv row
    valid = f.valid_mask(max_conf=50)
    u, v = 1100, 620                             # e.g. detection centre
    win = f.depth_mm[v-5:v+6, u-5:u+6][valid[v-5:v+6, u-5:u+6]]
    if win.size:
        X, Y, Z = s.backproject(u, v, np.median(win))
        Pw = s.camera_to_world(np.array([X, Y, Z]), f.row)   # None if pose not OK
        print(f.row["mkv_frame_idx"], f.row["image_timestamp_ns"], (X, Y, Z), Pw)
```

Command-line summary of a session: `python session_reader.py <session_folder>`.
