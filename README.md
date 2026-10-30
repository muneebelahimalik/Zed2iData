# Zed2iData

Field data collection for the ZED 2i stereo camera (WeedScan, Vidalia onion plots).
The data is meant for later frame extraction, depth analysis and leaf-emergence-point
localization, so the recorder is built to be lossless, fully documented, and to
report anything that goes wrong.

| File | Purpose |
|---|---|
| `Zed2i_video_datacollection.py` | GUI recorder (run this in the field) |
| `extract_svo.py` | Recompute depth from the SVO2 file and export frames, depth, confidence, poses, point clouds, or a gap-free `.mkv` set |
| `verify_session.py` | Check a session after recording (frame counts, gaps, CRCs, checksums for backups) |
| `record_headless.py` | Terminal recorder (same data as the GUI) for SSH / no monitor, e.g. on a Jetson |
| `zed_doctor.py` | Checks the machine, USB 3 link, disk and SDK; pre-builds NEURAL models; benchmarks loss-free settings |
| `jetson/` | `setup_jetson.sh` (one-time setup), `check_ssd.sh` (check/fix the data-SSD mount), `run_weedscan.sh` (desktop launcher). See **JETSON.md** |
| `.vscode/` | Run/debug configs and tasks: GUI, headless recorder, doctor, verify, SSD check (JETSON.md → Running from VS Code) |
| `session_reader.py` | Standalone Python reader for sessions (numpy + ffmpeg): decoding, joins, backprojection, world transform |
| `DATA_SPEC.md` | **Full data specification**: every file, format, code and column, plus guidance for extraction and localization |
| `video.py` | Older script, kept for reference |

## Install

1. ZED SDK 4.x or 5.x with its Python API (`pyzed`) and an NVIDIA GPU (needed for NEURAL depth).
2. FFmpeg and ffprobe on `PATH`.
3. `pip install -r requirements.txt`

## What is recorded

Each recording goes into its own folder named `Session_YYYYmmdd_HHMMSS_<plot>`:

* **`recording.svo2`** is the master record: the raw left and right images, IMU,
  magnetometer and barometer data, and hardware timestamps, stored with LOSSLESS
  (bit-exact) compression. Depth can be recomputed from it at any time with a better
  model (`NEURAL_PLUS`). Nothing here depends on the live processing keeping up.
* **`RGB_left.mkv`, `Depth.mkv`, `Confidence.mkv`** are live, pixel-aligned FFV1
  lossless videos. RGB is bit-exact. Depth is 16-bit at 0.1 mm per unit, which is about
  100× finer than ZED depth noise. Invalid, too-close and too-far pixels keep separate
  codes (0, 65534 and 65535), so they are never collapsed into a single 0. Confidence
  is stored too, so you can filter depth later. Depth is saved with no confidence
  filtering, no hole filling and no temporal smoothing, so it is the raw per-frame depth.
* **`frames.csv`** has one row per frame: hardware timestamp, gap/drop flags,
  exposure/gain/WB, 6-DoF pose, speed, distance, IMU, tilt from nadir.
* `markers.csv` (press F9 while recording), `events.log`, `calibration.json`
  (rectified and factory raw calibration, plus the IMU transform), a copy of the
  factory `SN<serial>.conf`, `session_metadata.json`, and a `README.txt` that explains
  how to decode every file.

Data-safety features:
* The camera image flip is forced OFF. With AUTO, a downward-facing camera can come
  out flipped differently from one session to the next.
* Each stream has its own encoder thread and a bounded RAM buffer. If the encoders
  can't keep up, a frame is dropped from all three .mkv files together, so they stay
  aligned, and the drop is flagged in `frames.csv`. The SVO2 file still has that frame.
* Gaps in the hardware timestamps (frames the camera dropped) are detected and logged.
* Recording refuses to start when free disk space is low, and stops and finalizes
  cleanly before the disk fills up.
* The PC is kept from sleeping while recording.
* Stopping waits until every queued frame is written. The .mkv frame counts are then
  verified with ffprobe and the result is saved in `session_metadata.json`.

## Recommended capture settings

* **HD2K @ 15 fps, NEURAL live depth, SVO2 LOSSLESS + live .mkv.** For analysis,
  recompute depth offline with `extract_svo.py --depth-mode NEURAL_PLUS`.
* **Camera pointing straight down.** The GUI shows the tilt from nadir and the depth
  at the image centre. Keep the lens at least about 0.5 m above the tallest leaves,
  because the ZED 2i minimum depth is about 0.3 m.
* **Set exposure manually and keep it short.** Auto exposure on a ZED 2i can reach
  most of the frame period (up to about 66 ms at 15 fps), which blurs the image at
  walking speed. Use *Exposure % ≈ 3–5* at 15 fps (about 2–3 ms) and raise *Gain*
  until the image is bright enough. Alternatively, let auto settle and press *Lock
  current auto values* to fix exposure, gain and white balance so that colour is
  consistent within a plot.
* **Click "Test disk speed" before going to the field.** HD2K with everything enabled
  writes roughly 250–300 MB/s, which is about 1 TB per hour. Record to a local NVMe SSD,
  not a network drive or USB stick. If the disk or CPU can't keep up, untick the live
  .mkv and record SVO2 only (about 120 MB/s). You can rebuild the .mkv files later
  with `extract_svo.py --mkv`.

### Golden hour, shade, low sun: use **Motion-safe auto** exposure
The ZED's own auto exposure brightens by **lengthening the exposure** first, up to about
61 ms at 15 fps, and that smears the image when the camera moves. Tick **Motion-safe
auto** in the GUI (section 2), or pass `record_headless.py --motion-safe
--target-speed 0.12`. It keeps the exposure at or below the longest value that gives
≤ the blur budget (section 4, default 1 px) at your **target speed**, and raises or
lowers **gain** automatically to hold brightness. In bright light it also shortens the
exposure. The stats line shows `motion-safe: exp …% gain … brightness …`.
* `⚠ too dark at max gain`: walk slower (set a lower target speed, which allows a longer
  exposure), add light, or accept more blur.
* Keep the live `BLUR ≈ … px` at ≤ 1–2 px.
* A diffuse, **flicker-free** LED light near the camera lets the camera use less gain
  (less noise) and fills in low-sun shadows.

### Travel speed

Motion blur, not overlap, sets the speed limit. The *Speed planner* panel works this
out from the real calibration, your lens height and the current exposure, then shows
your live speed from visual-inertial tracking (red means too fast). *Pace beep* gives
a beep every N metres at the target speed, which lets you walk in step with a
measuring wheel.

Maximum speed for at most 1 px of blur at the plant tops (HD2K, 2.1 mm lens, plants 0.2 m tall):

| Lens height | GSD at plant tops | 2 ms exposure | 3 ms | 5 ms | 8 ms |
|---|---|---|---|---|---|
| 0.8 m | 0.57 mm/px | 0.28 m/s | 0.19 | 0.11 | 0.07 |
| 1.0 m | 0.75 mm/px | 0.38 m/s | 0.25 | 0.15 | 0.09 |
| 1.2 m | 0.94 mm/px | 0.47 m/s | 0.31 | 0.19 | 0.12 |

**Default recommendation:** lens at about 1.0 m, exposure about 3 ms, and a walking
speed of **0.25 m/s** (0.9 km/h, about 0.8 ft/s, about 0.56 mph, or 1 m every 4 s).
At 15 fps that is about 17 mm of travel per frame, so each ground point appears in
about 70 frames. That is plenty of overlap for picking sharp frames and for
multi-view depth. Moving slower also reduces rolling-shutter shear.

## Windows laptop vs Jetson

The same code runs on both. Each machine gets the latest version only when you update
it there, with `git pull origin main` in its `Zed2iData` folder. All improvements apply on
both machines: SVO2 master record, raw 16-bit depth, separate capture process, frame-loss
checks, verification, Ctrl+C handling and the VS Code configs.

What differs automatically by platform:

| | Windows laptop | Jetson |
|---|---|---|
| Default data folder | `M:/Research/Data/...` | from `config_local.json` (the SSD) |
| Live FFV1 `.mkv` | **on** (needs FFmpeg on PATH) | off (CPU too weak; rebuild offline) |
| Live depth model | NEURAL | NEURAL_LIGHT |
| Sounds / keep-awake | winsound / SetThreadExecutionState | aplay / systemd-inhibit |
| USB 3 link check | not available (use ZED Diagnostic) | `zed_doctor.py` |

Per-machine settings go in a `config_local.json` next to the scripts, so you never edit the
code. Example for the laptop:
`{"base_directory": "M:/Research/Data/Vidalia_onions_visit3", "default_record_mkv": true}`

**Benchmark the laptop once**:
`python zed_doctor.py --bench --out M:/Research/Data --configs HD2K:15:NEURAL:LOSSLESS:mkv,HD2K:15:NEURAL:LOSSLESS`.
If the `:mkv` run reports dropped .mkv frames or missing camera frames, turn the live
.mkv off. The SVO2 still holds everything.

## After each field day

```
python verify_session.py <session>            # quick check
python verify_session.py <session> --decode   # full CRC check of every .mkv frame
python verify_session.py <session> --checksum # write SHA-256 manifest before copying
python verify_session.py <copy>   --check-copy  # verify the backup copy
```

## Extracting frames

```
# all frames, NEURAL_PLUS depth, lossless PNGs + 16-bit depth/confidence PNGs
python extract_svo.py <session>/recording.svo2 --out <session>/extracted

# one frame per 50 mm of travel, float depth (.npy) and coloured point clouds
python extract_svo.py rec.svo2 --out ex --every-mm 50 --float-depth --ply

# directly from the live .mkv files
ffmpeg -i RGB_left.mkv -start_number 0 rgb/%06d.png
ffmpeg -i Depth.mkv    -start_number 0 -pix_fmt gray16le depth/%06d.png
```

To decode depth: `depth_mm = png / 10`, then treat 0, 65534 and 65535 as invalid. Use
the intrinsics in `calibration.json → calibration_rectified.left_cam` for 3D:
`X = (u − cx)·Z/fx`, `Y = (v − cy)·Z/fy`, `Z = depth_mm`.
