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
