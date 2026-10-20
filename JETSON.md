# Running WeedScan on an NVIDIA Jetson with the ZED 2i

Tested design target: JetPack 5/6 with ZED SDK 4.x/5.x for L4T. The recorder writes
the same data on the Jetson as on the Windows PC. Downstream code needs no changes
(see `DATA_SPEC.md`).

## 1. One-time setup

```bash
git clone https://github.com/muneebelahimalik/Zed2iData.git ~/Zed2iData
cd ~/Zed2iData

# check the ZED Python API is installed for the python3 you will use
python3 -c "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())" \
  || python3 /usr/local/zed/get_python_api.py

# mount the data SSD first (see "Data SSD" below), then:
# system packages, ZED USB power rule, max performance, data folder + auto-mount, launcher
sudo ./jetson/setup_jetson.sh --max-perf --data-dir /media/$USER/<YOUR_SSD>/ZED_Data --fstab
```

### Data SSD

The recorder **refuses to record to `/media/...` or `/mnt/...` when no drive is
mounted there**. Otherwise the folder would silently be created on the small eMMC
system disk. The SSD must therefore be mounted at every boot. `--fstab` adds a UUID
entry with `nofail`, so the Jetson still boots if the SSD is removed. A backup of
`/etc/fstab` is kept. To do the same by hand:

```bash
sudo blkid /dev/nvme0n1p1                       # note the UUID
echo 'UUID=<uuid>  /media/pcp/Samsung_980_PRO  ext4  defaults,nofail,noatime,x-systemd.device-timeout=10  0  2' | sudo tee -a /etc/fstab
sudo mount -a && findmnt /media/pcp/Samsung_980_PRO
```

What `setup_jetson.sh` does:
* **Packages:** installs `python3-tk`, `python3-pil.imagetk`, `ffmpeg`, `alsa-utils`, and
  `python3-opencv` if `cv2` is missing. It does **not** touch `numpy` or `pyzed`:
  upgrading numpy can break the ZED Python wheel.
* **USB power:** adds a udev rule so the ZED is never USB-autosuspended. Autosuspend
  can cause mid-run disconnects.
* **Max performance (`--max-perf`):** runs `nvpmodel -m 0` and `jetson_clocks`, and
  re-applies `jetson_clocks` at every boot. Mode 0 is MAXN on most modules; check
  `nvpmodel -q --verbose` for yours.
* **Data folder (`--data-dir`):** writes `config_local.json`, which holds the data
  folder. That file is per-machine, git-ignored, and can override any key of
  `CONFIG` in `Zed2i_video_datacollection.py`.
* **Launcher:** adds a **WeedScan** launcher to the desktop.

Then, **with the camera plugged in and internet available once**:

```bash
python3 zed_doctor.py --optimize                 # builds NEURAL models for this GPU (5-20 min each, once)
python3 zed_doctor.py --bench --out /media/$USER/<YOUR_SSD>/ZED_Data
```

The first time a NEURAL depth model is used on a Jetson, the SDK optimizes it for the
GPU, which takes minutes. `--optimize` does this at home, so opening the camera in the
field is instant. On its first open, the camera also downloads its factory
calibration file (`/usr/local/zed/settings/SN<serial>.conf`), which needs internet once.

`zed_doctor.py` checks each item below and prints a fix for every problem:
* the Jetson model and power mode;
* `pyzed`, numpy, OpenCV, Tk and ffmpeg;
* that the **camera is on a USB 3 link** (≥ 5000 Mb/s);
* whether USB autosuspend is on;
* SDK detection;
* the factory calibration file;
* free space and sustained write speed of the data disk.

The SDK log line "Self-calibration was skipped due to low texture" is harmless when the
camera is pointing at a desk or wall. In the field, open the camera while it already
looks at the soil and plants, so self-calibration has texture to work with.

`--bench` records short test sessions at several settings. It then lists which
settings ran with **zero lost frames on your Jetson**. Use one of those.

## 2. Recording

**With a monitor:** double-click **WeedScan** on the desktop, or run
`./jetson/run_weedscan.sh`. The GUI is the same as on Windows, and its log is saved
to `~/ZED_Data/logs/`.

**Over SSH (laptop or phone) / no monitor:**

```bash
cd ~/Zed2iData
python3 record_headless.py --plot A3 --operator MM --height 1.0 --exposure 4 --gain 60
#   type text + Enter = marker ("start row 3"),  q + Enter or Ctrl+C = stop & save
#   --lock-auto 5   let auto exposure/WB settle 5 s then lock it (instead of --exposure)
#   --duration 600  stop automatically after 10 min
#   --res HD1080 --fps 30, --depth NEURAL, --mkv, --svo-compression H265_LOSSLESS ...
```

If the SSH connection drops, the recording is stopped and saved cleanly (SIGHUP is
handled). To keep a recording alive independently of SSH, run it inside `tmux`:
start with `tmux new -s rec`, detach with Ctrl+B then D, reattach with
`tmux attach -t rec`.

## 3. Jetson defaults and why

| Setting | Jetson default | Reason |
|---|---|---|
| SVO2 master record | **on, LOSSLESS** (bit-exact) | Holds everything raw. Depth is recomputed offline |
| Live FFV1 .mkv | **off** | The Jetson CPU cannot encode three HD2K lossless streams at 15 fps. Rebuild them later with `extract_svo.py --mkv` on a PC |
| Live depth | **NEURAL_LIGHT** (or NEURAL on SDK < 4.2) | Only used for preview, centre depth, tracking and speed. It leaves GPU headroom so no frames are dropped |
| Resolution | HD2K @ 15 fps | Finest detail. Drop to HD1080 if `--bench` shows losses |

You can change all of these in the GUI, with `record_headless.py` flags, or in
`config_local.json`, e.g.:
```json
{ "base_directory": "/media/me/SSD/ZED_Data", "default_depth_mode": "NEURAL", "default_record_mkv": false }
```

`H264_LOSSLESS` / `H265_LOSSLESS` SVO modes use the hardware video encoder. They are
**near**-lossless, not bit-exact, and **Jetson Orin Nano has no hardware encoder**, so
they fail there. Stay with `LOSSLESS` unless the benchmark shows it can't keep up.

## 4. Hardware checklist

* **USB:** plug the ZED 2i directly into a USB 3 port with the original cable, with
  no passive hub or extension. `zed_doctor.py` must show "camera on USB 3".
* **Disk:** record to an **NVMe or external USB 3 SSD**, not the eMMC or SD boot disk.
  HD2K SVO2 lossless writes about 120 MB/s, which is about 430 GB per hour.
* **Power:** use a power supply rated for MAXN. Undervoltage throttles the GPU and
  causes dropped frames. Check `sudo tegrastats` for throttling during the benchmark.
* **Cooling:** keep the fan on and shade the Jetson in the field. Thermal throttling
  looks like slowly rising "missing camera frames".

## 5. After the field day

```bash
python3 verify_session.py <session> --checksum      # integrity + SHA-256 manifest
rsync -a --progress <field-day-folder> <lab-storage>/
python3 verify_session.py <copy-of-session> --check-copy
```

Extraction (NEURAL_PLUS depth, PNGs, point clouds, .mkv rebuild) is best run on the
lab PC or GPU workstation with `extract_svo.py` (see `DATA_SPEC.md` §8). It also
works on the Jetson, only slower.
