#!/usr/bin/env bash
# One-time setup of a Jetson (JetPack + ZED SDK already installed) for WeedScan recording.
#
#   sudo ./jetson/setup_jetson.sh [--max-perf] [--data-dir /media/<user>/<SSD>/ZED_Data]
#
#   - installs the system packages the recorder needs (does NOT touch numpy / pyzed)
#   - installs a udev rule so the ZED 2i is never USB-autosuspended (avoids mid-run dropouts)
#   - optionally sets the max power mode + locks clocks now and at every boot (--max-perf)
#   - writes config_local.json with your data folder (--data-dir)
#   - adds a "WeedScan" desktop launcher
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_PERF=0
DATA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-perf) MAX_PERF=1; shift ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
RUN_USER="${SUDO_USER:-$USER}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

echo "== apt packages =="
apt-get update -qq
apt-get install -y python3-tk python3-pil python3-pil.imagetk ffmpeg alsa-utils
if ! sudo -u "$RUN_USER" python3 -c "import cv2" 2>/dev/null; then
  apt-get install -y python3-opencv
fi

echo "== ZED Python API =="
if sudo -u "$RUN_USER" python3 -c "import pyzed.sl as sl; print('pyzed OK, SDK', sl.Camera.get_sdk_version())"; then
  :
else
  echo "pyzed missing -> run:  python3 /usr/local/zed/get_python_api.py   (as $RUN_USER, then re-run this script)"
fi

echo "== USB: keep the ZED powered (no autosuspend) =="
cat > /etc/udev/rules.d/99-weedscan-zed-power.rules <<'RULE'
# Stereolabs ZED: never autosuspend (prevents frame drops / disconnects during long recordings)
ACTION=="add|change", SUBSYSTEM=="usb", ATTR{idVendor}=="2b03", TEST=="power/control", ATTR{power/control}="on"
RULE
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=2b03 || true
for d in /sys/bus/usb/devices/*; do
  if [[ -f "$d/idVendor" && "$(cat "$d/idVendor")" == "2b03" && -f "$d/power/control" ]]; then
    echo on > "$d/power/control"
  fi
done

if [[ $MAX_PERF -eq 1 ]]; then
  echo "== Max performance =="
  nvpmodel -m 0 || echo "nvpmodel -m 0 failed; check 'nvpmodel -q --verbose' for your module's max mode"
  jetson_clocks || true
  cat > /etc/systemd/system/weedscan-jetson-clocks.service <<'UNIT'
[Unit]
Description=Lock Jetson clocks for WeedScan recording
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable weedscan-jetson-clocks.service
fi

if [[ -n "$DATA_DIR" ]]; then
  echo "== Data folder: $DATA_DIR =="
  mkdir -p "$DATA_DIR"
  chown "$RUN_USER":"$RUN_USER" "$DATA_DIR"
  if [[ ! -f "$REPO/config_local.json" ]]; then
    printf '{\n  "base_directory": "%s"\n}\n' "$DATA_DIR" > "$REPO/config_local.json"
    chown "$RUN_USER":"$RUN_USER" "$REPO/config_local.json"
    echo "wrote $REPO/config_local.json"
  else
    echo "config_local.json exists - set \"base_directory\" in it yourself if needed"
  fi
fi

echo "== Desktop launcher =="
chmod +x "$REPO/jetson/run_weedscan.sh" "$REPO"/*.py || true
DESK="$RUN_HOME/Desktop"
mkdir -p "$DESK"
cat > "$DESK/WeedScan.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=WeedScan ZED 2i
Comment=Field data collection
Exec=$REPO/jetson/run_weedscan.sh
Terminal=true
Icon=camera-video
DESKTOP
chown "$RUN_USER":"$RUN_USER" "$DESK/WeedScan.desktop"
chmod +x "$DESK/WeedScan.desktop"
sudo -u "$RUN_USER" gio set "$DESK/WeedScan.desktop" metadata::trusted true 2>/dev/null || true

echo
echo "Done. Next (as $RUN_USER, camera plugged in, internet available once):"
echo "  python3 $REPO/zed_doctor.py --optimize          # one-time NEURAL model build (5-20 min each)"
echo "  python3 $REPO/zed_doctor.py --bench --out <data dir>   # what this Jetson sustains"
