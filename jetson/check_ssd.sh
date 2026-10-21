#!/usr/bin/env bash
# Check (and repair) the data-SSD mount used by WeedScan.
#
#   sudo ./jetson/check_ssd.sh                         # uses base_directory from config_local.json
#   sudo ./jetson/check_ssd.sh /media/pcp/Samsung_980_PRO
#
# Checks: the NVMe/SSD is visible, /etc/fstab has a valid entry for the mount point,
# it is mounted now (mounts it if not), it is writable by your user, and that the
# fstab entry will work at boot. Prints exactly what it changed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
ok()   { echo -e "[\e[32m OK \e[0m] $*"; }
warn() { echo -e "[\e[33mWARN\e[0m] $*"; }
fail() { echo -e "[\e[31mFAIL\e[0m] $*"; }
if [[ $EUID -ne 0 ]]; then echo "run with sudo:  sudo $0 $*"; exit 1; fi

# ---- which mount point?
MNT="${1:-}"
if [[ -z "$MNT" ]]; then
  DATA="$(python3 -c "import json;print(json.load(open('$REPO/config_local.json'))['base_directory'])" 2>/dev/null)"
  if [[ -z "$DATA" ]]; then fail "no argument and no base_directory in config_local.json"; exit 1; fi
  # mount point = the /media/<user>/<name> or /mnt/<name> prefix of the data folder
  MNT="$(echo "$DATA" | sed -E 's#^(/media/[^/]+/[^/]+|/mnt/[^/]+|/run/media/[^/]+/[^/]+).*#\1#')"
  echo "data folder: $DATA"
fi
echo "mount point: $MNT"
echo

# ---- 1. the disk
echo "== Disks =="
lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINT | grep -v '^loop\|^zram'
echo

# ---- 2. fstab entry
echo "== /etc/fstab =="
LINE="$(grep -E "^[^#]*[[:space:]]${MNT}[[:space:]]" /etc/fstab || true)"
if [[ -z "$LINE" ]]; then
  fail "no /etc/fstab entry for $MNT"
  echo "      fix: mount the SSD once by hand, then: sudo ./jetson/setup_jetson.sh --data-dir $MNT/ZED_Data --fstab"
else
  echo "  $LINE"
  SPEC="$(echo "$LINE" | awk '{print $1}')"
  DEV="$(findfs "$SPEC" 2>/dev/null || true)"
  if [[ -n "$DEV" ]]; then ok "fstab entry points to an existing device: $SPEC -> $DEV"
  else fail "fstab entry $SPEC does not match any connected disk (SSD missing, or reformatted -> new UUID)"
       echo "      current UUIDs:"; blkid | grep -v 'loop\|zram' | sed 's/^/        /'
       echo "      fix: edit /etc/fstab (sudo nano /etc/fstab) and put the SSD's current UUID in that line"
  fi
  echo "$LINE" | grep -q nofail && ok "'nofail' set (boot does not hang if the SSD is unplugged)" \
                                || warn "no 'nofail' option: the Jetson may stop booting if the SSD is missing"
fi
echo

# ---- 3. mounted now?
echo "== Mount state =="
systemctl daemon-reload 2>/dev/null || true
if findmnt -n "$MNT" >/dev/null; then
  ok "mounted: $(findmnt -n -o SOURCE,FSTYPE,OPTIONS "$MNT")"
else
  warn "$MNT is NOT mounted - trying to mount it from /etc/fstab"
  mkdir -p "$MNT"
  if mount "$MNT"; then ok "mounted now: $(findmnt -n -o SOURCE,FSTYPE "$MNT")"
  else fail "mount failed - see the error above and: dmesg | tail -20"; exit 1; fi
fi
# a second automount by the desktop (udisks) under a different name confuses things
OTHER="$(findmnt -rn -o TARGET -S "$(findmnt -n -o SOURCE "$MNT")" | grep -v "^${MNT}$" || true)"
[[ -n "$OTHER" ]] && warn "the same SSD is ALSO mounted at: $OTHER (harmless, but only record to $MNT)"
echo

# ---- 4. permissions + write test
echo "== Write test as $RUN_USER =="
OWNER="$(stat -c %U "$MNT")"
if [[ "$OWNER" != "$RUN_USER" ]]; then
  warn "$MNT is owned by $OWNER - giving it to $RUN_USER"
  chown "$RUN_USER":"$RUN_USER" "$MNT"
fi
if sudo -u "$RUN_USER" bash -c "f='$MNT/.weedscan_write_test'; echo ok > \"\$f\" && sync && rm \"\$f\""; then
  ok "$RUN_USER can write to $MNT"
else
  fail "$RUN_USER cannot write to $MNT"
fi
df -h "$MNT" | tail -1 | awk '{print "      size " $2 ", used " $3 ", free " $4}'
echo

# ---- 5. will it mount at boot?
echo "== Boot =="
UNIT="$(systemd-escape -p --suffix=mount "$MNT")"
if systemctl cat "$UNIT" >/dev/null 2>&1; then
  ok "systemd generated $UNIT from /etc/fstab: it will be mounted at every boot"
else
  warn "systemd has no mount unit for $MNT yet (check the fstab line above)"
fi
if VERIFY="$(findmnt --verify --tab-file /etc/fstab 2>&1)"; then
  ok "/etc/fstab syntax verified"
else
  warn "findmnt --verify found problems in /etc/fstab:"; echo "$VERIFY" | sed 's/^/        /'
fi
echo
echo "To be sure: sudo reboot, then run  findmnt $MNT  (it must show /dev/nvme0n1p1)."
