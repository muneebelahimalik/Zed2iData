#!/usr/bin/env bash
# Launch the WeedScan GUI with a log file. Used by the desktop launcher.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HOME/ZED_Data/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/weedscan_$(date +%Y%m%d_%H%M%S).log"
cd "$REPO"
echo "WeedScan log: $LOG"
python3 -u Zed2i_video_datacollection.py 2>&1 | tee "$LOG"
echo "WeedScan exited. Press Enter to close."
read -r _
