#!/usr/bin/env bash
# 在本机 Mac 跑：把 AutoDL (autodl-hand) 上的灵巧手视频拉回本地
# 用法：bash fetch_videos_b.sh
set -e

LOCAL_DIR="/Users/jinc_air/Documents/resume/demo/assets/videos/hand"
REMOTE_HOST="autodl-hand"
REMOTE_LOGDIR="/root/autodl-tmp/logs"

mkdir -p "$LOCAL_DIR"

echo "=== Pulling Leap Hand videos from ${REMOTE_HOST} ==="
sshpass -p 'XKvA55KdrnD5' rsync -avP --include="*/" --include="*.mp4" --exclude="*" \
    -e "ssh -p 26364 -o StrictHostKeyChecking=no" \
    root@connect.bjb2.seetacloud.com:${REMOTE_LOGDIR}/ \
    "$LOCAL_DIR/" 2>/dev/null || echo "(no videos yet)"

echo "=== Done ==="
find "$LOCAL_DIR" -name "*.mp4" -mtime -1
