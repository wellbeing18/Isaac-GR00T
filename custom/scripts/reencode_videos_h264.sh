#!/bin/bash
#
# Re-encode dataset videos from AV1 to H.264 for faster training
#
# AV1 codec is very efficient for storage but CPU-intensive to decode.
# H.264 has hardware decode support and is much faster for ML training.
#
# Usage:
#   bash reencode_videos_h264.sh /path/to/dataset
#   bash reencode_videos_h264.sh /path/to/dataset --dry-run  # Preview only
#
# Expected speedup: 15-25% faster training
#

set -e

DATASET_PATH="${1:?Usage: $0 /path/to/dataset [--dry-run]}"
DRY_RUN=false

if [[ "$2" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be modified ==="
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

# Find videos directory
VIDEOS_DIR="$DATASET_PATH/videos"
if [[ ! -d "$VIDEOS_DIR" ]]; then
    echo "ERROR: Videos directory not found: $VIDEOS_DIR"
    exit 1
fi

echo "========================================"
echo "Re-encode Videos: AV1 → H.264"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo ""

# Count videos
TOTAL_VIDEOS=$(find "$VIDEOS_DIR" -name "*.mp4" | wc -l)
echo "Found $TOTAL_VIDEOS video files"
echo ""

# Check current codec of first video
FIRST_VIDEO=$(find "$VIDEOS_DIR" -name "*.mp4" | head -1)
if [[ -n "$FIRST_VIDEO" ]]; then
    CURRENT_CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$FIRST_VIDEO" 2>/dev/null)
    echo "Current codec: $CURRENT_CODEC"

    if [[ "$CURRENT_CODEC" == "h264" ]]; then
        echo "Videos are already H.264 encoded. Nothing to do."
        exit 0
    fi
fi

echo ""
echo "This will:"
echo "  1. Re-encode all .mp4 files to H.264"
echo "  2. Save originals as .mp4.av1_backup"
echo "  3. Replace originals with H.264 versions"
echo ""

if [[ "$DRY_RUN" == false ]]; then
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Process each video
PROCESSED=0
FAILED=0
SKIPPED=0

while IFS= read -r VIDEO_FILE; do
    # Check codec
    CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$VIDEO_FILE" 2>/dev/null)

    if [[ "$CODEC" == "h264" ]]; then
        ((SKIPPED++))
        continue
    fi

    BACKUP_FILE="${VIDEO_FILE}.av1_backup"
    TEMP_FILE="${VIDEO_FILE}.h264_temp.mp4"

    echo -n "[$((PROCESSED + FAILED + 1))/$TOTAL_VIDEOS] $(basename "$VIDEO_FILE")... "

    if [[ "$DRY_RUN" == true ]]; then
        echo "would re-encode ($CODEC → h264)"
        ((PROCESSED++))
        continue
    fi

    # Re-encode to H.264
    # -c:v libx264: H.264 codec
    # -preset fast: Good balance of speed/quality
    # -crf 18: High quality (lower = better, 18-23 is good)
    # -pix_fmt yuv420p: Standard pixel format
    if ffmpeg -y -hide_banner -loglevel error \
        -i "$VIDEO_FILE" \
        -c:v libx264 \
        -preset fast \
        -crf 18 \
        -pix_fmt yuv420p \
        -an \
        "$TEMP_FILE" 2>/dev/null; then

        # Backup original and replace
        mv "$VIDEO_FILE" "$BACKUP_FILE"
        mv "$TEMP_FILE" "$VIDEO_FILE"

        # Get file sizes for reporting
        ORIG_SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE" 2>/dev/null)
        NEW_SIZE=$(stat -f%z "$VIDEO_FILE" 2>/dev/null || stat -c%s "$VIDEO_FILE" 2>/dev/null)

        echo "done ($(numfmt --to=iec $ORIG_SIZE) → $(numfmt --to=iec $NEW_SIZE))"
        ((PROCESSED++))
    else
        echo "FAILED"
        rm -f "$TEMP_FILE"
        ((FAILED++))
    fi

done < <(find "$VIDEOS_DIR" -name "*.mp4" -type f | sort)

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "  Processed: $PROCESSED"
echo "  Skipped (already H.264): $SKIPPED"
echo "  Failed: $FAILED"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "✅ All videos re-encoded successfully!"
    echo ""
    echo "Backups saved as *.av1_backup"
    echo "To remove backups and save space:"
    echo "  find $VIDEOS_DIR -name '*.av1_backup' -delete"
else
    echo "⚠️  Some videos failed to re-encode. Check the output above."
fi
