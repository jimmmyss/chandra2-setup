#!/usr/bin/env bash
# Process a directory of PDFs/images by sharding the file list across N
# already-running vLLM servers (launched by launch_servers.sh).
#
# Usage:
#   ./run_sharded.sh <input_dir> <output_dir> [num_servers] [extra chandra args...]
#
# Example:
#   ./run_sharded.sh ~/datasets/ecclesia/raw ~/datasets/ecclesia/out 8 --no-images
#
# Assumes servers are at http://localhost:8000/v1 ... http://localhost:(8000+N-1)/v1.
# Run inside the chandra-vllm conda env (so `chandra` is on PATH).

set -euo pipefail

INPUT="${1:?usage: $0 <input_dir> <output_dir> [num_servers] [extra args...]}"
OUTPUT="${2:?usage: $0 <input_dir> <output_dir> [num_servers] [extra args...]}"
N="${3:-$(nvidia-smi -L | wc -l)}"
shift 3 || shift 2 || true
EXTRA=("$@")

BASE_PORT="${BASE_PORT:-8000}"

# Per-shard chandra defaults — tuned to match olmocr's "--workers 128" total
# concurrency: 8 shards * MAX_WORKERS_PER_SHARD = 128 in-flight requests.
MAX_WORKERS_PER_SHARD="${MAX_WORKERS_PER_SHARD:-16}"
MAX_RETRIES="${MAX_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-28}"

mkdir -p "$OUTPUT"

# Build sorted file list of supported types
mapfile -t FILES < <(find "$INPUT" -maxdepth 1 -type f \
    \( -iname '*.pdf' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
       -o -iname '*.gif' -o -iname '*.webp' -o -iname '*.tiff' -o -iname '*.bmp' \) \
    | sort)

TOTAL="${#FILES[@]}"
if (( TOTAL == 0 )); then
    echo "No supported files in $INPUT"; exit 1
fi

# Skip files whose output already exists (resume support).
# Set SKIP_EXISTING=0 to reprocess everything.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TODO=()
SKIPPED=0
if [[ "$SKIP_EXISTING" == "1" ]]; then
    for f in "${FILES[@]}"; do
        stem="$(basename "$f")"; stem="${stem%.*}"
        if [[ -f "$OUTPUT/$stem/$stem.md" ]]; then
            SKIPPED=$((SKIPPED + 1))
        else
            TODO+=("$f")
        fi
    done
    echo "Resume mode: $SKIPPED already done, $((${#TODO[@]})) remaining (of $TOTAL)."
else
    TODO=("${FILES[@]}")
fi

REMAINING="${#TODO[@]}"
if (( REMAINING == 0 )); then
    echo "Nothing to do — all files already have output. (Set SKIP_EXISTING=0 to force reprocess.)"
    exit 0
fi

echo "Sharding $REMAINING files across $N server(s)..."

# Create a temp directory of N shard subdirs containing symlinks (so chandra
# treats each shard as a normal input directory).
WORK="$(mktemp -d -t chandra-shard-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

for ((i=0; i<N; i++)); do
    mkdir -p "$WORK/shard_$i"
done

for ((idx=0; idx<REMAINING; idx++)); do
    shard=$((idx % N))
    ln -s "${TODO[$idx]}" "$WORK/shard_${shard}/"
done

# Launch one chandra client per shard, each pointed at its dedicated server
PIDS=()
for ((i=0; i<N; i++)); do
    port=$((BASE_PORT + i))
    log="$OUTPUT/.shard_${i}.log"
    echo "  shard $i -> http://localhost:${port}/v1   ($(ls "$WORK/shard_$i" | wc -l) files)   log: $log"
    VLLM_API_BASE="http://localhost:${port}/v1" \
      chandra "$WORK/shard_$i" "$OUTPUT" \
        --method vllm \
        --max-workers "$MAX_WORKERS_PER_SHARD" \
        --max-retries "$MAX_RETRIES" \
        --batch-size "$BATCH_SIZE" \
        "${EXTRA[@]}" \
      > "$log" 2>&1 &
    PIDS+=($!)
done

echo
echo "Workers PIDs: ${PIDS[*]}"
echo "Tail any worker:  tail -f $OUTPUT/.shard_0.log"
echo "Waiting for completion..."

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done

if (( FAIL > 0 )); then
    echo "WARNING: $FAIL shard(s) reported errors. Check $OUTPUT/.shard_*.log"
    exit 1
fi
echo "Done. Output in $OUTPUT"
