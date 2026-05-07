#!/usr/bin/env bash
# End-to-end fire-and-forget pipeline for chandra:
#   1. (Re)launch one vLLM server per GPU with high-throughput defaults
#   2. Wait for all servers to become healthy
#   3. Run the sharded client over the input directory
#   4. (Optional) tear down servers when done
#
# Usage:
#   nohup ./scripts_local/run_pipeline.sh <input_dir> <output_dir> > pipeline.log 2>&1 &
#
# Env vars:
#   KEEP_SERVERS=1  -> leave vLLM servers running after the job finishes
#   GPUS="0,1,2,3"  -> only use these GPUs (defaults to all)
#   plus any of MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
#   GPU_MEM_UTIL, MAX_WORKERS_PER_SHARD, MAX_RETRIES, BATCH_SIZE.

set -euo pipefail

INPUT="${1:?usage: $0 <input_dir> <output_dir>}"
OUTPUT="${2:?usage: $0 <input_dir> <output_dir>}"

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_PORT="${BASE_PORT:-8000}"

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    N="${#GPU_LIST[@]}"
else
    N=$(nvidia-smi -L | wc -l)
fi

echo "=== [1/3] Launching $N vLLM server(s) ==="
"$HERE/launch_servers.sh"

echo
echo "=== [2/3] Waiting for servers to become healthy ==="
for ((i=0; i<N; i++)); do
    port=$((BASE_PORT + i))
    url="http://localhost:${port}/v1/models"
    printf "  server $i (port %d)... " "$port"
    for ((t=0; t<300; t++)); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "ready"
            break
        fi
        sleep 2
        if (( t == 299 )); then
            echo "TIMEOUT after 600s"
            echo "Check: sudo docker logs chandra-vllm-${i}"
            exit 1
        fi
    done
done

echo
echo "=== [3/3] Sharded processing: $INPUT -> $OUTPUT ==="
"$HERE/run_sharded.sh" "$INPUT" "$OUTPUT" "$N"

if [[ "${KEEP_SERVERS:-0}" != "1" ]]; then
    echo
    echo "=== Cleanup: stopping vLLM servers (set KEEP_SERVERS=1 to skip) ==="
    STOP=1 "$HERE/launch_servers.sh"
fi

echo
echo "Pipeline complete. Output: $OUTPUT"
