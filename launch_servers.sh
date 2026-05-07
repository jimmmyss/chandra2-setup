#!/usr/bin/env bash
# Launch one vLLM server per GPU. Each server gets its own port (8000+i)
# and its own container name (chandra-vllm-i) so they can be managed independently.
#
# Usage:
#   ./launch_servers.sh                # launches on GPUs 0..N-1, where N = num GPUs
#   GPUS="0,2,5" ./launch_servers.sh   # launches on a custom set of GPUs
#   STOP=1 ./launch_servers.sh         # stops all chandra-vllm-* containers
#
# Tunables (env vars, with sensible defaults for A100-40GB):
#   MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, GPU_MEM_UTIL,
#   IMAGE (vllm image tag), MODEL, SERVED_NAME, BASE_PORT.

set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:v0.17.0}"
MODEL="${MODEL:-datalab-to/chandra-ocr-2}"
SERVED_NAME="${SERVED_NAME:-chandra}"
BASE_PORT="${BASE_PORT:-8000}"

# A100-40GB high-throughput defaults (mirroring tuned olmocr settings:
#   data-parallel=8, gpu-memory-utilization=0.92, max_model_len=16384,
#   ~16 in-flight requests per server -> 128 total across 8 GPUs).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-40}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

if [[ "${STOP:-0}" == "1" ]]; then
    echo "Stopping all chandra-vllm-* containers..."
    sudo docker ps -a --filter "name=chandra-vllm-" --format "{{.Names}}" \
      | xargs -r sudo docker rm -f
    exit 0
fi

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    NUM=$(nvidia-smi -L | wc -l)
    GPU_LIST=()
    for ((i=0; i<NUM; i++)); do GPU_LIST+=("$i"); done
fi

echo "Launching ${#GPU_LIST[@]} vLLM server(s): ${GPU_LIST[*]}"

for idx in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$idx]}"
    port=$((BASE_PORT + idx))
    name="chandra-vllm-${idx}"

    sudo docker rm -f "$name" >/dev/null 2>&1 || true

    echo "  GPU $gpu  ->  http://localhost:${port}/v1   (container: $name)"

    sudo docker run -d \
        --name "$name" \
        --runtime nvidia \
        --gpus "device=${gpu}" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -p "${port}:8000" \
        --ipc=host \
        --restart unless-stopped \
        "$IMAGE" \
        --model "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --no-enforce-eager \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
        > /dev/null
done

echo
echo "All servers launched. Tail logs with:  sudo docker logs -f chandra-vllm-0"
echo "Stop all with:                          STOP=1 $0"
