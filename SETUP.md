# Chandra OCR — vLLM (server + client) setup on this machine

This document records **exactly** what was installed, where, and why, on this
host (`instance-20260420-202919-a100x8`) to run Chandra OCR 2 with the vLLM
backend. It is intentionally version-pinned to avoid the dependency drift that
caused trouble on a previous attempt.

## Host inventory (snapshot at install time on this machine)

- OS: Debian-based linux, kernel `6.1.0-44-cloud-amd64`
- GPUs: **8× NVIDIA A100-SXM4-40GB** (driver 575.57.08, CUDA 12.9)
- Docker: 20.10.24 (already installed; **no `nvidia` runtime registered**)
- Conda: miniforge3, conda 26.1.1, mamba available
- Repo: `/home/jimmys/chandra` (cloned from `https://github.com/datalab-to/chandra.git`)
- `chandra/.python-version` → `3.12`

> **Reproducing on a fresh box?** The "Install commands — from absolute zero"
> section below covers everything from a bare Debian/Ubuntu install with just
> an NVIDIA GPU, including the driver, Docker, the NVIDIA container runtime,
> Miniforge, the repo, the vLLM image, the conda env, and pinned Python deps.
> Each step has a "skip if…" so you can jump past anything already present.

## Why this layout

Chandra has two backends: `hf` (local HuggingFace, requires torch) and `vllm`
(talks to a vLLM OpenAI-compatible server over HTTP). We use **vllm**.

The repo's `chandra_vllm` script does **not** `pip install vllm`. It runs the
official Docker image:

```
chandra/scripts/vllm.py:77 → vllm/vllm-openai:v0.17.0
```

So vLLM with all its torch/CUDA/xformers/flashinfer constraints lives entirely
inside the Docker image. The Python env on the host is **only** the lightweight
client (the `chandra` CLI + an `openai` HTTP client). This separation is the
whole point: it sidesteps the dependency hell of installing vLLM via pip.

## What gets installed

### A. System (apt) — requires sudo
| Package | Version | Why |
|---|---|---|
| `nvidia-container-toolkit` | latest from NVIDIA repo | Registers the `nvidia` runtime in Docker so `docker run --runtime nvidia` (which `chandra_vllm` invokes) can expose GPUs to containers. Tracks the host driver, so pinning is unnecessary. |

Daemon change: `nvidia-ctk runtime configure --runtime=docker` (edits
`/etc/docker/daemon.json`) + `systemctl restart docker`.

### B. Docker images
| Image | Tag | Why this exact tag |
|---|---|---|
| `vllm/vllm-openai` | `v0.17.0` | Hard-pinned in `chandra/scripts/vllm.py`. Self-contained — its own torch/CUDA/xformers/flashinfer. |

### C. Conda env: `chandra-vllm` (Python 3.12)
Pinned versions taken from chandra's own `uv.lock` (the maintainers' tested
set) and the lower bounds in `pyproject.toml`. **No torch, no transformers,
no vllm.**

| Package | Version | Source |
|---|---|---|
| `chandra-ocr` | `0.2.0` (editable, `--no-deps`) | local `/home/jimmys/chandra` |
| `openai` | `2.2.0` | pyproject lower bound |
| `pydantic` | `2.12.0` | pyproject lower bound |
| `pydantic-settings` | `2.11.0` | pyproject lower bound |
| `pypdfium2` | `4.30.0` | pyproject lower bound |
| `pillow` | `10.4.0` | last stable 10.x (pyproject `>=10.2.0`) |
| `beautifulsoup4` | `4.14.2` | pyproject lower bound |
| `markdownify` | `1.1.0` | pyproject (already exact) |
| `click` | `8.1.7` | latest stable 8.x (pyproject `>=8.0.0`) |
| `filetype` | `1.2.0` | pyproject lower bound |
| `python-dotenv` | `1.1.1` | pyproject lower bound |
| `six` | `1.17.0` | pyproject lower bound |

### D. NOT installed (intentionally)
- `vllm` (pip) — lives only inside the Docker image
- `torch`, `torchvision`, `transformers`, `accelerate` — HF backend only, not used
- `flash-attn` — HF backend only
- `streamlit` — only needed for `chandra_app`

## Install commands — from absolute zero

The steps below assume a fresh Debian/Ubuntu host with **only an OS and an
NVIDIA GPU**. Every step has a "skip if…" check at the top so you can jump
past anything you already have. Tested on Debian 12 / Ubuntu 22.04+ with
A100-class GPUs.

> All commands assume your user has `sudo` privileges. Run them in order.

### 0. Verify you have an NVIDIA GPU
```bash
lspci | grep -i nvidia
# expect a line like: 00:04.0 3D controller: NVIDIA Corporation GA100 ...
```
If nothing prints, you don't have an NVIDIA GPU and chandra-vllm cannot run.

### 1. Install the NVIDIA driver
**Skip if** `nvidia-smi` already prints a table of GPUs.

```bash
# Pick the latest stable driver from your distro
sudo apt-get update
sudo apt-get install -y nvidia-driver firmware-misc-nonfree   # Debian
# or, on Ubuntu:
# sudo ubuntu-drivers install
sudo reboot
```
After reboot, verify:
```bash
nvidia-smi   # should show your GPU(s), driver version, CUDA version
```
Driver must be **>= 525** for vLLM v0.17 (CUDA 12.x). Driver 575 (this host) is fine.
You do **not** need to install CUDA or cuDNN on the host — they live inside the Docker image.

### 2. Install Docker Engine
**Skip if** `docker --version` prints something.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```
Verify:
```bash
sudo docker run --rm hello-world   # should print "Hello from Docker!"
```
Optional (avoid `sudo` for every docker command — re-login afterwards):
```bash
sudo usermod -aG docker "$USER"
```

### 3. Install NVIDIA Container Toolkit (gives Docker GPU access)
**Skip if** `docker info | grep -i runtimes` already shows `nvidia`.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Verify:
```bash
docker info | grep -i runtimes        # should now include 'nvidia'
sudo docker run --rm --runtime nvidia --gpus device=0 \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# expect to see GPU 0 listed from inside the container
```

### 4. Install Miniforge (conda + mamba)
**Skip if** `conda --version` already works.

```bash
curl -fsSL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
exec bash    # reload shell so conda is on PATH (or open a new terminal)
```
Verify:
```bash
conda --version    # e.g. 26.x
mamba --version    # ships with miniforge
```

### 5. Install git and clone the chandra repo
**Skip if** `/home/jimmys/chandra/pyproject.toml` already exists.

```bash
sudo apt-get install -y git
mkdir -p /home/jimmys
cd /home/jimmys
git clone https://github.com/datalab-to/chandra.git
cd chandra
cat .python-version    # should print 3.12 — confirms target Python version
```

### 6. Pull the pinned vLLM image
**Skip if** `sudo docker images vllm/vllm-openai:v0.17.0` already lists it.

```bash
sudo docker pull vllm/vllm-openai:v0.17.0   # ~20 GB; takes several minutes
```

### 7. Create the conda env
**Skip if** `conda env list` already shows `chandra-vllm`.

```bash
conda create -n chandra-vllm python=3.12 -y
conda activate chandra-vllm   # 'mamba activate' won't work unless mamba shell is initialized; conda is already initialized in this shell
```

### 8. Install pinned Python deps
```bash
cd /home/jimmys/chandra
pip install --no-deps -e .
pip install \
  "openai==2.2.0" \
  "pydantic==2.12.0" \
  "pydantic-settings==2.11.0" \
  "pypdfium2==4.30.0" \
  "pillow==10.4.0" \
  "beautifulsoup4==4.14.2" \
  "markdownify==1.1.0" \
  "click==8.1.7" \
  "filetype==1.2.0" \
  "python-dotenv==1.1.1" \
  "six==1.17.0"
```
Verify:
```bash
which chandra chandra_vllm
# /home/jimmys/miniforge3/envs/chandra-vllm/bin/chandra
# /home/jimmys/miniforge3/envs/chandra-vllm/bin/chandra_vllm
chandra --help
```

### 9. Make the helper scripts executable
**Skip if** they're already `+x`.

```bash
chmod +x /home/jimmys/chandra/scripts_local/launch_servers.sh
chmod +x /home/jimmys/chandra/scripts_local/run_sharded.sh
chmod +x /home/jimmys/chandra/scripts_local/run_pipeline.sh
```

### 10. Smoke test (single GPU, single PDF)
```bash
conda activate chandra-vllm
# launch one server on GPU 0
/home/jimmys/chandra/scripts_local/launch_servers.sh GPUS="0"
# wait until "Application startup complete." appears in the log
sudo docker logs -f chandra-vllm-0
# Ctrl-C to detach (server keeps running)

# in the same shell (or a new one with conda activate chandra-vllm):
chandra /path/to/some/file.pdf /tmp/chandra_smoketest --method vllm
ls /tmp/chandra_smoketest/   # should contain a subfolder per processed file
```
If that produces a `.md` file, **the entire stack works**. From here, see
"Scaling: many GPUs over a big folder of PDFs" below.

## Running it

### Start the server (one A100-40GB)
```bash
conda activate chandra-vllm
chandra_vllm --gpu a100-40
```
Listens on `http://localhost:8000/v1`, served model name `chandra`.
First run will download model weights `datalab-to/chandra-ocr-2` to
`~/.cache/huggingface` (mounted into the container).

To use a different GPU: `VLLM_GPUS=3 chandra_vllm --gpu a100-40`.

### Run the client (in another terminal)
```bash
conda activate chandra-vllm
chandra input.pdf ./output --method vllm
```

## Scaling: many GPUs over a big folder of PDFs

The bundled `chandra` CLI processes files **strictly sequentially** (one file
at a time, with within-file page batching). Talking to a single vLLM server it
will only ever saturate one GPU. To use all 8 A100s on a directory of PDFs,
shard the file list across 8 independent servers — this beats tensor-parallel
for an OCR-sized model.

Two helper scripts are provided in `scripts_local/`:

### Tuned defaults (mirroring known-good olmocr settings)

| Param | Value | Purpose |
|---|---|---|
| `--gpu-memory-utilization` | `0.92` | use as much VRAM as possible |
| `--max-model-len` | `24576` | room for very dense pages (math/tables); cap at 32768 |
| `--max-num-seqs` | `40` | concurrent sequences per server (lowered to fit larger context in 40GB) |
| `--max-num-batched-tokens` | `8192` | bigger prefill batches |
| client `--max-workers` per shard | `16` | 8 shards × 16 = 128 in-flight requests total (matches olmocr `--workers 128`) |
| client `--max-retries` | `3` | survive transient vLLM hiccups |
| client `--batch-size` | `28` | chandra default for vllm |

All overridable via env var (see script headers).

### Launch one server per GPU
```bash
cd /home/jimmys/chandra
./scripts_local/launch_servers.sh
# launches chandra-vllm-0..7 on ports 8000..8007, one per GPU
# (containers run with --restart unless-stopped, so they survive shell exit)
```

Verify they're up:
```bash
sudo docker ps --filter name=chandra-vllm-
sudo docker logs -f chandra-vllm-0   # watch first server boot
```

Stop them all:
```bash
STOP=1 ./scripts_local/launch_servers.sh
```

Tunable env vars: `GPUS="0,1,2,3"` (subset), `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL`, `BASE_PORT`. See script header.

### Process a directory across all servers
```bash
conda activate chandra-vllm

nohup ./scripts_local/run_sharded.sh \
    /path/to/pdfs \
    /path/to/out \
    > /home/jimmys/dataset_run.log 2>&1 &
disown
#./scripts_local/run_sharded.sh /path/to/pdfs /path/to/out
# optional 3rd arg = number of servers; default = number of GPUs
# any further args are forwarded to chandra (e.g. --no-images)
```

### Fire-and-forget end-to-end pipeline (recommended for big jobs)

Launches servers, waits for them to be healthy, runs sharded processing, then
tears down. Use exactly like the olmocr `nohup` invocation:

```bash
conda activate chandra-vllm
nohup ./scripts_local/run_pipeline.sh \
    /home/jimmys/datasets/archetai \
    /home/jimmys/datasets/archetai_chandra_out \
    > /home/jimmys/chandra_archetai.log 2>&1 &
```

Env vars:
- `KEEP_SERVERS=1` — don't tear down servers after the job (keep them warm for
  another run).
- Any of `GPUS`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`,
  `GPU_MEM_UTIL`, `MAX_WORKERS_PER_SHARD`, `MAX_RETRIES`, `BATCH_SIZE`.

Watch progress:
```bash
tail -f /home/jimmys/chandra_archetai.log              # pipeline-level
tail -f /path/to/out/.shard_0.log                      # one worker
watch -n 2 nvidia-smi                                  # GPU utilization
```

The script symlinks the file list into N round-robin shards, launches N
`chandra` clients each pointing at its own server via `VLLM_API_BASE`, and
waits for all of them. Per-shard logs are at `<output>/.shard_<i>.log`.

### Trade-offs
- **N independent servers (this approach)** — best for many small/medium PDFs.
  Near-linear scaling because file-level parallelism is what matters when
  chandra's client iterates serially.
- **Single tensor-parallel server (`--tensor-parallel-size 8`)** — usually
  worse for OCR-sized models. Only worth it if a single GPU can't fit your
  desired context/batch settings.
- **Single server + nginx/litellm load balancer** — clean one URL, but a
  single chandra client process won't keep N backends busy unless individual
  PDFs are very large.

## What this changes on your system

1. New apt package: `nvidia-container-toolkit` (+ libs).
2. New apt source list: `/etc/apt/sources.list.d/nvidia-container-toolkit.list`.
3. New keyring: `/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg`.
4. `/etc/docker/daemon.json` modified to register the `nvidia` runtime.
5. Docker daemon restarted.
6. New Docker image cached locally: `vllm/vllm-openai:v0.17.0`.
7. New conda env: `~/miniforge3/envs/chandra-vllm`.
8. Editable install of `chandra-ocr` pointing at `/home/jimmys/chandra`.

Nothing else on the host is modified. No global pip installs, no system Python
changes, no changes to other conda envs.

## Uninstall / rollback

```bash
conda env remove -n chandra-vllm
sudo docker rmi vllm/vllm-openai:v0.17.0
sudo apt-get remove -y nvidia-container-toolkit
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list \
        /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# Optional: revert /etc/docker/daemon.json and: sudo systemctl restart docker
```

## Install log

- [x] 1. nvidia-container-toolkit installed (apt)
- [x] 2. nvidia runtime registered in docker (`/etc/docker/daemon.json` updated, daemon restarted; `docker info` shows `Runtimes: ... nvidia ...`)
- [x] 3. nvidia runtime smoke test passes (`nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L` → `GPU 0: NVIDIA A100-SXM4-40GB`)
- [x] 4. vllm/vllm-openai:v0.17.0 pulled (image size 20.7GB, image id `700d8ac4f37a`)
- [x] 5. conda env `chandra-vllm` created (python 3.12.x, miniforge3)
- [x] 6. pinned deps installed (no resolver conflicts; full version list in `pip list` of the env)
- [x] 7. `chandra` and `chandra_vllm` on PATH at `/home/jimmys/miniforge3/envs/chandra-vllm/bin/`