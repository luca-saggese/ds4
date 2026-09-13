# M1 GB10 V4.1 Q2 Cold-Streaming Freeze

- Date: 2026-09-13T19:18:03+02:00
- Base tag: `m0-gb10-base-2026-09-13`
- Base SHA: `458e2277e8105a19f96aa3f761c3c49cb5d38946`
- origin/main observed SHA: `ae226d9dd16dbf31f5e4fc918863bf10fa8b3e77`
- upstream/main observed SHA: `bd66c402070042bf0a79ad6ece8242de4c93680c`
- Branch: `m1/v41-q2-cold`
- M1 specification source: `origin/main:M1_ds4_v41_Q2_cold_SSD_GB10.md`
- Kernel: `Linux gx10-da63 6.17.0-1008-nvidia #8-Ubuntu SMP PREEMPT_DYNAMIC Wed Jan 21 17:56:56 UTC 2026 aarch64`
- OS: Ubuntu 24.04.4 LTS (Noble Numbat)
- CUDA toolkit: CUDA 13.0, nvcc 13.0.88
- NVIDIA driver: 580.126.09
- GPU: NVIDIA GB10
- RAM: 121 GiB visible, 118 GiB available at freeze
- Swap: 15 GiB total, 258 MiB used at freeze
- Compiler: GCC 13.3.0 (`cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`)

## Storage

- NVMe device: `/dev/nvme0n1`, model `ESL01TBTLCZ-27J2-TYN`, `ROTA=0`
- NVMe filesystem: ext4 on `/dev/nvme0n1p2`, mounted at `/`
- NVMe capacity/free: 982,819,848,192 / 369,293,778,944 bytes
- HDD device: `/dev/sda`, model `Expansion HDD`, `ROTA=1`
- HDD filesystem: exFAT on `/dev/sda2`, mounted at `/mnt/seagate`
- HDD capacity/free: 10,000,434,987,008 / 9,047,492,263,936 bytes

The V4.1 artifact is 365,713,686,528 bytes. After moving V4 off the NVMe,
only 369,293,778,944 bytes were free there, which did not provide safe
headroom for the model, build products, logs, OS, and runtime files. V4.1 was
therefore downloaded directly to the verified rotating Seagate filesystem.
M1 runs from this device are described as cold backing-file streaming, not SSD
streaming.

## Models

### DeepSeek V4 regression model

- Path: `/mnt/seagate/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Size: 86,720,111,488 bytes
- SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Backing device: `/dev/sda2` (`Expansion HDD`, `ROTA=1`)
- Backing type: `7200-rpm HDD`
- Filesystem: exFAT

The source and destination SHA-256 values were compared before the original
NVMe copy was removed.

### DeepSeek V4.1 Flash Q2

- Path: `/mnt/seagate/ds4-models/v41/DeepSeek-V4.1-Flash-Q2.gguf`
- Size: 365,713,686,528 bytes
- SHA-256: `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`
- Backing device: `/dev/sda2` (`Expansion HDD`, `ROTA=1`)
- Backing type: `7200-rpm HDD`
- Filesystem: exFAT
- Download target: `ds41f-q2`

`download_model.sh` verified both the exact byte count and SHA-256 before
accepting the artifact.

## Frozen integration inputs

- PR #647 diagnostics: `062d994969885f8647d0915cc4d250ec0fd46c2e`
- PR #1031 selected-expert prefill: `9b8d8fdebe878ae188a315469989bc2bea02272b`
- PR #1036 ROCm V4.1 reference head observed:
  `c89c085f866b21ab5f8a5e4201ca9fca63fd43b6`

Only the first two commits are integration inputs. PR #1036 is a semantic
non-Metal reference and is not integrated wholesale.
