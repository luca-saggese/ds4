# M0 GB10 CUDA / SSD-Streaming Report

Date: 2026-09-13

## Result

M0 is complete on NVIDIA DGX Spark / GB10. The frozen DeepSeek V4 model builds
and runs on CUDA in resident and SSD-streaming modes. PR #647 was integrated
commit by commit, and only the selected-expert IQ2/Q2 prefill fix from PR #1031
was integrated. No DeepSeek V4.1 implementation or other out-of-scope feature
was added.

## Frozen revisions

- Branch: `m0/gb10-cache-integration`
- origin/main base: `1fd23042d1f4bc096c155fa721ee3e9cd3ef6551`
- upstream/main observed: `bd66c402070042bf0a79ad6ece8242de4c93680c`
- PR #647 frozen head: `ab9847347fdff2064e5fb4f0762d202a9862d71a`
- PR #1031 observed head: `a52c4a7dc7e996def6736908ae79ec88b774941f`
- Tested implementation SHA before this report: `b91846ba43b814172bc2516bc3ed0328ee1b64a9`
- Local baseline tag: `m0-gb10-base-2026-09-13`

PR #1031 had advanced to four commits when fetched. M0 selected only
`9b8d8fd`; `9abaf4a`, `4eca170`, and `a52c4a7` are excluded.

## Model

- File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Size: 86,720,111,488 bytes
- SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Location during testing: repository root, untracked

## Integrated commits

| Branch SHA | Source | Change |
| --- | --- | --- |
| `458e227` | M0 | Freeze GB10 environment and model manifest |
| `c247326` | M0 | Record baseline build and tests |
| `d0403cf` | PR #647 `062d994` | CUDA SSD-streaming counters |
| `e8f0980` | M0 compatibility | Emit counters from current GPU graph generation paths |
| `a6740c8` | PR #647 `d84d9f0` | Persistent per-(layer, expert) LRU |
| `26183f6` | PR #647 `e59f289` | Pooled CUDA expert buffers |
| `a00d5ce` | PR #647 `ab98473` | Byte-accurate device cache budget |
| `c0bd647` | PR #1031 `9b8d8fd` | Compact selected-expert IQ2/Q2 prefill |
| `b91846b` | M0 observability | Count compact versus full-table IQ2/Q2 MMQ calls |

## Baseline

`make clean && make -j20 cuda-spark` completed successfully. The build retains
one pre-existing signedness warning at `ds4.c:66997`; M0 does not modify that
unrelated code.

`DS4_TEST_MODEL=<model> ./ds4_test` and
`make test-cuda-q8-scratch CUDA_ARCH=sm_121` passed. Metal-only and
missing-optional-model cases were reported as not applicable by the runner.
Invoking bare `./ds4_test` without `DS4_TEST_MODEL` fails because the local
default `ds4flash.gguf` symlink is absent; this is classified as a pre-existing
local configuration issue.

The fixed short prompt used context 1024, temperature 0, a 32-token cap, and
thinking disabled.

| Mode | Prefill | Decode | Output SHA-256 |
| --- | ---: | ---: | --- |
| Resident | 14.49 t/s | 10.69 t/s | `acc525cb0c01ddc54f684286ff25341f2d13f252dc02efc07231b6546742b8a8` |
| SSD cold run 1 | 0.53 t/s | 1.01 t/s | `6f60887f51010b1485e0cd6dd16f201990d4abd0bc5ed8353d334992b44999c0` |
| SSD cold run 2 | 0.50 t/s | 1.04 t/s | `6f60887f51010b1485e0cd6dd16f201990d4abd0bc5ed8353d334992b44999c0` |

The two SSD-cold outputs are byte-identical and define the M0 SSD correctness
golden. Resident and SSD output differ at the frozen baseline; all subsequent
short SSD cache runs match the SSD golden exactly.

## PR #647 validation

Before the LRU, counters reported:

```text
hits=0 misses=10417 bytes_from_file=68.667 GiB bytes_from_cache=0.000 GiB
```

With the persistent LRU and an explicit 20 GiB request:

```text
hits=6430 misses=3987 hit_rate=0.617
bytes_from_file=26.281 GiB bytes_from_cache=42.385 GiB
```

The 1 GiB small-cache test resolved to one effective entry and produced 10,417
misses without a crash, allocation failure, CUDA error, or output drift. This
exercised continuous eviction and pooled-buffer reuse.

### Byte-accurate budget matrix

All rows produced the SSD golden SHA-256
`6f60887f51010b1485e0cd6dd16f201990d4abd0bc5ed8353d334992b44999c0`.
The runtime reserves 3.38 GiB of prefill headroom before deriving the effective
expert-cache budget.

| Requested | Effective budget | Counted | Pool parked | Device total | Hits / misses | Hit rate | File / cache bytes | Prefill / decode | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 GiB | 16.625 GiB | 16.625 GiB | 0 GiB | 16.625 GiB | 6430 / 3987 | 0.617 | 26.281 / 42.385 GiB | 0.41 / 2.48 t/s | 1,349,976 KiB |
| 48 GiB | 44.620 GiB | 24.772 GiB | 0 GiB | 24.772 GiB | 6659 / 3758 | 0.639 | 24.772 / 43.895 GiB | 0.40 / 1.10 t/s | 1,355,684 KiB |
| 64 GiB | 60.625 GiB | 24.772 GiB | 0 GiB | 24.772 GiB | 6659 / 3758 | 0.639 | 24.772 / 43.895 GiB | 0.45 / 2.92 t/s | 1,362,360 KiB |
| 70 GiB | 66.623 GiB | 24.772 GiB | 0 GiB | 24.772 GiB | 6659 / 3758 | 0.639 | 24.772 / 43.895 GiB | 0.47 / 2.72 t/s | 1,362,600 KiB |
| 80 GiB | 72.562 GiB | 24.772 GiB | 0 GiB | 24.772 GiB | 6659 / 3758 | 0.639 | 24.772 / 43.895 GiB | 0.48 / 3.03 t/s | 1,362,196 KiB |

Every measured `device_total` is at or below the effective byte budget. After
the 80 GiB run the host reported 118 GiB available RAM and no new swap growth
attributable to the run.

## PR #1031 selected-expert prefill validation

A 2,500-byte tracked long-prompt fixture was run with context 1024, 32 generated
tokens, SSD cold mode, and a 20 GiB cache request.

```text
prefill=10.04 t/s
decode=1.82 t/s
selected_mmq_compact_calls=43
selected_mmq_full_table_calls=0
hits=5574
misses=9486
bytes_from_file=62.530 GiB
bytes_from_cache=36.743 GiB
device_total=16.625 GiB
output_sha256=75568a8bc05655dae295c1ac5b454e249571c9f4542ee18e918008e031c5fcea
```

The counters demonstrate that every observed IQ2/Q2 MMQ prefill layer used the
compact selected-expert staging and none resolved the full MoE expert tables.

## Scope confirmation

Not integrated:

- PR #605 or any of its upload/cache commits
- PR #589
- PR #1031 commits after `9b8d8fd`
- DeepSeek V4.1 CUDA
- Engram, predictor, mixed per-expert quantization, neural prefetch
- multi-GPU, multi-Spark, MTP, or vision work

No M0 run reported OOM, deadlock, illegal CUDA access, or deterministic SSD
output drift.
