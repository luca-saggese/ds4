# M0 Baseline Results

## Frozen inputs

- Git SHA: `458e227` (freeze manifest commit)
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Model SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Backend: CUDA on NVIDIA GB10 (`sm_121a`)
- Prompt: `Explain in one short sentence why the sky appears blue.`
- Generation: context 1024, temperature 0, 32-token cap, thinking disabled

## Build and tests

| Check | Result | Classification |
| --- | --- | --- |
| `make clean && make -j20 cuda-spark` | PASS | Baseline |
| CUDA build warning at `ds4.c:66997` | Present | PRE-EXISTING |
| Bare `./ds4_test` using missing default `ds4flash.gguf` | FAIL | PRE-EXISTING / local configuration |
| `DS4_TEST_MODEL=... ./ds4_test` | PASS | Baseline |
| `make test-cuda-q8-scratch CUDA_ARCH=sm_121` | PASS | Baseline |
| Metal-only and optional-model tests | Skipped by test runner | NOT APPLICABLE |

## Deterministic generation

| Mode | Prefill | Decode | Output SHA-256 | Result |
| --- | ---: | ---: | --- | --- |
| Resident | 14.49 t/s | 10.69 t/s | `acc525cb0c01ddc54f684286ff25341f2d13f252dc02efc07231b6546742b8a8` | PASS |
| SSD cold, run 1 | 0.53 t/s | 1.01 t/s | `6f60887f51010b1485e0cd6dd16f201990d4abd0bc5ed8353d334992b44999c0` | PASS |
| SSD cold, run 2 | 0.50 t/s | 1.04 t/s | `6f60887f51010b1485e0cd6dd16f201990d4abd0bc5ed8353d334992b44999c0` | PASS |

The two independent SSD-cold runs are byte-identical. This SSD-cold output is
the M0 correctness reference for subsequent cache and staging integrations.

Resident and SSD-cold output differ at baseline despite both being greedy.
The difference is recorded rather than modified in M0; cache integrations must
match the frozen SSD-cold reference on the SSD-streaming path.

No baseline run reported OOM, deadlock, or illegal CUDA access.
