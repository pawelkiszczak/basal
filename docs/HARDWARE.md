# Hardware: which mode on which GPU

All numbers are for **basal-1.0-4.5B**, one decision = **both option orders** (the default), batch size 1 unless
stated. *dec/s*: two-order decisions per second with 32 option-order passes per forward (offline) or 32 concurrent
clients (HTTP). *Agreement*: share of decisions whose top option equals the fp32 reference on the same machine.
Measured with `basal-bench` / `basal-loadtest` (H100, RTX PRO 6000 and RTX 5090 with the equivalent research harness)
on a private 500-item sample of the Polish/English test split.

## Recommendation

| GPU class | recommended mode | why |
|---|---|---|
| Data-centre (B300, B200, H100, H200) | `fast` (bf16) | the model is limited by kernel launches at batch 1; quantisation adds work |
| Workstation / consumer (RTX PRO 6000, RTX 5090, RTX 4090) | `fp8` | compute-bound; FP8 gives 1.3–1.4× at 97% agreement |
| Desktop with unified memory (DGX Spark GB10) | `fp8` | memory-bandwidth-bound; FP8 halves latency |
| Batched high-throughput serving on Blackwell | `vllm` + `-NVFP4` checkpoint | up to 3× throughput, but −3 accuracy points |
| A100 and other GPUs without FP8 | `fast` | bf16 path works on any sm80+ GPU |

Add `--mode fast-exit` to let requests choose `"early_exit": "0.99"` (about 1.15× faster, agreement ≥ 0.99).

## Measured

| GPU | mode | 2 orders (ms) | dec/s | agreement | accuracy |
|---|---|---|---|---|---|
| **B300 SXM6** | `fast` (bf16) | **8.8** | 109 | 1.000 | 0.822 |
| | `fp8` | 9.7 | 102 | 0.974 | 0.824 |
| | `nvfp4` (torchao) | 13.9 | 63 | 0.904 | 0.790 |
| | `fast-exit`, 0.99 | **7.7** | **119** | 0.998 | 0.824 |
| | HTTP `fast` (p50) | **9.7** | 109 | – | – |
| | HTTP `fp8` (p50) | 10.4 | **119** | – | – |
| **H100 80GB** | `fast` (bf16) | 12.5 | 63 | 0.993 | 0.826 |
| | `fp8` | 11.3 | 80 | 0.976 | 0.830 |
| | `fast-exit`, 0.99 | 10.5 | – | 0.992 | – |
| | HTTP `fast` (p50) | 14.1 | 63 | – | – |
| | HTTP `fast-exit`, 0.99 (p50) | 12.1 | 77 | – | – |
| **RTX PRO 6000 Blackwell** | `fast` (bf16) | 18.9 | 39 | 0.994 | 0.822 |
| | `fp8` | 14.9 | 58 | 0.968 | 0.840 |
| | HTTP `fast` (p50) | 22.5 | 39 | – | – |
| **RTX 5090** | `fast` (bf16) | 27.3 | 24 | 0.999 | 0.824 |
| | `fp8` | 19.4 | 40 | 0.968 | 0.828 |
| | HTTP `fast` (p50) | 32.3 | 23 | – | – |
| **DGX Spark (GB10)** | `eager` fp32 | 800.4 | 0.4 | reference | 0.822 |
| | `fast` (bf16) | 92.0 | 7.0 | 0.996 | 0.826 |
| | `fp8` | **44.6** | 8.3 | 0.970 | 0.836 |
| | `nvfp4` (torchao) | 41.0 | 10.4 | 0.910 | 0.792 |
| | `vllm` + NVFP4 checkpoint | 46.1 | **26.6** | 0.896¹ | 0.792 |
| | `fast-exit`, 0.99 | 76.9 | 7.6 | 0.994 | 0.828 |

¹ agreement with bf16 (which agrees with fp32 on 99.6%).

basal-1.0-1.5B on H100 (`fast`): 6.4 ms, about 158 dec/s.

## Notes per platform

- **DGX Spark (GB10, aarch64, CUDA 13).** Install PyTorch for CUDA 13 (`--index-url https://download.pytorch.org/whl/cu130`)
  before `pip install ".[fp8]"`. The GPU shares LPDDR5X memory (about 273 GB/s) with the CPU; every forward pass reads
  all weights, which is why FP8 (half the bytes) halves latency.
- **B300 / B200 (sm_100/sm_103).** PyTorch cu130 wheels work. The `vllm` mode needs a CUDA toolkit (nvcc) of version
  12.9 or newer on the machine: vLLM's FlashInfer kernels are compiled on first use, and nvcc 12.8 (found in some
  container images) cannot target the B300 (`Unsupported gpu architecture 'compute_103a'`). Compilation of `nvfp4` takes long (about 50 minutes on
  the B300) and is not recommended there.
- **RTX 5090 / RTX PRO 6000.** A driver with CUDA 12.8 is enough (`--index-url https://download.pytorch.org/whl/cu128`).
- **Start-up.** `fast` compiles and captures CUDA graphs for all input shapes before serving: 4–10 minutes the first
  time, much less with a warm compile cache. `fast-exit` captures one graph per segment and takes longer (14–22
  minutes). `fast-nocompile` starts in seconds at about 1.3× the latency.
- **NVFP4 quality.** With or without calibration, 4-bit weights and activations change about 10% of decisions of this
  model and lower accuracy by about 3 points. Prefer bf16 or FP8 unless batch throughput matters more.
