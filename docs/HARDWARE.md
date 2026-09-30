# Hardware: which mode on which GPU

Numbers are for **basal-1.0-4.5B** unless the section says otherwise; one decision = **both option orders** (the default), batch size 1 unless
stated. *dec/s*: two-order decisions per second with 32 option-order passes per forward (offline) or 32 concurrent
clients (HTTP). *Agreement*: share of decisions whose top option equals the fp32 reference on the same machine (RTX 5090 and
RTX 4090: the bf16 reference, marked ²).
Measured with `basal-bench` / `basal-loadtest` (H100, RTX PRO 6000 and RTX 5090 with the equivalent research harness)
on a private 500-item sample of the Polish/English test split.

## Recommendation

| GPU class | recommended mode | why |
|---|---|---|
| Data-centre (B300, B200, H100, H200) | `fast` (bf16) | limited by kernel launches at batch 1: FP8 gains ≤ 10% on H100 and is slower on B300, bf16 keeps decisions unchanged |
| Workstation / consumer (RTX PRO 6000, RTX 5090) | `fp8` | compute-bound; FP8 gives 1.3–1.4× at 97% agreement |
| RTX 4090 | `fast` (bf16) | FP8 compilation stalled on our test machine; bf16 works (1.5B: 12.5 ms) |
| Desktop with unified memory (DGX Spark GB10) | `fp8` | memory-bandwidth-bound; FP8 halves latency |
| Batched high-throughput serving on Blackwell | `vllm` + `-NVFP4` checkpoint | 3.2× throughput on the DGX Spark (the only machine where it was measured), but −3 accuracy points |
| A100 and other GPUs without FP8 (not measured) | `fast` | the bf16 path runs on any sm80+ GPU |
| Apple Silicon (M-series Mac) | `mlx` (`mlx-q8` on 16 GB Macs) | compute-bound: bf16 is fastest, 8-bit weights halve memory at no speed gain; see [Apple Silicon](#apple-silicon) |

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
| | `fast-exit`, 0.99 | 10.5 | 65 | 0.992 | 0.820 |
| | HTTP `fast` (p50) | 14.1 | 63 | – | – |
| | HTTP `fast-exit`, 0.99 (p50) | 12.1 | 77 | – | – |
| **RTX PRO 6000 Blackwell** | `fast` (bf16) | 18.9 | 39 | 0.994 | 0.822 |
| | `fp8` | 14.9 | 58 | 0.968 | 0.840 |
| | HTTP `fast` (p50) | 22.5 | 39 | – | – |
| **RTX 5090** | `fast` (bf16) | 27.3 (28.0 on a second machine) | 24 | 0.999² | 0.824 |
| | `fp8` | 19.4 | 40 | 0.968² | 0.828 |
| | HTTP `fast` (p50) | 32.3 | 23 | – | – |
| **DGX Spark (GB10)** | `eager` fp32 | 800.4 | 0.4 | reference | 0.822 |
| | `fast` (bf16) | 92.0 | 7.0 | 0.996 | 0.826 |
| | `fp8` | **44.6** | 8.3 | 0.970 | 0.836 |
| | `nvfp4` (torchao) | 41.0 | 10.4 | 0.910 | 0.792 |
| | `vllm` + NVFP4 checkpoint | 46.1 | **26.6** | 0.896¹ | 0.792 |
| | `fast-exit`, 0.99 | 76.9 | 7.6 | 0.994 | 0.828 |

¹ agreement with bf16 (which agrees with fp32 on 99.6%). ² agreement with bf16 eager (no fp32 run on this card).

## basal-1.0-1.5B (lite)

| GPU | `fast` (bf16) | dec/s | `fp8` | dec/s | agreement bf16 / fp8 | vs 4.5B bf16 |
|---|---|---|---|---|---|---|
| B300 SXM6 | **4.7 ms** | **250** | 5.2 ms | 224 | 0.994 / 0.966 | 1.9× faster |
| H100 80GB | 6.2 ms | 157 | 6.3 ms | 177 | 0.992 / 0.964 | 2.0× |
| RTX PRO 6000 Blackwell | 8.8 ms | 101 | 7.6 ms | 128 | 0.998 / 0.968 | 2.2× |
| RTX 5090 | 12.7 ms | 67 | 9.3 ms | 99 | 0.996 / 0.960 | 2.2× |
| RTX 4090 | 12.5 ms | 51 | – | – | reference | – |
| DGX Spark (GB10) | 34.3 ms | 20 | **18.1 ms** | 31 | 0.996 / 0.962 | 2.7× |

Speed-up against the 4.5B model measured with the released engine on the same machine (RTX PRO 6000: 19.1 ms,
RTX 5090: 28.0 ms; the earlier runs above: 18.9 and 27.3 ms). Agreement with fp32 (RTX 4090: bf16 is the reference).

HTTP on H100 (`basal-serve --mode fast`): 7.7 ms p50, 147 decisions/s with 32 clients. NVFP4 on the DGX Spark: 16.5 ms
but agreement 0.87 — not recommended. RTX 4090: the FP8 compilation stalled on our test machine; bf16 works.

## Apple Silicon

Apple M4 Max (40-core GPU, 128 GB), macOS 27, torch 2.14 (MPS), MLX 0.32.3, mlx-lm 0.31.3. Measured with `basal-bench`
on the **44 bundled examples** (shorter prompts than the private 500-item sample above, so latencies are not directly
comparable with the CUDA tables; between two runs on this machine latencies differed by up to 30%). Agreement with `eager-fp32`
(PyTorch MPS, fp32) on the same machine; GB: peak memory (MLX: allocator peak, MPS: memory held by the Metal driver).

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| basal-1.0-4.5B | `eager-fp32` | 542.6 | 326.5 | 1.2 | reference | 0.795 | 23.0 |
| | `mps` (bf16) | 284.8 | 242.8 | 3.1 | 0.977 | 0.773 | 10.2 |
| | `mlx` (bf16) | **265.9** | **216.4** | **3.3** | 1.000 | 0.795 | 10.1 |
| | `mlx-q8` | 298.4 | 243.9 | 2.9 | 1.000 | 0.795 | 6.0 |
| | HTTP `mlx` (p50 / p95) | 207.9 / 255.8 | – | 3.6 | – | – | – |
| basal-1.0-1.5B | `eager-fp32` | 175.7 | 101.8 | 4.5 | reference | 0.727 | 10.2 |
| | `mps` (bf16) | 98.2 | 79.5 | 9.7 | 0.977 | 0.750 | 4.2 |
| | `mlx` (bf16) | **87.3** | **70.9** | **12.0** | 0.977 | 0.750 | 4.0 |
| | `mlx-q8` | 94.8 | 76.3 | 8.6 | 0.977 | 0.750 | 2.6 |

HTTP: `basal-loadtest` against `basal-serve --mode mlx` (default prompts of the tool, 32 concurrent clients).

- **Compute-bound.** About 90% of the forward time is in the linear layers, which MLX runs at 10–12 TFLOPS in bf16 on
  the M4 Max; bf16, fp16 and fp32 differ little, larger batches do not raise throughput, and `mx.compile` gave no
  speed-up (so the `mlx` backend pads only to the longest prompt of a batch instead of to fixed shape buckets). Other
  M-series chips were not measured; being compute-bound, latency should scale roughly with GPU core count and clock.
- **8-bit weights** (`mlx-q8`, MLX affine, group 64, decoder layers only) cut the resident weights of the 4.5B from
  8.9 to 4.8 GB (the checkpoint is loaded lazily, so the bf16 copy is never held) and are about 10% slower.
- **4-bit weights** were tried and dropped: agreement fell to 0.86 on the 1.5B without any speed gain.
- **Early exit** (`fast-exit`) and the `fp8` / `nvfp4` / `vllm` modes are CUDA-only.

## Notes per platform

- **DGX Spark (GB10, aarch64, CUDA 13).** In the README install steps use the CUDA 13 build of PyTorch
  (`uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`) instead of cu128. The GPU shares LPDDR5X memory (about 273 GB/s) with the CPU; every forward pass reads
  all weights, which is why FP8 (half the bytes) halves latency.
- **B300 / B200 (sm_100/sm_103).** PyTorch cu130 wheels work. The `vllm` mode needs a CUDA toolkit (nvcc) of version
  12.9 or newer on the machine: vLLM's FlashInfer kernels are compiled on first use, and nvcc 12.8 (found in some
  container images) cannot target the B300 (`Unsupported gpu architecture 'compute_103a'`). Compilation of `nvfp4` takes long (about 50 minutes on
  the B300) and is not recommended there.
- **RTX 5090 / RTX PRO 6000.** A driver with CUDA 12.8 is enough (`--index-url https://download.pytorch.org/whl/cu128`).
- **Start-up.** `fast` compiles and captures CUDA graphs for all input shapes before serving: 4–10 minutes the first
  time, much less with a warm compile cache. `fast-exit` captures one graph per segment and takes longer (14–22
  minutes). `fast-nocompile` starts in seconds at about 1.4× the latency.
- **NVFP4 quality.** With or without calibration, 4-bit weights and activations change about 10% of decisions of this
  model and lower accuracy by about 3 points. Prefer bf16 or FP8 unless batch throughput matters more.
