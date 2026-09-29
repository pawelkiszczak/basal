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
  8.9 to 4.8 GB (the checkpoint is loaded lazily, so the bf16 copy is never held); 0–10% slower.
- **4-bit weights** were tried and dropped: agreement fell to 0.86 on the 1.5B without any speed gain.
- **Throttling.** Under sustained load the MacBook's GPU slows down by up to 2× (automatic power mode); the table above
  was measured back to back, the comparison below after a cool-down before every engine.
- **Early exit** (`fast-exit`) and the `fp8` / `nvfp4` modes are CUDA-only.

### Inference engines on Apple Silicon

basal needs the next-token probabilities of the option letters after a prompt that ends in `{"answer": "`, not generated
text. Every engine that can return them was run on the same M4 Max: 44 bundled examples, both option orders, each
engine on prompts it had not seen (so prefix caches only help between the two orders of a question), 60 s cool-down
before each engine. *ms*: median per decision, one request at a time; *dec/s*: remaining 21 decisions in one call
(in-process) or with 4 concurrent clients (HTTP servers); *TV*: total-variation distance of the averaged two-order
probabilities to the fp32 PyTorch reference.

| engine | 4.5B ms | 4.5B dec/s | 1.5B ms | 1.5B dec/s | agreement 4.5B / 1.5B | TV mean / max (4.5B) |
|---|---|---|---|---|---|---|
| basal `mlx` | 198 | 5.1 | **67** | 15.8 | 1.000 / 0.977 | 0.0047 / 0.024 |
| basal `mlx-q8` | 200 | 4.8 | 70 | 14.3 | 1.000 / 0.977 | 0.0085 / 0.091 |
| basal `mps` | 239 | 4.4 | 89 | 12.4 | 0.977 / 0.977 | 0.0050 / 0.035 |
| basal `gguf` F16 | 200 | 5.5 | 69 | 16.7 | 1.000 / 1.000 | **0.0006** / 0.004 |
| basal `gguf` Q8_0 | 212 | 5.1 | 72 | 15.9 | 1.000 / 0.977 | 0.0039 / 0.038 |
| basal `gguf` Q4_K_M | 222 | 4.9 | 76 | 14.3 | 0.955 / 0.955 | 0.0465 / 0.307 |
| basal `vllm` on vllm-metal 0.30 ¹ | 210 | **6.9** | 80 | **21.4** | 0.977 / 0.977 | 0.0056 / 0.051 |
| mlx-lm 0.31 Python API, KV prompt cache, orders one after the other ¹ | 261 | 3.7 | 94 | 10.7 | 1.000 / 0.977 | 0.0050 / 0.042 |
| llama-cpp-python 0.3.35 F16, orders one after the other | 261 | 3.8 | 101 | 9.8 | 1.000 / 1.000 | 0.0005 / 0.005 |
| mlx_lm.server 0.31 (`top_logprobs` 11) ¹ | 374 | 3.1 | 153 | 7.6 | 1.000 / 0.955 | 0.0053 / 0.031 |
| llama-server b11146 F16 (token ids, `n_probs` 20) | **191** | 5.1 | 79 | 13.2 | 1.000 / 1.000 | 0.0005 / 0.005 |
| MTPLX 2.12, prompt-scoring lane (`echo`, `max_tokens` 0) ¹ | 339 | 2.9 | 129 | 8.0 | 0.977 / 0.977 | 0.0059 / 0.043 |
| Ollama 0.34, safetensors import ³ | 280 | 3.3 | 105 | 9.0 | 1.000 / 0.977 | 0.0055 / 0.059 |
| LM Studio, llama.cpp engine, F16 GGUF (chat with assistant prefill) ² | 211 | 5.4 | 81 | 16.2 | 0.955 / 0.977 | 0.0607 / 0.630 |

¹ These engines build the model with mlx-lm's Llama, which does not read `rope_parameters` (transformers 5) and falls
back to `rope_theta` 10000 instead of 1e6: unpatched, the mean TV is 0.10–0.12 for mlx_lm.server and 0.10–0.13 for
vllm-metal (4.5B–1.5B, max 0.73). They were run on a copy of the checkpoint whose `config.json` also has `"rope_theta": 1000000`
(basal's own `mlx` backend reads the value itself). ² LM Studio (and Ollama with a GGUF import, and llama-server with
text prompts) tokenize with the GGUF vocabulary, which splits basal prompts differently from the training tokenizer;
see [GGUF.md](GGUF.md#other-llamacpp-front-ends-send-token-ids). ³ Imported from the patched copy of ¹ (not tested
without).

![Latency and throughput of every engine](figures/apple_latency_throughput.png)

![Speed vs faithfulness](figures/apple_speed_vs_fidelity.png)

Per item, the same comparison shows where the deviations come from: the rope bug (bottom rows) and the GGUF text
tokenizer (LM Studio, llama-server with text) shift almost every item, Q4_K_M a few items strongly, the bf16 / 8-bit
paths stay at 1e-3–1e-2, and F16 through llama.cpp mostly below 1e-3. Items where fp32 itself is nearly tied (e.g.
`q-36` on the 1.5B) flip under any perturbation.

![Per-item distance to fp32](figures/apple_fidelity_heatmap.png)

- **Fastest single decision**: llama.cpp (llama-server with token ids, basal `gguf`) and basal `mlx`, within 5%.
  **Highest throughput**: vLLM's scheduler on vllm-metal. basal's `vllm` mode runs on it with the rope patch; it asks
  for the letter ids with `logprob_token_ids`, because vllm-metal ignores `logprobs_mode="processed_logprobs"` and
  returns the top-k of the whole vocabulary: with plain top-k the least likely option was missing (probability 0) on
  14 of the 44 items of the 1.5B.
  **Closest to fp32**: llama.cpp F16.
- Packing both option orders into one forward pass (basal `mlx` / `gguf`) is worth 25–30% against running them one
  after the other with a prompt cache (mlx-lm, llama-cpp-python rows).
- Not usable as shipped: **oMLX** 0.7.0rc1 returns no logprobs (checked with a request), **vllm-mlx** 0.5.0 has no
  logprobs in its API models; **mistral.rs** and **MLC-LLM** implement Llama without the q/k/v/o and MLP biases basal
  needs; **Swama**, **Exo** and Apple's Foundation Models framework expose no logprobs. **MTPLX** accelerates decoding
  with multi-token prediction, which a one-token decision does not use; plain Llama runs on its experimental AR path.

#### Quality

![Accuracy, log-loss and changed decisions per engine](figures/apple_quality.png)

Against the gold labels, the 44 bundled examples cannot rank the engines: every engine gets 33–36 of 44 right with the
4.5B and 31–33 with the 1.5B, and the 95% intervals are ±12 points wide. The engines with the rope bug even score
36/44 and a lower log-loss on the 4.5B (apparently flatter probabilities, which help on the confidently wrong
items), although they change the probabilities of almost every item. For a backend, quality therefore means
faithfulness to the reference model on which accuracy, calibration temperatures and confidence thresholds were
measured: the decisions it changes (right panel) and the TV distance above. The errors themselves belong to the model:
the same items are wrong in every faithful engine.

![Probability of the gold option per item](figures/apple_gold_prob_heatmap.png)

Figures: `python docs/figures/make_figures.py` (matplotlib) from the measurements in `docs/figures/apple_engines.json`.

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
