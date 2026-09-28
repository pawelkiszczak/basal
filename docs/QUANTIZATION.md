# Low precision: FP8 and NVFP4

basal-1.0 can run in three precisions. All modes read the same letter probabilities; lower precision changes a small
share of decisions, which we report as argmax agreement with the fp32 reference.

| precision | how | GPUs | typical agreement with fp32 |
|---|---|---|---|
| bf16 | `--mode fast` | any CUDA GPU, sm80+ | ≥ 0.99 (identical decisions up to bf16 noise) |
| FP8 (on the fly) | `--mode fp8` (torchao dynamic FP8, per-row scales) | Hopper, Blackwell (Ada: FP8 compilation stalled on an RTX 4090) | ≈ 0.96–0.98 |
| FP8 (checkpoint) | `--mode vllm --model Remek/basal-1.0-4.5B-FP8` | Hopper, Blackwell | not benchmarked yet (expected close to `--mode fp8`) |
| NVFP4 (checkpoint) | `--mode vllm --model Remek/basal-1.0-4.5B-NVFP4` | Blackwell (measured on DGX Spark GB10) | 0.90 (−3 accuracy points) |
| NVFP4 (on the fly) | `--mode nvfp4` (torchao, experimental) | Blackwell | 0.90–0.91 (−3 accuracy points) |

## When does low precision help?

At batch size 1 a 4.5B model on a 400-token prompt is limited by kernel-launch overhead on server GPUs, by compute on
workstation and consumer GPUs, and by memory bandwidth on the DGX Spark. FP8 therefore helps most where memory or
compute is the bottleneck (DGX Spark: 92.0 → 44.6 ms, RTX PRO 6000: 18.9 → 14.9 ms, RTX 5090: 27.3 → 19.4 ms per
decision), little on the H100 (12.5 → 11.3 ms) and not at all on the B300 (8.8 → 9.7 ms). The price is that 2–4% of
decisions change. NVFP4 changes about 10% and costs about 3 accuracy points on the 4.5B model (about 7 on the 1.5B).
Use bf16 when every decision counts, FP8 on workstation, consumer and desktop GPUs, and NVFP4 only when batch throughput
matters more than accuracy.

## The ModelOpt checkpoints

The `-FP8` and `-NVFP4` repositories were produced with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
post-training quantisation (`NVFP4_DEFAULT_CFG` / `FP8_DEFAULT_CFG`), calibrated on 1,024 decision prompts from the
calibration split (both option orders), with the output head kept in bf16 (the decision is read from its logits). They
are standard ModelOpt Hugging Face exports; they load in vLLM (tested) and are also supported by SGLang and
TensorRT-LLM (untested).

```bash
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz"   # separate environment: vLLM brings its own torch
basal-serve --model Remek/basal-1.0-4.5B-NVFP4 --mode vllm --port 8000
```

The `vllm` mode asks vLLM for one token restricted to the option letters and reads the log-probabilities computed
*after* that restriction (`logprobs_mode="processed_logprobs"`), which is exactly the letter softmax of the other modes.
vLLM's prefix caching shares the state between the two option orders.
