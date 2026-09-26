# Low precision: FP8 and NVFP4

basal-1.0 can run in three precisions. All modes read the same letter probabilities; lower precision changes a small
share of decisions, which we report as argmax agreement with the fp32 reference.

| precision | how | GPUs | typical agreement with fp32 |
|---|---|---|---|
| bf16 | `--mode fast` | any CUDA GPU, sm80+ | ≥ 0.99 (identical decisions up to bf16 noise) |
| FP8 (on the fly) | `--mode fp8` (torchao dynamic FP8, per-row scales) | Ada (RTX 40xx, L40S), Hopper, Blackwell | ≈ 0.97–0.98 |
| FP8 (checkpoint) | `--mode vllm --model Remek/basal-1.0-4.5B-FP8` | Hopper, Blackwell | see HARDWARE.md |
| NVFP4 (checkpoint) | `--mode vllm --model Remek/basal-1.0-4.5B-NVFP4` | Blackwell: B200/B300, RTX 50xx, RTX PRO 6000, GB10 | see HARDWARE.md |
| NVFP4 (on the fly) | `--mode nvfp4` (torchao, experimental) | Blackwell | see HARDWARE.md |

## When does low precision help?

At batch size 1 a 4.5B model on a 400-token prompt is limited by kernel-launch overhead on server GPUs and by compute on
workstation and consumer GPUs. FP8 therefore helps most where compute is the bottleneck (RTX PRO 6000: 18.9 → 14.9 ms,
RTX 5090: 27.3 → 19.4 ms per decision) and less on the H100 (12.5 → 11.3 ms). The price is that 2–3% of decisions change.
Use bf16 when every decision counts and FP8/NVFP4 when throughput matters more.

## The ModelOpt checkpoints

The `-FP8` and `-NVFP4` repositories were produced with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
post-training quantization (`NVFP4_DEFAULT_CFG` / `FP8_DEFAULT_CFG`), calibrated on 1,024 decision prompts from the
calibration split (both option orders), with the output head kept in bf16 (the decision is read from its logits). They
are standard ModelOpt Hugging Face exports and load in vLLM, SGLang and TensorRT-LLM.

```bash
pip install "basal[vllm]"
basal-serve --model Remek/basal-1.0-4.5B-NVFP4 --mode vllm --port 8000
```

The `vllm` mode asks vLLM for one token restricted to the option letters and reads the log-probabilities computed
*after* that restriction (`logprobs_mode="processed_logprobs"`), which is exactly the letter softmax of the other modes.
vLLM's prefix caching shares the state between the two option orders.
