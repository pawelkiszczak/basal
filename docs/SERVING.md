# Serving, batch inference and benchmarks

Flags of `basal-serve`, inference on a JSONL file with `basal-run`, a browser app that picks and serves a model for
your hardware (`basal-ui`), a look inside one decision with `basal-viz`, and benchmarking a GPU.

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | hardware |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + SOAM packing + compact batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fp8` | the `-FP8` ModelOpt checkpoints natively; bf16 weights quantised to FP8 at load time (torchao) | GPUs with FP8 (sm89+): Hopper, Blackwell |
| `nvfp4` | `fast` with bf16 weights quantised to NVFP4 at load time (torchao, experimental; costs accuracy; not the basal-1.5 / 1.0 `-NVFP4` checkpoints) | Blackwell |
| `vllm` | vLLM backend, readout from the option codes' log-probabilities; bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only), and the basal-1.5 / 1.0 `-NVFP4` checkpoints; `--vllm-preset deterministic` for repeatable answers, basal-omni media with `--media on` ([OMNI.md](OMNI.md#vllm---mode-vllm)) | CUDA GPUs supported by vLLM |
| `sglang` | SGLang backend, same readout | CUDA GPUs supported by SGLang |
| `eager` | plain PyTorch reference; `--device cuda\|mps\|cpu` (default: `BASAL_DEVICE`, else CUDA, else Apple Silicon, else CPU) | any GPU, Apple Silicon (MPS) or CPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen per request | models that ship `exit_heads/` (basal-1.0 only) |
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` ports or bf16 weights; shared-prefix KV cache (opt-in `--mlx-packed`: short Llama groups as one packed forward, [QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)) | Apple Silicon |
| `mlx-q8` | `mlx` with 8-bit weights made from bf16 at load time (about half the memory, not faster) | Apple Silicon |
| `mps` | the PyTorch engine without CUDA graphs: SOAM packing + token-budget batching; on the Mac's GPU, or on the CPU (`--device cpu`, the default without MPS), where a request's state runs once instead of once per prompt as in `eager` | Apple Silicon, CPU |
| `gguf` | a GGUF file in process through llama.cpp (`--gguf <file>`, extra `basal[gguf]`) | Apple Silicon (Metal), CUDA, CPU |
| `strix` *(experimental, community-maintained)* | basal-1.5 / basal-2.0 (Llama architecture): PyTorch + Triton kernels written for gfx1151: flash attention over the packed prefix trie, gate / up + SwiGLU fused, matmuls with the bias and residual add in the epilogue, all rows of a batch in one sequence (bf16; [HARDWARE.md](HARDWARE.md#amd-strix-halo-experimental-community-maintained), [benchmarks](benchmarks/STRIX_HALO.md)) | AMD Strix Halo (ROCm + Triton); measured only there |
| `ollama` | the basal server in front of Ollama (GGUF ports; `--ollama-model`, `--ollama-url`) | anything Ollama runs on |
| `llamacpp` | the basal server in front of `llama-server` (GGUF ports; `--llamacpp-url`) | anything llama.cpp runs on |
| `tpu` *(experimental, community-maintained)* | JAX / XLA on one TPU chip: SOAM packing, one compiled program per shape, `--max-len` (extra `basal[tpu]`, [install](INSTALL.md#google-tpu-experimental-community-maintained)) | Google TPU (v5e measured) |

- Without `--mode`: `fast` on CUDA, `mlx` on Apple Silicon (`mps` without `basal[mlx]`), `tpu` on a TPU VM with
  `basal[tpu]`, `eager` otherwise. On a CPU, `--mode mps` computes a request's state once for all its prompts and is
  faster for requests with several questions or option orders; `eager` stays the default there (the reference path).
- **Qwen3.5 models (Gated DeltaNet layers)**, such as basal-omni-pico: `fast` and `fast-nocompile` serve them since 2.4
  (2.3.2 refused them): the graph engine keeps each request state's recurrent state beside its keys and values and
  continues every question from it (one row per question and option order in the branch pass), with fla
  (flash-linear-attention) for the Gated DeltaNet layers. Without `--mode` they run `fast` on Hopper (H100, the GPU
  class the port was gated on) when fla is installed, `eager` elsewhere. Not with them yet: `fp8` / `nvfp4`, `fast-exit`,
  `mps`, `--batches compact`, `--state-cache-gb`, `--prefix-cache`, `--template-prefix on`, `--readout-layer on`,
  `--lean` and `--long-lane` (each refused before loading). `--mode vllm` and `--mode gguf` serve Qwen3.5 models
  too. Other models with recurrent-state layers (Qwen3-Next, Mamba and other hybrids) are refused by every
  shared-prefix engine and run `eager` without `--mode`.
- **`--mode fp8` below compute capability 8.9** (Ampere, e.g. RTX 3090 / A100) stops before loading the model (2.3.2;
  before, torchao failed after the load): serve bf16 with `--mode fast` there.
- **basal-omni (Gemma 4) in `eager`** (2.3): the engine's own Gemma 4 forward: a request's state (and media) once,
  then every question over it, and mini's last layers at the answer rows only; `--eager-forward hf` keeps the
  transformers forward of 2.2. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **basal-omni (Gemma 4) in `fast` / `fast-nocompile`** (2.3, text): the engine's own Gemma 4 layer code with SOAM
  over its sliding-window layers and the answer-rows-only last layers; `--gemma-attn auto` = FlashAttention 3 + flex,
  compiled, on Hopper, flex on Blackwell workstation GPUs (sm_120), else eager's math kernel uncompiled; `fp8` /
  `nvfp4` quantisation at load time is refused for it (serve an `-FP8` checkpoint). Without `--mode` it runs `fast` on
  Hopper with FlashAttention 3 and on sm_120 Blackwell, `eager` on other GPUs. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- `--calibration file.json` overrides the calibration file (default: `CALIBRATION.<mode>.json` when the model ships
  one for that engine, else `CALIBRATION.json`; `--mode gguf` loads `CALIBRATION.gguf.json` when the model ships one,
  else `CALIBRATION.json`; with `--orders 2`,
  `CALIBRATION.o2.json` when shipped). A file applies its thresholds only to the option orders it was fitted with. `basal-calibrate` fits one on your labelled requests
  ([CALIBRATION.md](CALIBRATION.md)). A missing file is refused at start-up.
- **Option orders** (`--orders 1 | 2 | auto`): `1` asks the order given (the default for prompt-format-2 models such as
  basal-2.0, which are trained with random option orders). `2` also asks the reversed order and averages the
  probabilities (the default for basal-1.5 and basal-1.0); both orders share the state and the question text, so the
  second order costs only its options. `auto` asks order 1 and the reversed order only where order 1 is uncertain
  (`--auto-tau`, `--auto-band`). See [FORMATS.md](FORMATS.md#option-orders).
- **SOAM** (`--soam on`, default): every question and option order of a request is one packed row over the state.
  `--soam off` restores the per-question path of basal-1.0 (same answers in fp32). See [SOAM.md](SOAM.md).
- **Start-up**: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (several
  minutes the first time, about half with a warm compile cache). A request with several branches (option orders,
  questions, `multi` labels) runs in two exact passes: the state once, keeping its keys and values, then all branches
  over them ([SOAM.md](SOAM.md#which-kernel-runs)).
- **Batching**: concurrent requests are batched into the same forward passes and every request is answered as soon as
  its passes are read out. Under load the server waits up to `--batch-window-ms` (default 3) for as many requests as
  the previous batch had; an idle server answers at once. `--max-batch` and `--wait-ms` control batching further.
- **`--deterministic`**: answers do not depend on load or on the other requests: every request runs in its own
  forward passes, never batched with others, with deterministic kernels (FP8 checkpoints: GEMMs with full
  accumulation). Costs about 4% of the throughput under load with one option order; in-process engines only
  (`basal-loadtest --determinism N` checks a running server, including decision flips under load). Batch-invariant
  kernels (2.3.4, tested, not shipped) made basal-omni's batched answers bit-identical, but were slower than one
  request per forward (H100, basal-omni-1.0-mini: 154 instead of 163 decisions/s), so `--deterministic` keeps one
  request per forward; a faster batch-invariant GEMM is future work.
- **Long states**: `--context-overflow truncate` (the default) cuts the middle of a state that does not fit the
  context (the first 1/3 and the last 2/3 of the kept tokens, `--truncate-head`; the question and options stay whole)
  and says what was cut in `usage.truncated`; `--context-overflow refuse` returns HTTP 422 instead. The context is
  32,768 tokens for basal-2.0 (4.5B) and max, 8,192 for mini (Ollama mode: 4,096; llama.cpp mode: `llama-server -c`;
  vLLM / SGLang: `--max-len`).
- **Engine v2.0.0 items** (bf16 weights in `--mode fast` on Hopper with FlashAttention 3; each item has a flag,
  `auto` = on):
  - `--batches compact`: single prompts packed back to back into one row (FA3 varlen), graphs per token count, no
    padding to bucket lengths or row counts. On other sm80+ GPUs `--batches compact` runs them through torch's varlen
    attention (Llama models; `auto` keeps them off there). basal-2.0-mini on an RTX 4060 Ti 16 GB, against
    `--batches default`: single questions 66.8 -> 109.4 decisions/s (template prefix on), the same top answer on 58
    of 58 answers; the compact graphs took 2.4 GB more GPU memory there (13.4 -> 15.8 GB, the card full; with room
    they take about 6 GB for mini and 7 GB for the 4.5B, see below), and with the card full
    five questions over ~1.5k tokens (a path compact rows do not change) ran 5.6% slower. The same kernel on an H100,
    forced with `--attn fa2 --batches compact` (an H100 runs FlashAttention 3 otherwise; 16 clients, 2.3.1 check):
    single questions +16% for basal-2.0-mini (79.0 -> 91.7 decisions/s) and basal-2.0-4.5B (27.9 -> 32.3), five
    questions over 3k-10k tokens unchanged, GPU memory +5.8 / +7.2 GB, start-up +51 / +88 s; on dev v2 every decision
    with margin >= 0.2 kept its top answer (4.5B with `--template-prefix off`, as for every engine; with it on, 4 of
    9,064 changed). On a Blackwell RTX PRO 6000 (Server Edition, sm_120, torch 2.13, where `auto` runs no compact
    rows): single questions +18% for mini (49.0 -> 57.9) and +13% for 4.5B (17.0 -> 19.1), five questions unchanged,
    GPU memory +5.8 / +7.2 GB, start-up +31 / +112 s; dev v2 at margin >= 0.2: 7,632 / 7,632 (mini) and 9,047 /
    9,049 (4.5B, template prefix off) kept their top answer. On an RTX 3090 (Ampere, sm_86; @krzysztofkaminski, #33):
    +68% on short prompts and +20% / +18% (mini / 4.5B) on ~1.6k-token states, GPU memory +5.9 GB for mini, the
    same top answer on 58 of 58; up to 2.3.2 the 4.5B with compact rows ran out of memory at start-up on 24 GB.
    From 2.3.3 (#35, #37, @Butterski) the graphs share their output buffers and are captured largest first, so
    compact rows cost only 0.1-0.2 GB more than default rows (H100 after warm-up: mini 7.1 / 7.3 GB, 4.5B 15.2 /
    15.4 GB, every probability as in 2.3.2), and the 4.5B with compact rows starts on a 24 GB RTX 4090. From 2.3.4
    basal-omni's graphs are captured largest first too (H100 after warm-up: basal-omni-1.0-mini 27.8 -> 18.9 GB,
    1.0-max 45.3 -> 31.3 GB, every probability as in 2.3.3).
  - `--template-prefix on`: the chat template, system prompt and user header of every prompt language are computed
    once at start-up; compact rows run only their own tokens over them (off for basal-2.0 4.5B, see the quick start).
  - `--scheduler on`: requests longer than `--long-tokens` (2,048) wait in a backlog and join engine calls in
    highest-response-ratio order, at most 8,192 of their tokens per call. It trades long-request latency (about +7% p50)
    for short-request latency; `--scheduler off` serves in arrival order.
  - Answers are not bit-identical to engine v1.5.0's (other GEMM row counts); `--deterministic` stays reproducible on a
    given GPU.
- **Kernels of the CUDA graph engine** (`fast`, `fast-nocompile`, `fp8`, `nvfp4`, `fast-exit`):
  - `--attn auto|fa2|fa3|fa4|cudnn` is the attention of causal forwards. `auto` uses FlashAttention 3 on Hopper when
    its wheel is installed, else torch's built-in FlashAttention 2; FP8 checkpoints and `--mode fp8` stay on
    FlashAttention 2 unless `--attn fa3`.
  - `--merge auto|on|off` merges the q/k/v and gate/up projections into one GEMM each (`auto`: on for bf16 weights,
    off for FP8 checkpoints).
  - `--fp8-accum fast|full`: FP8 GEMM accumulation in the default mode (`--deterministic` always uses `full`).
  - Opt-in, off by default: `--lean on|pass2` (one compiled layer function: faster start-up, slower under load),
    `--tree flex` (a FlexAttention mask for the branches of a request), `--readout-layer on`, `--state-cache-gb N`
    (a state cache across requests; 2.3.1: also for states that begin with a cached one, e.g. agent traces, which then
    compute only their new tokens), `--prefix-cache`, `--long-lane`. Measurements:
    [HARDWARE.md](HARDWARE.md#kernels-of-the-cuda-graph-engine-one-h100).
- **vLLM options (2.3):** `--vllm-preset default | deterministic` (deterministic: batch-invariant kernels, no prefix
  cache, no chunked prefill; repeatable answers at a throughput cost) loads `CALIBRATION.vllm.<preset>.json` when the
  model ships one, else `CALIBRATION.vllm.json`, else `CALIBRATION.json`, and is reported in `GET /v1/models`;
  `--vllm-fast-prefill auto | off` (auto, the default: the KV-shared layers of Gemma 4 E2B / E4B run at the answer
  position only during prefill). Measurements: [OMNI.md](OMNI.md#vllm---mode-vllm).
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096): `--max-len 32768` for long
  documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback).
- **Status codes and load (2.3.1)**: an invalid request answers HTTP 422, the GPU out of memory 503 (retry later or
  elsewhere) and any other server error 500, all with `{"error": "..."}`. `GET /health` returns the requests in flight
  as `running` and `queued` (waiting for the engine); both 0 means idle, for a process manager that stops or restarts
  the server. Definitions: [API.md](API.md).

## API access

`basal-serve` is open by default. To require API keys, pass `--api-keys PATH` with a JSON file containing SHA-256
digests (never plaintext secrets):

```json
{"keys": [{"name": "laptop", "sha256": "<64-character lowercase SHA-256 hex digest>", "admin": false}]}
```

Send the secret as `Authorization: Bearer <secret>` or `X-API-Key: <secret>`. `GET /health` remains open.
`GET /v1/keys` requires an admin key and reports each key's in-memory `last_used` time. File changes take effect
without restarting the server. MCP at `/mcp` uses the same authentication.

`--cors-origins` accepts a comma-separated list of browser origins, for example
`--cors-origins https://app.example,https://admin.example`. CORS is disabled by default; use `*` to allow all origins.
Allowed preflight methods are GET, POST and OPTIONS.

```bash
basal-serve --model Remek/basal-2.0-mini --api-keys ./api-keys.json --cors-origins https://app.example
```

## HTTP API reference

The full `basal-serve` HTTP contract, including System One and OpenAI Decisions requests, authentication, errors,
media, and copy-ready client setups, is documented in [API.md](API.md).

## Inference on a JSONL file

`basal-run` sends every line of a JSONL file to a running server (concurrently; the server batches the requests) and
writes one answer per line, in the same order. Start a server first (`basal-serve ...`), then:

```bash
EX=$(python -c "import basal, pathlib; print(pathlib.Path(basal.__file__).parent / 'examples')")   # installed examples
basal-run --input $EX/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
```

Each input line is either a **simple item** or a **full request** (the formats can be mixed):

```jsonc
// simple item: one question, options as a list; "type" defaults to "choice", "gold" (index) is optional
{"id": "t1", "state": "Klient: od wczoraj nie mogę zalogować się do bankowości ...", "question": "Do którego działu skierować zgłoszenie?",
 "options": ["Reklamacje kart", "Wsparcie bankowości elektronicznej", "Kredyty"], "gold": 1}
// yes/no item: options are [yes-text, no-text]
{"id": "t2", "type": "noul", "state": "...", "question": "Czy odstąpienie złożono w terminie?", "options": ["Tak", "Nie"]}
// full /v1/systemone request: several typed questions about one state
{"id": "t3", "state": {"ticket": "..."}, "questions": {"category": {"type": "choice", "instructions": "...", "criteria": {"complaint": "...", "other": "..."}},
                                                      "urgent": {"type": "noul", "instructions": "..."}}}
```

Simple items are sent with placeholder keys and `"option_keys": "hide"`, so the model sees exactly the option texts.
Each output line contains `id`, the full `answers` (probabilities, confidence) and the server `latency_ms`; simple items
also get `prediction` (index of the chosen option), `option`, `confidence` and, when `gold` is given, `correct`. At the
end `basal-run` prints a summary (items per second, median latency and accuracy if gold labels are present).
Ready-to-run examples (Polish and English) in `basal/examples/` of the repository, also installed with the package
(`$EX` above):

| file | what it shows |
|---|---|
| `choice.jsonl` | routing, document type, amounts, policy rules, sentiment, next step — one option out of 3–4 (with `gold`) |
| `noul.jsonl` | yes/no decisions: deadlines, approval thresholds, phishing, refunds, missing information, alerts (with `gold`) |
| `score.jsonl` | ordered scales: urgency, satisfaction, fraud risk, answer correctness; the answer includes the expected level |
| `complex.jsonl` | full requests: several typed questions about one JSON state (fan-out), a web-agent step with structured options, a deadline decomposed into simple questions, loan triage, a prompt-injection guard |
| `questions.jsonl` | 20 mixed items; with `choice`, `noul` and `score` the default set of `basal-bench` and `basal-loadtest` |
| `types_demo.py` | one request per decision type, sent to a running server, request and response printed |

```bash
for f in choice noul score complex; do basal-run --input $EX/$f.jsonl --output answers_$f.jsonl; done
```

## See how a decision is made

`basal-viz` (contributed by [@lukaspoloki](https://github.com/lukaspoloki), [#2](https://github.com/rkinas/basal/pull/2))
opens a local web page that runs the model and shows every step of a decision: the prompt of each option order the
server asks (prompt format and option codes from the model's `basal.json`; one order by default for basal-2.0), the
shared-prefix packing when there are two orders, the code probabilities after every layer ("logit lens"), the code
readout, the calibration and the decision against the thresholds the calibration file certifies. The same request is
answered by the server in the same process, and the page shows the difference (0 for a single question; a request with
several questions shares its state in one packed row, which moves bf16 probabilities by a few thousandths). The bundled
examples are in a drop-down, with the expected answer where the file has one.

```bash
basal-viz --model Remek/basal-2.0-mini --port 8080     # then open http://127.0.0.1:8080
```

It runs `--mode mps` on Apple Silicon and `--mode eager` elsewhere (`--device`, `--orders 1|2`, `--calibration` as for
`basal-serve`), and only the Llama-architecture models (basal-1.0, basal-1.5, basal-2.0; not basal-omni). It is meant
for exploring, not serving: every question runs one extra forward per option order to record all layers.

## Browser app: pick a model for your hardware and try it

`basal-ui` is a local web app in the style of LM Studio: it detects the machine, lists every basal model on Hugging
Face with what this machine can do with it, downloads the one you choose, starts `basal-serve` for it and gives you a
playground for decisions.

```bash
uv pip install "basal[ui,mlx]"        # [ui] adds FastAPI; add the engine extra of your machine (mlx, gguf, fp8, vllm)
basal-ui --port 7860                  # then open http://127.0.0.1:7860
```

The left menu has one view per task, each with its own address (`#overview`, `#models`, `#manager`, `#playground`,
`#api`, `#history`, `#logs`), so the browser's back button and bookmarks work. **Overview** shows the hardware, the
loaded models (one row each: state, mode, media, estimated → resident memory, requests and average confidence,
Restart / Logs / Unload) above a memory budget bar, the recommended model for this machine and short summaries of the
disk and the API server; **Models**, **Model manager**, **Playground**, **API server**, **History** and **Logs** hold
the features below. The menu's footer shows how many models are ready, their names, the API server's URL and an
**Unload all** button. Deleting a model or revoking a key asks for confirmation in a dialog, and actions report their
outcome in a short notification. On a narrow window the menu becomes a top bar.

- **Hardware.** NVIDIA GPU (name, compute capability, memory), Apple Silicon (MLX's recommended working set) or CPU
  (60% of RAM), the engines installed (`mlx`, `llama-cpp-python`, `vllm`, `media`), and the memory available now
  (free GPU memory on NVIDIA; free + inactive + speculative pages from `vm_stat` on macOS, which leaves out reclaimable
  file cache; `MemAvailable` on Linux).
- **All models.** Every `Remek/basal-*` repository and each `.gguf` file of a `-GGUF` repository, with weight size,
  estimated memory (weights × 1.2 + 1 GB), fit against the budget and *fit now* against the available memory (*fits*
  up to 85%, *tight* up to 100%), and the `basal-serve` mode it would run in, or why it cannot run here and which
  extra fixes that. Offline, the list is what is in the models directory.
- **Suggested**: cards for the models that fit both the budget and the memory available now, a comfortable fit before
  a tight one. *Recommended*: the basal-2.0 family in the format of the machine (after
  [HARDWARE.md](HARDWARE.md#recommendation): FP8 on NVIDIA GPUs with compute capability 8.9+ other than data-centre
  GPUs and the RTX 4090, bf16 there, `-MLX-8bit` on Apple Silicon, GGUF `Q8_0` on a CPU with `llama-cpp-python`,
  else bf16), 4.5B if it fits now, else mini. *Best quality*: the largest tier that fits now. *Fastest*: mini.
  *Multimodal*: the largest basal-omni checkpoint this machine can serve with media. Without a reading of the
  available memory the budget fit stands in for it.
- **Download** (a model not on disk yet) fetches only what serving needs (the chosen `.gguf` file of a `-GGUF`
  repository; not the 4.5B's `think/` model) into the models directory and shows the progress in the row; nothing is
  started. Once on disk the row offers **Load** instead.
- **Load** (downloading first when needed) runs `basal-serve` for the model as a subprocess on a free loopback port.
  **Several models can be loaded at once**: each gets its own private `basal-serve`, and a loaded model's row shows a
  *loaded* chip and **Unload**. A load that would exceed the memory budget is refused (below).
- **Load with media**, for the basal-omni-1.0 bf16 checkpoints (mini and max: images and audio; ultra: images; the
  FP8, GGUF and MLX ports and basal-omni-1.1-max have no encoders): `basal-serve --mode eager --media on`, with
  `--device mps` on Apple Silicon and `--device cpu` on a machine without a GPU. Needs the `basal[media]` extra
  (pillow, soundfile, scipy, torchvision); the row says so when it is missing.
- **Model manager**: the models directory (default: the Hugging Face cache, `HF_HUB_CACHE`) can be set to any
  writable directory; downloads, the downloaded marks, serving and deletes all use it. The setting is kept in
  `~/.config/basal-ui/settings.json` (`BASAL_UI_CONFIG` moves it). The variants on disk are listed with their size and
  can be deleted; for a `-GGUF` repository only the chosen `.gguf` file goes, and the repository once no `.gguf` is
  left. The directory cannot change while any model is loaded or downloading, and a loaded model cannot be deleted.
- **Playground**: a model selector (the ready models), the bundled examples, a question builder (`choice`, `noul`,
  `score`) or the raw JSON request (which carries `"model"`); answers with calibrated probabilities, confidence and
  latency. Requests go to the selected model's `POST /v1/systemone`. With a media server, up to 8 images or audio
  files (20 MiB each) go along as `media`, with the image `detail` and the audio `lang`.

### Memory budget

The budget is the hardware budget shown under Hardware unless set in **API server → Memory** (at least 1 GiB), and
**Keep free** (default 2 GiB) is held back from it. A load is refused with `409 over memory budget: … loaded + …
needed > …` when the estimates of the models already loaded or loading plus the new model's estimate exceed budget −
keep free; unload a model or raise the budget. The Overview bar shows each model's resident memory (summed over the
`basal-serve` process group with `ps`, so engine workers count; on Windows the estimates stand in), the committed
estimates and the memory available now. On Apple Silicon, MLX and MPS allocations live in unified memory and may show
only partly as resident memory; the estimate is what the budget check uses.

### Statistics and benchmarks

**Models → Loaded** lists, per model, the decisions basal-ui answered (playground, API server, MCP): requests,
questions, errors, average best-answer probability (the answer's `confidence`; `set_confidence` for `multi`; the
largest probability for `tool`), mean and p95 latency (over the last 1000 requests), input tokens and last use, with
one line per question type. They are kept in `<settings stem>.stats.json` next to the settings file, survive restarts
and can be reset per model or all at once.

**Benchmark** measures one loaded model at a time: 5 warm-up requests, 30 sequential requests (p50 / p95 / mean
latency) and 120 at concurrency 8 (decisions, questions and input tokens per second, errors). Every item with a `gold`
answer is scored once (accuracy overall and per file). The set is either the bundled examples (44 items in
`basal/examples/`) or your own JSONL file or directory: [BENCHMARK_SETS.md](BENCHMARK_SETS.md). The last 5 results
per model stay in the stats file and are shown side by side. Benchmark requests are not counted in the statistics or
the history.

### History and logs

**History** lists every decision basal-ui answered — from the playground, the API server and MCP — newest first:
time, model, source, a preview of the state, the answer of each question with its confidence, latency and HTTP
status. Filter by model and source, search the state, open a row for the full request and response, copy the request
or **Replay in Playground** (the request in the JSON editor with its model selected). The last 2000 decisions are kept
in `<settings stem>.history.jsonl` (owner-only) next to the settings file, with media bytes replaced by their size;
**Clear history** empties it.

**Logs** shows the output of each `basal-serve` (the last 1000 lines per model, kept after unload for the last 10
unloaded models) and basal-ui's own event log (loads, downloads, readiness, errors, unloads, API server starts and
stops, benchmarks, auto-load decisions). `#logs/<model id>` opens one model's tab (the Overview row's **Logs** link);
follow mode keeps the view at the newest line; the text can be copied or downloaded. Logs are kept in memory only.

### API server

One port serves every loaded model. The API server runs inside basal-ui and forwards each request to the private
`basal-serve` of the model it names, so every request body must carry `"model"`: `400 MODEL_REQUIRED` without it,
`404 MODEL_NOT_FOUND` for a model that is not loaded (both list the loaded models). `GET /v1/models` lists the ready
models (the basal `models` entries and the OpenAI `data` list), `GET /health` names them, and `GET /v1/stats` returns
the statistics above (admin key only when keys are required — that is, basal-ui's own key). Details:
[API.md](API.md#several-models-behind-basal-ui).

| Setting | Default | Effect |
|---|---|---|
| Serve the API | Off | Off, the models answer only basal-ui. On, the API server listens on the configured host and port. |
| Exposure | This computer (`127.0.0.1`) | Local network binds `0.0.0.0`; use this only when remote clients need access. |
| Port | `8000` | TCP port of the API server; valid range is 1024–65535. |
| Require an API key | Off | Requires a named key on every API route except `/health`. Supports `Authorization: Bearer <key>` and `X-API-Key`. |
| Max context length | `4096` | Passed to each vLLM and SGLang `basal-serve` (`512`–`131072`). |
| CORS origins | Empty | Optional allowed browser origins, one per line; `*` allows every origin. |
| Decision log | Empty (off) | Every `basal-serve` appends its decisions (propensity records) to this writable file. |
| MCP at `/mcp` | Off | Adds the MCP streamable-HTTP endpoint when the MCP extra is installed. |
| Auto-load models | Off | Loads every model that was loaded when basal-ui last ran (in load order). |

**Security.** The server is loopback-only by default. Local-network exposure is unauthenticated unless **Require an API
key** is enabled; use a key before selecting **Local network**. Keys are generated with the `bsl_` prefix, stored as
SHA-256 hashes in the settings file and keys file, and their plaintext is shown only once when created. Copy it then;
it cannot be retrieved later. Revoking a key rewrites the live keys file and takes effect without a restart. Each
`basal-serve` listens on loopback only and always requires basal-ui's internal admin key, so only basal-ui and the API
server reach it; that key's record is not exposed in the UI.

**Restart server** restarts only the API server (instant; the models stay loaded) and applies the saved host, port,
key, CORS and MCP settings; until then the status line shows **Restart to apply**. **Max context length** and
**Decision log** belong to each `basal-serve`: a model loaded before such a change shows **Restart** in its Overview
row, which reloads only that model. The app checks the chosen bind port before starting the API server and reports
`port <port> on <host> is in use` if another process owns it.

The **Connect** panel provides copy-ready examples for curl (both `/v1/systemone` and `/v1/decisions`), Python
(httpx), TypeSafe and OpenAI SDKs, LiteLLM's `hosted_vllm` provider, and MCP, each with the selected model's id in
`"model"`. When authentication is disabled, examples omit API-key headers. Turn on MCP to use the streamable-HTTP
endpoint at `/mcp`: its `decide` and `decide_many` tools take a `model` argument and `list_models` names the loaded
models; the `basal-mcp` stdio bridge connects with `--model <id>`. Full request and response shapes, error envelopes
and compatibility details are in the [basal-serve HTTP API reference](API.md).

### Serving models from basal-ui

1. Start basal-ui and load one or more models. In **API server**, turn on **Serve the API**.
2. Choose **This computer** (`127.0.0.1`) for local clients. Choose **Local network** (`0.0.0.0`) only when another
   device must connect; enable **Require an API key** first. An unauthenticated LAN listener is accessible to other
   devices on that network.
3. Set the port (default `8000`). Enable **Require an API key**, then save and click **Restart server**.
4. Under **API keys**, enter a name such as `laptop`, choose **Create key**, and copy the displayed `bsl_...` secret.
   It is shown once; basal-ui cannot display it again.
5. Use **Connect** for a copy-ready client snippet. For example:

   ```python
   import httpx
   r = httpx.post("http://127.0.0.1:8000/v1/systemone", headers={"Authorization": "Bearer <copied key>"}, json={
       "model": "Remek/basal-2.0-mini-MLX-8bit",
       "state": "A user asks for a refund.",
       "questions": {"route": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "Billing", "support": "Support"}}}})
   print(r.json()["answers"]["route"])
   ```

6. To revoke access, choose **Revoke** beside the key and confirm. basal-ui rewrites the key file; the API server
   refuses that key on its next request without a restart.

The default settings file is `~/.config/basal-ui/settings.json`; set `BASAL_UI_CONFIG` to change its path. Next to it,
each running basal-ui writes two key files: `<settings stem>.keys-<pid>.json` (for example, `settings.keys-1234.json`),
the API keys from the settings plus that process's internal admin key, read by the API server; and
`<settings stem>.internal-<pid>.json`, the internal key alone, read by every `basal-serve` child, so a user key never
opens a child directly. basal-ui uses the internal key for its own requests (playground, API server, benchmarks). All
files are owner-only (`0600`) and hold SHA-256 hashes, never secrets. Two basal-ui instances may share one settings
file: a key created or revoked in either is written into every matching live instance's key file, the files of exited
instances are removed, and each instance deletes its own files when it quits. The statistics (`<stem>.stats.json`) and
the history (`<stem>.history.jsonl`) live next to it as well; both are written under the settings lock, so two
instances add their counts up and neither rotation drops the other's decisions (each History view shows only the
decisions its own instance answered or found in the file at startup).

**Auto-load.** Enable **Load the models when basal-ui starts** to remember every model that reached ready (an unload
forgets it; quitting basal-ui does not). On startup, basal-ui loads them in their load order and starts the API server
with the current settings; a model that is unknown or cannot run any more is skipped and noted in the Logs view.

**Troubleshooting.**

- **Port in use:** choose another port, save, and restart the server. basal-ui checks the configured bind address
  before starting the API server and reports `port <port> on <host> is in use`.
- **400 MODEL_REQUIRED / 404 MODEL_NOT_FOUND:** add `"model": "<id>"` with an id from `GET /v1/models` (or the
  Overview) to the body; the model must be loaded and ready.
- **401 Unauthorized:** check that **Require an API key** is on and send the copied secret as
  `Authorization: Bearer <secret>` (or `X-API-Key: <secret>`). Revoked keys stop working immediately; create a new
  key if the original is lost.
- **409 over memory budget:** unload a model, or raise the budget / lower **Keep free** under **API server → Memory**.
- **CORS preflight returns 400:** add the browser page's exact origin (scheme and host, with any port) to **CORS
  origins**, save, then restart the server. A foreign origin is rejected without an allow-origin header.
- **MCP extra missing:** install `basal[mcp]` in the same environment as basal-ui, then enable MCP and restart the
  server. The UI reports when the extra is unavailable.

For API routes, request and response formats, authentication details and compatibility notes, see the
[HTTP API reference](API.md). The Connect panel also includes TypeSafe, OpenAI, LiteLLM and MCP examples.

The browser app binds `127.0.0.1` by default and refuses changes (load, delete, settings) sent from other web origins,
so a web page cannot drive it through the browser; `--host 0.0.0.0` lifts the host check and prints a warning. The
NVIDIA, CPU and GGUF rules are covered by tests with simulated hardware; end to end it was run on Apple Silicon (MLX,
and basal-omni-1.0-mini with an image on MPS).

## Benchmark your GPU

Two tools are installed with the package. Both run out of the box on the bundled examples (44 Polish and English items
with gold answers in `basal/examples/`):

```bash
# offline: latency, throughput and agreement of serving modes (no HTTP); the first mode is the reference
basal-bench --model Remek/basal-2.0-4.5B --modes eager-fp32 fast --out bench.json

# end to end over HTTP against a running server (basal-serve ...)
basal-loadtest --url http://127.0.0.1:8000/v1/systemone
basal-loadtest --url http://127.0.0.1:8000/v1/systemone --synthetic 1500 --concurrency 16   # ~1.5k-token states

# identity gate: every prefix-reuse path of the fast engine against the eager per-prompt reference
basal-bench --identity --model Remek/basal-2.0-4.5B                    # fp32; --dtype bfloat16 as served
```

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with
concurrent clients (`--concurrency`); `--synthetic N` uses generated items with states of about N tokens. `basal-bench
--identity` runs single prompts, option orders and multi-question requests over one state, alone and batched, and
compares every readout with an eager forward of its own prompt (argmax agreement, max |Δp|).

basal-ui runs the same measurements on a loaded model from **Models → Loaded → Benchmark**, on the bundled examples
or your own file ([BENCHMARK_SETS.md](BENCHMARK_SETS.md)).

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers. For numbers that
describe *your* workload, write a JSONL file with a few hundred items in the same simple format (`{"state": ...,
"question": ..., "options": [...], "gold": <index>}`, `gold` optional) with realistic state lengths, and pass it with
`--questions my_items.jsonl`.
