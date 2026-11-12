# Local LLM: own inference engine + web UI + statistics

Chat for `models/MiniCPM5-2B-Q8_0.gguf` where **inference is written by hand in
Rust**: GGUF parser, Q8_0 kernels, BPE tokenizer, 42-layer forward pass, KV
cache, sampler, streaming. No llama.cpp, candle, or other model server at
runtime — llama.cpp is used once, only to record the comparison fixtures.

```
apps/engine      lib  GGUF → Q8_0 → matmul → tokenizer → forward pass → sampler
apps/api         lib  domain / application / infrastructure / presentation (Kana)
apps/gateway     bin  Axum + serve SPA
apps/cli         bin  REPL terminal (next phase)
.migrations      crate sea-orm-migration (SQLite)
web              static SPA: index.html + app.js + markdown.js + style.css
```

## Running

```bash
cargo build --release
cargo run --release -p llm-gateway
# open http://127.0.0.1:8099
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `models/MiniCPM5-2B-Q8_0.gguf` | model path |
| `LLM_CONTEXT` | `8192` | context size (KV f16 ≈ 43 KB/token) |
| `LLM_THREADS` | number of cores | rayon threads for matmul (CPU path) |
| `LLM_GPU` | automatic | `0` forces the CPU even when a GPU is present |
| `LLM_CUDA_ARCH` | `sm_89` | PTX architecture at build time (needs `nvcc`); the prompt path uses `mma.sync`, so it needs sm_80 or newer |
| `LLM_DATABASE` | `sqlite://llm.db?mode=rwc` | statistics database |
| `LLM_ADDRESS` | `127.0.0.1:8099` | bind address |
| `LLM_LOG` | `info` | tracing filter |

SQLite migrations run automatically at start.

The web interface follows the shape of a conventional chat application: a
conversation sidebar + new-conversation button, a transcript with message
bubbles and markdown (language-labelled code blocks, copy button), a floating
composer with temperature and answer-length settings, and a collapsible
telemetry rail. Dark and light themes, both passing WCAG AA contrast for body
text. No build step and no CDN: `web/` is served as-is by the gateway.

Engine and HTTP request tracing use structured fields. Filter by diagnostic
need without changing benchmark output:

```bash
LLM_LOG=llm_engine=debug,llm_api=info cargo run --release -p llm-gateway
LLM_LOG=llm_engine=trace cargo run --release -p llm-cli -- bench -p 128 -n 32
```

The `debug` level logs the forward phase, layers, prefill chunks, and a
generation summary. The `trace` level adds every token and the CUDA kernel
launch geometry; tracing output is written to stderr.

## API

```
GET    /api/health
GET    /api/model
GET    /api/conversations            POST /api/conversations          201
GET    /api/conversations/{id}       DELETE /api/conversations/{id}   204
POST   /api/conversations/{id}/messages   → text/event-stream
POST   /api/conversations/{id}/stop       202
GET    /api/stats/overview
GET    /api/stats/runs?limit=20
GET    /api/stats/timeseries?days=30
```

Streaming uses SSE with a fixed order:

```
event: start   data: {"prompt_tokens":123}
event: token   data: {"text":"Halo"}
event: stats   data: {"generated_tokens":88,"tokens_per_second":17.1,...}
event: error   data: {"message":"..."}
```

A client that disconnects stops generation (cancel flag), and partial statistics
are still recorded.

## Statistics

Stored per run in SQLite: prompt tokens, generated tokens, prefill ms, TTFT ms,
decode ms, total ms, tok/s (computed from decode, not including prefill),
inter-token latency, ctx used, `stop_reason`, sampler summary, peak RSS.
Aggregates: total runs/tokens, mean + p50/p95 tok/s, mean TTFT, 30-day daily
buckets.

## Proof (not "looks like it runs")

All fixtures were recorded from llama.cpp b11429 and then committed; the tests
never call llama.cpp.

```bash
cargo test --workspace                      # unit + API (no model)
cargo test --release -p llm-engine -- --ignored   # needs the real model
./scripts/record-golden.sh tokenizer|logits|greedy  # re-record fixtures
```

| Test | Contents | Status |
|---|---|---|
| `tokenizer_golden` | 39 sentences (ASCII, double spaces, newlines, ZWJ emoji, CJK, Arabic, Thai, long numbers, CRLF, U+3000, ChatML markers) → ids must match `llama-tokenize` exactly | 39/39 identical |
| `logits_golden` | 130,560 logits compared in full against `llama-debug --save-logits`, two prompts | argmax + top-5 identical, max\|Δ\| ≈ 0.12 |
| `greedy_golden` | 32 greedy tokens (temp 0) vs llama-server `/completion` | identical |
| `ops`, `quant_q8`, `matmul` | rmsnorm, rope, softmax, silu, Q8_0 dot vs f64 reference, batched matmul vs naive | pass |
| `gpu_matmul` | CUDA kernel vs CPU kernel on real weights, n=1/8/64; the GPU pipeline must produce the same argmax | max\|Δ\| ≈ 1.5e-6, argmax 8181 |
| `http_api` | router + use cases + in-memory SQLite + fake engine: 201/204/404/400, SSE ordering, run recording | pass |
| `tactics`, `workspace`, `manifest` | the tactics table never selects a `Rejected` variant for any shape and every row carries evidence; the planner never stacks two buffers alive in the same phase; the build identity is 64 hex chars and stable | pass |

Why the logits tolerance is 0.2: ggml dequantizes Q8_0 weights to f16 inside its
matmul, whereas this engine computes an exact Q8_0 dot. The difference is ~0.1 on
logits of magnitude ~9, which is the accuracy of llama.cpp's own path. What must
be identical is the decision: argmax, top-5, and the 32 greedy tokens.

## GPU (CUDA)

If an NVIDIA GPU is present, the engine uses it automatically: weights, KV
cache, and all activations stay in VRAM, and the forward pass is run by CUDA
kernels written in this repo (`apps/engine/kernels/gpu.cu`), not through any
inference library. The runtime only needs the driver (`libcuda.so`); `nvcc` is
only needed at build time to compile the PTX, and if it is missing the crate
still builds and automatically runs on the CPU.

```bash
cargo build --release        # PTX for sm_89 is embedded
LLM_GPU=0 cargo run -p llm-gateway   # force CPU
```

An important design decision: the CPU/GPU boundary is in the **model**, not in
matmul. The first version only moved matmul to the device and kept activations
on the host; that made decode 67 ms/token (slower than the CPU's 46 ms) because
of 294 H2D/D2H round-trips + synchronization per token. Once the whole forward
pass moved to the device, decode became 27.6 ms/token. One forward pass, two
backends (`apps/engine/src/compute.rs`), so there are no two copies of the layer
logic.

If VRAM is not enough for the weights + 512 MB headroom, the model automatically
falls back to the CPU at load, instead of failing in the middle of an answer.

### Our own benchmark, shaped like llama-bench

`llm-cli bench` follows the `llama-bench` interface so that two numbers can be
compared directly: `-p` list of prompt lengths, `-n` list of generation lengths,
`-r` number of repetitions, `-o json`, and output as a `test | t/s` table with
`±` spread (sample standard deviation, the same formula as llama-bench).

### Measured performance

RTX 4050 Laptop 6 GB (sm_89, driver 615.78.08), 2.68 GB model, context 4096,
~3.0 GiB VRAM used, median of 9 decode steps per point, gateway shut down:

| Context | GPU decode | CPU decode | GPU prefill | CPU prefill |
|---|---:|---:|---:|---:|
| 0 tokens | **68.9 tok/s** (14.5 ms) | 21.7 tok/s (46.0 ms) | 16 ms | 50 ms |
| 128 tokens | **67.4 tok/s** (14.8 ms) | 19.5 tok/s (51.4 ms) | 3,843 tok/s (0.03 s) | 35 tok/s (3.63 s) |
| 512 tokens | **66.6 tok/s** (15.0 ms) | 15.6 tok/s (64.1 ms) | 4,856 tok/s (0.11 s) | 30 tok/s (16.7 s) |
| 1024 tokens | **63.5 tok/s** (15.7 ms) | 11.1 tok/s (90.3 ms) | 5,202 tok/s (0.20 s) | 27 tok/s (36.8 s) |
| 2048 tokens | 60.7 tok/s (16.5 ms) | — | — | — |
| 4090 tokens | 54.0 tok/s (18.5 ms) | — | — | — |

Decode is 3.0–5.4× faster, prefill 9.3–10.6× faster. One 120-token generation in
the CLI: 67.0 tok/s, TTFT ≈ 140 ms.

The decode floor on the 4050 (≈192 GB/s) is 2.68 GB of weights / 192 GB/s ≈ 14 ms
per token; the measured value is 14.5 ms at context 0, so decode sits at 96% of
that floor. Two things trim the remaining distance:

- **weight scales stored as f16, not f32** — the scales really are f16 in the
  GGUF file, so storing them as f16 changes not a single bit of the result, but
  trims the scale plane from 0.32 GB to 0.16 GB. Total weights return to 2.68 GB
  exactly as in the file, and decode drops 15.5 → 14.5 ms;
- **f16 KV cache, like llama.cpp** — at 4090 tokens the KV used to be 704 MB and
  was re-read every token (3.7 ms); now it is 352 MB (1.85 ms). The decode curve
  becomes 14.5 / 14.8 / 15.0 / 15.7 / 16.5 / 18.5 ms at context
  0 / 128 / 512 / 1024 / 2048 / 4090.

### Against llama.cpp, and how much VRAM it takes

`llm-cli bench` prints its own VRAM footprint on the last row of the table. The
numbers below were taken with the SM clock locked at 1800 MHz for both engines,
the same model, `-r 5`, llama.cpp `8ab29fa` with `-ngl 99`:

| | llm-cli | llama.cpp | ratio |
|---|---:|---:|---:|
| pp128 | 3,585 | 4,180 | 85.8% |
| pp512 | 4,659 | 5,316 | 87.7% |
| pp1024 | 4,489 | 5,246 | 85.6% |
| tg128 | 68.9 | 69.7 | 98.8% |

Context 4096, 1024-token prefill:

| | weights | KV | scratch | total on device |
|---|---:|---:|---:|---:|
| llm-cli | 2.50 GiB | 0.16 GiB | 0.12 GiB | **2.78 GiB** |
| llama.cpp | 2.23 GiB | 0.16 GiB | 0.055 GiB | 2.45 GiB |

The KV cache is exactly the same — 41 KiB per token, f16, same geometry. The
remaining 0.33 GiB difference has two causes, and both are llama.cpp's choices:
271 MiB of its weights stay in the mmapped file and are read from the host, and
all of its operations share a single 56 MiB compute buffer, whereas this engine
keeps nine separate scratch buffers. The percentages in the previous table were
measured without locking the clock and with a different llama.cpp build, so the
two tables are not comparable.

Prefill used to be ~670 GFLOPS (5–10% of fp32 peak); now the matmul itself runs
at 12.5 TMAC/s through int8 tensor cores and attention through f16 tensor cores
(see the sections below).

### Why decode needs weight repacking

The GGUF format stores Q8_0 as 34-byte blocks (`f16 scale` + 32 `i8`). As long as
a row is read as-is, one lane can load only 1 byte per instruction and one warp
moves 32 bytes — a quarter of a cache line. That is what made decode stop at
27.6 ms/token (≈97 GB/s effective).

On upload, each Q8_0 tensor is split into two planes: `scales` (k/32 `f32` per
row) and `quants` (k `i8` per row), both 32-byte aligned. That way one lane loads
8 quantized values at once through a single 64-bit load, plus two `float4` loads
for the activations. The result is 15.4 ms/token (≈174 GB/s effective).

The price: 1.125 bytes per parameter instead of 1.0625 — weight VRAM rises from
2.68 GB to ~2.83 GB, and the repack adds ~0.4 s at load.

### Prefill: int8 × int8, like ggml

The earlier 16× prefill gap came from a different algorithmic class, not from a
lack of tuning: the old version dequantized every weight to f32 and then
multiplied it with FMA, whereas ggml quantizes the activations to q8_0 and
multiplies int8 by int8. What was ported from llama.cpp:

- `quantize_row_q8_0` (ggml/src/ggml-cpu/arch/x86/quants.c) — one f32 scale per
  32 elements, `q = round(x·127/amax)`;
- `ggml_vec_dot_q8_0_q8_0` — `_mm256_sign_epi8` + `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16`, so 32 multiplies per instruction pair, not 8;
- `mmq` (ggml/src/ggml-cuda/mmq.cuh) — `__dp4a` on the GPU, 4-row tiles in shared
  memory as int8, and tile copies through two-stage `cp.async` so that the next
  tile is already in flight while the current one is multiplied.

The result: GPU prefill 275 → 534 tok/s at context 128 (and 235 → 824 tok/s
through the CLI on a 215-token prompt), CPU 28 → 35 tok/s. Decode untouched.

Then the prompt path moved to tensor cores, and this time the geometry was taken
from llama.cpp instead of guessed:

| Parameter | Value in llama.cpp | File |
|---|---|---|
| threads per block | 256 (8 warps) | `mmq-config-ampere.cuh:104` |
| resident blocks per SM | 1 | same |
| output rows per block (I) | 128 | same |
| tokens per block (J) | 8–128 | same |
| rows per warp | 32 when J ≥ 48 | `mmq.cuh:180` |
| k per iteration | 256, tile 32 | `mmq.cuh:9,116` |
| A fragment | `ldmatrix.x4` | `mma.cuh:855` |
| B fragment | `load_generic` ("faster than ldmatrix") | `mmq-vec-dot.cuh:250` |
| scale accumulation | `sum += C*dA*dB`, one fma per element | `mmq-vec-dot.cuh:195` |

What changed in this kernel: one block of 128 rows × 64 tokens with 8 warps, one
warp 32 rows (two m16 minitiles) so that one B fragment feeds two `mma`s, both
operands in shared memory, A through `ldmatrix.x4`, and per-block scales become
one fma per accumulator element. The 64-token capacity is tied to
`PREFILL_CHUNK` in `session.rs`; if the chunk is raised without raising the tile,
matmul returns `Unsupported` instead of silently dropping tokens.

The result: matmul alone at **5.5 TMAC/s** (up from 2.1), and prefill 818 → 1,829
tok/s at context 128. For comparison, llama.cpp's 4,431 tok/s prefill equals
≈7.5 TMAC/s — so its matmul is now at 73% of the llama.cpp path.

Two expensive traps, recorded because neither is visible from the code:

- `ldmatrix` takes a **64-bit generic address** (`"l"`), not the 32-bit shared
  address returned by `__cvta_generic_to_shared` the way `cp.async` does. Giving
  it a shared address makes ptxas read it as a global address and the whole
  launch hits an illegal memory access;
- measurements were misleading several times because another test process was
  still holding 2.8 GB of VRAM: `Compute::auto` then silently fell back to the
  CPU, and what looked like "a slower kernel" was really a different machine.

The remaining prefill gap now grows with context: 1,829 tok/s (ctx 128) drops to
1,071 tok/s (ctx 1024), while llama.cpp stays flat. The cause is measured — at
ctx 1024 roughly 47% of prefill time is in attention, and the prefill path still
uses the serial per-head `attention` kernel (only the decode path splits across
blocks). Tiling attention for prefill is the next piece of work.

### Weights read once per 1024 tokens, not per 64

This is the biggest cause of the prefill gap, and it is not about arithmetic. A
prefill chunk used to be 64 tokens, meaning all 2.8 GB of weights were re-read
every 64 tokens — 512 tokens means eight full passes. llama.cpp uses a much
larger ubatch.

Now the chunk is 1024 tokens, and the prompt kernel's grid is ordered with the
**token tile as the fastest dimension**, so blocks running concurrently use the
same weight tile and that tile stays in L2 instead of being re-read from DRAM.
Grid `(n/64, out/256)`.

Results at 8 warps: pp128 2,190 → 2,813, pp512 1,880 → 3,000, pp1024 1,576 →
2,557. After that the token tile was raised to 128 (one weight tile in shared
memory serves 128 tokens instead of 64): pp128 2,929, pp512 3,190, pp1024 2,747.

The kernel's final shape: 8 warps, 128 rows, 128 tokens per block, k chunk 256,
178 registers without spills, 77,824 bytes of shared memory per block.

### A larger k chunk: 128 → 256 elements

The prompt kernel stages one `k` slice into shared memory per iteration. At 128
elements a slice is only 24 KB and each iteration must wait for its copy to
finish before multiplying; at 256 elements the slice is twice as large, so that
wait covers less of the arithmetic work. The size exceeds the 48 KB limit for
static shared memory, so this kernel now uses dynamic allocation and
`cuFuncSetAttribute` to raise the limit (58,368 bytes).

Result: pp128 1,959 → 2,104 tok/s, pp512 1,607 → 1,738, pp1024 1,297 → 1,378.
Decode untouched.

What did **not** help, and why: enlarging the prefill attention tile from 16 to 32
queries so that K/V is read half as often. The result was actually slower
(1,932 / 1,574 / 1,261) for two reasons at once — the K and V buffers must be
used alternately, so values are staged in a separate pass, and registers per
thread rise 65 → 98, lowering occupancy. Reverted.

### Prefill attention: tiled queries, online softmax

The old kernel launched one block per (head, token) and wrote every score row
through global memory, so the longer the context the more expensive it got. Its
replacement follows the shape of `fattn-tile.cuh`: one block per head and 16
query positions, K/V staged into shared memory 32 positions at a time, and
softmax kept online (`running_max`/`running_sum` plus an `exp(m_old - m_new)`
correction on the accumulator) so that no buffer the size of `visible` is ever
formed.

Two things determine its shape:

- **K is staged transposed** (`k_tile[dim][key]`). With a row-major layout, one
  lane reads `k_tile[key][dim]` for its own key and the whole warp falls into the
  same bank; transposed, each lane touches a different bank and there are no
  conflicts. Side effect: each lane completes one full score, so there is no
  `warp_sum` in the inner loop.
- **One lane, one key**. My first version mixed two schemes — lanes computed dots
  for different keys and then `warp_sum` added them up, which means summing
  scores from unrelated keys. The result was garbage that only showed up on long
  prompts.

Result: prefill 1,829 → 1,954 tok/s (ctx 128), 1,437 → 1,600 (512), 1,071 → 1,292
(1024). Decode unchanged.

The remaining attention cost now comes from repeated K/V reads: one head has four
query blocks, so every K/V row is read four times per chunk per head. At ctx 1024
that is about 2.7 GB per chunk for 42 layers. Raising the query tile from 16 to
32 would halve it, but shared memory becomes 52 KB and it needs dynamic shared
memory (`cuFuncSetAttribute`) because of the 48 KB static limit.

### Per-kernel profile (`LLM_PROFILE=1`)

`LLM_PROFILE=1` wraps every kernel launch in CUDA events and prints a table of
time inside the GPU when the process finishes — kernel time, not queue time. For
one 512-token prefill:

| Kernel | Total ms | Calls | ms/call |
|---|---:|---:|---:|
| `q8_matmul_mma` | 81.9 | 294 | 0.279 |
| `attention_prefill` | 25.1 | 42 | 0.596 |
| `quantize_rows` | 4.8 | 168 | 0.029 |
| `silu_mul` | 7.8 | 42 | 0.186 |
| `residual_add` | 3.5 | 84 | 0.042 |
| `rmsnorm_rows` | 2.6 | 85 | 0.031 |
| `rope` | 2.1 | 84 | 0.025 |
| the rest | ~1.9 | — | — |
| **total** | **124.2** | | |

Prefill wall time is ~126 ms, so it is almost entirely kernels — there is no
leakage between launches. Matmul runs at 13.0 TMAC/s inside the model versus
13.5 TMAC/s measured alone, so that path is already at 96% of its own speed.

### Where prefill time goes (pp512, 149 ms)

Measured by switching off one part at a time, not estimated:

| Part | Time | Share | Note |
|---|---:|---:|---|
| Matmul (all, including the output projection) | 106 ms | 71% | the kernel itself 14.2 TMAC/s; measured here 9.6 TMAC/s |
| — of which: activation quantization | 20 ms | 13% | used to be 7 times per layer, now 4 (q/k/v share one) |
| Attention | 32 ms | 22% | down from 43 ms after four queries per key read |
| Elementwise (rmsnorm, rope, kv_append, silu, residual) | 17 ms | 12% | already at the DRAM bandwidth limit |

The gap between 14.2 TMAC/s (kernel alone, measured with 267 MB of weights
streamed from DRAM so that it is not an L2 case) and 9.6 TMAC/s (inside the
model) is **wave quantization**: `out = 2048` with a 128-row tile gives 16 blocks
per token tile, and 20 SMs means not even one wave is full — 64 blocks run as
3.2 waves, so 80% efficient. For `out = 6144` the figure is 96%.

To catch up with llama.cpp (87 ms at pp512) three things must come down together:
matmul to ≈75 ms (a full kernel, with no leftover wave), attention to ≈5 ms
(needs f16 tensor cores like ggml's `fattn`), and elementwise to ≈8 ms (fusion).

### Prefill attention: four heads per block

Sixteen heads share two KV heads, so one block per head reads the same K/V rows
eight times. One block now serves four heads that are guaranteed to share one KV
head (`FA_HEADS`), with eight warps: two warps per head, eight query positions
each. The K/V tile in shared memory becomes the only copy, and K/V reads per
chunk drop from 16 block-reads to 4.

The shared memory needed is 72 KB (four-head queries 32 KB, K 16 KB, V 16 KB,
probabilities 8 KB), so this kernel uses dynamic allocation and
`cuFuncSetAttribute` — same as the prompt kernel.

Result: pp128 2,104 → 2,190 tok/s, pp512 1,738 → 1,880, pp1024 1,378 → 1,576.
Decode untouched. Verified against a host reference: maximum difference 4.5e-7 on
40 tokens, 16 heads, with `start = 32`.

Eight heads per block (K/V then read only twice per chunk) needs 112 KB and
exceeds the 99 KB per-block limit on sm_89, so that step waits for a smaller
query tile or f16 queries.

After that the block was changed to **two heads with no probability buffer**:
probabilities stay in registers and reach the value pass through `__shfl_sync`,
so shared memory drops to 48 KB and two blocks fit per SM (16 warps, not 8).
Attention 27.8 → 25.1 ms. But K/V traffic becomes twice as frequent, so pp128
drops 3,518 → 3,438 while pp512 and pp1024 rise (4,063 → 4,099 and
3,734 → 3,758).

### Prefill attention: two queries per key read

The attention kernel ran at 0.52 TMAC/s — 11% of fp32 peak — because one scalar
dot needs two shared reads for every multiply-add, and shared memory can serve
only one transaction per cycle. That caps a warp at sixteen multiply-adds per
cycle, no matter how fast the arithmetic units are.

Now **four** queries run through the same key read, and four query dims are
fetched as a single `float4` (one broadcast for the whole warp). Keys stay
dim-major — key-major makes every lane land in the same bank and actually slows
things down (measured: 3,036 vs 3,143).

Result: pp128 2,929 → 3,190, pp512 3,190 → 3,605, pp1024 2,747 → 3,355.
Attention cost drops from 27% to 20% of pp512 prefill (measured by switching the
attention calls off: 3,488 → 4,382).

### Decode attention: positions split across blocks

This model has 16 heads, so one block per head gives only 16 blocks — on a card
with 20 SMs that means some SMs idle, and each block walks the whole KV serially
on its own. The cost shows up as a rising curve: 15.4 ms/token at context 0, but
26.2 ms at context 1024.

Now positions are split across blocks (`attention_split`): one block holds one
head and one slice of positions, keeping its own max, weight sum, and weighted V
sum, then `attention_combine` merges them with `exp(max_i - max)` like flash
attention. The number of slices is chosen so that each block gets ~64 positions
(bounded by the 256-position shared window), so context 1024 yields
16 × 16 = 256 blocks instead of 16.

Result: the decode curve flattens to 15.5 / 15.8 / 16.0 / 16.6 ms at context
0 / 128 / 512 / 1024, and 17.8 / 20.3 ms at 2048 / 4090. At context 1024 that is
38.1 → 60.1 tok/s.

A mistake that happened along the way, recorded because the shape is classic: the
first combine version multiplied the numerator by `sum_i` **and** used `sum_i` as
the denominator, so the normalizations cancelled each other out and the result
was the raw weighted V sum — output was correct only when `splits = 1` (context
≤ 64), which happened to be the first case I tested by hand. What caught it was
`greedy_golden`, not my eyes: the text still began with " Paris" and then
rambled. The correct formula puts `sum_i` only in the denominator.

### Why the CPU is still used as a reference

Both backends run the **same forward pass** and are tested against the same
llama.cpp fixtures. The `gpu_matmul` test compares the GPU kernel with the CPU
kernel on real weights (max |Δ| ≈ 1.5e-6), and `logits_golden`/`greedy_golden`
pass on both backends with the same numbers (max |Δ| 0.118 / 0.572 / 0.274; 32
greedy tokens identical). So the GPU is not a separate "roughly right" path.

### f16 KV, f16 weight scales, and elementwise fusion

Four small changes that close the decode gap and trim prefill:

- **f16 KV cache**. llama.cpp stores KV in f16, this engine used f32. Now
  `kv_append` converts on write, and all three attention kernels convert back on
  read. At context 4090 the KV re-read every token drops 704 → 352 MB, so decode
  20.3 → 18.5 ms — and the numerics actually move closer to the llama.cpp
  fixtures (all fixtures were recorded from their f16 cache).
- **f16 weight scales**. The Q8_0 repack used to store the f16 scales as f32.
  Since the scales are f16 in the file, storing them as f16 is identical
  bit-for-bit, and the scale plane drops 0.32 → 0.16 GB: decode 15.5 → 14.5 ms.
  There is one trap: the first version loaded the scales with `LDG.U16` per
  element in the prompt kernel and prefill actually dropped 7%; loading two
  scales at once as `half2` (a 32-bit access, the same shape as the old f32)
  restored everything.
- **`silu(gate)·up` quantized directly** (`silu_mul_quantize_rows`), without
  writing the f32 activation and then reading it back — saving one full n×6144
  pass per layer. Careful: the decode path (n < 8) reads f32 activations
  directly and never sees the prepared buffer, so below that threshold the
  product is still written as before.
- **`rmsnorm` quantized directly** (`rmsnorm_quantize_rows`) for the same reason.
  Explicit float4 vector loads are needed here — the scalar version runs twice as
  slow as the pair of kernels it replaces. The matmul residual was also tried
  fused into the mma kernel epilogue, but the scattered bias loads in the
  epilogue turned out to be more expensive (81 → 87 ms) than the separate
  residual pass it replaced (3 ms), so that fusion is only used for decode
  (n = 1), where one load per row does not scatter.

The prompt embedding was also made a single launch (`embedding_rows`) instead of
one launch per token.

### Prefill attention: f16 tensor cores

The previous fp32 kernel was bound by **shared memory** bandwidth, not DRAM: a
scalar dot needs two shared reads per MAC, so it does not matter how fast the
arithmetic units are. The real fix is to multiply on tensor cores, like ggml's
`fattn`:

- one block holds **eight heads** (this model's entire GQA group) and 16 query
  positions; one warp = one head, 16 queries = one m16 tile;
- Q and K/V are staged into shared memory as f16 (48 KB, two blocks per SM); K
  stays dim-major so that the B fragment (=Kᵀ) loads directly with `ldmatrix`;
- QKᵀ is computed with `mma.sync.m16n8k16.f16` and the scores stay in registers —
  **the scores that have already been exp'd are the A fragment of PV**, so the
  value step is also `mma` with no extra shared reads, the same trick as ggml;
- softmax stays online: `running_max`/`running_sum` per query row plus an
  `exp(m_old − m_new)` correction on the O accumulator, exactly like the fp32
  kernel.

Result: prefill attention 25.5 → 10.5 ms at pp512 (and proportionally at other
sizes), which lifts pp128 3,438 → 3,843, pp512 4,099 → 4,856, and pp1024
3,758 → 5,202 tok/s. The logits delta against llama.cpp actually shrinks
(logits_golden_long 0.321 → 0.274) because f16 is their attention precision.

The two traps that cost the most time, recorded because neither is visible from
the code:

- **the `m16n8` C fragment is d0/d1 = (row r, columns 2t and 2t+1), d2/d3 = row
  r+8** — not d0/d2 = row r as written in the PTX ISA text I read. The symptom:
  every value wrong except the first row;
- **the B fragment needs `ldmatrix…x2.trans`**. Without the transpose, the lane
  distribution gives B[lane/4][2t] whereas mma asks for B[2t][lane/4]; row 0
  happens to be right, so the error looks like a masking problem.

Both were pinned down with a standalone micro-probe comparing the kernel against
a CPU reference on random matrices, not through the full forward pass.

### Comparison with llama.cpp

Both are measured with a harness of the same shape: `llama-bench` on the
llama.cpp side, and `llm-cli bench` on this engine's side — the same test names
(`pp`/`tg`), the same number of repetitions, and spread reported as `±`. The
commands correspond exactly:

```text
llama-bench -m models/MiniCPM5-2B-Q8_0.gguf -ngl 99 -p 128,512,1024 -n 128 -r 3
llm-cli bench -m models/MiniCPM5-2B-Q8_0.gguf -p 128,512,1024 -n 128 -r 3
```

`pp` measures prefill of N tokens only; `tg` prefills a short prompt and then
measures the decode phase only, so a slow prefill cannot hide in the generation
number. llama.cpp b11429 was built from source with CUDA (`-DGGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=89`), same model, same machine, no other process using
the GPU.

Both sets of numbers below were measured with the **SM clock locked at 1800 MHz**
(`nvidia-smi -lgc 1800,1800`). Without the lock, the laptop 4050's boost drifts
20–25% between processes and the comparison loses its meaning: the same
measurement can give 4,000 or 5,200 tok/s purely because of the card's starting
temperature.

The engine column was re-measured after the memory-plan and CUDA graph work
(`bench -p 128,512,1024 -n 128 -r 3`), the llama.cpp column stays from the same
locked session: only this side changed, and the spread is written down too.

| Test | llm GPU | llama.cpp GPU | Ratio |
|---|---:|---:|---:|
| pp128 | 3,718 ± 36 tok/s | 3,788 ± 235 tok/s | 98% |
| pp512 | 4,740 ± 2 tok/s | 4,735 ± 73 tok/s | 100% |
| pp1024 | 4,566 ± 2 tok/s | 4,677 ± 92 tok/s | 98% |
| tg128 | 70.27 ± 0.15 tok/s | 69.3 ± 0.04 tok/s | 101% |

The 100% and 101% figures sit inside llama.cpp's measurement spread (±73 tok/s at
pp512, ±0.04 at tg128), so the reading is "equal within noise", not "faster".

On CPU (llama.cpp `-ngl 0 -t 20`, this engine `LLM_GPU=0`): decode 16.9–17.2 vs
17.1–19.5 tok/s, prefill 70–99 vs 27–35 tok/s.

| Metric | llm GPU | llama.cpp GPU |
|---|---:|---:|
| VRAM (ctx 4096) | 2.76 GiB | 2.49 GiB |
| Load + 1 token (wall) | 0.9 s | 1.32 s |

### What has already closed part of the gap

- **Matmuls that share activations leave in one launch** (`MMA_GROUP_MAX`). q, k
  and v read the same normalized rows; as three launches each only fills 4 + 1 +
  1 waves out of 34 waves per layer, so their row space is merged into 4.
  gate|up too, but only up to 512 tokens: at 1,024 tokens the merged form
  measures 2.40 ms against 2 × 0.70 ms separate (`GROUP_FFN_MAX_TOKENS`).
- **Per-lane weight scales hoisted out of the token loop** — sixteen values per k
  slice, not two loads per mma — and activation scales stored block-major so that
  a lane's two columns arrive in one 64-bit load. Six shared loads per mma become
  three.
- **All loads for one k slice are issued before the first store.** Weights,
  activations, and both scale planes depart together, so a block waits once per
  slice instead of four times: 99.9 → 90.5 ms matmul at pp512. This is what the
  profiler reads as `long_scoreboard` with 8 warps.
- **Sixteen warps and 256 rows per block.** The activation tile used to be staged
  once per 128-row block, so at 512 tokens activations were copied as often as
  weights: 200 MB of tile copies per layer against 201 MB of weights, 16.8 GB for
  the whole prefill. With 256 rows, half of that traffic disappears, and the SM
  has 16 warps to hide the staging latency: 90.5 → 83.3 ms, tensor pipe
  29% → 35%.
- **B fragment in one load.** The quantizer writes each 32-element block as (0-3,
  16-19, 4-7, 20-23, 8-11, 24-27, 12-15, 28-31), so the two words a lane feeds to
  mma are adjacent and arrive in one 64-bit load instead of two. That permutation
  happens during quantization, so the tile copy stays a straight 16 bytes; only
  the prompt path sees it, decode reads raw f32 activations.
- **Leftover tiles are cut along the token axis** (`wave_plan` + `tail_slices`)
  so that the last wave is not empty: +4% at pp512. The slices write different
  columns, so there are no partial sums to merge.
- **Prefill attention: heaviest tile first.** The causal mask makes the last
  query tile see the whole context, so load across blocks rises linearly while
  blocks are scheduled in index order. Reversed: 10.5 → 8.4 ms at pp512.
- **Decode**: q|k|v and gate|up in one launch (295 → 169 launches per token),
  attention split twice as fine, rmsnorm with one barrier and vector loads.

### What came from netra-kernel

Four mechanisms were taken from the Netra AOT compiler (its research write-up is
in `docs/research/netra-kernel.md`; the inventory of the CUDA surface available
to the kernels in this repo is in `docs/research/cuda-surface.md`); what was not
taken is the compiler itself, because here the "compiler" is `build.rs` + nvcc +
one `const` table.

- **Build identity.** `build.rs` records the sha256 of `gpu.cu`, the sha256 of
  the PTX that is actually compiled, the arch, the nvcc version, and every
  `.entry` in the PTX. `BuildManifest::identity()` hashes all of it, `/api/model`
  reports it, and `bench` prints it below the table — so the numbers in this
  README point at one artifact, not at a commit that may since have been rebuilt.
  Every run in SQLite also stores `build_hash`, through its own migration.
- **Kernel plan at load time.** `attention_path()` used to decide per call (shape
  test + `LLM_ATTN_OLD` env + token threshold), while `attention_scores()`
  recomputed the same decision for scratch sizing — two places that could diverge
  unnoticed. Now `tactics.rs` chooses once when the model loads, from a table in
  which every row carries the measurement that justifies it: the fp32 tile
  `Verified` behind the f16 tensor cores, eight-heads-per-block `Rejected`
  because 112 KB of shared memory does not fit in 99 KB, and shapes without a
  tile fall back to the serial path **with the reason recorded**. Three rules the
  tests guard: a `Rejected` variant is never chosen for any shape, the regime on
  the common path always states its reason, and merging stops at 512 tokens — at
  1,024 tokens it measures 2.40 ms against 2 × 0.70 ms.
- **Static memory plan.** `workspace.rs` declares every scratch buffer with its
  lifetime phase range, then places them: two buffers whose ranges do not meet
  share one region, and buffers alive in the same phase are never stacked. One
  arena per session, so a token allocates nothing — previously 13 allocations +
  13 frees per token. Sizes follow the kernel plan, so the path that keeps
  probabilities in registers plans a zero-sized score grid, not the 537 MB the
  serial path needs.
- **Decode graph.** One decode step = 169 launches; now it is recorded once and
  replayed with a single driver call. Three things had to move for the graph to
  be valid: the engine has its own stream (the legacy default cannot be
  captured), the values that change every token (`position`, `visible`) are read
  by the kernels through pointers to device memory, and embedding + position
  upload stay outside the graph because both need a host copy. Because the
  attention grid widens as the context fills, the session keeps one graph per
  split count — Netra's "profile per shape" idea. Pinning the grid at capacity
  yields a single graph, but it costs 69.0 → 59.0 tok/s, so that cache is not
  decoration.

### The remaining gap, and where to go

- **Prefill matmul is within ±2–4%** (83 ms against ~80 ms at pp512). What is
  still visible in the profiler: `wait` and `mio_throttle` in the inner loop,
  i.e. shared memory queuing — no longer tile copies. What we do not yet have
  from llama.cpp: *stream-k*, which splits the k axis so that no wave is half
  empty (most noticeable at pp128: a 2048-row projection is only 8 tiles).
- **Prefill attention is still 2.3× slower** (11.1 ms against ~4.9 ms) and it is
  now 10% of prefill runtime. The profile is clear: tensor pipe 11%, and the two
  top stalls are `long_scoreboard` 5.6 and `mio_throttle` 5.2 — blocks idling
  while waiting for K/V tiles that are staged scalar-wise and re-read 32× per KV
  head. What is needed is a larger query tile (K/V read 16× instead of 32×) with
  enough warps to hide its staging, not a one-line patch.
- **Decode 101%**: DRAM floor 2.39 GB/token ÷ 187.8 GB/s = 12.7 ms, its matmul
  13.4 ms (178 GB/s). The gap between launches is already gone thanks to the
  graph; the remaining ~0.9 ms is small kernels actually doing work, and what is
  left above the DRAM floor is only the difference between 178 and 187.8 GB/s.
- **CPU**: prefill is still 2.5–3× behind and decode degrades with context (the
  CPU path does not use attention tiling yet).

### Two bugs found while chasing these numbers

- **`quantize_rows` did not check bounds.** Its grid is rounded up to a multiple
  of 128 threads, so leftover threads quantized row `n` and wrote over the scale
  plane: a 65-token prompt (or 129, 193, …) produced rambling logits, while 64
  and 66 tokens were correct. The `gpu_matmul` test now loads 65, 129, and 193
  precisely because of that.
- **The tail slice width was applied to every block.** When a launch needs to cut
  leftover tiles, blocks holding a full tile also counted a quarter of the
  tokens. Prefill numbers briefly looked 25% higher because of it; the numbers in
  the table above are the ones that count all tokens.

## Loader: whatever the file declares

An important separation: the **loader** can be generic across all GGUF files, the
**forward pass** cannot. The loader reads what the file says; the forward pass
only knows the blocks that have already been implemented.

What the loader reads, with no architecture name in the code:

- **header and metadata** — keys are read under `general.architecture`
  (`qwen2.block_count`, `llama.block_count`, …), with `llama.` as fallback, and
  optional fields (kv head, key length, rope, epsilon, vocab) use the same
  defaults as llama.cpp;
- **tokenizer** — two families according to `tokenizer.ggml.model`: `gpt2`
  (byte-level BPE, pre-tokenizer selected from `tokenizer.ggml.pre`: minicpm5,
  qwen2, gpt2, llama-bpe) and `llama`/`spm` (SentencePiece: pieces + scores, merge
  by ascending score, `<0xNN>` byte fallback); `bos_token_id`/`eos_token_id` come
  from metadata, not from constants;
- **chat template** — `tokenizer.chat_template` is classified from its markers
  into `chatml`, `llama3`, or `mistral`, then rendered natively. An unrecognized
  template is rejected by name, not silently rendered as ChatML.

`llm-cli inspect <file.gguf>` answers for any file, including ones that cannot be
run: architecture, geometry, tensor type histogram, tokenizer family, template
shape, then a `runnable` verdict taken from an actual load attempt — so it cannot
drift from the engine's behavior.

The limits are stated, not hidden: models with blocks that do not exist yet
(state-space, head_dim > 128) fail at load with a specific reason. A real example
from this repo:

```
$ llm-cli inspect models/moe-expert-coder-4b-Q8_0.gguf
architecture  qwen35
geometry      layers 32, embedding 2560, head dim 256, experts 128
tokenizer     family gpt2, vocab 248320, pre-tokenizer qwen35
chat template chatml
verdict       runnable — yes, on the cpu path
```

## State-space blocks: gated delta net

The `qwen35` model interleaves full attention blocks with recurrent blocks: every
4th block is attention, the rest are gated delta net. What is read from the
header: `{arch}.full_attention_interval`, `ssm.conv_kernel`, `ssm.state_size`,
`ssm.inner_size`, `ssm.time_step_rank`, `ssm.group_count`. Tensors per recurrent
block: `attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_alpha`,
`ssm_beta`, `ssm_norm`, `ssm_out`; its attention blocks use
`attn_q_norm`/`attn_k_norm` and a q projection that is twice as wide (query +
sigmoid gate next to it).

The flow lives in `apps/engine/src/deltanet.rs`, its spec in
`docs/research/qwen35-deltanet.md`: a causal per-channel convolution over the
packed `q|k|v`, per-head L2 normalization of q/k, a delta rule per value head
(`S *= exp(g)`, `delta = (v - S·k)·beta`, `S += k·delta`, `out = S·q/√d`), then a
gated RMSNorm with the `silu(z)` gate and the `ssm_out` projection.

On the GPU, all four projections of this block run on the device; only the state
arithmetic is read back to the host, because the delta rule has no kernel yet. Per
token that is about 50 KB down and 16 KB up, while the weights the projections
skip are gigabytes — so this host island is cheap, and the model is no longer
forced onto the CPU. The consequence is that recurrent blocks cannot enter graph
capture; models that have such blocks still run decode without a graph.

The `qwen35` attention block, 256 wide per head, also runs on the device: the
decode-split kernel and the serial kernel stage up to 256 elements per head.
Prefill at that width uses the serial kernel (the tiled one is bound to four dims
per lane, i.e. 128); a tiled shape for 256 is future work, not a requirement for
correctness.

Verification: `llama-eval-callback` from llama.cpp `d812350493` (the build that
runs this file at 10.5 t/s) dumps the value of every tensor, and
`scripts/compare_reference.py` compares them with the engine's dump
(`LLM_DUMP_LAYERS=<dir> llm-cli ...` writes each block as raw f32). Result:
embedding and normalization identical to print precision (4e-5), the `attn_qkv`
projection 3e-4, delta rule output 5e-5, block 0 output ~1e-3 — the noise level
of Q8_0 activation quantization. Greedy output for the test prompt matches
llama.cpp ("The capital of France is" → " Paris." then EOS).

Two things that are easy to get wrong and have been fixed here: the `ssm_conv1d`
weights are stored `[channel, tap]` (tap fastest), not `[tap, channel]`; and the
`ssm_dt` bias must be added before softplus. Either one produces a model that
still runs and emits convincing but wrong text.

Speed, 4.2B Q8_0, context 1024, measured with `llm-cli bench`:

| | CPU | GPU (RTX 4050 Laptop) |
|---|---|---|
| pp128 | 13.8 t/s | 89.6 t/s |
| pp512 | 18.9 t/s | 91.1 t/s |
| tg32 | 7.9 t/s | 26.5 t/s |

4.17 GiB VRAM used, 4.49 GiB peak at context 8192. The text produced by GPU and
CPU is identical for the test prompt (39 tokens, stopping at EOS), and every block
matches to ~5e-3 — the noise level of activation quantization, not an arithmetic
difference.

### How much context fits

This file declares `qwen35.context_length = 262144`, but the window that can be
served is limited by the card, not by the model. The KV cache only grows in 8
attention blocks (the other 24 keep a fixed state), so it costs 32 KB per token
in f16:

| context | KV | weights | total | RTX 4050 (6 GiB) |
|---|---|---|---|---|
| 8192 | 0.25 GiB | 4.18 GiB | 4.43 GiB | fits |
| 32768 | 1.00 GiB | 4.18 GiB | 5.18 GiB | fits, 5.30 GiB peak |
| 40960 | 1.25 GiB | 4.18 GiB | 5.43 GiB | allocation failed |
| 131072 | 4.00 GiB | 4.18 GiB | 8.18 GiB | impossible |

So the practical limit on this card is 32k. On the CPU 128k does work (KV in RAM,
31 GB), but 18 t/s prefill means a 128k prompt takes about two hours — not an
option for use.

One more thing that factors in: the prefill kernel for 256-wide heads is still
the serial one, and that kernel writes a score grid of `n_head × token ×
capacity`. With a fixed 1024-token chunk, a 32k context would ask for 2 GiB for
the grid alone. That is why the prefill chunk shrinks with capacity
(`Session::prefill_chunk`), with a 128 MiB cap: 32k uses a 64-token chunk and a
137 MiB grid. The cost is small — pp128 drops from 89.6 to 85.7 t/s.

## Mixture of experts

The feed-forward block can be a router over a stack of experts, read from GGUF
like any other dense model: `{arch}.expert_count`, `expert_used_count`,
`expert_feed_forward_length`, and the tensors `ffn_gate_inp.weight` (router) +
`ffn_{gate,up,down}_exps.weight` (experts stacked along the outermost dimension,
so each expert is a contiguous slice).

Forward per token: one router matmul (n_expert logits, read back to the host to
select experts), then for each selected expert gate/up matmul + silu + down matmul
scaled by its routing weight before being summed into the hidden state. Routing
weights are computed with a softmax over the selected top-k, so they sum to one.

Verified with four small fixtures (`scripts/make_tiny_moe.py`) that share one set
of weights: a dense model vs a model with four identical experts — if the routing
softmax does not sum to one, the two cannot match. Six tests cover selection,
normalization, weighted accumulation, and the end-to-end path.

The known cost: routing and expert selection happen per token per layer, with one
host copy in between, and prefill runs experts per row rather than grouped per
expert — correct first, fast later. Models that interleave state-space blocks
with attention (e.g. the `qwen35.ssm` architecture) are still rejected at load:
those blocks do not exist in this forward pass.

## Implementation notes

- **No allocation on the token path.** Per-token and per-chunk scratch is planned
  once per session (`workspace.rs`), and `Model::forward` does not call `alloc` at
  all. That is also what makes graph capture possible.
- **Kernel decisions live in one place.** `tactics.rs` picks the attention path
  and the matmul merge policy when the model loads; `compute.rs` only asks for
  that plan. No shape thresholds are scattered across call sites.
- **Numbers point at an artifact.** Every `bench` output prints the build
  identity and the plan identity; without that, one number cannot be compared
  with another.
- **The loader reads keys belonging to the model's own architecture**, not just
  `llama.*`, and rejects models whose blocks cannot run with a clear reason. A
  real example: `models/moe-expert-coder-4b-Q8_0.gguf` is a hybrid model
  (`qwen35.ssm.*` — state-space layers interleaved with attention, head_dim 256),
  and it now fails at load with `architecture qwen35 interleaves state-space
  blocks with attention; this engine implements dense attention blocks only`,
  not `MissingMetadata`. Dense models from other families (qwen2, llama, mistral)
  will load as long as head_dim ≤ 128.
- **This 2B model's limits, and where they lie.** In conversations that contain a
  user name, the model sometimes calls itself by that name ("namaku Budi" →
  "Saya Budi!"), and in some conversations it freezes by repeating a previous
  answer (measured 1 in 5 two-turn conversations). This is not an engine bug: the
  exact same prompt, given to llama.cpp b11429, produces the same mess ("Saya
  ingerti kamu mungkin ngisiné saka aplikasi AI seperti ini"). System prompts (9
  variants tested), BOS, and narrower sampling reduce the symptom but do not
  eliminate it — what remains is the Indonesian-language ability of a 2B model.
  The fix is a different model, not code: the loader reads any llama-architecture
  GGUF.
- **Truncation sampling is narrower than llama.cpp's default, and that is
  measured.** At `top_p 0.95` / `min_p 0.05` this model leaves Indonesian: it
  answers in Javanese and denies facts it was just given. `0.8 / 0.15` is correct
  but still leaves a Javanese tail ("iki ana"), `0.75 / 0.25` is clean on every
  question tested, across two seeds. The API default and
  `SamplingParams::default()` use 0.75 / 0.25.
- **Conversation prompts start with BOS**, because the model's template asks for
  it (`{{- bos_token }}`) and this engine never sent it before. Without BOS the
  recall answer comes out rambling; with BOS the answer is "Kode proyek NX-42.".
- **The repeat-penalty window deliberately contains only generated tokens**, not
  the prompt tail like llama.cpp: copying that habit measured worse because it
  suppresses exactly the tokens that must be remembered (the answer turns into a
  Javanese refusal).
- **The system prompt follows the conversation's language, and that is not a
  matter of taste.** Measured on this model: with `"You are a helpful
  assistant."` a two-turn conversation collapses — asked "Kode proyeknya apa?"
  after being told NX-42, the answer is `"I'm sorry, I can't assist with that."`
  and that refusal is repeated for the next turn. With `"Anda adalah asisten AI.
  Jawab singkat dalam bahasa Indonesia."` the model answers one turn then freezes
  on the same fragment ("Aku inggah."). With `"Kamu asisten yang membantu."` all
  four questions are answered and both facts are remembered; longer instructions
  actually make it ramble, so the prompt is left at one sentence.
- **Launch shapes are validated against the device, not assumed.** At load,
  `cuFuncGetAttribute` and `cuOccupancyMaxActiveBlocksPerMultiprocessor` are read
  for the three decisive kernels: prompt matmul 127 registers / 77,824 bytes
  shared / 512 threads / 1 block per SM, prefill attention 122 / 49,152 / 256 / 2,
  decode matmul 42 / 0 / 128 / 10. The wave plan uses that blocks-per-SM figure,
  not `1`. In debug builds, `launch()` also compares the argument count against
  `cuFuncGetParamCount` — `cuLaunchKernel` does not check that, so a call site
  that is one argument short will send garbage without an error.

- Weights stay quantized in the mmap; dequantization is done per row inside the
  matmul, never for the whole model.
- Attention uses one key/value read per GQA group (8 query heads share 1 key), and
  its dot product uses AVX2 FMA with 4 accumulators: a plain `iter().sum()`
  depends on the previous FMA, so it runs at one MAC per latency.
- Matmul has been batched from the start (`n` tokens in one call) — prefill uses
  the same kernel as decode.
- RoPE: rotation of **consecutive** pairs `(2i, 2i+1)`, matching
  `LLAMA_ROPE_TYPE_NORM` for the `llama` architecture (half-split rotation
  produces wrong logits).
- The `minicpm5` pre-tokenizer runs as **two sequential regexes** like llama.cpp,
  with `\s` = ASCII whitespace (not Unicode).
- `zod-rs` is used without the `axum` feature (that feature pins axum 0.7);
  validation is called once at the handler boundary.
- The chat template is rendered in the engine (a single template owner), not in
  the application layer.
