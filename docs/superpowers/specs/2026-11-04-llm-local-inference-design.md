# Desain: LLM Inference Engine Sendiri + Web UI + Statistik Penuh

Tanggal: 2026-11-04
Status: menunggu review

## 1. Tujuan

Aplikasi lokal (single user, `127.0.0.1`) untuk mengobrol dengan
`models/MiniCPM5-2B-Q8_0.gguf`, di mana **inferensi diimplementasikan sendiri di
Rust** — tanpa llama.cpp, tanpa candle, tanpa server LLM eksternal — plus UI web
dan statistik lengkap yang dipersist ke SQLite.

Non-tujuan (eksplisit, fase lain): GPU/CUDA, multi-user/auth, multi-model,
tool calling, vision, fine-tuning.

## 2. Fakta terverifikasi (bukan asumsi)

| Fakta | Nilai | Cara verifikasi |
|---|---|---|
| Arsitektur | `llama`, 42 layer, emb 2048, ffn 6144 | parse header GGUF |
| Attention | 16 q head, 2 kv head (GQA rasio 8), head_dim 128 | header GGUF |
| RoPE | `freq_base` 5e6, `dimension_count` 128, tanpa scaling | header GGUF |
| Norm | RMSNorm, eps 1e-6 | header GGUF |
| Vocab | 130560 token, 129794 merges, `tokenizer.ggml.model = gpt2` | header GGUF |
| Pre-tokenizer | `minicpm5` = 2 regex diterapkan **berurutan** | `src/llama-vocab.cpp`, `src/unicode.cpp` |
| Tensor | 381: 296 Q8_0 + 85 F32 (semua norm F32) | header GGUF |
| Output | `output.weight` ada (tidak tied), Q8_0 | header GGUF |
| Chat template | ChatML `<|im_start|>role\n…<|im_end|>\n`, bos `<s>` | header GGUF |
| ctx native | 131072 (kita batasi default 8192) | header GGUF |
| Tooling pembanding | `llama-tokenize`, `llama-eval-callback`, `llama-cli` b11429 | `/usr/bin` |
| Host | 20 core, 31 GB RAM, tanpa GPU path | `nproc`, `free` |

Model tidak akan diubah, hanya dibaca (mmap read-only).

## 3. Layout workspace

```
llm/
├── Cargo.toml                # workspace, resolver 3, edition 2024
├── apps/
│   ├── engine/               # lib `llm-engine`: inferensi murni, tanpa tokio/axum
│   ├── api/                  # lib `llm-api`: domain/application/infrastructure/presentation
│   ├── gateway/              # bin `llm-gateway`: Axum + serve SPA
│   └── cli/                  # bin `llm-cli`: REPL + harness golden
├── .migrations/              # crate sea-orm-migration (SQLite)
├── web/                      # index.html, app.js, style.css (ES modules, tanpa build)
├── scripts/                  # record-golden.sh (rekam fixture dari llama.cpp)
├── tests/fixtures/           # golden: tokenizer/*.json, logits/*.json
├── models/                   # gitignored
└── docs/superpowers/specs/
```

Aturan dependensi Kana berlaku di `apps/api`:

```
presentation → application → domain
infrastructure → domain            (implementasi trait domain/port)
presentation → infrastructure      (hanya wiring AppState)
```

`apps/engine` berdiri sendiri: hanya `memmap2`, `rayon`, `half`, `fancy-regex`.
Ia tidak tahu HTTP, DB, atau tokio.

## 4. Engine (`apps/engine`)

### 4.1 `gguf.rs` — pembaca GGUF

- `GgufFile::open(path)` → mmap read-only (`memmap2`), parse header v3, KV, tensor info.
- `Meta` accessor bertipe: `u32/u64/f32/str/array`, dengan error jelas saat key hilang.
- `TensorInfo { name, dims: Vec<u64>, dtype, offset }`; data = `&mmap[data_start+offset ..][..nbytes]`.
- Validasi saat load: `general.architecture == "llama"`, jumlah layer cocok dengan
  jumlah tensor `blk.N.*`, semua tensor yang dibutuhkan ada, `nbytes` tidak melewati file.

### 4.2 `quant.rs` — Q8_0

Blok Q8_0 = 34 byte: `d: f16` + `qs: [i8; 32]`.

- `dot_q8_0_f32(block: &[u8], x: &[f32]) -> f32` = `d * Σ q_i·x_i`, akumulasi f32.
- Jalur AVX2 (`_mm256_cvtepi32_ps`, `_mm256_fmadd_ps`) + fallback scalar, dipilih runtime
  (`is_x86_feature_detected!`). Keduanya wajib menghasilkan nilai identik-ish
  (test: max|Δ| < 1e-4 relatif terhadap referensi f64).
- F32 dan F16 dibaca langsung (norm & bias).

### 4.3 `tensor.rs` — matmul

`matmul(weights: &Tensor, x: &[f32] /* n×k */, n: usize, y: &mut [f32] /* n×out */)`

- Iterasi output row `o` di-paralel dengan `rayon`; tiap thread punya buffer `x` bersama
  (read-only). Batch `n` sejak awal — prefill memakai jalur yang sama dengan decode
  (`n=1`), jadi tidak ada dua implementasi yang bisa menyimpang.
- Q8_0: baris = `k/32` blok berurutan.

### 4.4 `tokenizer.rs` — byte-level BPE

- Byte-level mapping GPT-2 (256 entri) untuk encode; decode memakai tabel balik.
- Vocab: `tokens[i] -> id`, `merges[(a,b)] -> rank`.
- Pre-tokenisasi `minicpm5`, **dua pola berurutan** (persis seperti llama.cpp):
  1. `\p{N}{1,3}`
  2. `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

  Split pertama menghasilkan segmen; tiap segmen di-split pola kedua. `fancy-regex`
  dipakai karena pola memakai lookahead `(?!\S)` yang tidak didukung `regex` crate.
- Token spesial (`<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`, `<think>`, dst.) dicocokkan
  langsung sebelum BPE; `add_bos = false`.
- BPE: priority queue bigram (rank terendah dulu), sama seperti llama.cpp.
- API: `encode(&str) -> Vec<u32>`, `decode(&[u32]) -> String` (byte-level, lossy aman untuk
  UTF-8 parsial saat streaming).

### 4.5 `chat.rs` — template

Fixed ChatML, bukan jinja:

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n
```

System default: `"You are a helpful assistant."`. Render menghasilkan `String`, lalu
di-tokenize. Stop token: `<|im_end|>` (id dari vocab) dan `</s>`.

### 4.6 `model.rs` — forward pass

```
x = dequant_row(token_embd, token)                      # 2048
for l in 0..42:
    h = rmsnorm(x, attn_norm[l], 1e-6)                  # 2048
    q = Wq·h (2048→2048) ; k = Wk·h (2048→256) ; v = Wv·h (2048→256)
    rope(q, k, pos, base=5e6, n_rot=128)                 # rotasi pasangan (2i, 2i+1)
    kv_cache[l].push(k, v)                              # f32
    a = gqa_attention(q, kv_cache[l], 16 head / 2 kv, skala 1/√128, causal)
    x += Wo·a                                           # 2048→2048
    h = rmsnorm(x, ffn_norm[l], 1e-6)
    x += Wdown·(silu(Wgate·h) * (Wup·h))                # 6144 → 2048
x = rmsnorm(x, output_norm, 1e-6)
logits = Wout·x                                         # 130560 f32
```

Catatan numerik: semua aktivasi f32; softmax dengan pengurangan maksimum; bobot
di-dequant per baris saat dipakai (tidak ada dequant seluruh model ke RAM).

### 4.6b `compute.rs` + `cuda/` — backend CPU dan GPU

Satu forward pass, dua perangkat. Setiap tensor adalah `Activation`
(`Host(Vec<f32>)` atau `Device(DeviceBuffer)`); backend dipilih sekali saat load,
jadi loop layer di `model.rs` tidak tahu sedang bicara ke siapa.

```
Compute::matmul / rmsnorm / rope / kv_append / attend / silu_mul / residual_add / embedding
  ├─ None  → CPU: tensor::matmul, ops::*, attend_host
  └─ Some  → CUDA: q8_matmul, rmsnorm_rows, rope, kv_append, attention, silu_mul, residual_add, embedding_row
```

Keputusan yang diambil dari pengukuran, bukan dari selera:

| Percobaan | Decode ctx 0 | Kesimpulan |
|---|---:|---|
| hanya matmul di GPU, aktivasi di host | 67,7 ms/token | kalah dari CPU (46 ms): 294 round-trip + sinkronisasi per token |
| seluruh forward pass di device, kernel matmul lama | 65,4 ms/token | masih lambat: n=1 memakai 1 dari 8 warp, shared 24 KB membatasi occupancy |
| + jalur n<8 tanpa shared, semua warp pada satu dot | 32,6 ms/token | occupancy dan util warp adalah masalahnya |
| + attention 4 warp (posisi dipecah antar warp) | 27,6 ms/token | kurva decode rata: 27,6 → 38,3 ms dari ctx 0 ke 1024 |
| + repack Q8_0 jadi `scales[] + quants[]` sejajar 32 byte | 15,4 ms/token | decode 91% dari lantai bandwidth (14 ms); 8 kuantisasi per load 64-bit |
| + jalur prompt 4 baris × 8 token, k dipecah 1024 | prefill 278 tok/s | satu baca shared = 32 MAC; 242 tok/s pada ctx 1024 |
| + attention decode dipecah antar blok (flash-decoding) | 16,6 ms @ ctx 1024 | dari 26,2 ms; kurva decode rata 15,5 → 20,3 ms sampai ctx 4090 |

Catatan jalur prompt: `__syncthreads` di dalam loop chunk mengharuskan **semua**
warp menjalankan jumlah iterasi yang sama. Versi pertama memakai
`for (group = warp*8; group < n; ...)`, jadi pada n < 64 sebagian warp tidak
pernah masuk loop dan hasilnya sampah (max |Δ| 2,5e38). Perbaikannya: loop atas
`base` yang seragam, `group = base + warp*8` di dalamnya, tulis hasilnya dijaga
`group + slot < n`.

Detail yang mudah salah:

- `cuCtxSetCurrent` dipanggil di setiap operasi publik, karena generasi berjalan
  di thread `spawn_blocking` yang berbeda dari thread pembuat context.
- Bobot diunggah sekali saat load (bukan lazily), supaya TTFT tidak memuat
  0,3 s upload dan VRAM yang tidak cukup ketahuan di awal → fallback CPU.
- Shared memory hanya diminta untuk jalur batch (n ≥ 8); jalur decode 0 byte.
- Skor attention diberi stride `start + tokens` per (head, token), bukan satu
  baris per head, karena baris batch berjalan paralel.
- PTX di-embed saat build (`build.rs`, `nvcc -ptx -arch sm_89`); tanpa `nvcc`
  crate tetap build dan `Gpu::open()` melaporkan tidak ada device.

### 4.7 `sampler.rs`

Urutan tetap (mengikuti default llama.cpp): repeat penalty (last 64, 1.1) → temperature →
top-k (40) → top-p (0.95) → min-p (0.05) → sampel. RNG deterministik (`xorshift64*` dengan
seed). `temperature = 0` → greedy.

### 4.8 `session.rs`

- `Session::new(&Model, ctx_size)`; KV cache `Vec<f32>` per layer, kapasitas `ctx_size`.
  f16 menghemat separuh memori tetapi memaksa konversi f16→f32 di dalam loop
  attention (16 head × posisi × 128); pengukuran: f32 3–4× lebih cepat pada
  konteks panjang, jadi f32 yang dipakai.
- `prefill(&[u32]) -> Result<(), EngineError>` (batched matmul, chunk 64 token).
- `decode_step(token) -> &[f32]` (logits), `sample(&mut self, params) -> u32`.
- `reset()`, `pos()`, `ctx_size()`, `kv_bytes()`.
- Batas: prompt + generate > ctx → `EngineError::ContextFull` (bukan panic).

### 4.9 `generate.rs` — driver streaming

`generate(session, prompt_tokens, params, cancel: &AtomicBool, sink: impl FnMut(Event))`
di `engine` berupa fungsi sinkron; lapisan tokio ada di `apps/api`. Event:
`Start { prompt_tokens }`, `Token { id, text }`, `Done { stats }`, `Cancelled`.

## 5. Lapisan API (`apps/api`, pola Kana)

### 5.1 Domain (tanpa dependensi luar selain uuid/chrono)

- `domain/conversation/entity.rs`: `Conversation { id, title, created_at, updated_at }`,
  `Message { id, conversation_id, role, content, token_count, created_at }`,
  `NewConversation`, `NewMessage`, `Role { System, User, Assistant }`.
- `domain/stats/entity.rs`: `RunStats { id, conversation_id, model, prompt_tokens,
  gen_tokens, prefill_ms, ttft_ms, decode_ms, total_ms, tokens_per_second, ctx_used,
  stop_reason, sampler, token_latencies_ms, peak_rss_bytes, created_at }`,
  `StatsOverview`, `DayBucket`, `RunFilter`.
- Repository trait `ConversationRepository`, `RunRepository` (gaya `impl Future`, `Send + Sync`),
  `RepositoryError { NotFound, Conflict(String), Database(String) }`.

### 5.2 Application

Port (bentuknya primitif + data owned, **bukan** closure — pelajaran `impl Future` di trait):

```rust
pub trait ChatEngine: Send + Sync {
    fn info(&self) -> impl Future<Output = ModelInfo> + Send;
    fn start(&self, req: GenerateRequest)
        -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send;
}

pub struct GenerationHandle {
    pub run_id: Uuid,
    pub events: tokio::sync::mpsc::Receiver<GenerationEvent>,
    pub cancel: Arc<AtomicBool>,
}
```

`ModelInfo`/`GenerateRequest`/`GenerationEvent` didefinisikan di application (bukan
meminjam tipe engine) supaya presentation tidak menyentuh crate engine.

Use case: `CreateConversation`, `ListConversations`, `ConversationDetail`,
`DeleteConversation`, `SendMessage` (menyimpan pesan user, memulai generate, mencatat
stats saat selesai), `StopGeneration`, `ModelInfo`, `StatsOverview`, `ListRuns`,
`StatsTimeseries`.

### 5.3 Infrastructure

- `SeaOrmConversationRepository`, `SeaOrmRunRepository` (SQLite), `map_db_err` +
  deteksi `"unique"` → `Conflict`.
- `LocalChatEngine`: `Arc<Mutex<Session>>` + `Semaphore(1)` (satu generate sekaligus),
  `spawn_blocking` menjalankan driver engine, token dikirim ke `mpsc` (buffer 32),
  `cancel: AtomicBool` dicek per token.
- `ProcessMetrics`: RSS dari `/proc/self/statm`, jumlah thread, load average.

### 5.4 Presentation

```
GET    /api/model
GET    /api/conversations                 POST /api/conversations        201
GET    /api/conversations/:id             DELETE /api/conversations/:id  204
POST   /api/conversations/:id/messages    → SSE
POST   /api/conversations/:id/stop        202
GET    /api/stats/overview?range=7d
GET    /api/stats/runs?limit=&offset=
GET    /api/stats/timeseries?range=30d&bucket=day
GET    /api/health
GET    /                                  SPA statis (tower-http ServeDir)
```

SSE (`text/event-stream`, keep-alive 15 s, cancel saat klien putus):

```
event: start   data: {"run_id":"…","prompt_tokens":123}
event: token   data: {"text":"Halo"}
event: stats   data: {"gen_tokens":88,"tokens_per_second":6.1,"ttft_ms":1200,…}
event: error   data: {"message":"context penuh"}
```

Aturan Kana: `#[instrument(skip_all, fields(...))]` di setiap handler, `AppError` +
`IntoResponse`, `From` impl untuk error domain/repo/engine, 201 untuk POST create,
204 untuk DELETE, `warn!` untuk kegagalan terduga, `error!` sebelum 500 generik.
DTO request: `Deserialize + Serialize + ZodSchema` (`#[zod(...)]`), response `Serialize`
saja, konversi via `From`. Fitur `axum` milik zod-rs dipin ke axum 0.7, jadi kita pakai
schema-nya saja dan memanggil validasi **sekali** di boundary handler (bukan per field
tersebar).

## 6. Statistik

### 6.1 Skema SQLite (migration `m20261104_000001_init`)

```
conversations(id TEXT PK, title TEXT, created_at TEXT, updated_at TEXT)
messages(id TEXT PK, conversation_id TEXT FK→conversations ON DELETE CASCADE,
         role TEXT, content TEXT, token_count INTEGER, created_at TEXT)
runs(id TEXT PK, conversation_id TEXT FK→conversations ON DELETE CASCADE,
     model TEXT, prompt_tokens INTEGER, gen_tokens INTEGER,
     prefill_ms INTEGER, ttft_ms INTEGER, decode_ms INTEGER, total_ms INTEGER,
     tokens_per_second REAL, ctx_used INTEGER, stop_reason TEXT,
     sampler_json TEXT, token_latencies_json TEXT, peak_rss_bytes INTEGER,
     created_at TEXT)
idx_runs_created_at, idx_messages_conversation
```

### 6.2 Definisi metrik (harus eksplisit, kalau tidak angkanya bohong)

- `ttft_ms` — dari mulai prefill sampai token pertama siap dikirim.
- `prefill_ms` — waktu seluruh prompt diproses; `prompt_tokens / prefill_ms` = prefill tok/s.
- `decode_ms` — token pertama → token terakhir; `tokens_per_second` dihitung dari sini
  (bukan termasuk prefill).
- `token_latencies_ms` — selisih waktu antar token (array, untuk sparkline).
- `peak_rss_bytes` — RSS proses saat run selesai.
- `stop_reason` — `eos | max_tokens | cancelled | context_full | error`.

### 6.3 Agregat

Overview: total run, total token, avg/p50/p95 tok/s, avg TTFT, run hari ini, model aktif,
info engine (layer, ctx, KV bytes, thread, quant mix). Timeseries: bucket harian
`{date, runs, tokens, avg_tps}`. Runs: daftar terbaru dengan semua kolom.

## 7. Web UI (`web/`, tanpa build step)

Satu halaman, tiga panel, ES modules JS murni + CSS (tanpa framework, tanpa bundler):

- Kiri: daftar percakapan, tombol baru, hapus.
- Tengah: transkrip, streaming token (SSE `EventSource` lewat `fetch` + `ReadableStream`
  karena SSE butuh POST), input + tombol Stop, pilihan temperature/top-p/max token.
- Kanan: panel statistik live (tok/s, TTFT, token, ctx) + grafik SVG inline
  (token/hari 30 hari, tok/s per run terakhir) + kartu info model.

Aksesibilitas minimum: kontras teks ≥ 4.5:1, semua kontrol bisa keyboard, `aria-live`
pada area streaming, tidak ada layout shift saat token masuk.

## 8. Strategi pembuktian

1. **Tokenizer golden** — `scripts/record-golden.sh` menjalankan
   `llama-tokenize -m models/… --ids` atas korpus (~40 string: ASCII, spasi ganda,
   newline, emoji, CJK, Arab, kode, angka panjang, apostrof `'s`/`'ll`) → fixture JSON.
   Test: id kita == id llama.cpp untuk **semua** kasus.
2. **Logits golden** — `llama-debug --save-logits` merekam 130 560 logits penuh untuk
   dua prompt (1 token dan 5 token) → test: `argmax` sama, top-3 sama, top-10 overlap
   ≥ 7/10, `max|Δ| < 0.2` (single) dan `< 0.7` (batched). Batas ini bukan kelonggaran
   sembarangan: ggml men-dequant bobot Q8_0 ke f16 di dalam matmul-nya, jadi llama.cpp
   adalah pihak yang kurang tepat di jalur batched. Keputusan (argmax + 32 token greedy)
   tetap harus identik.
3. **Greedy continuation** — `llama-cli --temp 0 -n 32` vs engine kita: 32 token identik.
4. **Unit** — dot Q8_0 vs f64, rmsnorm, rope, softmax, sampler determinisme,
   chat template string.
5. **API** — `tower::ServiceExt::oneshot` + SQLite `:memory:` + `FakeChatEngine`;
   termasuk SSE menghasilkan urutan `start → token… → stats` dan cancel.
6. **Gate** — `cargo clippy --workspace --all-targets` (bukan `cargo check`),
   `cargo test --workspace`, plus test `--ignored` yang butuh model asli.

## 9. Kinerja & batas (jujur)

- Target decode ≥ 5 tok/s, prefill ≥ 30 tok/s, TTFT < 3 s untuk prompt ~200 token
  di 20 core. Batas atas teoretis ~15 tok/s (2,68 GB bobot per token / bandwidth RAM).
- Default `ctx_size` 8192 → KV f32 ≈ 704 MB (42 layer × 512 elemen × 4 byte × 8192).
  `ctx` bisa dinaikkan lewat env, RAM naik linear (~86 KB/token).
- Satu generate pada satu waktu; permintaan kedua menunggu (bukan antre tanpa batas:
  maksimum 4 menunggu, lalu 429).
- Tanpa GPU. Jalur CUDA adalah fase terpisah dan tidak dirancang di sini.

## 10. Fase

| Fase | Isi | Gate |
|---|---|---|
| P0 | workspace, `gguf.rs`, `quant.rs`, `tokenizer.rs` | tokenizer golden 100% |
| P1 | `tensor.rs`, `model.rs`, `session.rs` | logits golden + greedy match |
| P2 | `sampler.rs`, `generate.rs`, `cli` REPL | chat jalan di terminal, tok/s terukur |
| P3 | `apps/api` domain/application/infra/presentation, migration, SSE | test API + SSE hijau |
| P4 | `web/` UI + statistik live | dipakai manual, angka cocok dengan DB |
| P5 | hardening: ctx penuh, cancel, error, clippy bersih | semua gate hijau |

## 11. Risiko

| Risiko | Mitigasi |
|---|---|
| Regex pre-tokenizer beda halus (lookahead, `\p{N}`) | korpus golden dari `llama-tokenize`, bukan tebakan |
| Varian RoPE salah | **Terjadi dan sudah diperbaiki**: `LLAMA_ROPE_TYPE_NORM` untuk arch `llama` merotasi pasangan berurutan `(2i, 2i+1)`, bukan half-split. Ditemukan oleh logits golden (argmax 280 vs 8181), bukan oleh tebakan. |
| Akurasi f32 menumpuk di 42 layer | toleransi eksplisit + argmax; kalau gagal, akumulasi f64 pada attention |
| Kecepatan di bawah target | ukur dulu, optimasi terakhir (AVX2, chunk prefill), bukan tebak |
| SQLite + stream bersamaan | satu writer, `max_connections(1)` untuk writer, WAL |
