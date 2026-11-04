# LLM Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Engine inferensi GGUF dari nol di Rust (tanpa llama.cpp/candle) + API Axum pola Kana + UI web dengan statistik penuh, terbukti benar lewat golden fixture dari llama.cpp.

**Architecture:** `apps/engine` (lib murni: gguf → quant → tensor → tokenizer → model → session → sampler → generate) tidak tahu HTTP/DB/tokio. `apps/api` berlapis domain/application/infrastructure/presentation; port `ChatEngine` mengembalikan channel + cancel flag (bukan closure). `apps/gateway` merakit Axum + serve `web/`. Statistik dipersist ke SQLite lewat SeaORM.

**Tech Stack:** Rust edition 2024, memmap2, rayon, half, fancy-regex, axum 0.8, tokio, sea-orm 1.1 (sqlite), sea-orm-migration, tracing, uuid, chrono, zod-rs (tanpa fitur axum).

**Spec:** `docs/superpowers/specs/2026-11-04-llm-local-inference-design.md`

**Status (2026-11-04):** Task 1–14 selesai. Bukti: tokenizer 39/39 identik dengan
`llama-tokenize`; logits 130 560 nilai (argmax + top-3 sama, max|Δ| 0.12/0.57);
32 token greedy identik dengan llama-server; 9 test HTTP hijau dengan SQLite
in-memory; UI diverifikasi di browser 1440×900 dan 390×844. Dua koreksi terhadap
rencana awal: RoPE memakai pasangan berurutan (bukan half-split), dan
`apps/cli` ditambahkan agar engine bisa dipakai tanpa web.

**Status lanjutan (perf pass):** profil per posisi (`tests/perf_profile.rs`)
menemukan attention sebagai biaya yang tumbuh 4× dari ctx 0 ke ctx 1024. Tiga
perbaikan: KV cache f32 (menghapus konversi f16 di loop), satu pembacaan
key/value per grup GQA, dan dot AVX2 FMA 4 akumulator. Hasil: decode ctx 128
190→54 ms/token, ctx 512 283→63 ms, ctx 1024 400→162 ms. Sampler juga memakai
`select_nth_unstable_by` sehingga tidak menyortir 130 560 kandidat per token.
Fixture logits ketiga (172 token) membuktikan prefill multi-chunk tetap benar:
argmax + top-3 sama, max|Δ| 0,26.

## Global Constraints

- Edition 2024; trait async pakai `fn f(..) -> impl Future<Output = T> + Send`, bukan `#[async_trait]`.
- `apps/engine` tidak boleh bergantung pada tokio, axum, sea-orm, atau tracing.
- Domain (`apps/api/src/domain`) hanya `uuid` + `chrono`.
- Bobot tetap terkuantisasi di mmap; dequant per baris di dalam matmul. Tidak ada dequant model penuh.
- Satu matmul ber-batch (`n` token) untuk prefill dan decode.
- Default `ctx_size = 8192`; KV f16.
- Model: `models/MiniCPM5-2B-Q8_0.gguf`, 42 layer, emb 2048, ffn 6144, 16 q head / 2 kv head, head_dim 128, rope base 5e6, eps 1e-6, vocab 130560, pre-tokenizer `minicpm5` (2 pola berurutan), byte_encode = true.
- Gate akhir tiap task: `cargo clippy --workspace --all-targets` (bukan `cargo check`).
- Commit mengikuti `kana-commit-convention` (feat/fix ber-scope, docs/chore tanpa scope).

## Review Focus

1. Teks yang mengandung angka (`"tahun 2024 ada 3"`) — pass pertama `\p{N}{1,3}` memotong digit jadi ≤3 lebih dulu; menggabung dua pola menghasilkan id salah.
2. Token yang terpotong di tengah codepoint UTF-8 saat streaming — harus di-decode lossy per emisi, bukan menunggu string utuh.
3. Prompt lebih panjang dari ctx, atau prompt+max_tokens > ctx — harus error `ContextFull`, bukan panic atau overwrite KV.
4. Dua permintaan generate bersamaan — harus antre/serial, bukan dua writer ke satu KV cache.
5. Klien memutus SSE di tengah generasi — generasi harus berhenti (cancel) dan statistik parsial tetap tercatat.

---

### Task 1: Workspace skeleton + gate

**Files:**
- Create: `Cargo.toml`, `apps/engine/Cargo.toml`, `apps/engine/src/lib.rs`, `.gitignore`
- Test: `apps/engine/src/lib.rs` (smoke)

**Interfaces:**
- Produces: workspace dengan member `apps/engine`, `apps/api`, `apps/gateway`, `apps/cli`, `.migrations`.

- [ ] **Step 1:** Tulis `Cargo.toml` workspace (resolver 3, `[workspace.package] edition = "2024"`), `.gitignore` (`/target`, `models/`, `*.db`), `apps/engine` dengan deps `memmap2`, `rayon`, `half`, `fancy-regex`.
- [ ] **Step 2:** `cargo clippy --workspace --all-targets` → PASS.
- [ ] **Step 3:** Commit `feat(engine): scaffold workspace with engine crate`.

### Task 2: `gguf.rs` — header, KV, tabel tensor

**Files:**
- Create: `apps/engine/src/gguf.rs`, `apps/engine/src/error.rs`
- Test: `apps/engine/tests/gguf_header.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct GgufFile { /* mmap */ }
  impl GgufFile {
      pub fn open(path: &Path) -> Result<Self, EngineError>;
      pub fn meta_u64(&self, key: &str) -> Result<u64, EngineError>;
      pub fn meta_f32(&self, key: &str) -> Result<f32, EngineError>;
      pub fn meta_str(&self, key: &str) -> Result<&str, EngineError>;
      pub fn tensor(&self, name: &str) -> Result<TensorInfo<'_>, EngineError>;
      pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo>;
  }
  pub struct TensorInfo { pub name: String, pub dims: Vec<u64>, pub dtype: DType, pub offset: u64, /* data: &'a [u8] */ }
  pub enum DType { F32, F16, Q8_0 }
  ```
  Accessor data: `fn data(&self, t: &TensorInfo) -> &[u8]` (dipotong dari mmap).

- [ ] **Step 1:** Test header: 42 layer, 2048, 6144, 16/2 head, 130560 vocab, 296 tensor Q8_0, `output.weight` ada.
- [ ] **Step 2:** Jalankan → FAIL (belum ada).
- [ ] **Step 3:** Implementasi parser: magic `GGUF`, versi 3, KV dengan `seek` untuk array (jangan materialisasi 129794 merges), tabel tensor, `data_start` = align(32).
- [ ] **Step 4:** Test PASS + clippy.
- [ ] **Step 5:** Commit `feat(engine): parse gguf header, metadata and tensor table`.

### Task 3: `quant.rs` — Q8_0 dot product

**Files:** Create `apps/engine/src/quant.rs`; Test: `apps/engine/tests/quant_q8.rs`

**Interfaces:** `pub fn dot_q8_0(row: &[u8], x: &[f32]) -> f32` (panjang `x` = 32·nblok), `pub fn dequant_q8_0_row(row: &[u8], out: &mut [f32])`.

- [ ] **Step 1:** Test: blok `d=0.5`, `qs=[1..32]` → dot dengan `x=1.0` = `0.5·Σq`. Bandingkan jalur AVX2 dan scalar (max|Δ| < 1e-4).
- [ ] **Step 2:** FAIL → implementasi 34-byte block, AVX2 + fallback → PASS.
- [ ] **Step 3:** Commit `feat(engine): add q8_0 dot product and dequant`.

### Task 4: `tensor.rs` — matmul ber-batch

**Files:** Create `apps/engine/src/tensor.rs`; Test: `apps/engine/tests/matmul.rs`

**Interfaces:** `pub fn matmul(w: &Weight, x: &[f32], n: usize, y: &mut [f32])` dengan `Weight { dtype, k, out, data }`; `y[n·out + o] = Σ_i x[n·k + i]·W[o·k + i]`.

- [ ] **Step 1:** Test: matriks 4×8 Q8_0 (dibuat manual) vs referensi naive f64, `n=1` dan `n=3`.
- [ ] **Step 2:** Implementasi: rayon atas `o`, blok Q8_0, F32 langsung.
- [ ] **Step 3:** PASS + commit `feat(engine): add batched quantized matmul`.

### Task 5: `tokenizer.rs` + golden corpus

**Files:** Create `apps/engine/src/tokenizer.rs`, `apps/engine/src/byte_level.rs`, `scripts/record-golden.sh`, `tests/fixtures/tokenizer.json`; Test: `apps/engine/tests/tokenizer_golden.rs`

**Interfaces:** `Tokenizer::from_gguf(&GgufFile) -> Result<Self>`, `encode(&self, &str) -> Vec<u32>`, `decode(&self, &[u32]) -> String`, `token_id(&self, &str) -> Option<u32>`.

- [ ] **Step 1:** `scripts/record-golden.sh` menulis fixture dari `llama-tokenize -m models/… --ids "<text>"` untuk 40 string (ASCII, spasi ganda, newline, `'s`/`'ll`, emoji, CJK, Arab, angka panjang, kode, template ChatML).
- [ ] **Step 2:** Test golden (assert semua id sama) → FAIL.
- [ ] **Step 3:** Implementasi: byte-level map 256, vocab+merges dari GGUF, pre-tokenizer 2 pola berurutan via `fancy-regex`, BPE priority-queue, special token match dulu, decode lossy.
- [ ] **Step 4:** PASS (semua 40 kasus) + commit `feat(engine): implement minicpm5 byte-level bpe tokenizer`.

### Task 6: `chat.rs` — template ChatML

**Files:** Create `apps/engine/src/chat.rs`; Test: `apps/engine/tests/chat_template.rs`

**Interfaces:** `pub fn render(messages: &[(Role, &str)], system: Option<&str>) -> String`.

- [ ] **Step 1:** Test string persis untuk system+user, dan multi-turn.
- [ ] **Step 2:** Implementasi → PASS → commit `feat(engine): render chatml template`.

### Task 7: `model.rs` — forward pass

**Files:** Create `apps/engine/src/model.rs`, `apps/engine/src/ops.rs`; Test: `apps/engine/tests/ops.rs`, `apps/engine/tests/logits_golden.rs`

**Interfaces:** `Model::load(&GgufFile) -> Result<Model>`, `Model::forward(&self, tokens: &[u32], pos0: usize, kv: &mut KvCache) -> Vec<f32>` (logits token terakhir).

- [ ] **Step 1:** Unit test ops: rmsnorm, rope half-split, softmax, silu (nilai referensi tangan).
- [ ] **Step 2:** Fixture logits dari `llama-eval-callback` (`scripts/record-golden.sh`), test `#[ignore]`: argmax sama, max|Δ| < 0.05, top-10 sama.
- [ ] **Step 3:** Implementasi forward: embd → 42 layer (rmsnorm, qkv, rope, kv append, GQA causal softmax, attn_out, ffn) → norm → output.
- [ ] **Step 4:** Test PASS (atau, kalau meleset, uji varian RoPE NEOX sebelum menyalahkan hal lain) + commit `feat(engine): implement llama forward pass`.

### Task 8: `session.rs` + `sampler.rs`

**Files:** Create `apps/engine/src/session.rs`, `apps/engine/src/sampler.rs`; Test: `apps/engine/tests/greedy_golden.rs`

**Interfaces:** `Session::new(&Model, ctx)`, `prefill(&mut self, &[u32])`, `step(&mut self, u32) -> Vec<f32>`, `sample(&mut self, &SamplingParams) -> u32`, `Session::kv_bytes()`.

- [ ] **Step 1:** Test greedy `#[ignore]`: 32 token pertama identik dengan `llama-cli --temp 0 -n 32`.
- [ ] **Step 2:** Test sampler: temp 0 deterministik; top-k=1 → argmax; seed sama → urutan sama.
- [ ] **Step 3:** Implementasi + PASS + commit `feat(engine): add kv cache session and sampler`.

### Task 9: `generate.rs` + CLI

**Files:** Create `apps/engine/src/generate.rs`, `apps/cli/src/main.rs`, `apps/cli/Cargo.toml`; Test: manual + `apps/engine/tests/generate_events.rs`

**Interfaces:** `generate(&mut Session, &[u32], &Params, &AtomicBool, impl FnMut(Event))`; `Event::{Start{prompt_tokens}, Token{id,text}, Done{stats}, Cancelled}`; `RunStats` di engine.

- [ ] **Step 1:** Test event order + cancel (set flag setelah 3 token → `Cancelled`).
- [ ] **Step 2:** Implementasi driver + REPL `llm-cli chat --model models/…`.
- [ ] **Step 3:** Jalankan nyata: prompt "Siapa presiden Indonesia?" → balasan koheren, tok/s tercetak.
- [ ] **Step 4:** Commit `feat(engine): stream generation with cancellation and stats`.

### Task 10: `apps/api` domain + application

**Files:** Create `apps/api/src/{domain,application}/…`, `apps/api/Cargo.toml`; Test: `apps/api/tests/use_cases.rs`

**Interfaces:** entity `Conversation`, `Message`, `RunStats`; trait `ConversationRepository`, `RunRepository`, `ChatEngine`; use case per operasi.

- [ ] **Step 1:** Test use case `SendMessage` dengan `FakeChatEngine` + repo in-memory: pesan user tersimpan, stats tercatat saat selesai.
- [ ] **Step 2:** Implementasi domain + application → PASS → commit `feat(api): add chat domain and use cases`.

### Task 11: infrastructure — SQLite + engine adapter

**Files:** Create `.migrations/src/*`, `apps/api/src/infrastructure/…`; Test: `apps/api/tests/repositories.rs` (`#[ignore]` untuk SQLite file; in-memory untuk CI)

- [ ] **Step 1:** Migration `m20261104_000001_init` (3 tabel + index).
- [ ] **Step 2:** Test repo: create/list/get/delete conversation, insert+aggregate runs.
- [ ] **Step 3:** Implementasi `SeaOrm*Repository`, `LocalChatEngine` (spawn_blocking + mpsc + cancel), `ProcessMetrics`.
- [ ] **Step 4:** PASS + commit `feat(api): persist conversations and run stats in sqlite`.

### Task 12: presentation — router + SSE

**Files:** Create `apps/api/src/presentation/…`; Test: `apps/api/tests/http_api.rs`

- [ ] **Step 1:** Test `oneshot`: `POST /api/conversations` → 201; `GET /api/conversations` → list; `DELETE` → 204; SSE memuat `start`, `token`, `stats`; `stop` membatalkan.
- [ ] **Step 2:** Implementasi handler + DTO + `AppError` + router → PASS.
- [ ] **Step 3:** Commit `feat(api): expose conversation, chat and stats endpoints`.

### Task 13: `web/` SPA + gateway

**Files:** Create `web/index.html`, `web/app.js`, `web/style.css`, `apps/gateway/src/main.rs`; Test: manual browser.

- [ ] **Step 1:** Gateway: bind `127.0.0.1:8099`, serve `web/`, nest router `/api`.
- [ ] **Step 2:** UI: daftar percakapan, streaming, Stop, panel statistik live + 2 grafik SVG.
- [ ] **Step 3:** Verifikasi manual di browser: kirim pesan, token streaming, angka tok/s cocok dengan DB (`sqlite3`), hapus percakapan.
- [ ] **Step 4:** Commit `feat(web): add chat ui with live statistics`.

### Task 14: hardening + gate penuh

- [ ] **Step 1:** Batas ctx → 400/`ContextFull`; antrean generate maks 4 → 429; cancel saat disconnect; error engine → 500 dengan log.
- [ ] **Step 2:** `cargo clippy --workspace --all-targets` bersih, `cargo test --workspace` hijau, test `--ignored` (butuh model) hijau.
- [ ] **Step 3:** README singkat (cara jalan, batas kinerja jujur).
- [ ] **Step 4:** Commit `docs: document how to run the local inference service`.
