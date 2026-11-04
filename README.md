# LLM lokal: engine inferensi sendiri + web UI + statistik

Chat untuk `models/MiniCPM5-2B-Q8_0.gguf` di mana **inferensinya ditulis sendiri
di Rust**: parser GGUF, kernel Q8_0, tokenizer BPE, forward pass 42 layer, KV
cache, sampler, streaming. Tidak ada llama.cpp, candle, atau server model lain
saat runtime — llama.cpp hanya dipakai sekali untuk merekam fixture pembanding.

```
apps/engine      lib  GGUF → Q8_0 → matmul → tokenizer → forward pass → sampler
apps/api         lib  domain / application / infrastructure / presentation (Kana)
apps/gateway     bin  Axum + serve SPA
apps/cli         bin  REPL terminal (fase berikut)
.migrations      crate sea-orm-migration (SQLite)
web              SPA statis: index.html + app.js + style.css
```

## Menjalankan

```bash
cargo build --release
cargo run --release -p llm-gateway
# buka http://127.0.0.1:8099
```

Variabel lingkungan:

| Variabel | Default | Arti |
|---|---|---|
| `LLM_MODEL` | `models/MiniCPM5-2B-Q8_0.gguf` | path model |
| `LLM_CONTEXT` | `8192` | ukuran konteks (KV f16 ≈ 43 KB/token) |
| `LLM_THREADS` | jumlah core | thread rayon untuk matmul (jalur CPU) |
| `LLM_GPU` | otomatis | `0` memaksa CPU walau ada GPU |
| `LLM_CUDA_ARCH` | `sm_89` | arsitektur PTX saat build (butuh `nvcc`) |
| `LLM_DATABASE` | `sqlite://llm.db?mode=rwc` | database statistik |
| `LLM_ADDRESS` | `127.0.0.1:8099` | alamat bind |
| `LLM_LOG` | `info` | filter tracing |

Migrasi SQLite dijalankan otomatis saat start.

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

Streaming memakai SSE dengan urutan tetap:

```
event: start   data: {"prompt_tokens":123}
event: token   data: {"text":"Halo"}
event: stats   data: {"generated_tokens":88,"tokens_per_second":17.1,...}
event: error   data: {"message":"..."}
```

Klien yang memutus koneksi menghentikan generasi (flag cancel), dan statistik
parsial tetap dicatat.

## Statistik

Disimpan per run di SQLite: prompt token, token dihasilkan, prefill ms, TTFT ms,
decode ms, total ms, tok/s (dihitung dari decode, bukan termasuk prefill),
latensi antar token, ctx terpakai, `stop_reason`, ringkasan sampler, RSS puncak.
Agregat: total run/token, rata-rata + p50/p95 tok/s, rata-rata TTFT, bucket
harian 30 hari.

## Pembuktian (bukan "kelihatan jalan")

Semua fixture direkam dari llama.cpp b11429 lalu di-commit; test tidak pernah
memanggil llama.cpp.

```bash
cargo test --workspace                      # unit + API (tanpa model)
cargo test --release -p llm-engine -- --ignored   # butuh model asli
./scripts/record-golden.sh tokenizer|logits|greedy  # rekam ulang fixture
```

| Test | Isi | Status |
|---|---|---|
| `tokenizer_golden` | 39 kalimat (ASCII, spasi ganda, newline, emoji ZWJ, CJK, Arab, Thai, angka panjang, CRLF, U+3000, penanda ChatML) → id harus sama persis dengan `llama-tokenize` | 39/39 sama |
| `logits_golden` | 130 560 logits dibandingkan penuh dengan `llama-debug --save-logits`, dua prompt | argmax + top-5 sama, max\|Δ\| ≈ 0.12 |
| `greedy_golden` | 32 token greedy (temp 0) vs `/completion` llama-server | identik |
| `ops`, `quant_q8`, `matmul` | rmsnorm, rope, softmax, silu, dot Q8_0 vs referensi f64, matmul batch vs naive | lulus |
| `gpu_matmul` | kernel CUDA vs kernel CPU pada bobot asli, n=1/8/64; pipeline GPU harus menghasilkan argmax yang sama | max\|Δ\| ≈ 1,5e-6, argmax 8181 |
| `http_api` | router + use case + SQLite in-memory + engine palsu: 201/204/404/400, urutan SSE, pencatatan run | lulus |

Kenapa toleransi logits 0.2: ggml men-dequant bobot Q8_0 ke f16 di dalam
matmul-nya, sedangkan engine ini menghitung dot Q8_0 eksak. Selisihnya ~0.1 pada
logit bermagnitudo ~9, yaitu akurasi jalur llama.cpp itu sendiri. Yang harus
identik adalah keputusannya: argmax, top-5, dan 32 token greedy.

## GPU (CUDA)

Kalau ada GPU NVIDIA, engine memakainya otomatis: bobot, KV cache, dan seluruh
aktivasi tinggal di VRAM, dan forward pass dijalankan oleh kernel CUDA yang
ditulis di repo ini (`apps/engine/kernels/gpu.cu`), bukan lewat library
inferensi mana pun. Runtime hanya butuh driver (`libcuda.so`); `nvcc` cuma
dibutuhkan saat build untuk meng-compile PTX, dan kalau tidak ada, crate tetap
build dan otomatis jalan di CPU.

```bash
cargo build --release        # PTX untuk sm_89 ikut di-embed
LLM_GPU=0 cargo run -p llm-gateway   # paksa CPU
```

Keputusan desain yang penting: batas CPU/GPU ada di **model**, bukan di matmul.
Versi pertama hanya memindahkan matmul ke device dan menyimpan aktivasi di host;
itu membuat decode 67 ms/token (lebih lambat dari CPU 46 ms) karena 294
round-trip H2D/D2H + sinkronisasi per token. Setelah seluruh forward pass pindah
ke device, decode jadi 27,6 ms/token. Satu forward pass, dua backend
(`apps/engine/src/compute.rs`), jadi tidak ada dua salinan logika layer.

Kalau VRAM tidak cukup untuk bobot + 512 MB headroom, model otomatis kembali ke
CPU saat load, bukan gagal di tengah jawaban.

### Kinerja terukur

RTX 4050 Laptop 6 GB (sm_89, driver 615.78.08), model 2,68 GB, konteks 4096,
VRAM terpakai ±3,1 GiB, median 9 langkah decode per titik, gateway dimatikan:

| Konteks | Decode GPU | Decode CPU | Prefill GPU | Prefill CPU |
|---|---:|---:|---:|---:|
| 0 token | **64,8 tok/s** (15,4 ms) | 21,7 tok/s (46,0 ms) | 16 ms | 50 ms |
| 128 token | **59,8 tok/s** (16,7 ms) | 19,5 tok/s (51,4 ms) | 278 tok/s (0,46 s) | 30 tok/s (4,26 s) |
| 512 token | **48,3 tok/s** (20,7 ms) | 15,6 tok/s (64,1 ms) | 259 tok/s (1,98 s) | 27 tok/s (18,8 s) |
| 1024 token | **38,4 tok/s** (26,0 ms) | 11,1 tok/s (90,3 ms) | 242 tok/s (4,22 s) | 23 tok/s (44,6 s) |

Decode 3,0–3,5× lebih cepat, prefill 9,3–10,6× lebih cepat. Satu generasi 120
token di CLI: 60,6 tok/s, TTFT 175 ms.

Lantai decode di 4050 (≈192 GB/s) adalah 2,68 GB bobot / 192 GB/s ≈ 14 ms per
token; yang terukur 15,4 ms, jadi decode sudah 91% dari lantai itu. Tidak ada
ruang berarti lagi di jalur decode tanpa mengubah presisi bobot.

Prefill masih di ±670 GFLOPS (5–10% dari puncak fp32 kartu ini). Jalur prompt
memakai 4 baris bobot × 8 token per blok dengan k dipecah 1024 elemen, jadi satu
baca shared menghasilkan 32 MAC; menaikkan lagi berarti tile 8×8 (64 akumulator
per lane, register tidak cukup) atau tensor core lewat bobot f16/bf16 — yang
mengubah numerik dan karenanya belum dilakukan.

### Kenapa decode butuh repack bobot

Format GGUF menyimpan Q8_0 sebagai blok 34 byte (`f16 skala` + 32 `i8`). Selama
baris dibaca apa adanya, satu lane hanya bisa memuat 1 byte per instruksi dan
satu warp memindahkan 32 byte — seperempat dari satu cache line. Itu yang
membuat decode berhenti di 27,6 ms/token (≈97 GB/s efektif).

Saat diunggah, tiap tensor Q8_0 dipecah jadi dua bidang: `scales` (k/32 `f32`
per baris) dan `quants` (k `i8` per baris), keduanya sejajar 32 byte. Dengan
begitu satu lane memuat 8 kuantisasi sekaligus lewat satu load 64-bit dan dua
load `float4` untuk aktivasi. Hasilnya 15,4 ms/token (≈174 GB/s efektif).

Harganya: 1,125 byte per parameter, bukan 1,0625 — VRAM bobot naik dari 2,68 GB
ke ±2,83 GB, dan repack menambah ±0,4 s saat load.

### Kenapa CPU masih dipakai sebagai pembanding

Kedua backend menjalankan **forward pass yang sama** dan diuji terhadap fixture
llama.cpp yang sama. Test `gpu_matmul` membandingkan kernel GPU dengan kernel CPU
pada bobot asli (max |Δ| ≈ 1,5e-6), dan `logits_golden`/`greedy_golden` lulus di
kedua backend dengan angka yang sama (max |Δ| 0,118 / 0,572 / 0,260; 32 token
greedy identik). Jadi GPU bukan jalur "kira-kira benar" yang terpisah.

### Kinerja CPU (untuk pembanding)

i5-13500HX (20 thread), model 2,68 GB, diukur dengan
`cargo test --release -p llm-engine --test perf_profile -- --ignored --nocapture`
(median 9 langkah decode per titik, gateway dimatikan agar tidak bersaing):

| Konteks | Decode | Prefill |
|---|---:|---:|
| 0 token | 21,4 tok/s (46,6 ms/token) | — |
| 128 token | 18,6 tok/s | 30,6 tok/s (4,2 s) |
| 512 token | 16,0 tok/s | 27,5 tok/s (18,6 s) |
| 1024 token | 6,2 tok/s | 24,4 tok/s (42 s) |

Pembanding llama.cpp b11429 pada mesin yang sama (Q8_0 yang sama, CPU):
decode 24,4 tok/s, prefill 81 tok/s. Jadi decode ≈ 88% dari llama.cpp,
prefill ≈ 1/3.

Kenapa prefill tertinggal: ggml memakai dot int8×int8 (`_mm256_maddubs_epi16`)
setelah mengkuantisasi aktivasi, sedangkan engine ini memakai int8×f32
(`_mm256_cvtepi8_epi32`) supaya tidak perlu langkah kuantisasi aktivasi. Itu
pilihan kesederhanaan, bukan batas perangkat keras.

KV cache f32: 86 KB/token → 704 MB pada ctx 8192, 1,4 GB pada ctx 16384.
f32 dipilih setelah pengukuran: versi f16 menyimpan setengah memori tapi
memaksa konversi f16→f32 di dalam loop attention (16 head × posisi × 128),
dan itu membuat decode 3–4× lebih lambat pada konteks panjang.

Batas jujur: satu generasi pada satu waktu (permintaan kedua ditolak 429
`engine busy`), tanpa batching antar-permintaan, tanpa multi-model, tanpa auth
(bind ke loopback). Bobot diunggah ke VRAM saat load, jadi startup menambah
±0,3 s dan VRAM terpakai walau belum ada permintaan.

## Catatan implementasi

- Bobot tetap terkuantisasi di mmap; dequant dilakukan per baris di dalam
  matmul, tidak pernah seluruh model.
- Attention memakai satu pembacaan key/value per grup GQA (8 head query berbagi
  1 key), dan dot-nya memakai AVX2 FMA dengan 4 akumulator: `iter().sum()` biasa
  bergantung pada FMA sebelumnya sehingga berjalan satu MAC per latensi.
- Matmul ber-batch sejak awal (`n` token sekali panggil) — prefill memakai kernel
  yang sama dengan decode.
- RoPE: rotasi pasangan **berurutan** `(2i, 2i+1)`, sesuai `LLAMA_ROPE_TYPE_NORM`
  untuk arsitektur `llama` (rotasi half-split menghasilkan logits yang salah).
- Pre-tokenizer `minicpm5` dijalankan sebagai **dua regex berurutan** seperti
  llama.cpp, dengan `\s` = whitespace ASCII (bukan Unicode).
- `zod-rs` dipakai tanpa fitur `axum` (fitur itu memaku axum 0.7); validasi
  dipanggil sekali di boundary handler.
- Template chat dirender di engine (satu pemilik template), bukan di lapisan
  application.
