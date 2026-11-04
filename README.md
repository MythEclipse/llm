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
| `LLM_CUDA_ARCH` | `sm_89` | arsitektur PTX saat build (butuh `nvcc`); jalur prompt memakai `mma.sync` jadi butuh sm_80 ke atas |
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
| 0 token | **64,4 tok/s** (15,5 ms) | 21,7 tok/s (46,0 ms) | 16 ms | 50 ms |
| 128 token | **63,3 tok/s** (15,8 ms) | 19,5 tok/s (51,4 ms) | 1.829 tok/s (0,07 s) | 35 tok/s (3,63 s) |
| 512 token | **62,6 tok/s** (16,0 ms) | 15,6 tok/s (64,1 ms) | 1.437 tok/s (0,36 s) | 30 tok/s (16,7 s) |
| 1024 token | **59,5 tok/s** (16,8 ms) | 11,1 tok/s (90,3 ms) | 1.071 tok/s (0,96 s) | 27 tok/s (36,8 s) |
| 2048 token | 56,2 tok/s (17,8 ms) | — | — | — |
| 4090 token | 49,3 tok/s (20,3 ms) | — | — | — |

Decode 3,0–5,4× lebih cepat, prefill 9,3–10,6× lebih cepat. Satu generasi 120
token di CLI: 60,6 tok/s, TTFT 175 ms.

Lantai decode di 4050 (≈192 GB/s) adalah 2,68 GB bobot / 192 GB/s ≈ 14 ms per
token; yang terukur 15,5 ms pada konteks 0, jadi decode sudah 90% dari lantai
itu. Kenaikan 15,5 → 20,3 ms sampai konteks 4090 juga bukan kernel lagi: KV f32
pada 4090 token berukuran 704 MB dan dibaca ulang tiap token, yaitu 704 MB /
192 GB/s ≈ 3,7 ms — persis selisih yang terukur. Menekannya berarti KV f16 atau
KV terkuantisasi, yang mengubah numerik.

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

### Prefill: int8 × int8, seperti ggml

Selisih prefill sebelumnya 16× karena kelas algoritmanya beda, bukan karena
kurang tuning: versi lama men-dequant setiap bobot ke f32 lalu mengalikannya
dengan FMA, sedangkan ggml mengkuantisasi aktivasi ke q8_0 dan mengalikan int8
dengan int8. Yang diport dari llama.cpp:

- `quantize_row_q8_0` (ggml/src/ggml-cpu/arch/x86/quants.c) — satu skala f32 per
  32 elemen, `q = round(x·127/amax)`;
- `ggml_vec_dot_q8_0_q8_0` — `_mm256_sign_epi8` + `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16`, jadi 32 perkalian per pasang instruksi, bukan 8;
- `mmq` (ggml/src/ggml-cuda/mmq.cuh) — `__dp4a` di GPU, tile 4 baris di shared
  memory sebagai int8, dan salinan tile lewat `cp.async` dua tahap supaya tile
  berikutnya sudah dalam perjalanan saat tile sekarang dikalikan.

Hasilnya prefill GPU 275 → 534 tok/s pada konteks 128 (dan 235 → 824 tok/s lewat
CLI pada prompt 215 token), CPU 28 → 35 tok/s. Decode tidak tersentuh.

Lalu jalur prompt dipindah ke tensor core, dan kali ini geometrinya diambil dari
llama.cpp, bukan dikira-kira:

| Parameter | Nilai di llama.cpp | Berkas |
|---|---|---|
| thread per blok | 256 (8 warp) | `mmq-config-ampere.cuh:104` |
| blok resident per SM | 1 | sama |
| baris keluaran per blok (I) | 128 | sama |
| token per blok (J) | 8–128 | sama |
| baris per warp | 32 saat J ≥ 48 | `mmq.cuh:180` |
| k per iterasi | 256, tile 32 | `mmq.cuh:9,116` |
| fragmen A | `ldmatrix.x4` | `mma.cuh:855` |
| fragmen B | `load_generic` ("faster than ldmatrix") | `mmq-vec-dot.cuh:250` |
| akumulasi skala | `sum += C*dA*dB`, satu fma per elemen | `mmq-vec-dot.cuh:195` |

Yang berubah di kernel ini: satu blok 128 baris × 64 token dengan 8 warp, satu
warp 32 baris (dua minitile m16) sehingga satu fragmen B dipakai dua `mma`,
kedua operan di shared memory, A lewat `ldmatrix.x4`, dan skala per blok menjadi
satu fma per elemen akumulator. Kapasitas 64 token itu terikat pada
`PREFILL_CHUNK` di `session.rs`; kalau chunk dinaikkan tanpa menaikkan tile,
matmul mengembalikan `Unsupported` alih-alih diam-diam membuang token.

Hasilnya matmul sendirian **5,5 TMAC/s** (dari 2,1), dan prefill 818 → 1.829
tok/s pada konteks 128. Sebagai pembanding, prefill llama.cpp 4.431 tok/s setara
≈7,5 TMAC/s — jadi matmul-nya kini di 73% jalur llama.cpp.

Dua jebakan yang mahal, dicatat karena keduanya tidak terlihat dari kode:

- `ldmatrix` menerima **alamat generik 64-bit** (`"l"`), bukan alamat shared
  32-bit hasil `__cvta_generic_to_shared` seperti `cp.async`. Memberinya alamat
  shared membuat ptxas membacanya sebagai alamat global dan seluruh launch
  kena illegal memory access;
- pengukuran sempat menyesatkan berkali-kali karena satu proses uji lain masih
  memegang 2,8 GB VRAM: `Compute::auto` lalu turun ke CPU tanpa suara, dan yang
  terlihat seperti "kernel lebih lambat" sebenarnya mesin yang berbeda.

Sisa selisih prefill sekarang tumbuh dengan konteks: 1.829 tok/s (ctx 128) turun
ke 1.071 tok/s (ctx 1024), sedangkan llama.cpp tetap datar. Penyebabnya terukur —
pada ctx 1024 kira-kira 47% waktu prefill ada di attention, dan jalur prefill
masih memakai kernel `attention` serial per head (yang dipecah antar blok baru
jalur decode). Tiling attention untuk prefill adalah pekerjaan berikutnya.

### Attention decode: posisi dipecah antar blok

Model ini punya 16 head, jadi satu blok per head hanya memberi 16 blok — di kartu
dengan 20 SM itu berarti sebagian SM menganggur, dan tiap blok berjalan sendiri
menyusuri seluruh KV secara serial. Biayanya terlihat sebagai kurva yang menanjak:
15,4 ms/token di konteks 0, tapi 26,2 ms di konteks 1024.

Sekarang posisi dipecah antar blok (`attention_split`): satu blok memegang satu
head dan satu potongan posisi, menyimpan max, jumlah bobot, dan jumlah V
terbobotnya sendiri, lalu `attention_combine` menyatukannya dengan
`exp(max_i - max)` seperti flash attention. Jumlah potongan dipilih agar tiap blok
mendapat ±64 posisi (dibatasi jendela shared 256 posisi), jadi konteks 1024
menghasilkan 16 × 16 = 256 blok, bukan 16.

Hasilnya kurva decode jadi rata: 15,5 / 15,8 / 16,0 / 16,6 ms pada konteks
0 / 128 / 512 / 1024, dan 17,8 / 20,3 ms pada 2048 / 4090. Pada konteks 1024 itu
38,1 → 60,1 tok/s.

Kesalahan yang sempat terjadi, dicatat karena bentuknya klasik: versi pertama
combine mengalikan pembilang dengan `sum_i` **dan** memakai `sum_i` sebagai
penyebut, sehingga normalisasinya saling meniadakan dan hasilnya jumlah V
terbobot mentah — keluarannya benar hanya saat `splits = 1` (konteks ≤ 64), yang
kebetulan adalah kasus pertama yang saya uji secara manual. Yang menangkapnya
`greedy_golden`, bukan mata: teksnya masih mulai " Paris" lalu melantur. Rumus
yang benar menaruh `sum_i` hanya di penyebut.

### Kenapa CPU masih dipakai sebagai pembanding

Kedua backend menjalankan **forward pass yang sama** dan diuji terhadap fixture
llama.cpp yang sama. Test `gpu_matmul` membandingkan kernel GPU dengan kernel CPU
pada bobot asli (max |Δ| ≈ 1,5e-6), dan `logits_golden`/`greedy_golden` lulus di
kedua backend dengan angka yang sama (max |Δ| 0,118 / 0,572 / 0,260; 32 token
greedy identik). Jadi GPU bukan jalur "kira-kira benar" yang terpisah.

### Perbandingan dengan llama.cpp

llama.cpp b11429 di-build dari source dengan CUDA (`-DGGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=89`) supaya GPU vs GPU-nya sebanding, bukan GPU lawan
CPU. Berkas model sama (`MiniCPM5-2B-Q8_0.gguf`), mesin sama, tidak ada proses
lain yang memakai GPU, dan tiap angka diambil setelah satu putaran pemanasan.
Sisi llama.cpp memakai `llama-bench` (`-r 2`/`-r 3`, `-t 20`); sisi engine ini
memakai `perf_profile` (median 9 langkah decode per titik).

| Metrik | llm GPU | llama.cpp GPU | llm CPU | llama.cpp CPU |
|---|---:|---:|---:|---:|
| Prefill 128 token | 275 tok/s | 4.431 tok/s | 28 tok/s | 99 tok/s |
| Prefill 512 token | 257 tok/s | 5.836 tok/s | 18 tok/s | 98 tok/s |
| Prefill 1024 token | 241 tok/s | 5.765 tok/s | 16 tok/s | 70 tok/s |
| Decode ctx 0 | **64,4 tok/s** | 71,2 tok/s | 17,1 tok/s | 16,9 tok/s |
| Decode ctx 128 | 63,2 tok/s | 71,3 tok/s | 13,9 tok/s | 17,2 tok/s |
| Decode ctx 512 | 62,6 tok/s | 71,2 tok/s | 9,8 tok/s | 15,9 tok/s |
| Decode ctx 1024 | 60,1 tok/s | 71,2 tok/s | 7,6 tok/s | 15,1 tok/s |
| VRAM (ctx 4096) | 3,1 GiB | 2,49 GiB | — | — |
| Load + 1 token (wall) | 1,07 s | 1,32 s | 0,24 s | 2,90 s |

Uji ujung-ke-ujung lewat CLI, `-n 128`, `--temp 0`, dua putaran bergantian supaya
kondisi termal sama. Prompt pendek 215 token, prompt panjang 1071 token (teks
sama diulang lima kali):

| Prompt | Metrik | llm GPU | llama.cpp GPU | llm CPU | llama.cpp CPU |
|---|---|---:|---:|---:|---:|
| 215 token | Prompt | 824 tok/s | 3.743–4.263 tok/s | 31,0 tok/s | 78,3 tok/s |
| 215 token | Generation | 62,2 tok/s | 69,0 tok/s | 16,4 tok/s | 22,7 tok/s |
| 1071 token | Prompt | 664 tok/s | 5.346–5.401 tok/s | 24,8 tok/s | 74,5 tok/s |
| 1071 token | Generation | 59,3 tok/s | 67,9 tok/s | 6,4 tok/s | 20,4 tok/s |

Bacaan jujurnya:

- **Decode sepadan di semua konteks yang diuji** (90% di ctx 0, 84% di ctx
  1024, dan kurvanya sekarang rata seperti llama.cpp). Selisih 10–16% itu ongkos
  repack (1,125 vs 1,0625 byte per parameter) plus attention yang belum di-tile
  dengan cara yang sama. Sebelum attention decode dipecah antar blok, angkanya
  38,1 tok/s pada ctx 1024; lihat catatan di bagian GPU.
- **Prefill 18× tertinggal di GPU.** ggml memakai kernel MMQ: aktivasi
  dikuantisasi ke int8 lalu di-dot dengan `dp4a`/MMA tensor core, sedangkan
  engine ini menjumlahkan f32 lewat FMA biasa. Selisih 4.431 vs 275 tok/s itu
  kelas algoritma, bukan tuning.
- **Di CPU prefill tertinggal 3,5×, dan decode masih menurun dengan konteks**
  (14,5 → 7,4 tok/s dari prompt 215 ke 1071 token) sementara llama.cpp tetap
  19,2–19,3. Pemecahan attention antar blok baru dikerjakan di jalur CUDA; jalur
  CPU masih menyusuri KV secara serial per (head, token). Prefill tertinggal
  karena sebab yang sama seperti di GPU: ggml memakai `_mm256_maddubs_epi16`
  (int8×int8) setelah aktivasi dikuantisasi, engine ini memakai
  `_mm256_cvtepi8_epi32` (int8×f32) supaya tidak perlu langkah kuantisasi.
- **VRAM 3,1 vs 2,49 GiB** karena bidang `scales` f32 menyimpan 4 byte per 32
  elemen yang di GGUF hanya 2 byte (f16).

Yang belum dikerjakan untuk menutup selisih itu: dot int8 (`dp4a`) plus akumulasi
f16 di jalur prompt, tensor core MMA untuk prefill, dan attention bertile ala
flash attention untuk konteks panjang. Ketiganya mengubah numerik, jadi butuh
fixture baru sebelum bisa diterima.

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
