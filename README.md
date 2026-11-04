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

### Benchmark sendiri, bentuknya seperti llama-bench

`llm-cli bench` mengikuti antarmuka `llama-bench` supaya dua angka bisa
dibandingkan langsung: `-p` daftar panjang prompt, `-n` daftar panjang generasi,
`-r` jumlah pengulangan, `-o json`, dan keluaran berupa tabel `test | t/s` dengan
sebaran `±` (deviasi standar sampel, rumus yang sama dengan llama-bench).

### Kinerja terukur

RTX 4050 Laptop 6 GB (sm_89, driver 615.78.08), model 2,68 GB, konteks 4096,
VRAM terpakai ±3,1 GiB, median 9 langkah decode per titik, gateway dimatikan:

| Konteks | Decode GPU | Decode CPU | Prefill GPU | Prefill CPU |
|---|---:|---:|---:|---:|
| 0 token | **64,4 tok/s** (15,5 ms) | 21,7 tok/s (46,0 ms) | 16 ms | 50 ms |
| 128 token | **63,5 tok/s** (15,8 ms) | 19,5 tok/s (51,4 ms) | 2.190 tok/s (0,06 s) | 35 tok/s (3,63 s) |
| 512 token | **62,7 tok/s** (15,9 ms) | 15,6 tok/s (64,1 ms) | 1.880 tok/s (0,27 s) | 30 tok/s (16,7 s) |
| 1024 token | **60,2 tok/s** (16,6 ms) | 11,1 tok/s (90,3 ms) | 1.576 tok/s (0,65 s) | 27 tok/s (36,8 s) |
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

### Satu chunk k lebih besar: 128 → 256 elemen

Kernel prompt men-stage satu potong `k` ke shared memory per iterasi. Dengan 128
elemen, satu potong hanya 24 KB dan tiap iterasi harus menunggu salinannya selesai
sebelum mengalikan; dengan 256 elemen potongannya dua kali lebih besar sehingga
waktu tunggu itu lebih sedikit menutupi pekerjaan aritmetika. Ukurannya melewati
batas 48 KB untuk shared memory statis, jadi kernel ini sekarang memakai alokasi
dinamis dan `cuFuncSetAttribute` untuk menaikkan batasnya (58.368 byte).

Hasilnya pp128 1.959 → 2.104 tok/s, pp512 1.607 → 1.738, pp1024 1.297 → 1.378.
Decode tidak tersentuh.

Yang **tidak** menolong, dan sebabnya: memperbesar tile attention prefill dari 16
ke 32 query supaya K/V dibaca separuh lebih jarang. Hasilnya justru lebih lambat
(1.932 / 1.574 / 1.261) karena dua hal sekaligus — buffer K dan V harus dipakai
bergantian sehingga nilai di-stage di pass terpisah, dan register per thread naik
65 → 98 sehingga okupansi turun. Dibatalkan.

### Attention prefill: query di-tile, softmax online

Kernel lama meluncurkan satu blok per (head, token) dan menulis setiap baris skor
lewat global memory, jadi makin panjang konteks makin mahal. Yang menggantikannya
mengikuti bentuk `fattn-tile.cuh`: satu blok per head dan 16 posisi query, K/V
di-stage ke shared memory 32 posisi sekaligus, dan softmax dijaga online
(`running_max`/`running_sum` plus koreksi `exp(m_lama - m_baru)` pada akumulator)
sehingga tidak ada buffer seukuran `visible` yang pernah dibentuk.

Dua hal yang menentukan bentuknya:

- **K di-stage transpose** (`k_tile[dim][key]`). Dengan layout baris-major, satu
  lane membaca `k_tile[key][dim]` untuk key-nya sendiri dan seluruh warp jatuh ke
  bank yang sama; ditranspose, tiap lane menyentuh bank berbeda dan tidak ada
  konflik. Efek sampingnya: tiap lane menyelesaikan satu skor penuh, jadi tidak
  ada `warp_sum` di loop dalam.
- **Satu lane satu kunci**. Versi pertama saya mencampur dua skema — lane
  menghitung dot untuk kunci berbeda lalu `warp_sum` menjumlahkannya, yang
  artinya menjumlahkan skor dari kunci yang tidak berhubungan. Hasilnya sampah
  yang hanya terlihat pada prompt panjang.

Hasilnya prefill 1.829 → 1.954 tok/s (ctx 128), 1.437 → 1.600 (512), 1.071 →
1.292 (1024). Decode tidak berubah.

Sisa biaya attention sekarang berasal dari pembacaan K/V berulang: satu head punya
empat blok query, jadi setiap baris K/V dibaca empat kali per chunk per head. Pada
ctx 1024 itu sekitar 2,7 GB per chunk untuk 42 layer. Menaikkan tile query dari 16
ke 32 memangkasnya setengah, tetapi shared memory-nya menjadi 52 KB dan butuh
dynamic shared memory (`cuFuncSetAttribute`) karena batas statis 48 KB.

### Attention prefill: empat head per blok

Enam belas head berbagi dua KV head, jadi satu blok per head membaca baris K/V
yang sama delapan kali. Satu blok sekarang melayani empat head yang pasti berbagi
satu KV head (`FA_HEADS`), dengan delapan warp: dua warp per head, masing-masing
delapan posisi query. Tile K/V di shared memory jadi satu-satunya salinan, dan
pembacaan K/V per chunk turun dari 16 blok-baca menjadi 4.

Shared memory yang dibutuhkan 72 KB (query empat head 32 KB, K 16 KB, V 16 KB,
probabilitas 8 KB), jadi kernel ini memakai alokasi dinamis dan
`cuFuncSetAttribute` — sama seperti kernel prompt.

Hasilnya pp128 2.104 → 2.190 tok/s, pp512 1.738 → 1.880, pp1024 1.378 → 1.576.
Decode tidak tersentuh. Diverifikasi terhadap referensi host: selisih maksimum
4,5e-7 pada 40 token, 16 head, dengan `start = 32`.

Delapan head per blok (K/V tinggal dibaca dua kali per chunk) butuh 112 KB dan
melewati batas 99 KB per blok di sm_89, jadi langkah itu menunggu tile query yang
lebih kecil atau query dalam f16.

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

Keduanya diukur dengan harness yang sama bentuknya: `llama-bench` di sisi
llama.cpp, dan `llm-cli bench` di sisi engine ini — nama tes yang sama (`pp`/`tg`),
jumlah pengulangan yang sama, dan sebaran yang dilaporkan sebagai `±`. Perintahnya
persis sepadan:

```text
llama-bench -m models/MiniCPM5-2B-Q8_0.gguf -ngl 99 -p 128,512,1024 -n 128 -r 3
llm-cli bench -m models/MiniCPM5-2B-Q8_0.gguf -p 128,512,1024 -n 128 -r 3
```

`pp` mengukur prefill N token saja; `tg` mem-prefill prompt pendek lalu mengukur
fase decode saja, jadi prefill yang lambat tidak bisa bersembunyi di angka
generation. llama.cpp b11429 di-build dari source dengan CUDA
(`-DGGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=89`), model sama, mesin sama, tidak
ada proses lain yang memakai GPU.

| Tes | llm GPU | llama.cpp GPU | Rasio |
|---|---:|---:|---:|
| pp128 | 2.190 ± 10 tok/s | 4.546 ± 482 tok/s | 48% |
| pp512 | 1.880 ± 1,7 tok/s | 5.890 ± 161 tok/s | 32% |
| pp1024 | 1.576 ± 1,9 tok/s | 5.796 ± 187 tok/s | 27% |
| tg128 | **63,2 ± 0,02 tok/s** | 71,2 ± 0,03 tok/s | 89% |

Di CPU (llama.cpp `-ngl 0 -t 20`, engine ini `LLM_GPU=0`): decode 16,9–17,2 vs
17,1–19,5 tok/s, prefill 70–99 vs 27–35 tok/s.

| Metrik | llm GPU | llama.cpp GPU |
|---|---:|---:|
| VRAM (ctx 4096) | 3,1 GiB | 2,49 GiB |
| Load + 1 token (wall) | 1,07 s | 1,32 s |

Bacaan jujurnya:

- **Decode sepadan**: 89% dan kurvanya rata terhadap konteks seperti llama.cpp.
  Selisihnya ongkos repack (1,125 vs 1,0625 byte per parameter) ditambah attention
  yang belum di-tile dengan cara yang sama.
- **Prefill 2–3,7× tertinggal**, dan selisihnya melebar dengan konteks: 48% pada
  128 token, 32% pada 512, 27% pada 1024. Dua penyebab yang sudah terukur:
  matmul sendirian berjalan 5,5 TMAC/s sedangkan prefill llama.cpp setara
  ≈7,5 TMAC/s, dan attention prefill membaca ulang K/V untuk setiap blok query.
- **CPU**: prefill masih 2,5–3× tertinggal dan decode turun dengan konteks
  (jalur CPU belum memakai tiling attention).

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
