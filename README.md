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
web              SPA statis: index.html + app.js + markdown.js + style.css
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

Antarmuka web mengikuti bentuk aplikasi chat yang lazim: sidebar percakapan +
tombol percakapan baru, transkrip dengan gelembung pesan dan markdown (blok kode
berlabel bahasa, tombol salin), komposer mengambang dengan pengaturan suhu dan
panjang jawaban, serta rail telemetri yang bisa dibuka-tutup. Tema gelap dan
terang, keduanya lolos kontras WCAG AA untuk teks yang dibaca. Tanpa build step
dan tanpa CDN: `web/` disajikan apa adanya oleh gateway.

Tracing engine dan request HTTP memakai field terstruktur. Gunakan filter
berdasarkan kebutuhan diagnosis tanpa mengubah output benchmark:

```bash
LLM_LOG=llm_engine=debug,llm_api=info cargo run --release -p llm-gateway
LLM_LOG=llm_engine=trace cargo run --release -p llm-cli -- bench -p 128 -n 32
```

Level `debug` mencatat fase forward, layer, chunk prefill, dan ringkasan
generation. Level `trace` menambahkan setiap token dan geometri peluncuran
kernel CUDA; output tracing ditulis ke stderr.

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
| `tactics`, `workspace`, `manifest` | tabel taktik tidak pernah memilih varian `Rejected` untuk bentuk mana pun dan setiap barisnya membawa bukti; perencana tidak pernah menumpuk dua buffer yang hidup di fase sama; identitas build 64 hex dan stabil | lulus |

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
VRAM terpakai ±3,0 GiB, median 9 langkah decode per titik, gateway dimatikan:

| Konteks | Decode GPU | Decode CPU | Prefill GPU | Prefill CPU |
|---|---:|---:|---:|---:|
| 0 token | **68,9 tok/s** (14,5 ms) | 21,7 tok/s (46,0 ms) | 16 ms | 50 ms |
| 128 token | **67,4 tok/s** (14,8 ms) | 19,5 tok/s (51,4 ms) | 3.843 tok/s (0,03 s) | 35 tok/s (3,63 s) |
| 512 token | **66,6 tok/s** (15,0 ms) | 15,6 tok/s (64,1 ms) | 4.856 tok/s (0,11 s) | 30 tok/s (16,7 s) |
| 1024 token | **63,5 tok/s** (15,7 ms) | 11,1 tok/s (90,3 ms) | 5.202 tok/s (0,20 s) | 27 tok/s (36,8 s) |
| 2048 token | 60,7 tok/s (16,5 ms) | — | — | — |
| 4090 token | 54,0 tok/s (18,5 ms) | — | — | — |

Decode 3,0–5,4× lebih cepat, prefill 9,3–10,6× lebih cepat. Satu generasi 120
token di CLI: 67,0 tok/s, TTFT ≈ 140 ms.

Lantai decode di 4050 (≈192 GB/s) adalah 2,68 GB bobot / 192 GB/s ≈ 14 ms per
token; yang terukur 14,5 ms pada konteks 0, jadi decode berada di 96% dari
lantai itu. Dua hal yang memangkas sisa jarak:

- **skala bobot disimpan f16, bukan f32** — skala memang f16 di file GGUF, jadi
  menyimpannya sebagai f16 tidak mengubah satu bit pun dari hasil, tetapi
  memangkas bidang skala dari 0,32 GB menjadi 0,16 GB. Bobot total kembali ke
  2,68 GB persis seperti di file, dan decode turun 15,5 → 14,5 ms;
- **KV cache f16, seperti llama.cpp** — pada 4090 token KV dulu 704 MB dan
  dibaca ulang tiap token (3,7 ms); sekarang 352 MB (1,85 ms). Kurva decode
  jadi 14,5 / 14,8 / 15,0 / 15,7 / 16,5 / 18,5 ms pada konteks
  0 / 128 / 512 / 1024 / 2048 / 4090.

### Dibanding llama.cpp, dan berapa VRAM-nya

`llm-cli bench` mencetak jejak VRAM-nya sendiri di baris terakhir tabel. Angka di
bawah diambil pada clock SM yang dikunci 1800 MHz untuk kedua engine, model yang
sama, `-r 5`, llama.cpp `8ab29fa` dengan `-ngl 99`:

| | llm-cli | llama.cpp | rasio |
|---|---:|---:|---:|
| pp128 | 3.585 | 4.180 | 85,8% |
| pp512 | 4.659 | 5.316 | 87,7% |
| pp1024 | 4.489 | 5.246 | 85,6% |
| tg128 | 68,9 | 69,7 | 98,8% |

Konteks 4096, prefill 1024 token:

| | bobot | KV | scratch | total di device |
|---|---:|---:|---:|---:|
| llm-cli | 2,50 GiB | 0,16 GiB | 0,12 GiB | **2,78 GiB** |
| llama.cpp | 2,23 GiB | 0,16 GiB | 0,055 GiB | 2,45 GiB |

KV cache sama persis — 41 KiB per token, f16, geometri yang sama. Sisa selisih
0,33 GiB punya dua sebab, dan keduanya pilihan llama.cpp: 271 MiB bobotnya tetap
di berkas yang di-mmap dan dibaca dari host, dan seluruh operasinya berbagi satu
compute buffer 56 MiB, sedangkan engine ini menyimpan sembilan buffer scratch
terpisah. Persentase di tabel sebelumnya diukur tanpa mengunci clock dan dengan
build llama.cpp lain, jadi kedua tabel tidak sebanding.

Prefill dulu ±670 GFLOPS (5–10% dari puncak fp32); sekarang matmul-nya sendiri
berjalan 12,5 TMAC/s lewat tensor core int8 dan attention lewat tensor core f16
(lihat bagian-bagian berikut).

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

### Bobot dibaca sekali per 1024 token, bukan per 64

Ini penyebab terbesar jarak prefill, dan bukan soal aritmetika. Satu chunk prefill
dulu 64 token, artinya seluruh 2,8 GB bobot dibaca ulang setiap 64 token — 512
token berarti delapan lintasan penuh. llama.cpp memakai ubatch jauh lebih besar.

Sekarang chunk 1024 token, dan grid kernel prompt diurutkan dengan **tile token
sebagai dimensi tercepat**, sehingga blok-blok yang berjalan bersamaan memakai
tile bobot yang sama dan tile itu tetap di L2, bukan dibaca ulang dari DRAM.
Grid `(n/64, out/256)`.

Hasil pada 8 warp: pp128 2.190 → 2.813, pp512 1.880 → 3.000, pp1024 1.576 →
2.557. Setelah itu tile token dinaikkan ke 128 (satu tile bobot di shared dipakai
untuk 128 token, bukan 64): pp128 2.929, pp512 3.190, pp1024 2.747.

Bentuk akhir kernel: 8 warp, 128 baris, 128 token per blok, potongan k 256,
178 register tanpa spill, 77.824 byte shared per blok.

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

### Profil per kernel (`LLM_PROFILE=1`)

`LLM_PROFILE=1` membungkus setiap peluncuran kernel dengan CUDA event dan
mencetak tabel waktu di dalam GPU saat proses selesai — waktu kernel, bukan waktu
antre. Untuk satu prefill 512 token:

| Kernel | Total ms | Panggilan | ms/panggilan |
|---|---:|---:|---:|
| `q8_matmul_mma` | 81,9 | 294 | 0,279 |
| `attention_prefill` | 25,1 | 42 | 0,596 |
| `quantize_rows` | 4,8 | 168 | 0,029 |
| `silu_mul` | 7,8 | 42 | 0,186 |
| `residual_add` | 3,5 | 84 | 0,042 |
| `rmsnorm_rows` | 2,6 | 85 | 0,031 |
| `rope` | 2,1 | 84 | 0,025 |
| sisanya | ~1,9 | — | — |
| **jumlah** | **124,2** | | |

Wall prefill ~126 ms, jadi hampir seluruhnya kernel — tidak ada kebocoran di
antara peluncuran. Matmul berjalan 13,0 TMAC/s di dalam model versus 13,5 TMAC/s
saat diukur sendirian, jadi jalur itu sudah 96% dari kecepatannya sendiri.

### Ke mana waktu prefill pergi (pp512, 149 ms)

Diukur dengan mematikan satu bagian sekali jalan, bukan diperkirakan:

| Bagian | Waktu | Porsi | Catatan |
|---|---:|---:|---|
| Matmul (semua, termasuk proyeksi output) | 106 ms | 71% | kernelnya sendiri 14,2 TMAC/s; yang terukur di sini 9,6 TMAC/s |
| — di dalamnya: kuantisasi aktivasi | 20 ms | 13% | dulu 7 kali per layer, sekarang 4 (q/k/v berbagi satu) |
| Attention | 32 ms | 22% | turun dari 43 ms setelah empat query per pembacaan kunci |
| Elementwise (rmsnorm, rope, kv_append, silu, residual) | 17 ms | 12% | sudah di batas bandwidth DRAM |

Selisih antara 14,2 TMAC/s (kernel sendirian, diukur dengan bobot 267 MB yang
dialirkan dari DRAM supaya bukan kasus L2) dan 9,6 TMAC/s (di dalam model) adalah
**kuantisasi gelombang**: `out = 2048` dengan tile 128 baris memberi 16 blok per
tile token, dan 20 SM berarti satu gelombang pun belum penuh — 64 blok berjalan
sebagai 3,2 gelombang, jadi 80% efisien. Untuk `out = 6144` angkanya 96%.

Untuk mengejar llama.cpp (87 ms di pp512) tiga hal harus turun bersamaan:
matmul ke ≈75 ms (kernel penuh, tanpa sisa gelombang), attention ke ≈5 ms (butuh
tensor core f16 seperti `fattn` milik ggml), dan elementwise ke ≈8 ms (fusion).

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

Sesudah itu blok diubah ke **dua head tanpa buffer probabilitas**: probabilitas
tinggal di register dan sampai ke pass value lewat `__shfl_sync`, jadi shared turun
ke 48 KB dan dua blok muat per SM (16 warp, bukan 8). Attention 27,8 → 25,1 ms.
Tapi trafik K/V jadi dua kali lebih sering, sehingga pp128 turun 3.518 → 3.438
sementara pp512 dan pp1024 naik (4.063 → 4.099 dan 3.734 → 3.758).

### Attention prefill: dua query per pembacaan kunci

Kernel attention berjalan 0,52 TMAC/s — 11% puncak fp32 — karena satu dot skalar
butuh dua pembacaan shared untuk setiap perkalian-tambah, dan shared hanya bisa
melayani satu transaksi per siklus. Itu membatasi warp di enam belas
perkalian-tambah per siklus, seberapa cepat pun unit aritmetikanya.

Sekarang **empat** query dijalankan lewat pembacaan kunci yang sama, dan empat dim
query diambil sebagai satu `float4` (satu broadcast untuk seluruh warp). Kuncinya
tetap dim-major — key-major membuat semua lane mendarat di bank yang sama dan
justru melambat (terukur: 3.036 vs 3.143).

Hasilnya pp128 2.929 → 3.190, pp512 3.190 → 3.605, pp1024 2.747 → 3.355.
Biaya attention turun dari 27% menjadi 20% dari prefill pp512 (diukur dengan
mematikan panggilan attention: 3.488 → 4.382).

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
kedua backend dengan angka yang sama (max |Δ| 0,118 / 0,572 / 0,274; 32 token
greedy identik). Jadi GPU bukan jalur "kira-kira benar" yang terpisah.

### KV f16, skala bobot f16, dan fusi elementwise

Empat perubahan kecil yang menutup celah decode dan memangkas prefill:

- **KV cache f16**. llama.cpp menyimpan KV dalam f16, engine ini dulu f32.
  Sekarang `kv_append` mengonversi saat menulis, dan ketiga kernel attention
  mengonversi balik saat membaca. Pada konteks 4090 KV yang dibaca ulang tiap
  token turun 704 → 352 MB, jadi decode 20,3 → 18,5 ms — dan numeriknya justru
  makin dekat ke fixture llama.cpp (semua fixture direkam dari cache f16 mereka).
- **Skala bobot f16**. Repack Q8_0 dulu menyimpan skala f16 sebagai f32. Karena
  skala memang f16 di file, menyimpannya sebagai f16 identik bit-per-bit, dan
  bidang skala turun 0,32 → 0,16 GB: decode 15,5 → 14,5 ms. Ada satu jebakan:
  versi pertama memuat skala dengan `LDG.U16` per elemen di kernel prompt dan
  prefill justru turun 7%; memuat dua skala sekaligus sebagai `half2` (akses
  32-bit, bentuk yang sama seperti f32 lama) mengembalikan semuanya.
- **`silu(gate)·up` langsung dikuantisasi** (`silu_mul_quantize_rows`), tanpa
  menulis aktivasi f32 lalu membacanya kembali — hemat satu pass penuh n×6144
  per layer. Perhatian: jalur decode (n < 8) membaca aktivasi f32 langsung dan
  tidak pernah melihat buffer prepared, jadi di bawah ambang itu produk tetap
  ditulis seperti sebelumnya.
- **`rmsnorm` langsung dikuantisasi** (`rmsnorm_quantize_rows`) dengan alasan
  yang sama. Vektor load float4 eksplisit diperlukan di sini — versi skalar
  berjalan dua kali lebih lambat daripada pasangan kernel yang digantikannya.
  Residual matmul juga dicoba dilebur ke epilog kernel mma, tetapi load bias
  yang tersebar di epilog ternyata lebih mahal (81 → 87 ms) daripada pass
  residual terpisah yang digantikannya (3 ms), jadi fusi itu hanya dipakai
  untuk decode (n = 1), di mana satu load per baris tidak menyebar.

Embedding prompt juga dijadikan satu launch (`embedding_rows`) alih-alih satu
launch per token.

### Attention prefill: tensor core f16

Kernel fp32 sebelumnya terikat bandwidth **shared memory**, bukan DRAM: dot
skalar butuh dua pembacaan shared per MAC, jadi tidak peduli seberapa cepat
unit aritmetikanya. Perbaikan yang sebenarnya adalah mengalikan di tensor
core, seperti `fattn` milik ggml:

- satu blok memegang **delapan head** (seluruh grup GQA model ini) dan 16 posisi
  query; satu warp = satu head, 16 query = satu ubin m16;
- Q dan K/V di-stage ke shared sebagai f16 (48 KB, dua blok per SM); K tetap
  dim-major sehingga fragmen B (=Kᵀ) dimuat langsung dengan `ldmatrix`;
- QKᵀ dihitung dengan `mma.sync.m16n8k16.f16` dan skor tinggal di register —
  **skor yang sudah di-exp adalah fragmen A dari PV**, jadi langkah value juga
  `mma` tanpa pembacaan shared tambahan, trik yang sama dengan ggml;
- softmax tetap online: `running_max`/`running_sum` per baris query plus
  koreksi `exp(m_lama − m_baru)` pada akumulator O, persis seperti kernel fp32.

Hasilnya attention prefill 25,5 → 10,5 ms pada pp512 (dan proporsional pada
ukuran lain), yang mengangkat pp128 3.438 → 3.843, pp512 4.099 → 4.856, dan
pp1024 3.758 → 5.202 tok/s. Delta logits terhadap llama.cpp justru mengecil
(logits_golden_long 0,321 → 0,274) karena f16 adalah presisi attention mereka.

Dua jebakan yang memakan waktu paling lama, dicatat karena keduanya tidak
terlihat dari kode:

- **fragmen C `m16n8` adalah d0/d1 = (baris r, kolom 2t dan 2t+1), d2/d3 =
  baris r+8** — bukan d0/d2 = baris r seperti yang tertulis di teks PTX ISA
  yang saya baca. Gejalanya: semua nilai salah kecuali baris pertama;
- **fragmen B butuh `ldmatrix…x2.trans`**. Tanpa transpos, distribusi lane
  memberikan B[lane/4][2t] sedangkan mma meminta B[2t][lane/4]; baris 0
  kebetulan benar sehingga kesalahan terlihat seperti masalah masking.

Keduanya dipastikan lewat micro-probe standalone yang membandingkan kernel
melawan referensi CPU pada matriks acak, bukan lewat forward pass penuh.

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

Kedua angka di bawah diukur dengan **clock SM dikunci 1800 MHz**
(`nvidia-smi -lgc 1800,1800`). Tanpa kunci, boost 4050 laptop bergeser 20–25%
antar proses dan perbandingan kehilangan artinya: pengukuran yang sama bisa
memberi 4.000 atau 5.200 tok/s hanya karena suhu awal kartu.

Kolom engine diukur ulang setelah kerja rencana memori dan CUDA graph
(`bench -p 128,512,1024 -n 128 -r 3`), kolom llama.cpp tetap dari sesi terkunci
yang sama: yang berubah hanya sisi ini, dan sebarannya ikut ditulis.

| Tes | llm GPU | llama.cpp GPU | Rasio |
|---|---:|---:|---:|
| pp128 | 3.718 ± 36 tok/s | 3.788 ± 235 tok/s | 98% |
| pp512 | 4.740 ± 2 tok/s | 4.735 ± 73 tok/s | 100% |
| pp1024 | 4.566 ± 2 tok/s | 4.677 ± 92 tok/s | 98% |
| tg128 | 70,27 ± 0,15 tok/s | 69,3 ± 0,04 tok/s | 101% |

Angka 100% dan 101% ada di dalam sebaran pengukuran llama.cpp (±73 tok/s pada
pp512, ±0,04 pada tg128), jadi bacaannya "setara dalam noise", bukan "lebih
cepat".

Di CPU (llama.cpp `-ngl 0 -t 20`, engine ini `LLM_GPU=0`): decode 16,9–17,2 vs
17,1–19,5 tok/s, prefill 70–99 vs 27–35 tok/s.

| Metrik | llm GPU | llama.cpp GPU |
|---|---:|---:|
| VRAM (ctx 4096) | 2,76 GiB | 2,49 GiB |
| Load + 1 token (wall) | 0,9 s | 1,32 s |

### Apa yang sudah menutup sebagian jarak

- **Matmul yang berbagi aktivasi berangkat dalam satu launch** (`MMA_GROUP_MAX`).
  q, k dan v membaca baris ternormalisasi yang sama; sebagai tiga launch masing-
  masing hanya mengisi 4 + 1 + 1 gelombang dari 34 gelombang per layer, ruang
  barisnya digabung jadi 4. gate|up juga, tetapi hanya sampai 512 token: pada
  1.024 token gabungan itu terukur 2,40 ms melawan 2 × 0,70 ms terpisah
  (`GROUP_FFN_MAX_TOKENS`).
- **Skala bobot tiap lane diangkat keluar dari loop token** — enam belas nilai per
  potongan k, bukan dua load per mma — dan skala aktivasi disimpan block-major
  supaya dua kolom satu lane tiba dalam satu load 64-bit. Enam load shared per mma
  jadi tiga.
- **Semua load satu potongan k diterbitkan sebelum store pertama.** Bobot,
  aktivasi, dan kedua bidang skala berangkat bersama, jadi blok menunggu sekali per
  potongan, bukan empat kali: 99,9 → 90,5 ms matmul pada pp512. Ini yang di
  profiler terbaca sebagai `long_scoreboard` dengan 8 warp.
- **Enam belas warp dan 256 baris per blok.** Tile aktivasi dulu di-stage sekali
  per blok 128 baris, jadi pada 512 token aktivasi disalin sesering bobot: 200 MB
  salinan tile per layer melawan 201 MB bobot, 16,8 GB untuk seluruh prefill.
  Dengan 256 baris, separuh trafik itu hilang, dan SM punya 16 warp untuk
  menyembunyikan latensi staging: 90,5 → 83,3 ms, tensor pipe 29% → 35%.
- **Fragmen B satu load.** Kuantiser menulis tiap blok 32 elemen sebagai (0-3,
  16-19, 4-7, 20-23, 8-11, 24-27, 12-15, 28-31), jadi dua word yang satu lane
  suapkan ke mma bersebelahan dan tiba dalam satu load 64-bit, bukan dua. Permutasi
  itu terjadi saat kuantisasi, jadi salinan tile tetap 16 byte lurus; hanya jalur
  prompt yang melihatnya, decode membaca aktivasi f32 mentah.
- **Tile sisa dipotong di sumbu token** (`wave_plan` + `tail_slices`) supaya
  gelombang terakhir tidak kosong: +4% pada pp512. Potongannya menulis kolom
  berbeda, jadi tidak ada penjumlahan parsial yang harus digabung.
- **Attention prefill: tile terberat dulu.** Masker kausal membuat tile query
  terakhir melihat seluruh konteks, jadi beban antar blok naik linear sementara
  blok dijadwalkan urut indeks. Dibalik: 10,5 → 8,4 ms pada pp512.
- **Decode**: q|k|v dan gate|up satu launch (295 → 169 peluncuran per token),
  attention split dua kali lebih halus, rmsnorm satu barrier dengan load vektor.

### Yang datang dari netra-kernel

Empat mekanisme diambil dari compiler AOT Netra (risetnya di
`docs/research/netra-kernel.md`; inventaris permukaan CUDA yang tersedia untuk
kernel di repo ini ada di `docs/research/cuda-surface.md`); yang tidak diambil adalah compilernya sendiri,
karena di sini "compiler"-nya `build.rs` + nvcc + satu tabel `const`.

- **Identitas build.** `build.rs` mencatat sha256 `gpu.cu`, sha256 PTX yang
  benar-benar dikompilasi, arch, versi nvcc, dan setiap `.entry` di PTX.
  `BuildManifest::identity()` menghash semuanya, `/api/model` melaporkannya, dan
  `bench` mencetaknya di bawah tabel — jadi angka di README ini menunjuk satu
  artefak, bukan satu commit yang mungkin sudah dibangun ulang. Setiap run di
  SQLite juga menyimpan `build_hash`, lewat migrasi tersendiri.
- **Rencana kernel saat load.** Dulu `attention_path()` memutuskan per panggilan
  (uji bentuk + env `LLM_ATTN_OLD` + ambang token), sementara
  `attention_scores()` menghitung ulang keputusan yang sama untuk ukuran scratch
  — dua tempat yang bisa berbeda tanpa ketahuan. Sekarang `tactics.rs` memilih
  sekali saat model dimuat dari tabel yang setiap barisnya membawa pengukuran
  yang menerimanya: tile fp32 `Verified` di belakang tensor core f16,
  delapan-head-per-blok `Rejected` karena 112 KB shared tidak muat di 99 KB,
  dan bentuk yang tidak punya tile jatuh ke jalur serial **dengan alasannya
  dicatat**. Tiga aturan yang dijaga test: varian `Rejected` tidak pernah
  terpilih untuk bentuk apa pun, regime di jalur umum selalu menyebut alasannya,
  dan penggabungan berhenti di 512 token — di 1.024 token terukur 2,40 ms
  melawan 2 × 0,70 ms.
- **Rencana memori statis.** `workspace.rs` mendeklarasikan setiap buffer
  scratch dengan rentang fase hidupnya, lalu menempatkannya: dua buffer yang
  rentangnya tidak bertemu berbagi satu wilayah, dan yang hidup di fase sama
  tidak pernah ditumpuk. Satu arena per sesi, jadi satu token tidak
  mengalokasikan apa pun — sebelumnya 13 alokasi + 13 pembebasan per token.
  Ukuran mengikuti rencana kernel, jadi jalur yang menyimpan probabilitas di
  register merencanakan grid skor nol, bukan 537 MB yang dibutuhkan jalur serial.
- **Graph decode.** Satu langkah decode = 169 peluncuran; sekarang direkam sekali
  dan diputar ulang dengan satu panggilan driver. Tiga hal harus pindah agar
  graph-nya sah: engine punya stream sendiri (legacy default tidak bisa
  di-capture), nilai yang berubah tiap token (`posisi`, `visible`) dibaca kernel
  lewat pointer ke device, dan embedding + unggah posisi tetap di luar graph
  karena keduanya butuh salinan host. Karena grid attention melebar saat konteks
  terisi, sesi menyimpan satu graph per jumlah split — ide "profil per bentuk"
  dari Netra. Mengunci grid di kapasitas menghasilkan satu graph, tetapi
  biayanya 69,0 → 59,0 tok/s, jadi cache itu bukan hiasan.

### Sisa jarak, dan ke mana harus pergi

- **Matmul prefill tinggal ±2–4%** (83 ms lawan ±80 ms pada pp512). Sisa yang
  terukur di profiler: `wait` dan `mio_throttle` di loop dalam, yaitu antrean
  shared memory — bukan lagi salinan tile. Yang belum kita punya dari llama.cpp:
  *stream-k*, yang membagi sumbu k supaya tidak ada gelombang setengah kosong
  (pp128 paling terasa: proyeksi 2048 baris hanya 8 tile).
- **Attention prefill masih 2,3× lebih lambat** (11,1 ms lawan ±4,9 ms) dan itu
  sekarang 10% dari runtime prefill. Profilnya jelas: tensor pipe 11%, dan dua
  stall teratas `long_scoreboard` 5,6 serta `mio_throttle` 5,2 — blok menganggur
  menunggu tile K/V yang di-stage skalar dan dibaca ulang 32× per KV head. Yang
  dibutuhkan tile query lebih besar (K/V dibaca 16× bukan 32×) dengan warp yang
  cukup untuk menyembunyikan staging-nya, bukan tambalan satu baris.
- **Decode 101%**: lantai DRAM 2,39 GB/token ÷ 187,8 GB/s = 12,7 ms, matmul-nya
  13,4 ms (178 GB/s). Celah antar peluncuran sudah hilang oleh graph; sisa ±0,9 ms
  adalah kernel kecil yang benar-benar bekerja, dan yang tersisa di atas lantai
  DRAM hanya selisih 178 vs 187,8 GB/s.
- **CPU**: prefill masih 2,5–3× tertinggal dan decode turun dengan konteks
  (jalur CPU belum memakai tiling attention).

### Dua bug yang ditemukan saat mengejar angka ini

- **`quantize_rows` tidak memeriksa batas.** Grid-nya dibulatkan ke kelipatan 128
  thread, jadi sisa thread mengkuantisasi baris `n` dan menulisnya menimpa bidang
  skala: prompt 65 token (atau 129, 193, …) menghasilkan logits yang melantur,
  sementara 64 dan 66 token benar. Test `gpu_matmul` sekarang memuat 65, 129, dan
  193 justru karena itu.
- **Lebar potongan tail dipakai untuk semua blok.** Saat sebuah launch perlu
  memotong tile sisa, blok yang memegang tile penuh ikut menghitung seperempat
  token. Angka prefill sempat terlihat 25% lebih tinggi karenanya; angka di tabel
  atas adalah yang menghitung semua token.

## Catatan implementasi

- **Tidak ada alokasi di jalur token.** Scratch satu token dan satu chunk
  direncanakan sekali per sesi (`workspace.rs`), dan `Model::forward` tidak
  memanggil `alloc` sama sekali. Itu juga yang membuat capture graph mungkin.
- **Keputusan kernel ada di satu tempat.** `tactics.rs` memilih jalur attention
  dan kebijakan penggabungan matmul saat model dimuat; `compute.rs` hanya
  menanyakan rencana itu. Tidak ada ambang bentuk yang tersebar di call site.
- **Angka menunjuk artefak.** Setiap keluaran `bench` mencetak identitas build dan
  identitas rencana; tanpa itu sebuah angka tidak bisa dibandingkan dengan angka
  lain.
- **Loader membaca kunci milik arsitektur modelnya sendiri**, bukan `llama.*`
  saja, dan menolak model yang bloknya tidak bisa dijalankan dengan alasan yang
  jelas. Contoh nyata: `models/moe-expert-coder-4b-Q8_0.gguf` adalah model
  hibrida (`qwen35.ssm.*` — lapisan state-space diselang-seling dengan
  attention, head_dim 256), dan sekarang gagal saat load dengan
  `architecture qwen35 interleaves state-space blocks with attention; this
  engine implements dense attention blocks only`, bukan `MissingMetadata`.
  Model dense dari keluarga lain (qwen2, llama, mistral) akan termuat selama
  head_dim ≤ 128.
- **Batas model 2B ini, dan di mana batasnya.** Pada percakapan yang memuat
  nama pengguna, model kadang menyebut dirinya dengan nama itu ("namaku Budi" →
  "Saya Budi!"), dan pada sebagian percakapan ia membeku mengulang jawaban
  sebelumnya (terukur 1 dari 5 percakapan dua giliran). Ini bukan bug engine:
  prompt yang sama persis, diberikan ke llama.cpp b11429, menghasilkan
  kekacauan yang sama ("Saya ingerti kamu mungkin ngisiné saka aplikasi AI
  seperti ini"). Prompt sistem (9 varian diuji), BOS, dan penyempitan sampling
  mengurangi gejalanya tetapi tidak menghilangkannya — yang tersisa adalah
  kemampuan bahasa Indonesia model 2B. Jalur perbaikannya model lain, bukan
  kode: loader membaca GGUF arsitektur llama apa pun.
- **Truncation sampling lebih sempit dari default llama.cpp, dan itu diukur.**
  Pada `top_p 0.95` / `min_p 0.05` model ini keluar dari bahasa Indonesia:
  menjawab dengan bahasa Jawa dan menyangkal fakta yang baru saja diberikan.
  `0.8 / 0.15` benar tapi masih menyisakan ekor Jawa ("iki ana"), `0.75 / 0.25`
  bersih di semua pertanyaan yang diuji, pada dua seed. Default API dan
  `SamplingParams::default()` memakai 0.75 / 0.25.
- **Prompt percakapan diawali BOS**, karena template model memintanya
  (`{{- bos_token }}`) dan sebelumnya engine ini tidak pernah mengirimnya.
  Tanpa BOS jawaban recall keluar melantur; dengan BOS jawabannya
  "Kode proyek NX-42.".
- **Jendela repeat-penalty sengaja hanya berisi token yang dihasilkan**, bukan
  ekor prompt seperti llama.cpp: menyalin kebiasaan itu terukur lebih buruk
  karena menekan justru token yang harus diingat (jawaban berubah menjadi
  penolakan berbahasa Jawa).
- **System prompt ikut bahasa percakapan, dan itu bukan selera.** Diukur pada
  model ini: dengan `"You are a helpful assistant."` percakapan dua giliran
  runtuh — ditanya "Kode proyeknya apa?" setelah diberi tahu NX-42, jawabannya
  `"I'm sorry, I can't assist with that."` lalu penolakan itu diulang untuk
  giliran berikutnya. Dengan `"Anda adalah asisten AI. Jawab singkat dalam
  bahasa Indonesia."` model menjawab satu giliran lalu membeku pada fragmen yang
  sama ("Aku inggah."). Dengan `"Kamu asisten yang membantu."` keempat pertanyaan
  dijawab dan kedua fakta diingat; instruksi yang lebih panjang justru membuatnya
  melantur, jadi prompt dibiarkan satu kalimat.
- **Bentuk launch divalidasi terhadap device, bukan diasumsikan.** Saat load,
  `cuFuncGetAttribute` dan `cuOccupancyMaxActiveBlocksPerMultiprocessor` dibaca
  untuk tiga kernel yang menentukan: matmul prompt 127 register / 77.824 byte
  shared / 512 thread / 1 blok per SM, attention prefill 122 / 49.152 / 256 / 2,
  matmul decode 42 / 0 / 128 / 10. Wave plan memakai angka blok-per-SM itu,
  bukan `1`. Di build debug, `launch()` juga membandingkan jumlah argumen dengan
  `cuFuncGetParamCount` — `cuLaunchKernel` tidak memeriksa itu, jadi call site
  yang tertinggal satu argumen akan mengirim sampah tanpa error.

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
