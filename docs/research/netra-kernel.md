# Riset: NetraRuntime/netra-kernel — apa yang layak diambil

Riset terhadap <https://github.com/NetraRuntime/netra-kernel> (clone lokal, commit
`main` per 2026-11-05, 84 MB) dibandingkan dengan engine di repo ini.

Kesimpulan satu baris: **tidak ada kode yang bisa dipakai langsung** (compiler-nya
Python, targetnya assembly AMD gfx950). Yang berguna adalah lima invarian
desainnya, dan tiga di antaranya menutup lubang yang sudah tercatat di README
repo ini.

## 1. Apa itu netra-kernel

Compiler kernel GPU *ahead-of-time* untuk inferensi LLM di AMD, gaya TensorRT:
profil → taktik → layout → rencana memori statis → engine. Kernel yang dilayani
bukan kernel generik, melainkan assembly mentah yang sudah terspesialisasi untuk
satu kontrak servis. Target produksi `gfx950`/wave64, FP8, Qwen3.6-35B di 8×
MI350X (78.498 tok/s output, 15.360/15.360 cek request identik).

```
compiler/netra_compiler/   2.753 LOC Python: ir, contracts, profiles, layouts,
                           tactics, memory, planner, engine, validation
kernels/gfx950/            template assembly + include tree (guards di assembler)
runtime/gfx950/engine/     C ABI di atas HIP (~netra_engine.hip)
schemas/                   6 JSON schema (engine, contract, profile, model, bundle)
docs/notes/                ~200 file bukti pengukuran per kernel
```

Alurnya:

```
frontend → IR → profile (guard eksak) → layout plan → tactic planner
        → static memory plan → instansiasi template → engine dir
        → runtime init (parse, load, resolve, alloc, bind, capture graph)
        → launch (guard + handle cache saja)
```

Artefak satu engine: `engine.json` + `contracts.json`, `graph.json`,
`graph_recipe.json`, `memory_plan.json`, `layout_plan.json`, `symbols.json`,
`validation_plan.json`, `explain.json`.

## 2. Invarian yang mereka pertahankan

1. **Generality tinggal di compiler, tidak pernah di kernel.** Shape, dtype,
   layout, quantization, epilogue, grid, block, LDS, offset workspace — semuanya
   konstanta sebelum serving. Tidak ada cabang shape saat runtime. Kalau butuh
   nilai di luar set tervalidasi, harus ada leaf taktik baru yang diukur.
2. **Kontrak = identitas komputasi immutable.** Hash dari JSON kanonik; nama
   model, timestamp, path absolut, evidence tidak masuk hash. Karena itu Qwen dan
   Gemma bisa memakai kontrak yang sama persis.
3. **Guard profil eksak; di luar itu fallback eksplisit.**
   `NETRA_STATUS_FALLBACK_REQUIRED` (kode 9), bukan dispatch kira-kira.
4. **Ledger kematangan taktik.** `experiment | verified | accepted | rejected`,
   plus `rank`, `evidence_refs`, `rejection_note`, dan `explain.json` yang mencatat
   setiap kandidat yang ditolak beserta alasannya. Taktik `rejected` tidak pernah
   dipilih. Poin penting: kernel dense mentah mereka menghasilkan `.text` yang
   byte-identik dan 0/2.048 mismatch BF16 — **tetap rejected** karena A/B serving
   regresi 4,51% / 10,02%. Angka kalah = veto permanen, bukan catatan.
5. **Rencana memori statis.** `BufferRequest{size, alignment, first_use, last_use,
   persistent, alias_of}` → offset + high-water; `workspace_bytes` masuk manifest.
   Nol alokasi di hot path.
6. **Determinisme artefak.** Kompilasi ke direktori berbeda menghasilkan file
   semantik byte-identik. Dua hash terpisah: integritas sumber (hash seluruh
   include closure) vs identitas komputasi (instruksi ternormalisasi, tanpa
   komentar/alias/nama model).
7. **Init melakukan segalanya, launch tidak melakukan apa pun.** Setelah binding
   dibekukan, launch tidak boleh alokasi, parsing, muat modul, akses file/env,
   cari string, atau registrasi mutable. Hanya cek device + guard numerik, lalu
   launch dari handle cache di stream milik pemanggil. Kernarg disimpan
   caller-local, jadi panggilan bersamaan tidak bisa saling menimpa pointer.
8. **Graph recipe = data deterministik**, bukan HIP graph terserialisasi. Capture
   dan instantiate per proses setelah binding; resepnya berisi index, depends_on,
   symbol, launch, bindings, workspace_offset.
9. **Gerbang bukti sebelum promosi**, berurutan: statis → cross-assembly
   (`.text` identik) → correctness di hardware → graph replay (20×) → A/B serving
   sepadan dari proses segar. `promotion_eligible: false` sebagai default.

## 3. Yang memetakan ke repo ini

Repo ini: engine Rust + CUDA sendiri untuk MiniCPM5-2B Q8_0, sm_89, sudah 93–98%
llama.cpp, kernel di satu `gpu.cu` (80 KB), PTX di-embed `build.rs`.

| # | Mekanisme netra | Keadaan di sini | Perubahan konkret | Nilai |
|---|---|---|---|---|
| A | Dispatch diputuskan saat load, guard eksak saat launch | `attention_path()` memutuskan per panggilan, termasuk cabang env `LLM_ATTN_OLD`; `group_worth_it()` pakai ambang `n <= 512`; `prepare()` pakai `n < 8` | satu `Plan`/`KernelSet` dibangun saat model load dari (arch, n_head, n_head_kv, head_dim, quant, ctx) → kernel terpilih + geometri launch + byte scratch; `attend()` hanya cek guard murah | tinggi |
| B | Manifest + hash artefak | `build.rs` sudah panggil nvcc, tidak mencatat apa pun | `engine.json` ringkas: arch, versi nvcc, sha256(`gpu.cu`+`build.rs`), daftar symbol + grid/block/dynamic-smem per launch site, high-water memori, versi ledger taktik. Hash-nya ikut ditulis ke row SQLite tiap run | tinggi |
| C | Ledger taktik + `explain` | ~20 eksperimen (termasuk yang negatif) hanya hidup sebagai prosa README + komentar `const`; `LLM_ATTN_OLD` masih bisa memilih varian yang kalah | tabel data: nama varian, predikat guard, maturity, delta terukur, rujukan bukti, rank. `LLM_PROFILE=1`/bench mencetak apa yang dipilih dan apa yang ditolak | tinggi |
| D | Rencana memori statis | `cuMemAlloc_v2` per buffer, counter footprint, headroom 512 MB heuristik; VRAM 2,9 vs 2,49 GiB llama.cpp | arena + offset dari lifetime per fase layer (q/k/v, gate/up, out attention); `attention_scores()` jadi keluaran plan, bukan query terpisah | sedang-tinggi |
| E | Graph recipe → capture saat init | belum ada (`cuGraph*` nol); README sudah menyebut ini sisa 0,9–1,3 ms decode | capture per bentuk decode setelah binding beku; posisi token di buffer device yang di-update di luar graph; prasyarat: D (tidak boleh alokasi saat capture) | sedang |
| F | Fallback eksplisit | head_dim ≠ 128 atau rasio GQA ≠ 8 diam-diam jatuh ke `Serial`/`Host` | status kapabilitas saat load + alasan; tampilkan di `/api/model` dan warning | rendah |
| G | Gerbang promosi | sudah ada fixture golden, bench `pp/tg`, ± dari 3 run | satu perintah yang gagal kalau taktik berstatus `accepted` tanpa field bukti lengkap | rendah (ikut C) |

Urutan yang masuk akal: **B → C → A → D → E**. B dulu karena semua klaim
sesudahnya bergantung pada "angka ini milik kode yang mana". C karena
pengetahuannya sudah ada, cuma belum berbentuk yang bisa menolak. A karena
menghapus cabang di hot path sekaligus jadi prasyarat E. D menutup 0,4 GiB dan
membuat E aman. E terakhir, dan hanya setelah D.

## 4. Yang tidak layak diambil

- **Compiler Python + IR + frontend + schema + bundle/aktivasi + adapter
  SGLang.** Netra butuh itu karena harus melayani banyak kontrak lintas model dan
  derajat TP. Di sini "compiler"-nya adalah `build.rs` + nvcc + satu tabel const.
  Menyalinnya berarti menulis lebih banyak kode daripada engine-nya sendiri.
- **Spesialisasi lewat assembly mentah.** nvcc/ptxas sudah memberi spesialisasi
  lewat template; menulis SASS tangan untuk sm_89 tidak membeli apa pun. Alasan
  mereka menulis assembly adalah jalur FP8 tensor core di gfx950, bukan skala.
- **Kekhususan AMD**: wave64, MFMA, anggaran LDS, HSACO, HIP graph. Padanan CUDA
  sudah dipakai (`cuFuncSetAttribute` untuk dynamic smem).
- **C ABI `.so` + packaging pkg-config.** Ini satu binary Rust; batas C tanpa
  konsumen.
- **Delapan file manifest per engine.** Satu `engine.json` cukup.
- **"Byte-identik tanpa timestamp" sebagai tujuan itu sendiri.** Ambil hash-nya,
  buang upacaranya.

## 5. Status implementasi

Kelima mekanisme yang layak sudah masuk. Ukuran di bawah dari `bench -p
128,512,1024 -n 128 -r 3` pada RTX 4050 dengan clock SM dikunci 1800 MHz,
dibandingkan angka README sebelum kerja ini.

| # | Mekanisme | Di mana | Hasil |
|---|---|---|---|
| B | identitas build (sha256 sumber + PTX + arch + nvcc + daftar `.entry`) | `apps/engine/build.rs`, `src/manifest.rs`, `GET /api/model`, kolom `runs.build_hash` | angka bench menunjuk artefak; run di dashboard bisa ditelusuri ke build |
| C | ledger taktik + guard eksak + fallback beralasan | `src/tactics.rs` | `LLM_ATTN_OLD` dan ambang bentuk di call site hilang; varian yang kalah tidak bisa terpilih lagi |
| A | rencana kernel diputuskan saat load | `Compute::plan`, `KernelPlan::for_model` | satu tempat memutuskan, ukuran scratch diturunkan dari keputusan itu |
| D | rencana memori statis + satu arena | `src/workspace.rs`, `Compute::workspace` | 13 alokasi + 13 free per token → 0; pp512 4.601 → 4.740, tg128 68,2 → 69,0, VRAM puncak 2,9 → 2,76 GiB |
| E | graph decode, satu per bentuk grid | `cuda::Graph`, `model::DecodeGraphs` | tg128 69,0 → 70,3 (+1,8%), 8 graph untuk 128 token pada ctx 4096 |
| F/G | fallback eksplisit + gerbang bukti | `KernelPlan::fallbacks`, test `every_variant_states_its_evidence` | regime di jalur umum selalu menyebut alasan; taktik `accepted` tanpa bukti menggagalkan test |

Yang ditolak setelah diukur, dan alasannya dicatat di sini supaya tidak dicoba
lagi tanpa angka baru:

- **Grid attention dikunci di kapasitas** (satu graph untuk semua posisi):
  menghapus cache per bentuk, tetapi `splits` ikut kapasitas sehingga pada ctx
  4096 ada 16 × 64 blok per layer yang sebagian besar kosong. tg128 69,0 → 59,0
  (−14,5%). Cache per jumlah split yang dipakai.
- **Graph untuk prefill**: bentuk `n` berubah tiap chunk dan chunk terakhir
  selalu berbeda, jadi capture-nya tidak pernah terpakai dua kali.
- **Arena dengan aliasing agresif**: rentang hidup di layer loop ini hampir
  semuanya bertumpuk (`x` hidup dari awal sampai akhir), jadi penghematannya
  hanya ±7% dari scratch; yang benar-benar berharga adalah satu alokasi dan
  pointer yang tidak berpindah.

## 6. Catatan kejujuran

Angka netra (78,5k tok/s, 15.360 cek) tidak bisa dipindahkan ke sini: hardware,
model, dan presisi berbeda. Yang bisa dipindahkan hanya bentuk invariannya.
Prediksi itu terbukti: A/B/C/F/G tidak mengubah tok/s sama sekali (dan memang
tidak boleh — `logits_golden` dan `greedy_golden` lulus tanpa perubahan), D dan E
yang mengubah angka: prefill +3%, decode +3%, VRAM puncak −0,14 GiB.
