"""Record the final logit vector and prompt ids from llama-debug.

Two fixtures: a five-token prompt (batched prefill, llama.cpp dequantizes the
weights to f16 there) and a single-token prompt (gemv path, where llama.cpp uses
the same exact Q8_0 dot product this engine implements).
"""

import json
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

PROMPTS = {
    "logits_golden": "The capital of France is",
    "logits_golden_single": "The",
    # A rendered ChatML turn: 19 tokens, so prefill has to keep positions,
    # attention and the batched matmul all correct over more than one chunk.
    # 100 plain tokens: two prefill chunks, so position handling and attention
    # over a long prompt are covered too.
    "logits_golden_long": (
        "Di pagi yang tenang, seorang nelayan menurunkan perahunya ke laut dan "
        "mulai mendayung menuju tempat yang biasa ia kunjungi selama bertahun-tahun. "
        "Angin laut berhembus pelan, membawa bau garam dan suara ombak yang memecah "
        "di batu karang. Ia menghitung setiap dayungan sambil mengingat nasihat "
        "ayahnya tentang membaca arah angin dan mengenali tanda-tanda cuaca. "
        "Langit di ufuk timur mulai berwarna keemasan, tanda bahwa matahari akan "
        "segera terbit dan menemani perjalanan panjang hari itu."
    ),
}


def record(model: str, root: str, stem: str, prompt: str) -> None:
    outdir = tempfile.mkdtemp()
    subprocess.run(
        [
            "llama-debug", "-m", model, "-p", prompt, "-n", "1",
            "--save-logits", "--logits-output-dir", outdir,
        ],
        check=True,
        capture_output=True,
    )
    bins = sorted(
        pathlib.Path(outdir).glob("*.bin"), key=lambda p: p.stat().st_size, reverse=True
    )
    logits_bin, tokens_bin = bins[0], bins[-1]
    tokens = list(
        struct.unpack(f"<{tokens_bin.stat().st_size // 4}i", tokens_bin.read_bytes())
    )
    shutil.copy(logits_bin, f"{root}/tests/fixtures/{stem}.bin")
    json.dump(
        {"prompt": prompt, "tokens": tokens, "logits_file": f"{stem}.bin"},
        open(f"{root}/tests/fixtures/{stem}.json", "w", encoding="utf-8"),
        indent=1,
    )
    print(f"{stem}: {logits_bin.stat().st_size // 4} logits, {len(tokens)} prompt tokens")


def main() -> None:
    model, root = sys.argv[1], sys.argv[2]
    for stem, prompt in PROMPTS.items():
        record(model, root, stem, prompt)


if __name__ == "__main__":
    main()
