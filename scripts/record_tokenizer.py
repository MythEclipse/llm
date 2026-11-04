"""Record llama.cpp's token ids for every line of the corpus."""

import ast
import json
import subprocess
import sys


def main() -> None:
    model, root = sys.argv[1], sys.argv[2]
    corpus = json.load(open(f"{root}/tests/fixtures/tokenizer_corpus.json", encoding="utf-8"))
    golden = []
    for text in corpus:
        out = subprocess.run(
            ["llama-tokenize", "-m", model, "--ids", "--no-escape", "-p", text],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        golden.append({"text": text, "ids": ast.literal_eval(out)})
    path = f"{root}/tests/fixtures/tokenizer_golden.json"
    json.dump(golden, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"recorded {len(golden)} tokenizer cases")


if __name__ == "__main__":
    main()
