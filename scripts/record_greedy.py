"""Record a greedy (temperature 0) continuation from llama-server.

`llama-completion` applies the chat template, so it answers a *different*
prompt; the `/completion` endpoint of llama-server is the raw-completion path
and matches what this engine does.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = "The capital of France is"
MAX_TOKENS = 32
PORT = 8123


def wait_until_ready(timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise SystemExit("llama-server did not become ready")


def main() -> None:
    model, root = sys.argv[1], sys.argv[2]
    server = subprocess.Popen(
        [
            "llama-server", "-m", model, "-c", "1024", "--port", str(PORT),
            "--no-webui", "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_ready()
        body = json.dumps(
            {
                "prompt": PROMPT,
                "n_predict": MAX_TOKENS,
                "temperature": 0,
                "top_k": 1,
                "repeat_penalty": 1.0,
                "cache_prompt": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=body,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
    finally:
        server.terminate()
        server.wait(timeout=30)

    text = payload["content"]
    json.dump(
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "text": text},
        open(f"{root}/tests/fixtures/greedy_golden.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(f"recorded {len(text)} chars of greedy continuation")


if __name__ == "__main__":
    main()
