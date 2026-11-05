"""Compare llama.cpp's eval-callback dump with this engine's raw f32 dumps.

llama.cpp prints, per row, the first three and the last three values of the row
and elides the middle, so only those edges can be compared. A row is `ne0` wide
(or fewer when the row is short enough that nothing was elided).

The engine's dumps are flat f32 arrays: token-major, then the tensor's own
innermost dimension. `--heads` says how many `ne0`-wide rows one token holds, so
row `(token, head)` can be found on both sides.

    python3 scripts/compare_reference.py /tmp/ref.txt /tmp/ours \
        r_raw-0-n5 attn_output-0 --width 4096 --heads 32 --head-dim 128
"""

import argparse
import re
import struct


def tensor_edges(text, name):
    """(shape, rows) where each row is the printed first/last values."""
    found = re.search(
        rf"common_debug_cb_eval:\s+{re.escape(name)} = .*= \{{([\d, ]+)\}}", text
    )
    if found is None:
        raise SystemExit(f"{name} not found")
    shape = [int(part) for part in found.group(1).split(",")]
    start = found.start()
    block = text[start:text.index("sum = ", start)]
    numbers = [float(value) for value in re.findall(r"-?\d+\.\d+", block)]
    row_width = min(shape[0], 6)
    return shape, [numbers[i:i + row_width] for i in range(0, len(numbers), row_width)]


def our_flat(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return struct.unpack(f"<{len(data) // 4}f", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("directory")
    parser.add_argument("ours")
    parser.add_argument("theirs")
    parser.add_argument("--width", type=int, required=True, help="floats per token in our dump")
    parser.add_argument("--heads", type=int, default=1, help="rows per token")
    parser.add_argument("--head-dim", type=int, default=None, help="floats per row")
    parser.add_argument("--tokens", type=int, default=5)
    args = parser.parse_args()

    text = open(args.reference, encoding="utf-8", errors="replace").read()
    shape, rows = tensor_edges(text, args.theirs)
    ours = our_flat(f"{args.directory}/{args.ours}.f32")
    head_dim = args.head_dim or args.width
    print(f"{args.ours} vs {args.theirs}: shape {shape}, {len(rows)} reference rows")

    for token in range(min(args.tokens, len(ours) // args.width)):
        worst = 0.0
        for head in range(args.heads):
            row = rows[token * args.heads + head]
            if len(row) < 6:
                continue
            base = token * args.width + head * head_dim
            mine = ours[base:base + head_dim]
            if len(mine) < head_dim:
                continue
            pairs = list(zip(mine[:3], row[:3])) + list(zip(mine[-3:], row[3:6]))
            worst = max(worst, max(abs(a - b) for a, b in pairs))
        print(f"  token {token}: max |diff| over the edges {worst:.5f}")


if __name__ == "__main__":
    main()
