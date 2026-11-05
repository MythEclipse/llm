"""Write a tiny SentencePiece GGUF so the spm path has a fixture.

The vocabulary is built from two words with every prefix present, because the
SentencePiece encoder starts from single characters and merges upward by score;
without the intermediate pieces nothing would merge and the test would only
exercise the byte fallback.

    python3 scripts/make_tiny_spm.py
"""

import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

SPACE = "\u2581"
WORDS = ["hello", "world"]


def vocabulary():
    """`<unk>`, `<s>`, `</s>`, then every prefix of every word with the marker."""
    pieces = ["<unk>", "<s>", "</s>"]
    scores = [0.0, 0.0, 0.0]
    for word in WORDS:
        for length in range(1, len(word) + 1):
            pieces.append(SPACE + word[:length])
            # Longer pieces score higher, so the merge walk climbs to the word.
            scores.append(float(length))
    return pieces, scores


def meta_str(value):
    encoded = value.encode()
    return 8, struct.pack("<Q", len(encoded)) + encoded


def meta_u32(value):
    return 4, struct.pack("<I", value)


def meta_i32_array(values):
    return 9, struct.pack("<I", 5) + struct.pack("<Q", len(values)) + b"".join(
        struct.pack("<i", value) for value in values
    )


def meta_f32_array(values):
    return 9, struct.pack("<I", 6) + struct.pack("<Q", len(values)) + b"".join(
        struct.pack("<f", value) for value in values
    )


def meta_str_array(values):
    payload = struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for value in values:
        encoded = value.encode()
        payload += struct.pack("<Q", len(encoded)) + encoded
    return 9, payload


def write_gguf(path, metadata):
    header = bytearray()
    header += struct.pack("<I", 0x4655_4747)
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 0)  # no tensors
    header += struct.pack("<Q", len(metadata))
    for key, (kind, payload) in metadata:
        encoded = key.encode()
        header += struct.pack("<Q", len(encoded)) + encoded
        header += struct.pack("<I", kind) + payload
    # The reader aligns the start of the tensor data section, and a file with no
    # tensors still has to be long enough for that section to exist.
    padding = (32 - len(header) % 32) % 32
    path.write_bytes(bytes(header) + b"\x00" * (padding + 32))


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    pieces, scores = vocabulary()
    types = [2] + [3, 3] + [1] * (len(pieces) - 3)
    metadata = [
        ("general.architecture", meta_str("llama")),
        ("general.name", meta_str("tiny-spm")),
        ("tokenizer.ggml.model", meta_str("llama")),
        ("tokenizer.ggml.tokens", meta_str_array(pieces)),
        ("tokenizer.ggml.scores", meta_f32_array(scores)),
        ("tokenizer.ggml.token_type", meta_i32_array(types)),
        ("tokenizer.ggml.bos_token_id", meta_u32(1)),
        ("tokenizer.ggml.eos_token_id", meta_u32(2)),
        ("tokenizer.ggml.add_space_prefix", meta_u32(1)),
        ("tokenizer.chat_template", meta_str("[INST] {{ messages }} [/INST]")),
    ]
    path = FIXTURES / "tiny_spm.gguf"
    write_gguf(path, metadata)
    print(f"{path.name}: {path.stat().st_size} bytes, {len(pieces)} pieces")


if __name__ == "__main__":
    main()
