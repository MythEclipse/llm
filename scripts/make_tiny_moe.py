"""Write the tiny GGUF fixtures the MoE test needs.

Three files, same seeded weights:

- `tiny_dense.gguf` — two dense layers, the baseline every other file must match.
- `tiny_moe.gguf` — the same feed-forward split into four identical experts, so
  the routed block must reproduce the dense one exactly. This is what proves the
  router weights sum to one and the accumulation is not double-counted.

Run from the repository root:

    python3 scripts/make_tiny_moe.py
"""

import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

EMBD = 32
HEADS = 4
HEADS_KV = 2
HEAD_DIM = 8
FF = 32
EXPERTS = 4
EXPERTS_USED = 2
VOCAB = 64
LAYERS = 2
EPS = 1e-5
ROPE_BASE = 10_000.0
SEED = 0x5EED

# ggml tensor type ids this writer emits.
F32 = 0


def meta_u32(value):
    return 4, struct.pack("<I", value)


def meta_f32(value):
    return 6, struct.pack("<f", value)


def meta_str(value):
    encoded = value.encode()
    return 8, struct.pack("<Q", len(encoded)) + encoded


def write_gguf(path, metadata, tensors):
    """`tensors` is a list of (name, dims, float32 array) in file order."""
    header = bytearray()
    header += struct.pack("<I", 0x4655_4747)  # "GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))

    for key, (kind, payload) in metadata:
        encoded = key.encode()
        header += struct.pack("<Q", len(encoded)) + encoded
        header += struct.pack("<I", kind) + payload

    # Tensor data offsets are relative to the aligned start of the data section.
    offset = 0
    infos = []
    blobs = []
    for name, dims, array in tensors:
        encoded = name.encode()
        info = struct.pack("<Q", len(encoded)) + encoded
        info += struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", dim) for dim in dims)
        info += struct.pack("<I", F32)
        info += struct.pack("<Q", offset)
        infos.append(info)
        blob = np.ascontiguousarray(array, dtype="<f4").tobytes()
        blobs.append(blob)
        offset += len(blob)
        offset = (offset + 31) // 32 * 32

    header += b"".join(infos)
    padding = (32 - len(header) % 32) % 32
    data = bytearray()
    for blob in blobs:
        data += blob
        data += b"\x00" * ((32 - len(data) % 32) % 32)

    path.write_bytes(bytes(header) + b"\x00" * padding + bytes(data))


def weights(rng):
    """Every matrix the tiny model needs, seeded so the files are reproducible."""

    def matrix(k, out):
        return (rng.standard_normal((out, k)) * 0.2).astype(np.float32)

    layers = []
    for _ in range(LAYERS):
        layers.append(
            {
                "attn_norm": (1.0 + rng.standard_normal(EMBD) * 0.02).astype(np.float32),
                "attn_q": matrix(EMBD, EMBD),
                "attn_k": matrix(EMBD, HEADS_KV * HEAD_DIM),
                "attn_v": matrix(EMBD, HEADS_KV * HEAD_DIM),
                "attn_output": matrix(EMBD, EMBD),
                "ffn_norm": (1.0 + rng.standard_normal(EMBD) * 0.02).astype(np.float32),
                "ffn_gate": matrix(EMBD, FF),
                "ffn_up": matrix(EMBD, FF),
                "ffn_down": matrix(FF, EMBD),
            }
        )
    return {
        "token_embd": matrix(EMBD, VOCAB),
        "output": matrix(EMBD, VOCAB),
        "output_norm": (1.0 + rng.standard_normal(EMBD) * 0.02).astype(np.float32),
        "layers": layers,
    }


def base_metadata(experts, used=EXPERTS_USED):
    metadata = [
        ("general.architecture", meta_str("llama")),
        ("general.name", meta_str("tiny")),
        ("llama.block_count", meta_u32(LAYERS)),
        ("llama.embedding_length", meta_u32(EMBD)),
        ("llama.feed_forward_length", meta_u32(FF)),
        ("llama.attention.head_count", meta_u32(HEADS)),
        ("llama.attention.head_count_kv", meta_u32(HEADS_KV)),
        ("llama.attention.key_length", meta_u32(HEAD_DIM)),
        ("llama.rope.dimension_count", meta_u32(HEAD_DIM)),
        ("llama.rope.freq_base", meta_f32(ROPE_BASE)),
        ("llama.attention.layer_norm_rms_epsilon", meta_f32(EPS)),
        ("llama.vocab_size", meta_u32(VOCAB)),
    ]
    if experts:
        metadata += [
            ("llama.expert_count", meta_u32(EXPERTS)),
            ("llama.expert_used_count", meta_u32(used)),
            ("llama.expert_feed_forward_length", meta_u32(FF)),
        ]
    return metadata


def dense_tensors(model):
    tensors = [
        ("token_embd.weight", [EMBD, VOCAB], model["token_embd"]),
        ("output_norm.weight", [EMBD], model["output_norm"]),
        ("output.weight", [EMBD, VOCAB], model["output"]),
    ]
    for index, layer in enumerate(model["layers"]):
        prefix = f"blk.{index}"
        tensors += [
            (f"{prefix}.attn_norm.weight", [EMBD], layer["attn_norm"]),
            (f"{prefix}.attn_q.weight", [EMBD, EMBD], layer["attn_q"]),
            (f"{prefix}.attn_k.weight", [EMBD, HEADS_KV * HEAD_DIM], layer["attn_k"]),
            (f"{prefix}.attn_v.weight", [EMBD, HEADS_KV * HEAD_DIM], layer["attn_v"]),
            (f"{prefix}.attn_output.weight", [EMBD, EMBD], layer["attn_output"]),
            (f"{prefix}.ffn_norm.weight", [EMBD], layer["ffn_norm"]),
            (f"{prefix}.ffn_gate.weight", [EMBD, FF], layer["ffn_gate"]),
            (f"{prefix}.ffn_up.weight", [EMBD, FF], layer["ffn_up"]),
            (f"{prefix}.ffn_down.weight", [FF, EMBD], layer["ffn_down"]),
        ]
    return tensors


def moe_tensors(model):
    """Stacked experts. `identical_experts` copies the dense matrices into every
    slot; otherwise expert 0 keeps them and the rest are zeros."""
    tensors = [
        ("token_embd.weight", [EMBD, VOCAB], model["token_embd"]),
        ("output_norm.weight", [EMBD], model["output_norm"]),
        ("output.weight", [EMBD, VOCAB], model["output"]),
    ]
    for index, layer in enumerate(model["layers"]):
        prefix = f"blk.{index}"
        gate = np.zeros((EXPERTS, FF, EMBD), dtype=np.float32)
        up = np.zeros((EXPERTS, FF, EMBD), dtype=np.float32)
        down = np.zeros((EXPERTS, EMBD, FF), dtype=np.float32)
        for expert in range(EXPERTS):
            gate[expert] = layer["ffn_gate"]
            up[expert] = layer["ffn_up"]
            down[expert] = layer["ffn_down"]
        # A zero router makes every logit equal, so the chosen set is arbitrary —
        # which is exactly what a fixture of identical experts wants: whichever
        # experts are picked, the routing weights must still sum to one.
        router = np.zeros((EXPERTS, EMBD), dtype=np.float32)
        tensors += [
            (f"{prefix}.attn_norm.weight", [EMBD], layer["attn_norm"]),
            (f"{prefix}.attn_q.weight", [EMBD, EMBD], layer["attn_q"]),
            (f"{prefix}.attn_k.weight", [EMBD, HEADS_KV * HEAD_DIM], layer["attn_k"]),
            (f"{prefix}.attn_v.weight", [EMBD, HEADS_KV * HEAD_DIM], layer["attn_v"]),
            (f"{prefix}.attn_output.weight", [EMBD, EMBD], layer["attn_output"]),
            (f"{prefix}.ffn_norm.weight", [EMBD], layer["ffn_norm"]),
            (f"{prefix}.ffn_gate_inp.weight", [EMBD, EXPERTS], router),
            (f"{prefix}.ffn_gate_exps.weight", [EMBD, FF, EXPERTS], gate),
            (f"{prefix}.ffn_up_exps.weight", [EMBD, FF, EXPERTS], up),
            (f"{prefix}.ffn_down_exps.weight", [FF, EMBD, EXPERTS], down),
        ]
    return tensors


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    model = weights(np.random.default_rng(SEED))

    write_gguf(FIXTURES / "tiny_dense.gguf", base_metadata(False), dense_tensors(model))
    # Four identical experts routed two at a time: the softmax weights have to sum
    # to one or this file answers twice as strongly as the dense one.
    write_gguf(
        FIXTURES / "tiny_moe.gguf",
        base_metadata(True),
        moe_tensors(model),
    )
    # The same experts, one at a time. Comparing the two files isolates the
    # routing weights from everything else.
    write_gguf(
        FIXTURES / "tiny_moe_one.gguf",
        base_metadata(True, used=1),
        moe_tensors(model),
    )
    for name in ("tiny_dense", "tiny_moe", "tiny_moe_one"):
        size = (FIXTURES / f"{name}.gguf").stat().st_size
        print(f"{name}.gguf: {size} bytes")


if __name__ == "__main__":
    main()
