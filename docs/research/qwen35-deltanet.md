# Qwen3.5 (`qwen35`): gated delta net layers, as llama.cpp builds them

Reference: llama.cpp `d812350493` (the build that runs the 4B file at 10.5 t/s).
Files read: `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
`src/models/models.h`, `ggml/src/ggml-cpu/ops.cpp`.

This is the specification this engine implements. Every formula below was taken
from the reference, not reconstructed from a paper, because the details that
decide whether logits match are the small ones: the order of `q|k|v` in one
packed projection, whether the state is stored transposed, where the `1/sqrt(d)`
scale sits, and whether the convolution is reversed.

## Geometry of `moe-expert-coder-4b-Q8_0.gguf`

| key | value |
| --- | --- |
| `block_count` | 32 |
| `embedding_length` | 2560 |
| `feed_forward_length` | 9216 |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | 4 |
| `attention.key_length` | 256 |
| `rope.dimension_count` | 64 |
| `rope.dimension_sections` | `[11, 11, 10, 0]` |
| `rope.freq_base` | 1e7 |
| `attention.layer_norm_rms_epsilon` | 1e-6 |
| `full_attention_interval` | 4 |
| `ssm.conv_kernel` | 4 |
| `ssm.state_size` | 128 |
| `ssm.inner_size` | 4096 |
| `ssm.time_step_rank` | 32 |
| `ssm.group_count` | 16 |

Derived: `head_k_dim = state_size = 128`, `n_v_heads = time_step_rank = 32`,
`head_v_dim = inner_size / n_v_heads = 128`, `key_dim = 128*16 = 2048`,
`value_dim = 128*32 = 4096`, `conv_dim = 2*key_dim + value_dim = 8192`.

Layer kinds: recurrent when `(il + 1) % 4 != 0`, so layers 0,1,2,4,5,6,… are
delta net and 3,7,…,31 are full attention — 24 and 8.

## Block shape (differs from a llama block)

```
h   = x + attn(rmsnorm(x, attn_norm))
ffn = ffn(rmsnorm(h, post_attention_norm))
x   = h + ffn
```

There is no `ffn_norm`: `post_attention_norm` is the FFN's input norm, and the
FFN's residual is the value *before* that norm. Getting this wrong yields a
model that loads, runs, and produces garbage.

## Full attention layer (8 of them)

- `attn_q` projects to `2 * head_dim * n_head`: per head, the first `head_dim`
  is the query and the second is a gate.
- `attn_q_norm` / `attn_k_norm` are RMS norms over `head_dim` (256), applied
  after the reshape, before RoPE.
- RoPE covers `rope.dimension_count = 64` dims. The file declares
  `dimension_sections = [11, 11, 10, 0]`, i.e. MRoPE; but for text every section
  uses the same position, so the sections only choose which position feeds each
  sector. With one position, `ggml_mrope_cache_init` degenerates to plain RoPE
  over 64 dims with `theta = freq_base^(-2i/64)`. Frequencies are *not* restarted
  per section unless `indep_sects`, which this model does not set.
- `attn_output` consumes `head_dim * n_head = 4096` and returns `n_embd`.
- After attention: `out = attn * sigmoid(gate)`, then `attn_output`.

## Recurrent layer (24 of them)

1. `qkv = attn_qkv · x` → `[2*key_dim + value_dim]`, laid out `q | k | v`.
2. `z = attn_gate · x` → `[value_dim]` (the gate for the final norm).
3. `beta = sigmoid(ssm_beta · x)` → `[n_v_heads]`.
4. `alpha = ssm_alpha · x` → `[n_v_heads]`, then
   `g = ssm_a * softplus(alpha + ssm_dt)` per head. `ssm_a` in the file already
   holds `-exp(A_log)`, so `g` is negative: a decay.
5. Depthwise causal convolution over the `conv_dim` channels of `qkv`, kernel 4,
   with a `kernel-1` row state carried between tokens:
   `out[t][c] = Σ_k input[t+k][c] * w[k][c]` (forward, not reversed), then SiLU.
6. Split the result into `q [head_k_dim, 16]`, `k`, `v [head_v_dim, 32]`.
7. L2-normalise `q` and `k` per head. In the reference this is
   `rms_norm(x, eps/n) * 1/sqrt(n)`, which equals `x / sqrt(Σx² + eps)`.
8. Repeat the 16 k-heads to 32 so they line up with the v-heads.
9. Delta rule, per v-head, sequential over tokens. State `S` is `128×128` per
   head, stored **transposed** (`M[j][i] = S[i][j]`) so rows are contiguous:

   ```
   M     *= exp(g)                       // scalar per head, decay
   delta[j] = (v[j] - Σ_i M[j][i] * k[i]) * beta
   M[j][i] += k[i] * delta[j]
   out[j]   = (Σ_i M[j][i] * q[i]) * (1/sqrt(128))
   ```

10. Gated norm over `head_v_dim` with `ssm_norm`, gated by `silu(z)`.
11. `out = ssm_out · out_flat` → `n_embd`.

Note the scale sits on the attention output, and `q` is *not* pre-scaled in the
reference's CPU op. Pre-scaling `q` gives the same numbers only if it is applied
once — the reference graph builder does pre-scale `q`, the CPU op scales `out`;
both are equivalent, and this engine scales `out`.

## What this engine deliberately does not do yet

- Prefill processes recurrent layers token by token. The reference has a
  chunked-parallel form (`build_delta_net_chunking`) that is much faster on long
  prompts; correctness first, then the chunked form.
- MTP / NextN draft blocks (`n_layer_nextn`) are not loaded. The 4B file has
  none.
