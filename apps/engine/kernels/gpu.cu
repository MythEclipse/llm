// Q8_0 kernels for the CUDA backend.
//
// One block owns one output row: it dequantizes that row into shared memory
// once, then every warp walks a different token over the shared copy. That is
// what keeps the weight read at one pass per launch for both decode (n = 1) and
// prompt prefill (n = chunk), which is the whole point of the CPU kernel too.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define Q8_BLOCK_ELEMS 32
#define Q8_BLOCK_BYTES 34
#define WARPS_PER_BLOCK 8

// Sum across the matmul block (2D: 32 lanes x 8 warps), result in thread 0.
__device__ __forceinline__ float warp_block_sum(float value, float* scratch) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    for (int step = 16; step > 0; step >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, step);
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = (threadIdx.x + threadIdx.y == 0) ? 0.0f : 0.0f;
    if (warp == 0) {
        total = (lane < WARPS_PER_BLOCK) ? scratch[lane] : 0.0f;
        for (int step = 4; step > 0; step >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, step);
        }
    }
    __syncthreads();
    return total;
}

// Sign-extend byte `index` of `word` to a float.
__device__ __forceinline__ float q8_element(unsigned int word, int index) {
    return (float)((int)(word << (24 - index * 8)) >> 24);
}

// y[t * out + o] = sum_k dequant(w[o][k]) * x[t * k + k]
//
// The row arrives repacked at upload time: `scales` holds k/32 f32 per row and
// `quants` holds k int8 per row, both 32-byte aligned. The 34-byte GGUF block
// layout cannot be loaded wider than one byte per element without straddling a
// block boundary, which is why decode sat at half the memory bandwidth.
//
//   n < warps   one dot per row, eight elements per lane via a 64-bit load and
//               two 128-bit activation loads, block-wide reduction.
//   n >= warps  one warp per token over the shared row, so the weight row is
//               read once and reused by every token in the chunk.
extern "C" __global__ void q8_matmul(const float* __restrict__ scales,
                                    const signed char* __restrict__ quants,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int k, int out, int n) {
    extern __shared__ float row[];
    __shared__ float reduction[WARPS_PER_BLOCK];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * blockDim.x + lane;
    const int o = blockIdx.x;
    const float* row_scales = scales + (size_t)o * (k / Q8_BLOCK_ELEMS);
    const signed char* row_quants = quants + (size_t)o * k;

    if (n < WARPS_PER_BLOCK) {
        for (int token = 0; token < n; ++token) {
            const float* activation = x + (size_t)token * k;
            float sum = 0.0f;
            for (int base = tid * 8; base < k; base += threads * 8) {
                const uint2 raw = *reinterpret_cast<const uint2*>(row_quants + base);
                const float4 low = *reinterpret_cast<const float4*>(activation + base);
                const float4 high = *reinterpret_cast<const float4*>(activation + base + 4);
                const float dot = q8_element(raw.x, 0) * low.x + q8_element(raw.x, 1) * low.y +
                                  q8_element(raw.x, 2) * low.z + q8_element(raw.x, 3) * low.w +
                                  q8_element(raw.y, 0) * high.x + q8_element(raw.y, 1) * high.y +
                                  q8_element(raw.y, 2) * high.z + q8_element(raw.y, 3) * high.w;
                sum += row_scales[base >> 5] * dot;
            }
            sum = warp_block_sum(sum, reduction);
            if (tid == 0) {
                y[(size_t)token * out + o] = sum;
            }
        }
        return;
    }

    for (int i = tid; i < k; i += threads) {
        row[i] = row_scales[i >> 5] * (float)row_quants[i];
    }
    __syncthreads();

    // Eight tokens per warp, eight independent accumulators: one shared read of
    // the weight row feeds eight fused multiply-adds, and the chains do not
    // depend on each other, so the pipeline stays full.
    const int group_stride = blockDim.y * 8;
    for (int group = warp * 8; group < n; group += group_stride) {
        float sums[8];
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            sums[slot] = 0.0f;
        }
        for (int i = lane; i < k; i += 32) {
            const float weight = row[i];
            #pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                if (group + slot < n) {
                    sums[slot] += weight * x[(size_t)(group + slot) * k + i];
                }
            }
        }
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            if (group + slot < n) {
                float total = sums[slot];
                for (int offset = 16; offset > 0; offset >>= 1) {
                    total += __shfl_down_sync(0xffffffffu, total, offset);
                }
                if (lane == 0) {
                    y[(size_t)(group + slot) * out + o] = total;
                }
            }
        }
    }
}
extern "C" __global__ void rmsnorm_rows(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       const float* __restrict__ weight,
                                       int n, int rows, float eps) {
    __shared__ float partial[256];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* source = src + (size_t)row * n;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += source[i] * source[i];
    }
    partial[tid] = sum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (tid < width) {
            partial[tid] += partial[tid + width];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    float* target = dst + (size_t)row * n;
    for (int i = tid; i < n; i += blockDim.x) {
        target[i] = source[i] * scale * weight[i];
    }
}

// act[i] = silu(gate[i]) * up[i]
extern "C" __global__ void silu_mul(float* __restrict__ act,
                                    const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        act[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// y[i] += x[i]
extern "C" __global__ void residual_add(float* __restrict__ y,
                                        const float* __restrict__ x,
                                        int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        y[i] += x[i];
    }
}

// Rotate consecutive (2i, 2i+1) pairs inside every head, llama.cpp's
// LLAMA_ROPE_TYPE_NORM for the llama architecture. Token `t` of the batch sits
// at position `start + t`.
extern "C" __global__ void rope(float* __restrict__ x,
                                int tokens, int heads, int head_dim, int half,
                                int n_rot, float base, int start) {
    const int total = tokens * heads * half;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += gridDim.x * blockDim.x) {
        const int token = index / (heads * half);
        const int rest = index % (heads * half);
        const int head = rest / half;
        const int pair = rest % half;
        const int position = start + token;
        // Same frequency as the CPU table: base^(-2 * pair / n_rot).
        const float angle = (float)position * powf(base, -2.0f * (float)pair / (float)n_rot);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        float* base = x + ((size_t)token * heads + head) * head_dim + 2 * pair;
        const float a = *base;
        const float b = *(base + 1);
        *base = a * cosine - b * sine;
        *(base + 1) = a * sine + b * cosine;
    }
}

// Copy this layer's keys and values for `tokens` new positions into the cache.
// Layout per position: [kv_head][head_dim].
extern "C" __global__ void kv_append(float* __restrict__ keys,
                                     float* __restrict__ values,
                                     const float* __restrict__ source_keys,
                                     const float* __restrict__ source_values,
                                     int tokens, int kv_dim, int start) {
    const int total = tokens * kv_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int token = i / kv_dim;
        const int offset = i % kv_dim;
        const size_t target = (size_t)(start + token) * kv_dim + offset;
        keys[target] = source_keys[i];
        values[target] = source_values[i];
    }
}

// Causal grouped-query attention for `tokens` query rows starting at `start`.
//
// One block per (head, query token), 4 warps: the positions of the KV cache are
// split across the warps instead of walked serially by one of them, which is
// what the profile showed dominating decode past a few hundred positions.
// `scores` holds n_head * tokens rows of `scores_stride` floats so batched rows
// never share a softmax buffer.
#define ATTENTION_WARPS 4
#define MAX_HEAD_DIM 128

__device__ __forceinline__ float warp_max(float value) {
    for (int step = 16; step > 0; step >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, step));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int step = 16; step > 0; step >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, step);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float result = scratch[0];
    for (int index = 1; index < ATTENTION_WARPS; ++index) {
        result = fmaxf(result, scratch[index]);
    }
    return result;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float result = 0.0f;
    for (int index = 0; index < ATTENTION_WARPS; ++index) {
        result += scratch[index];
    }
    return result;
}

// Decode attention, split across blocks: one block owns one head and one slice
// of the visible positions, so a 16-head model still fills 20 SMs at long
// context instead of launching 16 blocks that each walk the whole KV serially.
//
// Each block emits a normalised partial sum plus its own max and sum, and the
// combine kernel rescales them by exp(max_i - max). That is the same arithmetic
// as a single-pass softmax, only summed in a different order.
#define MAX_SPLIT_POSITIONS 256

extern "C" __global__ void attention_split(const float* __restrict__ query,
                                           const float* __restrict__ keys,
                                           const float* __restrict__ values,
                                           float* __restrict__ partial_acc,
                                           float* __restrict__ partial_max,
                                           float* __restrict__ partial_sum,
                                           int n_head, int n_head_kv, int head_dim,
                                           int visible, int splits, float scale) {
    extern __shared__ float weights[];
    __shared__ float scratch[ATTENTION_WARPS];
    __shared__ float partial[ATTENTION_WARPS][MAX_HEAD_DIM];

    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int elements = head_dim / 32;

    const int per_split = (visible + splits - 1) / splits;
    const int begin = split * per_split;
    const int end = min(begin + per_split, visible);
    const float* query_head = query + (size_t)head * head_dim;

    float local_max = -INFINITY;
    for (int offset = begin + warp; offset < end; offset += ATTENTION_WARPS) {
        const float* key = keys + (size_t)offset * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            dot += query_head[i] * key[i];
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            const float score = dot * scale;
            weights[offset - begin] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    __syncthreads();
    if (end > begin) {
        local_max = block_max(local_max, scratch);
    }
    __syncthreads();

    float accumulator[4];
    #pragma unroll
    for (int element = 0; element < 4; ++element) {
        accumulator[element] = 0.0f;
    }
    float total = 0.0f;
    for (int offset = begin + warp; offset < end; offset += ATTENTION_WARPS) {
        float weight = 0.0f;
        if (lane == 0) {
            weight = expf(weights[offset - begin] - local_max);
            weights[offset - begin] = weight;
            total += weight;
        }
        weight = __shfl_sync(0xffffffffu, weight, 0);
        const float* value = values + (size_t)offset * kv_dim + kv_head * head_dim;
        for (int element = 0; element < elements; ++element) {
            accumulator[element] += weight * value[lane + element * 32];
        }
    }
    for (int element = 0; element < elements; ++element) {
        partial[warp][lane + element * 32] = accumulator[element];
    }
    total = block_sum(total, scratch);
    __syncthreads();

    const size_t base = ((size_t)head * splits + split) * head_dim;
    if (threadIdx.x < head_dim) {
        float sum = 0.0f;
        for (int source = 0; source < ATTENTION_WARPS; ++source) {
            sum += partial[source][threadIdx.x];
        }
        partial_acc[base + threadIdx.x] = sum;
    }
    if (threadIdx.x == 0) {
        partial_max[(size_t)head * splits + split] = local_max;
        partial_sum[(size_t)head * splits + split] = total;
    }
}

extern "C" __global__ void attention_combine(const float* __restrict__ partial_acc,
                                             const float* __restrict__ partial_max,
                                             const float* __restrict__ partial_sum,
                                             float* __restrict__ out,
                                             int head_dim, int splits) {
    const int head = blockIdx.x;
    if (threadIdx.x >= head_dim) {
        return;
    }
    float global_max = -INFINITY;
    for (int split = 0; split < splits; ++split) {
        global_max = fmaxf(global_max, partial_max[(size_t)head * splits + split]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const size_t index = (size_t)head * splits + split;
        // The partial sums are unnormalised, so the split's weight is its own
        // sum rescaled against the global max: numerator and denominator both
        // scale by it, and only the denominator carries the sum.
        const float factor = expf(partial_max[index] - global_max);
        denominator += partial_sum[index] * factor;
        numerator += partial_acc[index * head_dim + threadIdx.x] * factor;
    }
    out[(size_t)head * head_dim + threadIdx.x] = numerator / denominator;
}

// Quantize activations to Q8_0, one warp per 32-element block. Doing this once
// per matmul is what lets the prompt kernel multiply int8 by int8 on the tensor
// cores instead of widening every weight to f32 first.
extern "C" __global__ void quantize_rows(const float* __restrict__ x,
                                        signed char* __restrict__ quants,
                                        float* __restrict__ scales,
                                        int k) {
    const int row = blockIdx.y;
    const int block = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t offset = (size_t)row * k + block * 32 + lane;

    const float value = x[offset];
    float peak = fabsf(value);
    #pragma unroll
    for (int step = 16; step > 0; step >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, step));
    }
    const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    if (lane == 0) {
        scales[(size_t)row * (k >> 5) + block] = peak / 127.0f;
    }
    quants[offset] = (signed char)roundf(value * inverse);
}

#define MMA_ROWS 16
#define MMA_WARPS 8
#define MMA_BLOCK_ROWS 128
#define MMA_ROWS_PER_WARP 32
#define MMA_MINI_TILES (MMA_ROWS_PER_WARP / MMA_ROWS)
#define MMA_BLOCK_TOKENS 64
#define MMA_TOKEN_GROUPS 4
#define MMA_TOKENS_PER_GROUP 16
#define MMA_CHUNK 256
#define MMA_CHUNK_BLOCKS (MMA_CHUNK / 32)
#define MMA_PAD 16

__device__ __forceinline__ void mma_s8(int& d0, int& d1, int& d2, int& d3,
                                      int a0, int a1, int a2, int a3,
                                      int b0, int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Loads a 16x32 int8 A fragment the way ggml does it: one ldmatrix.x4 per
// fragment instead of four scalar shared loads, with lanes 0-15 feeding the rows
// at byte 0 and lanes 16-31 the rows at byte 16.
__device__ __forceinline__ void load_a_fragment(int* fragment, const signed char* base,
                                               int stride, int lane) {
    // ldmatrix takes a generic address, not the 32-bit shared offset that cp.async
    // wants; handing it a converted offset makes ptxas read global memory at that
    // address instead, which faults.
    const signed char* address = base + (lane % MMA_ROWS) * stride + (lane / MMA_ROWS) * 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "l"(address));
}

// Prompt matmul on the int8 tensor cores, following the geometry ggml picks for
// Q8_0 on Ampere (mmq-config-ampere.cuh, mmq-vec-dot.cuh):
//
//   * one block owns 128 output rows and up to 64 tokens, with 8 warps and one
//     block resident per SM;
//   * a warp covers 32 rows (two m16 minitiles) and 8 tokens per mma, so a single
//     B fragment feeds two mma and the activation read is halved;
//   * both operands sit in shared memory — A is read with ldmatrix.x4, B with the
//     two-word-per-lane layout — and the per-block scales collapse into one fma
//     per accumulator element: sum += C * dA * dB.
extern "C" __global__ void q8_matmul_mma(const float* __restrict__ weight_scales,
                                        const signed char* __restrict__ weight_quants,
                                        const float* __restrict__ act_scales,
                                        const signed char* __restrict__ act_quants,
                                        float* __restrict__ y,
                                        int k, int out, int n) {
    // ldmatrix needs sixteen-byte-aligned shared addresses, and a plain char array
    // is only guaranteed one-byte alignment.
    // One dynamic allocation instead of four static arrays: a 256-element k chunk
    // needs 57 KB, and a kernel may only take 48 KB statically.
    extern __shared__ __align__(16) signed char smem[];
    constexpr int ROW_STRIDE = MMA_CHUNK + MMA_PAD;
    signed char* a_tile = smem;
    signed char* b_tile = a_tile + MMA_BLOCK_ROWS * ROW_STRIDE;
    float* a_scales = reinterpret_cast<float*>(b_tile + MMA_BLOCK_TOKENS * ROW_STRIDE);
    float* b_scales = a_scales + MMA_BLOCK_ROWS * MMA_CHUNK_BLOCKS;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * blockDim.x + lane;
    const int k_blocks = k >> 5;
    const int row_base = blockIdx.x * MMA_BLOCK_ROWS;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    const int row_group = (warp / 2) * MMA_ROWS_PER_WARP;
    const int token_offset = (warp % 2) * 8;

    float sums[MMA_TOKEN_GROUPS][MMA_MINI_TILES][4];
    #pragma unroll
    for (int j = 0; j < MMA_TOKEN_GROUPS; ++j) {
        #pragma unroll
        for (int t = 0; t < MMA_MINI_TILES; ++t) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                sums[j][t][e] = 0.0f;
            }
        }
    }

    for (int chunk = 0; chunk < k; chunk += MMA_CHUNK) {
        for (int index = tid * 16; index < MMA_BLOCK_ROWS * MMA_CHUNK; index += threads * 16) {
            const int row = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            const int position = chunk + element;
            const int target = row_base + row;
            int4* destination = reinterpret_cast<int4*>(a_tile + row * ROW_STRIDE + element);
            if (position + 16 <= k && target < out) {
                *destination =
                    *reinterpret_cast<const int4*>(weight_quants + (size_t)target * k + position);
            } else {
                *destination = make_int4(0, 0, 0, 0);
            }
        }
        for (int index = tid * 16; index < MMA_BLOCK_TOKENS * MMA_CHUNK; index += threads * 16) {
            const int token = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            const int position = chunk + element;
            int4* destination = reinterpret_cast<int4*>(b_tile + token * ROW_STRIDE + element);
            if (position + 16 <= k && token < n) {
                *destination =
                    *reinterpret_cast<const int4*>(act_quants + (size_t)token * k + position);
            } else {
                *destination = make_int4(0, 0, 0, 0);
            }
        }
        for (int index = tid; index < MMA_BLOCK_ROWS * MMA_CHUNK_BLOCKS; index += threads) {
            const int row = index / MMA_CHUNK_BLOCKS;
            const int block = index % MMA_CHUNK_BLOCKS;
            const int position = chunk + block * 32;
            const int target = row_base + row;
            a_scales[row * MMA_CHUNK_BLOCKS + block] = (position < k && target < out)
                ? weight_scales[(size_t)target * k_blocks + (position >> 5)]
                : 0.0f;
        }
        for (int index = tid; index < MMA_BLOCK_TOKENS * MMA_CHUNK_BLOCKS; index += threads) {
            const int token = index / MMA_CHUNK_BLOCKS;
            const int block = index % MMA_CHUNK_BLOCKS;
            const int position = chunk + block * 32;
            b_scales[token * MMA_CHUNK_BLOCKS + block] = (position < k && token < n)
                ? act_scales[(size_t)token * k_blocks + (position >> 5)]
                : 0.0f;
        }
        __syncthreads();

        int fragments[MMA_MINI_TILES][MMA_CHUNK_BLOCKS][4];
        #pragma unroll
        for (int t = 0; t < MMA_MINI_TILES; ++t) {
            #pragma unroll
            for (int block = 0; block < MMA_CHUNK_BLOCKS; ++block) {
                load_a_fragment(fragments[t][block],
                                a_tile + (row_group + t * MMA_ROWS) * ROW_STRIDE + block * 32,
                                ROW_STRIDE, lane);
            }
        }

        #pragma unroll
        for (int j = 0; j < MMA_TOKEN_GROUPS; ++j) {
            const int first_token = j * MMA_TOKENS_PER_GROUP + token_offset;
            #pragma unroll
            for (int block = 0; block < MMA_CHUNK_BLOCKS; ++block) {
                const int fragment_token = first_token + group;
                const int* activations = reinterpret_cast<const int*>(
                    b_tile + fragment_token * ROW_STRIDE + block * 32 + lane4 * 4);
                const int b0 = activations[0];
                const int b1 = activations[4];
                const int column = first_token + lane4 * 2;
                const float act_low = b_scales[column * MMA_CHUNK_BLOCKS + block];
                const float act_high = b_scales[(column + 1) * MMA_CHUNK_BLOCKS + block];

                #pragma unroll
                for (int t = 0; t < MMA_MINI_TILES; ++t) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_s8(d0, d1, d2, d3, fragments[t][block][0], fragments[t][block][1],
                           fragments[t][block][2], fragments[t][block][3], b0, b1);
                    const int row = row_group + t * MMA_ROWS;
                    const float weight_low = a_scales[(row + group) * MMA_CHUNK_BLOCKS + block];
                    const float weight_high =
                        a_scales[(row + group + 8) * MMA_CHUNK_BLOCKS + block];
                    sums[j][t][0] += (float)d0 * (weight_low * act_low);
                    sums[j][t][1] += (float)d1 * (weight_low * act_high);
                    sums[j][t][2] += (float)d2 * (weight_high * act_low);
                    sums[j][t][3] += (float)d3 * (weight_high * act_high);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < MMA_TOKEN_GROUPS; ++j) {
        const int first_token = j * MMA_TOKENS_PER_GROUP + token_offset;
        const int column_low = first_token + lane4 * 2;
        const int column_high = column_low + 1;
        #pragma unroll
        for (int t = 0; t < MMA_MINI_TILES; ++t) {
            const int row = row_base + row_group + t * MMA_ROWS + group;
            if (column_low < n && row < out) {
                y[(size_t)column_low * out + row] = sums[j][t][0];
            }
            if (column_high < n && row < out) {
                y[(size_t)column_high * out + row] = sums[j][t][1];
            }
            if (column_low < n && row + 8 < out) {
                y[(size_t)column_low * out + row + 8] = sums[j][t][2];
            }
            if (column_high < n && row + 8 < out) {
                y[(size_t)column_high * out + row + 8] = sums[j][t][3];
            }
        }
    }
}

// Prefill attention, tiled the way ggml's fattn-tile does it: one block per head
// and per sixteen query positions, keys and values staged in shared memory
// thirty-two positions at a time, and the softmax kept online so nothing of size
// `visible` is ever materialised. The serial kernel it replaces launched one block
// per (head, token) and wrote every score row through global memory, which made a
// 64-token prompt at context 1024 spend about half its prefill time there.
//
// Keys are staged transposed, dim-major, so a lane reads k_tile[dim][lane] and the
// whole warp touches thirty-two different banks; keeping them row-major instead
// gives every lane the same bank and serialises the read.
#define FA_QUERY_TILE 16
#define FA_KEY_TILE 32
#define FA_WARPS 4
#define FA_QUERIES_PER_WARP (FA_QUERY_TILE / FA_WARPS)
#define FA_MAX_HEAD_DIM 128
#define FA_DIMS_PER_LANE 4

extern "C" __global__ void attention_prefill(const float* __restrict__ query,
                                            const float* __restrict__ keys,
                                            const float* __restrict__ values,
                                            float* __restrict__ out,
                                            int n_head, int n_head_kv, int head_dim,
                                            int start, int tokens, float scale) {
    __shared__ float q_tile[FA_QUERY_TILE][FA_MAX_HEAD_DIM];
    __shared__ float k_tile[FA_MAX_HEAD_DIM][FA_KEY_TILE];
    __shared__ float v_tile[FA_KEY_TILE][FA_MAX_HEAD_DIM];
    __shared__ float weight[FA_QUERY_TILE][FA_KEY_TILE];

    const int head = blockIdx.x;
    const int tile = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int threads = blockDim.x;
    const int first_query = tile * FA_QUERY_TILE;
    const int queries = min(FA_QUERY_TILE, tokens - first_query);
    if (queries <= 0) {
        return;
    }

    for (int index = threadIdx.x; index < queries * head_dim; index += threads) {
        const int row = index / head_dim;
        const int dim = index % head_dim;
        q_tile[row][dim] = query[((size_t)(first_query + row) * n_head + head) * head_dim + dim];
    }

    float acc[FA_QUERIES_PER_WARP][FA_DIMS_PER_LANE];
    float running_max[FA_QUERIES_PER_WARP];
    float running_sum[FA_QUERIES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < FA_QUERIES_PER_WARP; ++i) {
        running_max[i] = -INFINITY;
        running_sum[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < FA_DIMS_PER_LANE; ++d) {
            acc[i][d] = 0.0f;
        }
    }
    __syncthreads();

    const int max_visible = start + min(first_query + queries, tokens);
    for (int base = 0; base < max_visible; base += FA_KEY_TILE) {
        const int keys_here = min(FA_KEY_TILE, max_visible - base);
        for (int index = threadIdx.x; index < keys_here * head_dim; index += threads) {
            const int row = index / head_dim;
            const int dim = index % head_dim;
            const size_t source = (size_t)(base + row) * kv_dim + kv_head * head_dim + dim;
            k_tile[dim][row] = keys[source];
            v_tile[row][dim] = values[source];
        }
        __syncthreads();

        // One lane per key: the lane walks the whole head dimension, so no warp
        // reduction is needed to finish a score.
        #pragma unroll
        for (int i = 0; i < FA_QUERIES_PER_WARP; ++i) {
            const int query_index = warp * FA_QUERIES_PER_WARP + i;
            if (query_index >= queries) {
                continue;
            }
            const int visible = start + first_query + query_index + 1;
            float score = -INFINITY;
            if (base + lane < visible) {
                float dot = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim) {
                    dot += q_tile[query_index][dim] * k_tile[dim][lane];
                }
                score = dot * scale;
            }
            const float tile_max = warp_max(score);
            const float new_max = fmaxf(running_max[i], tile_max);
            const float probability = score == -INFINITY ? 0.0f : expf(score - new_max);
            const float tile_sum = warp_sum(probability);
            const float correction = running_max[i] == -INFINITY
                ? 0.0f
                : expf(running_max[i] - new_max);
            running_sum[i] = running_sum[i] * correction + tile_sum;
            running_max[i] = new_max;
            #pragma unroll
            for (int d = 0; d < FA_DIMS_PER_LANE; ++d) {
                acc[i][d] *= correction;
            }
            weight[query_index][lane] = probability;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < FA_QUERIES_PER_WARP; ++i) {
            const int query_index = warp * FA_QUERIES_PER_WARP + i;
            if (query_index >= queries) {
                continue;
            }
            for (int row = 0; row < keys_here; ++row) {
                const float probability = weight[query_index][row];
                const float4 value =
                    *reinterpret_cast<const float4*>(&v_tile[row][lane * FA_DIMS_PER_LANE]);
                acc[i][0] += probability * value.x;
                acc[i][1] += probability * value.y;
                acc[i][2] += probability * value.z;
                acc[i][3] += probability * value.w;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FA_QUERIES_PER_WARP; ++i) {
        const int query_index = warp * FA_QUERIES_PER_WARP + i;
        if (query_index >= queries) {
            continue;
        }
        const float inverse = running_sum[i] > 0.0f ? 1.0f / running_sum[i] : 0.0f;
        float* destination =
            out + ((size_t)(first_query + query_index) * n_head + head) * head_dim;
        const int dim = lane * FA_DIMS_PER_LANE;
        destination[dim] = acc[i][0] * inverse;
        destination[dim + 1] = acc[i][1] * inverse;
        destination[dim + 2] = acc[i][2] * inverse;
        destination[dim + 3] = acc[i][3] * inverse;
    }
}

extern "C" __global__ void attention(const float* __restrict__ query,
                                     const float* __restrict__ keys,
                                     const float* __restrict__ values,
                                     float* __restrict__ out,
                                     float* __restrict__ scores,
                                     int n_head, int n_head_kv, int head_dim,
                                     int scores_stride, int start, int tokens, float scale) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int visible = start + token + 1;
    const int elements = head_dim / 32;

    __shared__ float scratch[ATTENTION_WARPS];
    __shared__ float partial[ATTENTION_WARPS][MAX_HEAD_DIM];

    float* row = scores + ((size_t)head * tokens + token) * scores_stride;
    const float* query_head = query + ((size_t)token * n_head + head) * head_dim;

    for (int offset = warp; offset < visible; offset += ATTENTION_WARPS) {
        const float* key = keys + (size_t)offset * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            dot += query_head[i] * key[i];
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            row[offset] = dot * scale;
        }
    }
    __syncthreads();

    float max = -INFINITY;
    for (int offset = threadIdx.x; offset < visible; offset += blockDim.x) {
        max = fmaxf(max, row[offset]);
    }
    max = block_max(max, scratch);
    __syncthreads();

    float total = 0.0f;
    for (int offset = threadIdx.x; offset < visible; offset += blockDim.x) {
        const float weight = expf(row[offset] - max);
        row[offset] = weight;
        total += weight;
    }
    total = block_sum(total, scratch);
    __syncthreads();

    const float inverse = 1.0f / total;
    for (int offset = threadIdx.x; offset < visible; offset += blockDim.x) {
        row[offset] *= inverse;
    }
    __syncthreads();

    float accumulator[4];
    for (int element = 0; element < elements; ++element) {
        accumulator[element] = 0.0f;
    }
    for (int offset = warp; offset < visible; offset += ATTENTION_WARPS) {
        const float* value = values + (size_t)offset * kv_dim + kv_head * head_dim;
        const float weight = row[offset];
        for (int element = 0; element < elements; ++element) {
            accumulator[element] += weight * value[lane + element * 32];
        }
    }
    for (int element = 0; element < elements; ++element) {
        partial[warp][lane + element * 32] = accumulator[element];
    }
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float sum = 0.0f;
        for (int source = 0; source < ATTENTION_WARPS; ++source) {
            sum += partial[source][threadIdx.x];
        }
        out[((size_t)token * n_head + head) * head_dim + threadIdx.x] = sum;
    }
}

// Embedding row dequantized straight into the activation buffer.
extern "C" __global__ void embedding_row(const float* __restrict__ scales,
                                         const signed char* __restrict__ quants,
                                         float* __restrict__ out,
                                         int row, int k) {
    const float* row_scales = scales + (size_t)row * (k / Q8_BLOCK_ELEMS);
    const signed char* row_quants = quants + (size_t)row * k;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        out[i] = row_scales[i >> 5] * (float)row_quants[i];
    }
}
