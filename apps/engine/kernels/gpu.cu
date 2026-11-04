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

// Prompt path: four output rows and eight tokens per block, so one shared read
// of a weight feeds 32 multiply-adds. The k loop is chunked because four rows of
// 6144 floats would not fit in shared memory.
#define ROWS_PER_BLOCK 4
#define K_CHUNK 1024
#define TOKENS_PER_WARP 8

extern "C" __global__ void q8_matmul_batch(const float* __restrict__ scales,
                                          const signed char* __restrict__ quants,
                                          const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int k, int out, int n) {
    __shared__ float tile[ROWS_PER_BLOCK][K_CHUNK];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * blockDim.x + lane;
    const int first_row = blockIdx.x * ROWS_PER_BLOCK;

    // Every warp runs the same number of group iterations: the __syncthreads in
    // the chunk loop would deadlock if a warp with no tokens skipped out early.
    const int group_stride = blockDim.y * TOKENS_PER_WARP;
    for (int base = 0; base < n; base += group_stride) {
        const int group = base + warp * TOKENS_PER_WARP;
        float sums[ROWS_PER_BLOCK][TOKENS_PER_WARP];
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            #pragma unroll
            for (int slot = 0; slot < TOKENS_PER_WARP; ++slot) {
                sums[row][slot] = 0.0f;
            }
        }

        for (int chunk = 0; chunk < k; chunk += K_CHUNK) {
            for (int index = tid; index < ROWS_PER_BLOCK * K_CHUNK; index += threads) {
                const int row = index / K_CHUNK;
                const int element = index % K_CHUNK;
                const int position = chunk + element;
                const int target = first_row + row;
                float value = 0.0f;
                if (position < k && target < out) {
                    const float scale =
                        scales[(size_t)target * (k / Q8_BLOCK_ELEMS) + (position >> 5)];
                    value = scale * (float)quants[(size_t)target * k + position];
                }
                tile[row][element] = value;
            }
            __syncthreads();

            for (int element = lane; element < K_CHUNK; element += 32) {
                if (chunk + element < k) {
                    #pragma unroll
                    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
                        const float weight = tile[row][element];
                        #pragma unroll
                        for (int slot = 0; slot < TOKENS_PER_WARP; ++slot) {
                            if (group + slot < n) {
                                sums[row][slot] +=
                                    weight * x[(size_t)(group + slot) * k + chunk + element];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            #pragma unroll
            for (int slot = 0; slot < TOKENS_PER_WARP; ++slot) {
                if (group + slot < n && first_row + row < out) {
                    float total = sums[row][slot];
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        total += __shfl_down_sync(0xffffffffu, total, offset);
                    }
                    if (lane == 0) {
                        y[(size_t)(group + slot) * out + first_row + row] = total;
                    }
                }
            }
        }
    }
}

// dst[row * n + i] = weight[i] * src[row * n + i] / sqrt(mean(src^2) + eps)
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
