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

// Weight matrices one matmul launch can carry, concatenated along the row axis.
#define MMA_GROUP_MAX 4

// The prompt matmul reads a lane's B fragment as two 32-bit words sixteen bytes
// apart in the natural layout, which is two shared loads per mma. Storing each
// thirty-two-element block as (0-3, 16-19, 4-7, 20-23, 8-11, 24-27, 12-15,
// 28-31) puts them next to each other instead, so one 64-bit load serves both.
// The permutation is applied here, at quantisation, so the staging copy stays a
// plain sixteen-byte copy. Only the prompt path sees this layout: decode reads
// raw f32 activations.
__device__ __forceinline__ int mma_activation_slot(int group) {
    const int order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    return order[group];
}

// Sign-extend byte `index` of `word` to a float.
__device__ __forceinline__ float q8_element(unsigned int word, int index) {
    return (float)((int)(word << (24 - index * 8)) >> 24);
}

__device__ __forceinline__ float q8_dot8(uint2 raw, float4 low, float4 high) {
    float dot = q8_element(raw.x, 0) * low.x + q8_element(raw.x, 1) * low.y +
                q8_element(raw.x, 2) * low.z + q8_element(raw.x, 3) * low.w;
    dot += q8_element(raw.y, 0) * high.x + q8_element(raw.y, 1) * high.y +
           q8_element(raw.y, 2) * high.z + q8_element(raw.y, 3) * high.w;
    return dot;
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
extern "C" __global__ void q8_matmul(const __half* __restrict__ scales,
                                    const signed char* __restrict__ quants,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const float* bias,
                                    int use_bias,
                                    int k, int out, int n) {
    extern __shared__ float row[];
    __shared__ float reduction[WARPS_PER_BLOCK];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * blockDim.x + lane;
    const int o = blockIdx.x;
    const __half* row_scales = scales + (size_t)o * (k / Q8_BLOCK_ELEMS);
    const signed char* row_quants = quants + (size_t)o * k;

    if (n < WARPS_PER_BLOCK) {
        for (int token = 0; token < n; ++token) {
            const float* activation = x + (size_t)token * k;
            float sum = 0.0f;
            #pragma unroll 2
            for (int base = tid * 8; base < k; base += threads * 8) {
                const uint2 raw = *reinterpret_cast<const uint2*>(row_quants + base);
                const float4 low = *reinterpret_cast<const float4*>(activation + base);
                const float4 high = *reinterpret_cast<const float4*>(activation + base + 4);
                const float dot = q8_element(raw.x, 0) * low.x + q8_element(raw.x, 1) * low.y +
                                  q8_element(raw.x, 2) * low.z + q8_element(raw.x, 3) * low.w +
                                  q8_element(raw.y, 0) * high.x + q8_element(raw.y, 1) * high.y +
                                  q8_element(raw.y, 2) * high.z + q8_element(raw.y, 3) * high.w;
                sum += __half2float(row_scales[base >> 5]) * dot;
            }
            sum = warp_block_sum(sum, reduction);
            if (tid == 0) {
                const size_t index = (size_t)token * out + o;
                y[index] = sum + (use_bias ? bias[index] : 0.0f);
            }
        }

        return;
    }

    for (int i = tid; i < k; i += threads) {
        row[i] = __half2float(row_scales[i >> 5]) * (float)row_quants[i];
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
                    const size_t index = (size_t)(group + slot) * out + o;
                    y[index] = total + (use_bias ? bias[index] : 0.0f);
                }
            }
        }
    }
}

__device__ __forceinline__ float warp_block_sum4(float value, float* scratch) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    for (int step = 16; step > 0; step >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, step);
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = (warp == 0 && lane < 4) ? scratch[lane] : 0.0f;
    if (warp == 0) {
        for (int step = 2; step > 0; step >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, step);
        }
    }
    return total;
}

// Rows one decode block owns: one warp each, so the activation vector is read
// once per four rows instead of once per row, and the reduction never leaves the
// warp (no shared memory, no __syncthreads).
#define DECODE_ROWS_PER_BLOCK 4

// Up to four matrices that read the same activation vector travel in one launch,
// their row spaces concatenated: q, k and v are three launches of a few hundred
// rows each, and a launch that small cannot keep the memory system busy on its
// own. The group also removes two launches from every layer of every token.
extern "C" __global__ void q8_matmul_decode(const __half* __restrict__ scales0,
                                            const signed char* __restrict__ quants0,
                                            const __half* __restrict__ scales1,
                                            const signed char* __restrict__ quants1,
                                            const __half* __restrict__ scales2,
                                            const signed char* __restrict__ quants2,
                                            const __half* __restrict__ scales3,
                                            const signed char* __restrict__ quants3,
                                            float* __restrict__ y0,
                                            float* __restrict__ y1,
                                            float* __restrict__ y2,
                                            float* __restrict__ y3,
                                            int out0, int out1, int out2, int out3, int count,
                                            const float* __restrict__ x,
                                            const float* __restrict__ bias,
                                            int use_bias,
                                            int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int flat = blockIdx.x * DECODE_ROWS_PER_BLOCK;
    const int outs[MMA_GROUP_MAX] = {out0, out1, out2, out3};
    int matrix = 0;
    int row = flat;
    for (int candidate = 0, first = 0; candidate < count; ++candidate) {
        if (flat < first + outs[candidate]) {
            matrix = candidate;
            row = flat - first;
            break;
        }
        first += outs[candidate];
    }
    const __half* scales = scales0;
    const signed char* quants = quants0;
    float* y = y0;
    if (matrix == 1) {
        scales = scales1;
        quants = quants1;
        y = y1;
    } else if (matrix == 2) {
        scales = scales2;
        quants = quants2;
        y = y2;
    } else if (matrix == 3) {
        scales = scales3;
        quants = quants3;
        y = y3;
    }
    const int out = outs[matrix];
    const int o = row + warp;
    if (o >= out) {
        return;
    }
    const __half* row_scales = scales + (size_t)o * (k / Q8_BLOCK_ELEMS);
    const signed char* row_quants = quants + (size_t)o * k;
    float sum = 0.0f;
    // One warp per row: four weight loads in flight per lane for k = 2048, and
    // the activation vector comes from the read-only path so four blocks sharing
    // it stay in L1 while the weight stream passes through.
    #pragma unroll 4
    for (int base = lane * 16; base < k; base += 32 * 16) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row_quants + base);
        const float4 v0 = __ldg(reinterpret_cast<const float4*>(x + base));
        const float4 v1 = __ldg(reinterpret_cast<const float4*>(x + base + 4));
        const float4 v2 = __ldg(reinterpret_cast<const float4*>(x + base + 8));
        const float4 v3 = __ldg(reinterpret_cast<const float4*>(x + base + 12));
        sum += __half2float(row_scales[base >> 5]) *
            (q8_dot8(make_uint2(raw.x, raw.y), v0, v1) +
             q8_dot8(make_uint2(raw.z, raw.w), v2, v3));
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, step);
    }
    if (lane == 0) {
        y[o] = sum + (use_bias ? bias[o] : 0.0f);
    }
}

// Decode calls this once per layer for a single row, so its cost is latency, not
// bandwidth: the sum is reduced inside each warp and across warps with one
// barrier, instead of a shared-memory tree that needs eight.
extern "C" __global__ void rmsnorm_rows(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       const float* __restrict__ weight,
                                       int n, int rows, float eps) {
    __shared__ float partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* source = src + (size_t)row * n;
    float sum = 0.0f;
    // Four elements per load: a single row is 8 KB, so the cost here is the
    // number of round trips, not bandwidth.
    const int vectors = n / 4;
    for (int i = tid; i < vectors; i += blockDim.x) {
        const float4 value = *reinterpret_cast<const float4*>(source + i * 4);
        sum += value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, step);
    }
    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();
    float total = (tid < warps) ? partial[tid] : 0.0f;
    if (warp == 0) {
        for (int step = 16; step > 0; step >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, step);
        }
        if (lane == 0) {
            partial[0] = total;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    float* target = dst + (size_t)row * n;
    for (int i = tid; i < vectors; i += blockDim.x) {
        const float4 value = *reinterpret_cast<const float4*>(source + i * 4);
        const float4 gain = *reinterpret_cast<const float4*>(weight + i * 4);
        *reinterpret_cast<float4*>(target + i * 4) = make_float4(
            value.x * scale * gain.x,
            value.y * scale * gain.y,
            value.z * scale * gain.z,
            value.w * scale * gain.w);
    }
}

// out *= sigmoid(gate): the gate of a gated attention block, applied to the
// attention output elementwise.
extern "C" __global__ void sigmoid_mul(float* __restrict__ act,
                                       const float* __restrict__ gate, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    act[index] *= 1.0f / (1.0f + expf(-gate[index]));
}

// Pull a query projection that packs a gate beside each query apart: per head the
// file stores `head_dim` query values then `head_dim` gate values, and the two
// halves go to separate buffers.
extern "C" __global__ void split_query_gate(const float* __restrict__ packed,
                                            float* __restrict__ query,
                                            float* __restrict__ gate,
                                            int heads, int head_dim, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int dim = index % head_dim;
    const int head = (index / head_dim) % heads;
    const int token = index / (head_dim * heads);
    const int source = (token * heads + head) * head_dim * 2;
    query[index] = packed[source + dim];
    gate[index] = packed[source + head_dim + dim];
}

// quants = quantize(rmsnorm(src)): the normalised rows go straight to int8 for
// the projections that consume them, so the f32 normalisation is never written
// out and read back.
extern "C" __global__ void rmsnorm_quantize_rows(const float* __restrict__ src,
                                                 const float* __restrict__ weight,
                                                 signed char* __restrict__ quants,
                                                 float* __restrict__ scales,
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
    const int blocks_per_row = n >> 5;
    if (tid >= blocks_per_row) {
        return;
    }
    const size_t offset = (size_t)tid * 32;
    float4 values[8];
    float peak = 0.0f;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const size_t i = offset + step * 4;
        const float4 s = *reinterpret_cast<const float4*>(source + i);
        const float4 w = *reinterpret_cast<const float4*>(weight + i);
        values[step].x = s.x * scale * w.x;
        values[step].y = s.y * scale * w.y;
        values[step].z = s.z * scale * w.z;
        values[step].w = s.w * scale * w.w;
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(values[step].x), fabsf(values[step].y)),
                                 fmaxf(fabsf(values[step].z), fabsf(values[step].w))));
    }
    const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    scales[(size_t)row * blocks_per_row + tid] = peak / 127.0f;
    signed char* quant = quants + (size_t)row * n;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        char4 quantized;
        quantized.x = (signed char)__float2int_rn(values[step].x * inverse);
        quantized.y = (signed char)__float2int_rn(values[step].y * inverse);
        quantized.z = (signed char)__float2int_rn(values[step].z * inverse);
        quantized.w = (signed char)__float2int_rn(values[step].w * inverse);
        *reinterpret_cast<char4*>(quant + offset + mma_activation_slot(step) * 4) = quantized;
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

// Reusable contiguous f32 primitives. The vector path covers the aligned
// transformer buffers while the scalar tail keeps the helpers valid for any n.
extern "C" __global__ void copy_f32(float* __restrict__ dst,
                                    const float* __restrict__ src,
                                    int n) {
    const int vector_count = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vector_count;
         i += gridDim.x * blockDim.x) {
        *reinterpret_cast<float4*>(dst + i * 4) =
            *reinterpret_cast<const float4*>(src + i * 4);
    }
    for (int i = vector_count * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void fill_f32(float* __restrict__ dst, float value, int n) {
    const float4 packed = make_float4(value, value, value, value);
    const int vector_count = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vector_count;
         i += gridDim.x * blockDim.x) {
        *reinterpret_cast<float4*>(dst + i * 4) = packed;
    }
    for (int i = vector_count * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] = value;
    }
}

// y[i] = alpha*y[i] + beta*x[i]. residual_add is this operation with both
// coefficients equal to one, and keeping it generic avoids another kernel for
// future residual, scaling, and blending paths.
extern "C" __global__ void axpby_f32(float* __restrict__ y,
                                     const float* __restrict__ x,
                                     float alpha,
                                     float beta,
                                     int n) {
    const int vector_count = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vector_count;
         i += gridDim.x * blockDim.x) {
        const float4 lhs = *reinterpret_cast<const float4*>(y + i * 4);
        const float4 rhs = *reinterpret_cast<const float4*>(x + i * 4);
        *reinterpret_cast<float4*>(y + i * 4) = make_float4(
            alpha * lhs.x + beta * rhs.x,
            alpha * lhs.y + beta * rhs.y,
            alpha * lhs.z + beta * rhs.z,
            alpha * lhs.w + beta * rhs.w);
    }
    for (int i = vector_count * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] = alpha * y[i] + beta * x[i];
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
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
        float* base = x + ((size_t)token * heads + head) * head_dim + 2 * pair;
        const float a = *base;
        const float b = *(base + 1);
        *base = a * cosine - b * sine;
        *(base + 1) = a * sine + b * cosine;
    }
}

// Copy this layer's keys and values for `tokens` new positions into the cache,
// narrowing f32 to f16 — the precision llama.cpp stores its KV cache in, so the
// cache costs half the bandwidth on every attention pass.
// Layout per position: [kv_head][head_dim].
extern "C" __global__ void kv_append(__half* __restrict__ keys,
                                     __half* __restrict__ values,
                                     const float* __restrict__ source_keys,
                                     const float* __restrict__ source_values,
                                     int tokens, int kv_dim, int start) {
    const int total = tokens * kv_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int token = i / kv_dim;
        const int offset = i % kv_dim;
        const size_t target = (size_t)(start + token) * kv_dim + offset;
        keys[target] = __float2half(source_keys[i]);
        values[target] = __float2half(source_values[i]);
    }
}

// Rotate the query, then rotate the key straight into the cache and copy the
// value beside it: one launch where the engine used three, and the rotated key is
// narrowed to f16 where it is computed instead of being written out as f32 and
// read back. The three passes touch disjoint buffers, so they share one grid.
extern "C" __global__ void rope_append(float* __restrict__ query,
                                       const float* __restrict__ key,
                                       const float* __restrict__ value,
                                       __half* __restrict__ keys,
                                       __half* __restrict__ values,
                                       int tokens, int q_heads, int kv_heads,
                                       int head_dim, int half, int n_rot,
                                       float base, const int* position_ptr, int kv_dim) {
    const int stride = gridDim.x * blockDim.x;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    // Read once per block: the engine writes this value before the launch, and a
    // captured graph replays the same launch at a new position.
    const int start = *position_ptr;

    const int query_total = tokens * q_heads * half;
    for (int index = first; index < query_total; index += stride) {
        const int token = index / (q_heads * half);
        const int rest = index % (q_heads * half);
        const int head = rest / half;
        const int pair = rest % half;
        const float angle =
            (float)(start + token) * powf(base, -2.0f * (float)pair / (float)n_rot);
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
        float* target = query + ((size_t)token * q_heads + head) * head_dim + 2 * pair;
        const float a = *target;
        const float b = *(target + 1);
        *target = a * cosine - b * sine;
        *(target + 1) = a * sine + b * cosine;
    }

    const int key_total = tokens * kv_heads * half;
    for (int index = first; index < key_total; index += stride) {
        const int token = index / (kv_heads * half);
        const int rest = index % (kv_heads * half);
        const int head = rest / half;
        const int pair = rest % half;
        const float angle =
            (float)(start + token) * powf(base, -2.0f * (float)pair / (float)n_rot);
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
        const size_t source = ((size_t)token * kv_heads + head) * head_dim + 2 * pair;
        const float a = key[source];
        const float b = key[source + 1];
        const size_t target = (size_t)(start + token) * kv_dim + head * head_dim + 2 * pair;
        keys[target] = __float2half(a * cosine - b * sine);
        keys[target + 1] = __float2half(a * sine + b * cosine);
    }

    const int value_total = tokens * kv_dim;
    for (int index = first; index < value_total; index += stride) {
        const int token = index / kv_dim;
        values[(size_t)(start + token) * kv_dim + index % kv_dim] = __float2half(value[index]);
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
// The widest head the split/serial kernels stage. qwen35 attention blocks are
// 256 wide, and both kernels index their per-warp accumulator by head_dim/32, so
// the constant bounds the register array as well as the shared partials.
#define MAX_HEAD_DIM 256

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
                                           const __half* __restrict__ keys,
                                           const __half* __restrict__ values,
                                           float* __restrict__ partial_acc,
                                           float* __restrict__ partial_max,
                                           float* __restrict__ partial_sum,
                                           int n_head, int n_head_kv, int head_dim,
                                           const int* visible_ptr, int splits, float scale) {
    extern __shared__ float weights[];
    // The visible count grows every token; a captured graph cannot bake it.
    const int visible = *visible_ptr;
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
        const __half* key = keys + (size_t)offset * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            dot += query_head[i] * __half2float(key[i]);
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

    float accumulator[MAX_HEAD_DIM / 32];
    #pragma unroll
    for (int element = 0; element < MAX_HEAD_DIM / 32; ++element) {
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
        const __half* value = values + (size_t)offset * kv_dim + kv_head * head_dim;
        for (int element = 0; element < elements; ++element) {
            accumulator[element] += weight * __half2float(value[lane + element * 32]);
        }
    }
    for (int element = 0; element < elements; ++element) {
        partial[warp][lane + element * 32] = accumulator[element];
    }
    total = block_sum(total, scratch);
    __syncthreads();

    const size_t base = ((size_t)head * splits + split) * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (int source = 0; source < ATTENTION_WARPS; ++source) {
            sum += partial[source][dim];
        }
        partial_acc[base + dim] = sum;
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
    // One block per head, but the head can be wider than the block: a 256-wide
    // head is two passes of a 128-thread block, and writing only the first half
    // leaves the rest of the output holding whatever the buffer had before.
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
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
            numerator += partial_acc[index * head_dim + dim] * factor;
        }
        out[(size_t)head * head_dim + dim] = numerator / denominator;
    }
}

// Quantize activations to Q8_0, one warp per 32-element block. Doing this once
// per matmul is what lets the prompt kernel multiply int8 by int8 on the tensor
// cores instead of widening every weight to f32 first.
extern "C" __global__ void quantize_rows(const float* __restrict__ x,
                                        signed char* __restrict__ quants,
                                        float* __restrict__ scales,
                                        int k, int total) {
    // One thread per thirty-two-element block. The warp version moved four bytes
    // per lane and needed a five-step butterfly for the block peak, which left the
    // kernel at half the bandwidth it should reach; a thread that reads its own
    // thirty-two floats pulls a hundred and twenty-eight contiguous bytes per
    // access and keeps the peak in registers.
    const int blocks_per_row = k >> 5;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The grid is rounded up to whole warps, so the last block has threads past
    // the end. Without this they quantise row `n` and write it over the scales
    // plane, which is what corrupted the first rows of a 65-token prompt.
    if (index >= total) {
        return;
    }
    const int row = index / blocks_per_row;
    const int block = index % blocks_per_row;
    const size_t offset = (size_t)row * k + block * 32;

    float4 values[8];
    float peak = 0.0f;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        values[step] = *reinterpret_cast<const float4*>(x + offset + step * 4);
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(values[step].x), fabsf(values[step].y)),
                                 fmaxf(fabsf(values[step].z), fabsf(values[step].w))));
    }
    const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    scales[(size_t)row * blocks_per_row + block] = peak / 127.0f;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        char4 quantized;
        quantized.x = (signed char)__float2int_rn(values[step].x * inverse);
        quantized.y = (signed char)__float2int_rn(values[step].y * inverse);
        quantized.z = (signed char)__float2int_rn(values[step].z * inverse);
        quantized.w = (signed char)__float2int_rn(values[step].w * inverse);
        *reinterpret_cast<char4*>(quants + offset + mma_activation_slot(step) * 4) = quantized;
    }
}

// quants = quantize(silu(gate) * up). The SwiGLU activation used to be written
// out and read back by quantize_rows for the down projection; quantising it
// where it is computed spares one full pass over n * ffn floats.
extern "C" __global__ void silu_mul_quantize_rows(const float* __restrict__ gate,
                                                  const float* __restrict__ up,
                                                  signed char* __restrict__ quants,
                                                  float* __restrict__ scales,
                                                  int k, int total) {
    const int blocks_per_row = k >> 5;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The grid is rounded up to whole warps, so the last block has threads past
    // the end. Without this they quantise row `n` and write it over the scales
    // plane, which is what corrupted the first rows of a 65-token prompt.
    if (index >= total) {
        return;
    }
    const int row = index / blocks_per_row;
    const int block = index % blocks_per_row;
    const size_t offset = (size_t)row * k + block * 32;

    float4 values[8];
    float peak = 0.0f;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const float4 g = *reinterpret_cast<const float4*>(gate + offset + step * 4);
        const float4 u = *reinterpret_cast<const float4*>(up + offset + step * 4);
        values[step].x = (g.x / (1.0f + expf(-g.x))) * u.x;
        values[step].y = (g.y / (1.0f + expf(-g.y))) * u.y;
        values[step].z = (g.z / (1.0f + expf(-g.z))) * u.z;
        values[step].w = (g.w / (1.0f + expf(-g.w))) * u.w;
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(values[step].x), fabsf(values[step].y)),
                                 fmaxf(fabsf(values[step].z), fabsf(values[step].w))));
    }
    const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    scales[(size_t)row * blocks_per_row + block] = peak / 127.0f;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        char4 quantized;
        quantized.x = (signed char)__float2int_rn(values[step].x * inverse);
        quantized.y = (signed char)__float2int_rn(values[step].y * inverse);
        quantized.z = (signed char)__float2int_rn(values[step].z * inverse);
        quantized.w = (signed char)__float2int_rn(values[step].w * inverse);
        *reinterpret_cast<char4*>(quants + offset + mma_activation_slot(step) * 4) = quantized;
    }
}

#define MMA_ROWS 16
#define MMA_WARPS 16
#define MMA_BLOCK_ROWS 256
#define MMA_ROWS_PER_WARP 16
#define MMA_MINI_TILES (MMA_ROWS_PER_WARP / MMA_ROWS)
#define MMA_BLOCK_TOKENS 128
#define MMA_TOKEN_GROUPS 16
#define MMA_TOKENS_PER_GROUP 8
#define MMA_CHUNK 128
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
// Eight warps of sixteen rows, so registers stay inside the file without spilling.
// Up to four weight matrices that read the same activations travel in one launch.
// Their row spaces are concatenated, so the tiles of q, k and v fill a wave
// together instead of one matrix's tiles filling a wave each: at 128 tokens a
// projection with sixteen tiles leaves four SMs idle, and three of those in a row
// is a fifth of the layer.
extern "C" __global__ __launch_bounds__(32 * MMA_WARPS, 1) void q8_matmul_mma(
                                        const __half* __restrict__ scales0,
                                        const signed char* __restrict__ quants0,
                                        const __half* __restrict__ scales1,
                                        const signed char* __restrict__ quants1,
                                        const __half* __restrict__ scales2,
                                        const signed char* __restrict__ quants2,
                                        const __half* __restrict__ scales3,
                                        const signed char* __restrict__ quants3,
                                        float* __restrict__ y0,
                                        float* __restrict__ y1,
                                        float* __restrict__ y2,
                                        float* __restrict__ y3,
                                        int out0, int out1, int out2, int out3, int count,
                                        const float* __restrict__ act_scales,
                                        const signed char* __restrict__ act_quants,
                                        int k, int n,
                                        int token_tiles, int full_blocks, int tail_slices,
                                        const float* __restrict__ bias, int use_bias) {
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
    // One flat grid of (row tile, token tile) pairs, token tiles varying fastest
    // so the blocks running together share one weight tile: it stays in L2 while
    // every token tile of those rows is computed.
    //
    // A launch that does not fill a whole wave would leave the last wave nearly
    // empty, and its cost is a whole tile per block however few blocks there are.
    // The first `full_blocks` blocks take one tile each; the remaining `tail`
    // tiles are cut along the token axis into `tail_slices` shorter blocks, so
    // the last wave is filled with work instead of idling. The slices write
    // disjoint columns, so no partial sums have to be combined afterwards.
    int tile = blockIdx.x;
    int slice = 0;
    // Only the tail blocks are cut short; a block that owns a whole tile keeps
    // the full token count. Applying the slice width to both was a silent
    // fifteen-sixteenths of a tile for every full block whenever a launch
    // happened to need slicing at all.
    int block_tokens = MMA_BLOCK_TOKENS;
    if (blockIdx.x >= (unsigned int)full_blocks) {
        const int index = blockIdx.x - full_blocks;
        tile = full_blocks + index / tail_slices;
        slice = index % tail_slices;
        block_tokens = MMA_BLOCK_TOKENS / tail_slices;
    }
    const int tile_row = tile / token_tiles;
    const int token_tile = tile % token_tiles;
    // Which matrix of the group this row tile belongs to, and where in it.
    const int outs[MMA_GROUP_MAX] = {out0, out1, out2, out3};
    int matrix = 0;
    int row_base = 0;
    for (int candidate = 0, first = 0; candidate < count; ++candidate) {
        const int tiles = (outs[candidate] + MMA_BLOCK_ROWS - 1) / MMA_BLOCK_ROWS;
        if (tile_row < first + tiles) {
            matrix = candidate;
            row_base = (tile_row - first) * MMA_BLOCK_ROWS;
            break;
        }
        first += tiles;
    }
    const int out = outs[matrix];
    const __half* weight_scales = scales0;
    const signed char* weight_quants = quants0;
    float* y = y0;
    if (matrix == 1) {
        weight_scales = scales1;
        weight_quants = quants1;
        y = y1;
    } else if (matrix == 2) {
        weight_scales = scales2;
        weight_quants = quants2;
        y = y2;
    } else if (matrix == 3) {
        weight_scales = scales3;
        weight_quants = quants3;
        y = y3;
    }
    const int slice_tokens = block_tokens;
    const int token_offset = slice * slice_tokens;
    const int token_base = token_tile * MMA_BLOCK_TOKENS + token_offset;
    const int tile_tokens = min(slice_tokens, n - token_base);
    const int first_group = token_offset / MMA_TOKENS_PER_GROUP;
    const int last_group = first_group + slice_tokens / MMA_TOKENS_PER_GROUP;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    // Every warp owns sixteen rows outright and walks all sixteen token groups, so
    // one weight tile staged in shared serves a hundred and twenty-eight tokens.
    const int row_group = warp * MMA_ROWS_PER_WARP;

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
        // Every load of the chunk is issued before any store: weights,
        // activations and both scale planes are then in flight together, and the
        // block stalls once per chunk instead of four times. Storing each plane as
        // it arrives is what the profiler reports as long-scoreboard stall.
        constexpr int ROW_COPIES = MMA_BLOCK_ROWS * MMA_CHUNK / (32 * MMA_WARPS * 16);
        constexpr int TOKEN_COPIES = MMA_BLOCK_TOKENS * MMA_CHUNK / (32 * MMA_WARPS * 16);
        constexpr int SCALE_PAIRS = MMA_CHUNK_BLOCKS / 2;
        constexpr int ROW_SCALES = MMA_BLOCK_ROWS * SCALE_PAIRS;
        constexpr int TOKEN_SCALES = MMA_BLOCK_TOKENS * MMA_CHUNK_BLOCKS;
        int4 weight_tile[ROW_COPIES];
        int4 act_tile[TOKEN_COPIES];
        float2 weight_scale_values[(ROW_SCALES + 32 * MMA_WARPS - 1) / (32 * MMA_WARPS)];
        float act_scale_values[(TOKEN_SCALES + 32 * MMA_WARPS - 1) / (32 * MMA_WARPS)];

        #pragma unroll
        for (int copy = 0; copy < ROW_COPIES; ++copy) {
            const int index = tid * 16 + copy * threads * 16;
            const int row = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            const int position = chunk + element;
            const int target = row_base + row;
            weight_tile[copy] = (position + 16 <= k && target < out)
                ? *reinterpret_cast<const int4*>(weight_quants + (size_t)target * k + position)
                : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int copy = 0; copy < TOKEN_COPIES; ++copy) {
            const int index = tid * 16 + copy * threads * 16;
            const int token = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            const int position = chunk + element;
            act_tile[copy] = (position + 16 <= k && token < slice_tokens && token < tile_tokens)
                ? *reinterpret_cast<const int4*>(
                      act_quants + (size_t)(token_base + token) * k + position)
                : make_int4(0, 0, 0, 0);
        }
        // One half2 load per thread pair of blocks, so the scale staging keeps
        // the same 32-bit access shape the f32 plane had.
        #pragma unroll
        for (int slot = 0; slot < ROW_SCALES / (32 * MMA_WARPS); ++slot) {
            const int index = tid + slot * 32 * MMA_WARPS;
            const int row = index / SCALE_PAIRS;
            const int block = (index % SCALE_PAIRS) * 2;
            const int position = chunk + block * 32;
            const int target = row_base + row;
            float2 values = make_float2(0.0f, 0.0f);
            if (position + 32 <= k) {
                values = __half22float2(*reinterpret_cast<const __half2*>(
                    weight_scales + (size_t)target * k_blocks + (position >> 5)));
                if (target >= out) {
                    values = make_float2(0.0f, 0.0f);
                }
            } else if (position < k && target < out) {
                values.x = __half2float(
                    weight_scales[(size_t)target * k_blocks + (position >> 5)]);
            }
            weight_scale_values[slot] = values;
        }
        #pragma unroll
        for (int slot = 0; slot < TOKEN_SCALES / (32 * MMA_WARPS); ++slot) {
            const int index = tid + slot * 32 * MMA_WARPS;
            const int token = index / MMA_CHUNK_BLOCKS;
            const int block = index % MMA_CHUNK_BLOCKS;
            const int position = chunk + block * 32;
            act_scale_values[slot] = (position < k && token < slice_tokens && token < tile_tokens)
                ? act_scales[(size_t)(token_base + token) * k_blocks + (position >> 5)]
                : 0.0f;
        }

        #pragma unroll
        for (int copy = 0; copy < ROW_COPIES; ++copy) {
            const int index = tid * 16 + copy * threads * 16;
            const int row = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            *reinterpret_cast<int4*>(a_tile + row * ROW_STRIDE + element) = weight_tile[copy];
        }
        #pragma unroll
        for (int copy = 0; copy < TOKEN_COPIES; ++copy) {
            const int index = tid * 16 + copy * threads * 16;
            const int token = index / MMA_CHUNK;
            const int element = index % MMA_CHUNK;
            *reinterpret_cast<int4*>(b_tile + token * ROW_STRIDE + element) = act_tile[copy];
        }
        #pragma unroll
        for (int slot = 0; slot < ROW_SCALES / (32 * MMA_WARPS); ++slot) {
            const int index = tid + slot * 32 * MMA_WARPS;
            const int row = index / SCALE_PAIRS;
            const int block = (index % SCALE_PAIRS) * 2;
            a_scales[row * MMA_CHUNK_BLOCKS + block] = weight_scale_values[slot].x;
            a_scales[row * MMA_CHUNK_BLOCKS + block + 1] = weight_scale_values[slot].y;
        }
        #pragma unroll
        for (int slot = 0; slot < TOKEN_SCALES / (32 * MMA_WARPS); ++slot) {
            const int index = tid + slot * 32 * MMA_WARPS;
            const int token = index / MMA_CHUNK_BLOCKS;
            const int block = index % MMA_CHUNK_BLOCKS;
            // Block-major: the two columns a lane multiplies sit next to each
            // other, so the inner loop reads them with one 64-bit load.
            b_scales[block * MMA_BLOCK_TOKENS + token] = act_scale_values[slot];
        }
        __syncthreads();

        // The rows this lane owns never change inside a chunk, so its weight
        // scales are loaded once instead of once per token group.
        float weight_low[MMA_CHUNK_BLOCKS];
        float weight_high[MMA_CHUNK_BLOCKS];
        #pragma unroll
        for (int block = 0; block < MMA_CHUNK_BLOCKS; ++block) {
            const int row = row_group + group;
            weight_low[block] = a_scales[row * MMA_CHUNK_BLOCKS + block];
            weight_high[block] = a_scales[(row + 8) * MMA_CHUNK_BLOCKS + block];
        }

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
            if (j < first_group || j >= last_group) {
                continue;
            }
            const int first_token = j * MMA_TOKENS_PER_GROUP - token_offset;
            #pragma unroll
            for (int block = 0; block < MMA_CHUNK_BLOCKS; ++block) {
                const int fragment_token = first_token + group;
                // The two words of a lane's B fragment sit next to each other
                // because the quantiser wrote them that way: one 64-bit load.
                const int2 pair = *reinterpret_cast<const int2*>(
                    b_tile + fragment_token * ROW_STRIDE + block * 32 + lane4 * 8);
                const int b0 = pair.x;
                const int b1 = pair.y;
                const int column = first_token + lane4 * 2;
                const float2 act = *reinterpret_cast<const float2*>(
                    b_scales + block * MMA_BLOCK_TOKENS + column);
                const float act_low = act.x;
                const float act_high = act.y;

                #pragma unroll
                for (int t = 0; t < MMA_MINI_TILES; ++t) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_s8(d0, d1, d2, d3, fragments[t][block][0], fragments[t][block][1],
                           fragments[t][block][2], fragments[t][block][3], b0, b1);
                    sums[j][t][0] += (float)d0 * (weight_low[block] * act_low);
                    sums[j][t][1] += (float)d1 * (weight_low[block] * act_high);
                    sums[j][t][2] += (float)d2 * (weight_high[block] * act_low);
                    sums[j][t][3] += (float)d3 * (weight_high[block] * act_high);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < MMA_TOKEN_GROUPS; ++j) {
        if (j < first_group || j >= last_group) {
            continue;
        }
        const int first_token = j * MMA_TOKENS_PER_GROUP - token_offset;
        const int column_low = first_token + lane4 * 2;
        const int column_high = column_low + 1;
        #pragma unroll
        for (int t = 0; t < MMA_MINI_TILES; ++t) {
            const int row = row_base + row_group + t * MMA_ROWS + group;
            if (column_low < tile_tokens && row < out) {
                const size_t index = (size_t)(token_base + column_low) * out + row;
                y[index] = sums[j][t][0] + (use_bias ? bias[index] : 0.0f);
            }
            if (column_high < tile_tokens && row < out) {
                const size_t index = (size_t)(token_base + column_high) * out + row;
                y[index] = sums[j][t][1] + (use_bias ? bias[index] : 0.0f);
            }
            if (column_low < tile_tokens && row + 8 < out) {
                const size_t index = (size_t)(token_base + column_low) * out + row + 8;
                y[index] = sums[j][t][2] + (use_bias ? bias[index] : 0.0f);
            }
            if (column_high < tile_tokens && row + 8 < out) {
                const size_t index = (size_t)(token_base + column_high) * out + row + 8;
                y[index] = sums[j][t][3] + (use_bias ? bias[index] : 0.0f);
            }
        }
    }
}

// Prefill attention, tiled the way ggml's fattn-tile does it: one block per head
// pair and per sixteen query positions, keys and values staged in shared memory
// thirty-two positions at a time, and the softmax kept online so nothing of size
// `visible` is ever materialised.
//
// Two heads to a block rather than four, and no probability buffer: the
// probabilities stay in registers and reach the value pass through `__shfl_sync`.
// That is what brings the block under 48 KB, so two of them fit on an SM instead
// of one — sixteen warps to hide shared-memory latency with, where four heads and
// a probability buffer left eight.
#define FA_HEADS 2
#define FA_QUERY_TILE 16
#define FA_KEY_TILE 32
#define FA_WARPS 8
#define FA_WARPS_PER_HEAD (FA_WARPS / FA_HEADS)
#define FA_QUERIES_PER_WARP (FA_QUERY_TILE / FA_WARPS_PER_HEAD)
#define FA_QUERY_GROUP 4
#define FA_MAX_HEAD_DIM 128
#define FA_DIMS_PER_LANE 4
#define FA_KEY_STRIDE (FA_MAX_HEAD_DIM)

extern "C" __global__ void attention_prefill(const float* __restrict__ query,
                                            const __half* __restrict__ keys,
                                            const __half* __restrict__ values,
                                            float* __restrict__ out,
                                            int n_head, int n_head_kv, int head_dim,
                                            int start, int tokens, float scale) {
    extern __shared__ __align__(16) signed char smem[];
    float* q_tile = reinterpret_cast<float*>(smem);
    float* k_tile = q_tile + FA_HEADS * FA_QUERY_TILE * FA_MAX_HEAD_DIM;
    float* v_tile = k_tile + FA_KEY_TILE * FA_KEY_STRIDE;

    const int group = blockIdx.x;
    // Causal masking makes the last query tile the most expensive one, by the
    // whole visible length. Blocks are scheduled in index order, so handing out
    // the heavy tiles first is what keeps the tail from deciding the makespan.
    const int tiles = (tokens + FA_QUERY_TILE - 1) / FA_QUERY_TILE;
    const int tile = tiles - 1 - blockIdx.y;
    const int heads_per_kv = n_head / n_head_kv;
    const int kv_head = (group * FA_HEADS) / heads_per_kv;
    const int kv_dim = n_head_kv * head_dim;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * blockDim.x + lane;
    const int head_slot = warp / FA_WARPS_PER_HEAD;
    const int head = group * FA_HEADS + head_slot;
    const int first_query = tile * FA_QUERY_TILE;
    const int queries = min(FA_QUERY_TILE, tokens - first_query);
    if (queries <= 0) {
        return;
    }

    for (int index = tid; index < FA_HEADS * queries * head_dim; index += threads) {
        const int slot = index / (queries * head_dim);
        const int rest = index % (queries * head_dim);
        const int row = rest / head_dim;
        const int dim = rest % head_dim;
        q_tile[(slot * FA_QUERY_TILE + row) * FA_MAX_HEAD_DIM + dim] =
            query[((size_t)(first_query + row) * n_head + group * FA_HEADS + slot) * head_dim + dim];
    }

    const int first_of_warp = (warp % FA_WARPS_PER_HEAD) * FA_QUERIES_PER_WARP;
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
        for (int index = tid; index < keys_here * head_dim; index += threads) {
            const int row = index / head_dim;
            const int dim = index % head_dim;
            const size_t source = (size_t)(base + row) * kv_dim + kv_head * head_dim + dim;
            // Keys stay dim-major: a lane owns one key and reads k_tile[dim][lane],
            // so the warp touches thirty-two different banks. The cache is f16,
            // the tile f32.
            k_tile[dim * FA_KEY_TILE + row] = __half2float(keys[source]);
            v_tile[row * FA_MAX_HEAD_DIM + dim] = __half2float(values[source]);
        }
        __syncthreads();

        // Each query group runs its scores and its value accumulation back to
        // back, so the probabilities never leave registers: the value pass reads
        // lane `row`'s probability with a shuffle instead of a shared load.
        #pragma unroll
        for (int base_query = 0; base_query < FA_QUERIES_PER_WARP; base_query += FA_QUERY_GROUP) {
            float dots[FA_QUERY_GROUP][2];
            #pragma unroll
            for (int q = 0; q < FA_QUERY_GROUP; ++q) {
                dots[q][0] = 0.0f;
                dots[q][1] = 0.0f;
            }
            if (base + lane < start + first_query + first_of_warp + base_query + FA_QUERY_GROUP) {
                for (int dim = 0; dim < head_dim; dim += FA_DIMS_PER_LANE) {
                    const float key0 = k_tile[(dim + 0) * FA_KEY_TILE + lane];
                    const float key1 = k_tile[(dim + 1) * FA_KEY_TILE + lane];
                    const float key2 = k_tile[(dim + 2) * FA_KEY_TILE + lane];
                    const float key3 = k_tile[(dim + 3) * FA_KEY_TILE + lane];
                    #pragma unroll
                    for (int q = 0; q < FA_QUERY_GROUP; ++q) {
                        const float* row = q_tile + head_slot * FA_QUERY_TILE * FA_MAX_HEAD_DIM
                            + (first_of_warp + base_query + q) * FA_MAX_HEAD_DIM;
                        const float4 values = *reinterpret_cast<const float4*>(row + dim);
                        dots[q][0] += values.x * key0 + values.y * key1;
                        dots[q][1] += values.z * key2 + values.w * key3;
                    }
                }
            }

            float probabilities[FA_QUERY_GROUP];
            #pragma unroll
            for (int q = 0; q < FA_QUERY_GROUP; ++q) {
                const int slot = base_query + q;
                const int query_index = first_of_warp + slot;
                const int visible = start + first_query + query_index + 1;
                const float score =
                    (base + lane < visible) ? (dots[q][0] + dots[q][1]) * scale : -INFINITY;
                const float tile_max = warp_max(score);
                const float new_max = fmaxf(running_max[slot], tile_max);
                probabilities[q] = score == -INFINITY ? 0.0f : expf(score - new_max);
                const float tile_sum = warp_sum(probabilities[q]);
                const float correction = running_max[slot] == -INFINITY
                    ? 0.0f
                    : expf(running_max[slot] - new_max);
                running_sum[slot] = running_sum[slot] * correction + tile_sum;
                running_max[slot] = new_max;
                #pragma unroll
                for (int d = 0; d < FA_DIMS_PER_LANE; ++d) {
                    acc[slot][d] *= correction;
                }
            }

            for (int row = 0; row < keys_here; ++row) {
                float probability[FA_QUERY_GROUP];
                #pragma unroll
                for (int q = 0; q < FA_QUERY_GROUP; ++q) {
                    probability[q] = __shfl_sync(0xffffffffu, probabilities[q], row);
                }
                const float4 value = *reinterpret_cast<const float4*>(
                    v_tile + row * FA_MAX_HEAD_DIM + lane * FA_DIMS_PER_LANE);
                #pragma unroll
                for (int q = 0; q < FA_QUERY_GROUP; ++q) {
                    const int slot = base_query + q;
                    acc[slot][0] += probability[q] * value.x;
                    acc[slot][1] += probability[q] * value.y;
                    acc[slot][2] += probability[q] * value.z;
                    acc[slot][3] += probability[q] * value.w;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FA_QUERIES_PER_WARP; ++i) {
        const int query_index = first_of_warp + i;
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

// Prefill attention on the f16 tensor cores. One block holds eight heads that
// share one KV head (this model's group size) and sixteen query positions, so
// the staged keys and values are read once per eight query heads. QK^T and PV
// both run as mma.m16n8k16.f16: a warp owns one head, its sixteen queries are
// one m-tile, and the scores never leave registers — the exp'ed score
// accumulators ARE the A fragments of the PV mma, which is the same trick
// ggml's fattn uses.
//
// Keys stay dim-major in shared memory so the K^T B fragments load straight
// with ldmatrix, and the online softmax keeps one running max/sum per query
// row across key tiles instead of materialising anything of size `visible`.
#define FA8_HEADS 8
#define FA8_QUERY_TILE 16
#define FA8_KEY_TILE 32
#define FA8_HEAD_DIM 128

__device__ __forceinline__ void mma_f16(float& d0, float& d1, float& d2, float& d3,
                                        int a0, int a1, int a2, int a3,
                                        int b0, int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A fragment: sixteen query rows of one head, sixteen dims, four 8x8 tiles.
__device__ __forceinline__ void load_q8_fragment(int* fragment, const __half* base, int lane) {
    const __half* address = base + (lane % 16) * FA8_HEAD_DIM + (lane / 16) * 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "l"(address));
}

// B fragment: sixteen rows of stride `stride` elements, eight columns, two
// 8x8 tiles — rows for K^T (dims), and for V (keys). The transpose matches
// what the mma B operand expects: b0 = B[2t][g], b1 = B[2t+1][g].
__device__ __forceinline__ void load_b8_fragment(int* fragment, const __half* base, int lane, int stride) {
    const __half* address = base + (lane % 16) * stride;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];\n"
                 : "=r"(fragment[0]), "=r"(fragment[1])
                 : "l"(address));
}

__device__ __forceinline__ int pack_half2(float low, float high) {
    __half2 packed = __floats2half2_rn(low, high);
    return *reinterpret_cast<int*>(&packed);
}

extern "C" __global__ __launch_bounds__(256, 2) void attention_prefill_mma(
    const float* __restrict__ query,
    const __half* __restrict__ keys,
    const __half* __restrict__ values,
    float* __restrict__ out,
    int n_head, int n_head_kv, int head_dim,
    int start, int tokens, float scale) {
    extern __shared__ __align__(16) signed char smem[];
    __half* q_tile = reinterpret_cast<__half*>(smem);
    __half* k_tile = q_tile + FA8_HEADS * FA8_QUERY_TILE * FA8_HEAD_DIM;
    __half* v_tile = k_tile + FA8_HEAD_DIM * FA8_KEY_TILE;

    const int group = blockIdx.x;
    // Causal masking makes the last query tile the most expensive one, by the
    // whole visible length. Blocks are scheduled in index order, so handing out
    // the heavy tiles first is what keeps the tail from deciding the makespan.
    const int tiles = (tokens + FA8_QUERY_TILE - 1) / FA8_QUERY_TILE;
    const int tile = tiles - 1 - blockIdx.y;
    const int heads_per_kv = n_head / n_head_kv;
    const int kv_head = (group * FA8_HEADS) / heads_per_kv;
    const int kv_dim = n_head_kv * head_dim;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int tid = warp * 32 + lane;
    const int head = group * FA8_HEADS + warp;
    const int first_query = tile * FA8_QUERY_TILE;
    const int queries = min(FA8_QUERY_TILE, tokens - first_query);
    if (queries <= 0) {
        return;
    }

    // Query rows are contiguous in memory. Load four f32 values at a time and
    // convert them into two half2 stores; this cuts the staging instruction
    // count without changing the tensor-core input layout.
    for (int index = tid * 4; index < FA8_HEADS * queries * head_dim; index += threads * 4) {
        const int slot = index / (queries * head_dim);
        const int rest = index % (queries * head_dim);
        const int row = rest / head_dim;
        const int dim = rest % head_dim;
        const float* source = query +
            ((size_t)(first_query + row) * n_head + group * FA8_HEADS + slot) * head_dim + dim;
        const float4 values = *reinterpret_cast<const float4*>(source);
        __half2* destination = reinterpret_cast<__half2*>(
            q_tile + (slot * FA8_QUERY_TILE + row) * FA8_HEAD_DIM + dim);
        destination[0] = __floats2half2_rn(values.x, values.y);
        destination[1] = __floats2half2_rn(values.z, values.w);
    }

    // Each lane owns query rows row_low and row_low + 8, and the two score
    // columns col_base and col_base + 1 of every 8-key tile.
    const int row_low = lane >> 2;
    const int col_base = (lane & 3) * 2;

    float accum[FA8_HEAD_DIM / 8][4];
    #pragma unroll
    for (int n8 = 0; n8 < FA8_HEAD_DIM / 8; ++n8) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            accum[n8][e] = 0.0f;
        }
    }
    float running_max_low = -INFINITY;
    float running_max_high = -INFINITY;
    float running_sum_low = 0.0f;
    float running_sum_high = 0.0f;
    __syncthreads();

    const int max_visible = start + min(first_query + queries, tokens);
    for (int base = 0; base < max_visible; base += FA8_KEY_TILE) {
        const int keys_here = min(FA8_KEY_TILE, max_visible - base);
        // The source K/V rows are contiguous half values. Keep K's dim-major
        // shared layout, but stage both operands in half2 pairs to reduce
        // global load and conversion overhead.
        for (int index = tid * 2; index < FA8_KEY_TILE * head_dim; index += threads * 2) {
            const int row = index / head_dim;
            const int dim = index % head_dim;
            if (row < keys_here) {
                const size_t source = (size_t)(base + row) * kv_dim + kv_head * head_dim + dim;
                // K is staged dim-major so the K^T fragments load untransposed;
                // V stays row-major for the PV fragments.
                const __half2 key_values = *reinterpret_cast<const __half2*>(keys + source);
                const __half2 value_values = *reinterpret_cast<const __half2*>(values + source);
                k_tile[dim * FA8_KEY_TILE + row] = key_values.x;
                k_tile[(dim + 1) * FA8_KEY_TILE + row] = key_values.y;
                *reinterpret_cast<__half2*>(v_tile + row * FA8_HEAD_DIM + dim) = value_values;
            } else {
                k_tile[dim * FA8_KEY_TILE + row] = __float2half(0.0f);
                k_tile[(dim + 1) * FA8_KEY_TILE + row] = __float2half(0.0f);
                *reinterpret_cast<__half2*>(v_tile + row * FA8_HEAD_DIM + dim) =
                    __floats2half2_rn(0.0f, 0.0f);
            }
        }
        __syncthreads();

        float scores[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                scores[nt][e] = 0.0f;
            }
        }
        #pragma unroll
        for (int dim = 0; dim < FA8_HEAD_DIM / 16; ++dim) {
            int a[4];
            load_q8_fragment(a, q_tile + warp * FA8_QUERY_TILE * FA8_HEAD_DIM + dim * 16, lane);
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int b[2];
                load_b8_fragment(b, k_tile + dim * 16 * FA8_KEY_TILE + nt * 8, lane, FA8_KEY_TILE);
                mma_f16(scores[nt][0], scores[nt][1], scores[nt][2], scores[nt][3],
                        a[0], a[1], a[2], a[3], b[0], b[1]);
            }
        }

        // Mask, scale, and run the online softmax statistics for this tile.
        const int visible_low = start + first_query + row_low + 1;
        const int visible_high = visible_low + 8;
        float tile_max_low = -INFINITY;
        float tile_max_high = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int column = base + col_base + nt * 8;
            scores[nt][0] = (column < visible_low) ? scores[nt][0] * scale : -INFINITY;
            scores[nt][1] = (column + 1 < visible_low) ? scores[nt][1] * scale : -INFINITY;
            scores[nt][2] = (column < visible_high) ? scores[nt][2] * scale : -INFINITY;
            scores[nt][3] = (column + 1 < visible_high) ? scores[nt][3] * scale : -INFINITY;
            tile_max_low = fmaxf(tile_max_low, fmaxf(scores[nt][0], scores[nt][1]));
            tile_max_high = fmaxf(tile_max_high, fmaxf(scores[nt][2], scores[nt][3]));
        }
        tile_max_low = fmaxf(tile_max_low, __shfl_xor_sync(0xffffffffu, tile_max_low, 1));
        tile_max_low = fmaxf(tile_max_low, __shfl_xor_sync(0xffffffffu, tile_max_low, 2));
        tile_max_high = fmaxf(tile_max_high, __shfl_xor_sync(0xffffffffu, tile_max_high, 1));
        tile_max_high = fmaxf(tile_max_high, __shfl_xor_sync(0xffffffffu, tile_max_high, 2));

        const float new_max_low = fmaxf(running_max_low, tile_max_low);
        const float new_max_high = fmaxf(running_max_high, tile_max_high);
        const float correction_low =
            running_max_low == -INFINITY ? 0.0f : expf(running_max_low - new_max_low);
        const float correction_high =
            running_max_high == -INFINITY ? 0.0f : expf(running_max_high - new_max_high);
        #pragma unroll
        for (int n8 = 0; n8 < FA8_HEAD_DIM / 8; ++n8) {
            accum[n8][0] *= correction_low;
            accum[n8][1] *= correction_low;
            accum[n8][2] *= correction_high;
            accum[n8][3] *= correction_high;
        }

        float tile_sum_low = 0.0f;
        float tile_sum_high = 0.0f;
        float probs[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            probs[nt][0] = scores[nt][0] == -INFINITY ? 0.0f : expf(scores[nt][0] - new_max_low);
            probs[nt][1] = scores[nt][1] == -INFINITY ? 0.0f : expf(scores[nt][1] - new_max_low);
            probs[nt][2] = scores[nt][2] == -INFINITY ? 0.0f : expf(scores[nt][2] - new_max_high);
            probs[nt][3] = scores[nt][3] == -INFINITY ? 0.0f : expf(scores[nt][3] - new_max_high);
            tile_sum_low += probs[nt][0] + probs[nt][1];
            tile_sum_high += probs[nt][2] + probs[nt][3];
        }
        tile_sum_low += __shfl_xor_sync(0xffffffffu, tile_sum_low, 1);
        tile_sum_low += __shfl_xor_sync(0xffffffffu, tile_sum_low, 2);
        tile_sum_high += __shfl_xor_sync(0xffffffffu, tile_sum_high, 1);
        tile_sum_high += __shfl_xor_sync(0xffffffffu, tile_sum_high, 2);
        running_sum_low = running_sum_low * correction_low + tile_sum_low;
        running_sum_high = running_sum_high * correction_high + tile_sum_high;
        running_max_low = new_max_low;
        running_max_high = new_max_high;

        // PV: the exp'ed scores ARE the A fragments — pack two neighbours per
        // register and multiply the value rows right away.
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            int pa[4];
            pa[0] = pack_half2(probs[kh * 2][0], probs[kh * 2][1]);
            pa[1] = pack_half2(probs[kh * 2][2], probs[kh * 2][3]);
            pa[2] = pack_half2(probs[kh * 2 + 1][0], probs[kh * 2 + 1][1]);
            pa[3] = pack_half2(probs[kh * 2 + 1][2], probs[kh * 2 + 1][3]);
            #pragma unroll
            for (int n8 = 0; n8 < FA8_HEAD_DIM / 8; ++n8) {
                int b[2];
                load_b8_fragment(b, v_tile + kh * 16 * FA8_HEAD_DIM + n8 * 8, lane, FA8_HEAD_DIM);
                mma_f16(accum[n8][0], accum[n8][1], accum[n8][2], accum[n8][3],
                        pa[0], pa[1], pa[2], pa[3], b[0], b[1]);
            }
        }
        __syncthreads();
    }

    const float inverse_low = running_sum_low > 0.0f ? 1.0f / running_sum_low : 0.0f;
    const float inverse_high = running_sum_high > 0.0f ? 1.0f / running_sum_high : 0.0f;
    if (first_query + row_low < tokens) {
        float* low = out + ((size_t)(first_query + row_low) * n_head + head) * head_dim;
        #pragma unroll
        for (int n8 = 0; n8 < FA8_HEAD_DIM / 8; ++n8) {
            const int dim = col_base + n8 * 8;
            low[dim] = accum[n8][0] * inverse_low;
            low[dim + 1] = accum[n8][1] * inverse_low;
        }
    }
    if (first_query + row_low + 8 < tokens) {
        float* high = out + ((size_t)(first_query + row_low + 8) * n_head + head) * head_dim;
        #pragma unroll
        for (int n8 = 0; n8 < FA8_HEAD_DIM / 8; ++n8) {
            const int dim = col_base + n8 * 8;
            high[dim] = accum[n8][2] * inverse_high;
            high[dim + 1] = accum[n8][3] * inverse_high;
        }
    }
}

extern "C" __global__ void attention(const float* __restrict__ query,
                                     const __half* __restrict__ keys,
                                     const __half* __restrict__ values,
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
        const __half* key = keys + (size_t)offset * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            dot += query_head[i] * __half2float(key[i]);
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

    float accumulator[MAX_HEAD_DIM / 32];
    for (int element = 0; element < elements; ++element) {
        accumulator[element] = 0.0f;
    }
    for (int offset = warp; offset < visible; offset += ATTENTION_WARPS) {
        const __half* value = values + (size_t)offset * kv_dim + kv_head * head_dim;
        const float weight = row[offset];
        for (int element = 0; element < elements; ++element) {
            accumulator[element] += weight * __half2float(value[lane + element * 32]);
        }
    }
    for (int element = 0; element < elements; ++element) {
        partial[warp][lane + element * 32] = accumulator[element];
    }
    __syncthreads();

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (int source = 0; source < ATTENTION_WARPS; ++source) {
            sum += partial[source][dim];
        }
        out[((size_t)token * n_head + head) * head_dim + dim] = sum;
    }
}

// Embedding row dequantized straight into the activation buffer.
extern "C" __global__ void embedding_row(const __half* __restrict__ scales,
                                         const signed char* __restrict__ quants,
                                         float* __restrict__ out,
                                         int row, int k) {
    const __half* row_scales = scales + (size_t)row * (k / Q8_BLOCK_ELEMS);
    const signed char* row_quants = quants + (size_t)row * k;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        out[i] = __half2float(row_scales[i >> 5]) * (float)row_quants[i];
    }
}

// Whole prompt embedding in one launch: one thread per (token, column).
extern "C" __global__ void embedding_rows(const __half* __restrict__ scales,
                                          const signed char* __restrict__ quants,
                                          const int* __restrict__ tokens,
                                          float* __restrict__ out,
                                          int n, int k) {
    const int blocks_per_row = k / Q8_BLOCK_ELEMS;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * k; i += gridDim.x * blockDim.x) {
        const int token = i / k;
        const int column = i % k;
        const int row = tokens[token];
        const __half* row_scales = scales + (size_t)row * blocks_per_row;
        const signed char* row_quants = quants + (size_t)row * k;
        out[i] = __half2float(row_scales[column >> 5]) * (float)row_quants[column];
    }
}
