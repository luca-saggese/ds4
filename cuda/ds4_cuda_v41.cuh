/* DeepSeek V4.1 CUDA baseline primitives. Float-addressable storage keeps
 * the released BF16/FP8/FP4 graph boundaries explicit. */

#ifndef DS4_V41_ACTIVATION_FORMAT_DEFINED
#define DS4_V41_ACTIVATION_FORMAT_DEFINED
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
#endif

#ifndef DS4_V41_CARRY_FORMAT_DEFINED
#define DS4_V41_CARRY_FORMAT_DEFINED
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };
#endif

__device__ static float v41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ static float v41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) +
                           ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__device__ static float v41_sum32(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, delta);
    return __shfl_sync(0xffffffffu, x, 0);
}

static bool v41_tensor_has_bytes(const ds4_gpu_tensor *tensor, uint64_t bytes) {
    return tensor && tensor->ptr && tensor->bytes >= bytes;
}

static bool v41_tensor_has_f32(const ds4_gpu_tensor *tensor, uint64_t count) {
    return count <= UINT64_MAX / sizeof(float) &&
           v41_tensor_has_bytes(tensor, count * sizeof(float));
}

__global__ static void v41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = v41_bf16(x[i]);
}

__global__ static void v41_quantize_kernel(float *x, uint32_t format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x;
    const uint64_t i = (uint64_t)blockIdx.x * block + lane;
    const float value = lane < block ? v41_bf16(x[i]) : 0.0f;
    float amax = fabsf(value);
    for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
    amax = __shfl_sync(0xffffffffu, amax, 0);

    float result;
    if (format == DS4_V41_FP8_E8M0) {
        const float scale = v41_pow2_ceil(fmaxf(amax, 1.0e-4f) / 448.0f);
        result = dsv4_e4m3fn_dequant_dev(fabsf(value) / scale) * scale;
    } else {
        const float scale = format == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(fmaxf(amax, 0.01171875f) / 6.0f) :
            v41_pow2_ceil(fmaxf(amax, 7.052966104933725e-38f) / 6.0f);
        result = dsv4_e2m1fn_dequant_dev(fabsf(value) / scale) * scale;
    }
    if (lane < block) {
        const uint32_t bits =
            (__float_as_uint(v41_bf16(result)) & 0x7fffffffu) |
            (__float_as_uint(value) & 0x80000000u);
        x[i] = __uint_as_float(bits);
    }
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width,
                                      uint32_t rows,
                                      ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (!x || !x->ptr || !width || !rows ||
        format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block)) return 0;
    const uint64_t count = (uint64_t)width * rows;
    if (count > UINT64_MAX / sizeof(float) ||
        x->bytes < count * sizeof(float)) return 0;
    if (format == DS4_V41_BF16) {
        if ((count + 255u) / 256u > UINT32_MAX) return 0;
        v41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
            (float *)x->ptr, count);
    } else {
        const uint64_t blocks = (uint64_t)(width / block) * rows;
        if (blocks > UINT32_MAX) return 0;
        v41_quantize_kernel<<<(unsigned)blocks, 32>>>(
            (float *)x->ptr, format);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 activation quantization");
}

struct v41_rope_args {
    uint32_t width, heads, start, stride, inverse;
    float frequencies[32];
};

__global__ static void v41_rope_kernel(float *x, v41_rope_args args) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x / args.heads;
    const float theta =
        (float)(args.start + row * args.stride) * args.frequencies[lane];
    float s, c;
    sincosf(theta, &s, &c);
    if (args.inverse) s = -s;
    const uint64_t i =
        (uint64_t)blockIdx.x * args.width + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = v41_bf16(re * c - im * s);
    x[i + 1u] = v41_bf16(re * s + im * c);
}

static float v41_rope_frequencies[2][32];
static pthread_once_t v41_rope_once = PTHREAD_ONCE_INIT;

static void v41_init_rope_frequencies(void) {
    for (int kind = 0; kind < 2; kind++) {
        const float base = kind ? 160000.0f : 10000.0f;
        const float low = (float)floor(
            64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) /
            (2.0 * log(base)));
        const float high = (float)ceil(
            64.0 * log(65536.0 / (2.0 * M_PI)) /
            (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (kind) {
                const float ramp =
                    fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
            }
            v41_rope_frequencies[kind][i] = f;
        }
    }
}

extern "C" int ds4_gpu_dsv41_rope_stride(
        ds4_gpu_tensor *x, uint32_t width, uint32_t heads, uint32_t rows,
        uint32_t start, uint32_t stride, bool compressed, bool inverse) {
    if (!x || !x->ptr || width < 64u || !heads || !rows ||
        rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT32_MAX) return 0;
    const uint64_t count = (uint64_t)width * heads * rows;
    if (count > UINT64_MAX / sizeof(float) ||
        x->bytes < count * sizeof(float) ||
        pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, stride, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0],
           sizeof(args.frequencies));
    v41_rope_kernel<<<heads * rows, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 unit-magnitude RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(
        ds4_gpu_tensor *x, uint32_t width, uint32_t heads, uint32_t rows,
        uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(
        x, width, heads, rows, start, 1u, compressed, inverse);
}

__global__ static void v41_engram_kernel(
        float *residual, const float *kv, const float *qw,
        const float *kw, const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t token = blockIdx.x, head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (mask && !mask[token]) return;
    const uint64_t offset = ((uint64_t)token * 4u + head) * width;
    const uint64_t key = ((uint64_t)token * 5u + head) * width;
    const uint64_t value = ((uint64_t)token * 5u + 4u) * width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i];
        const float k = v41_bf16(kv[key + i]);
        const uint64_t wi = (uint64_t)head * width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (qw[wi] * kw[wi]) * k;
    }
    h2 = v41_sum32(h2);
    k2 = v41_sum32(k2);
    dot = v41_sum32(dot) * rsqrtf(h2 / (float)width + eps);
    dot *= rsqrtf(k2 / (float)width + eps);
    dot *= rsqrtf((float)width);
    const float gate =
        1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    for (uint32_t i = lane; i < width; i += 32u) {
        residual[offset + i] = v41_bf16(
            residual[offset + i] + gate * v41_bf16(kv[value + i]));
    }
}

extern "C" int ds4_gpu_dsv41_engram_add(
        ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
        float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || !isfinite(eps) || eps <= 0.0f ||
        count > UINT64_MAX / 5u ||
        !v41_tensor_has_f32(residual, count * 4u) ||
        !v41_tensor_has_f32(kv, count * 5u) ||
        !v41_tensor_has_f32(q_weight, (uint64_t)width * 4u) ||
        !v41_tensor_has_f32(k_weight, (uint64_t)width * 4u) ||
        (mask && !v41_tensor_has_bytes(mask, rows)))
        return 0;
    v41_engram_kernel<<<dim3(rows, 4u), 32>>>(
        (float *)residual->ptr, (const float *)kv->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void v41_pool_kernel(
        float *out, const float *kv, const float *scores,
        const float *previous_kv, const float *previous_scores,
        uint32_t width, uint32_t tail) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const int64_t a = (int64_t)blockIdx.y * 2 - tail;
    const uint64_t b = (uint64_t)(a + 1) * width + col;
    const float ka = a < 0 ? previous_kv[col] :
        kv[(uint64_t)a * width + col];
    const float sa = a < 0 ? previous_scores[col] :
        scores[(uint64_t)a * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak), eb = expf(sb - peak);
    out[(uint64_t)blockIdx.y * width + col] =
        v41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

extern "C" int ds4_gpu_dsv41_pool2(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *scores, ds4_gpu_tensor *previous_kv,
        ds4_gpu_tensor *previous_scores, uint32_t width, uint32_t rows,
        uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs =
        (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!width || !rows || rows > UINT32_MAX - start ||
        !v41_tensor_has_f32(kv, count) ||
        !v41_tensor_has_f32(scores, count) ||
        !v41_tensor_has_f32(previous_kv, width) ||
        !v41_tensor_has_f32(previous_scores, width) ||
        (pairs && !v41_tensor_has_f32(out, (uint64_t)width * pairs)))
        return 0;
    if (pairs) {
        v41_pool_kernel<<<
            dim3((unsigned)(((uint64_t)width + 255u) / 256u), pairs), 256>>>(
            (float *)out->ptr, (const float *)kv->ptr,
            (const float *)scores->ptr, (const float *)previous_kv->ptr,
            (const float *)previous_scores->ptr, width, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 KV pair pooling")) return 0;
    }
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float);
        const uint64_t offset = (last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(
                previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(
                previous_scores, 0, scores, offset, bytes))
            return 0;
    }
    return 1;
}

template <bool FILTER>
__global__ static void v41_candidates_kernel(
        float *out, const float *scores, const float *mask,
        uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (FILTER) {
        if (col >= width) return;
        const uint64_t i = (uint64_t)row * width + col;
        out[i] = col < visible &&
                 mask[(uint64_t)row * blocks + col / 8u] == 0.0f ?
            scores[i] : -INFINITY;
    } else {
        if (col >= blocks) return;
        float best = -INFINITY;
        for (uint32_t i = col * 8u;
             i < min(visible, (col + 1u) * 8u); i++)
            best = fmaxf(best, scores[(uint64_t)row * width + i]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[(uint64_t)row * blocks + col] = best;
    }
}

static int v41_candidates(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
        uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio ||
        rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t out_width = mask ? width : blocks;
    if (!v41_tensor_has_f32(scores, (uint64_t)width * rows) ||
        !v41_tensor_has_f32(out, (uint64_t)out_width * rows) ||
        (mask && !v41_tensor_has_f32(mask, (uint64_t)blocks * rows)))
        return 0;
    const dim3 grid(
        (unsigned)(((uint64_t)out_width + 255u) / 256u), rows);
    if (mask) {
        v41_candidates_kernel<true><<<grid, 256>>>(
            (float *)out->ptr, (const float *)scores->ptr,
            (const float *)mask->ptr, width, start, ratio);
    } else {
        v41_candidates_kernel<false><<<grid, 256>>>(
            (float *)out->ptr, (const float *)scores->ptr,
            NULL, width, start, ratio);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 candidate selection");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(
        ds4_gpu_tensor *blocks, const ds4_gpu_tensor *scores,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return v41_candidates(
        blocks, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *block_mask,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return block_mask &&
        v41_candidates(
            scores, scores, block_mask, width, rows, start, ratio);
}

__global__ static void v41_gather_kernel(
        float *out, const float *source, const int32_t *ids,
        uint32_t source_rows) {
    const uint32_t row = blockIdx.x, col = threadIdx.x;
    const int32_t id = ids[row];
    if ((uint32_t)id >= source_rows) {
        out[(uint64_t)row * 512u + col] = NAN;
        out[(uint64_t)row * 512u + col + 256u] = NAN;
        return;
    }
    out[(uint64_t)row * 512u + col] =
        source[(uint64_t)id * 512u + col];
    out[(uint64_t)row * 512u + col + 256u] =
        source[(uint64_t)id * 512u + col + 256u];
}

extern "C" int ds4_gpu_dsv41_gather_kv(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
        const ds4_gpu_tensor *ids, uint32_t source_rows,
        uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u ||
        selected_rows > source_rows ||
        !v41_tensor_has_f32(source, (uint64_t)source_rows * 512u) ||
        !v41_tensor_has_f32(out, (uint64_t)selected_rows * 512u) ||
        !v41_tensor_has_bytes(
            ids, (uint64_t)selected_rows * sizeof(int32_t)))
        return 0;
    v41_gather_kernel<<<selected_rows, 256>>>(
        (float *)out->ptr, (const float *)source->ptr,
        (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

__global__ static void v41_carry_bf16_kernel(
        uint16_t *packed, float *plain, uint32_t width, uint32_t words,
        bool pack) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const uint64_t p = (uint64_t)blockIdx.y * words * 2u + col;
    const uint64_t f = (uint64_t)blockIdx.y * width + col;
    if (pack) packed[p] = (uint16_t)(__float_as_uint(plain[f]) >> 16u);
    else plain[f] = __uint_as_float((uint32_t)packed[p] << 16u);
}

__global__ static void v41_carry_mask_kernel(
        uint32_t *packed, float *plain, uint32_t width, uint32_t words,
        bool pack) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words) return;
    const uint64_t p = (uint64_t)blockIdx.y * words + word;
    uint32_t bits = pack ? 0u : packed[p];
    for (uint32_t bit = 0;
         bit < 32u && (uint64_t)word * 32u + bit < width; bit++) {
        const uint64_t f =
            (uint64_t)blockIdx.y * width + (uint64_t)word * 32u + bit;
        if (pack) bits |= plain[f] == 0.0f ? 1u << bit : 0u;
        else plain[f] = bits & (1u << bit) ? 0.0f : -INFINITY;
    }
    if (pack) packed[p] = bits;
}

extern "C" int ds4_gpu_dsv41_carry_copy(
        ds4_gpu_tensor *packed, uint32_t row_offset,
        ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
        uint32_t format, bool pack) {
    if (!width || !rows || rows > UINT32_MAX - row_offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ?
        (uint32_t)(((uint64_t)width + 1u) / 2u) :
        (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (!v41_tensor_has_bytes(
            packed, ((uint64_t)row_offset + rows) * words * 4u) ||
        !v41_tensor_has_f32(plain, (uint64_t)rows * width))
        return 0;
    uint32_t *p =
        (uint32_t *)packed->ptr + (uint64_t)row_offset * words;
    if (format == DS4_V41_CARRY_BF16) {
        v41_carry_bf16_kernel<<<
            dim3((unsigned)(((uint64_t)width + 255u) / 256u), rows), 256>>>(
            (uint16_t *)p, (float *)plain->ptr, width, words, pack);
    } else {
        v41_carry_mask_kernel<<<
            dim3((unsigned)(((uint64_t)words + 255u) / 256u), rows), 256>>>(
            p, (float *)plain->ptr, width, words, pack);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 compact prefill carry");
}

__global__ static void v41_indexer_kernel(
        float *scores, const float *q, const float *weights,
        const float *keys, uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x, token = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    if (key >= (start + token + 1u) / ratio) {
        if (!threadIdx.x)
            scores[(uint64_t)token * width + key] = -INFINITY;
        return;
    }
    __shared__ float head_values[4];
    float total = 0.0f;
    for (uint32_t head0 = 0; head0 < 32u; head0 += 4u) {
        const uint32_t head = head0 + warp;
        const float *query = q + ((uint64_t)token * 32u + head) * 128u;
        const float *kv = keys + (uint64_t)key * 128u;
        float dot = 0.0f;
        for (uint32_t col = lane; col < 128u; col += 32u)
            dot += query[col] * kv[col];
        dot = v41_sum32(dot);
        if (!lane)
            head_values[warp] =
                fmaxf(dot / 64.0f, 0.0f) *
                weights[(uint64_t)token * 32u + head];
        __syncthreads();
        if (!threadIdx.x)
            for (uint32_t h = 0; h < 4u; h++)
                total += head_values[h];
        __syncthreads();
    }
    if (!threadIdx.x)
        scores[(uint64_t)token * width + key] = total;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
        uint32_t source_rows, uint32_t rows, uint32_t start,
        uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || !rows ||
        rows > UINT32_MAX - start ||
        (start + rows) / ratio > source_rows ||
        source_rows > INT32_MAX || rows > INT32_MAX ||
        !v41_tensor_has_f32(scores, (uint64_t)source_rows * rows) ||
        !v41_tensor_has_f32(q, (uint64_t)rows * 32u * 128u) ||
        !v41_tensor_has_f32(keys, (uint64_t)source_rows * 128u) ||
        !v41_tensor_has_f32(weights, (uint64_t)rows * 32u))
        return 0;
    v41_indexer_kernel<<<dim3(source_rows, rows), 128>>>(
        (float *)scores->ptr, (const float *)q->ptr,
        (const float *)weights->ptr, (const float *)keys->ptr,
        source_rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 causal FP4 index scores");
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(
        ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows ||
        rows > UINT32_MAX - start || width > INT32_MAX ||
        rows > INT32_MAX || (start + rows) / ratio > width ||
        !v41_tensor_has_f32(scores, (uint64_t)width * rows) ||
        !v41_tensor_has_bytes(selected, (uint64_t)512u * rows * 4u))
        return 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        if (!visible) continue;
        const uint32_t top = min(visible, 512u);
        ds4_gpu_tensor in = {
            (float *)scores->ptr + (uint64_t)row * width,
            (uint64_t)visible * 4u, 0, scores->device_id
        };
        ds4_gpu_tensor out = {
            (uint32_t *)selected->ptr + (uint64_t)row * 512u,
            512u * 4u, 0, selected->device_id
        };
        if (!ds4_gpu_indexer_topk_tensor(
                &out, &in, visible, 1u, top))
            return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) {
    return 0;
}

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(
        uint32_t source_rows, uint32_t rows) {
    const uint64_t tiles = ((uint64_t)source_rows + 63u) / 64u;
    const uint64_t flags =
        (((uint64_t)rows + tiles) * 4u + 255u) & ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u +
           tiles * 64u * 128u * 2u;
}

extern "C" int ds4_gpu_dsv41_indexer_pack(
        ds4_gpu_tensor *packed, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *keys, uint32_t source_rows, uint32_t rows) {
    (void)packed;
    (void)q;
    (void)keys;
    (void)source_rows;
    (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
        const ds4_gpu_tensor *packed, uint32_t source_rows, uint32_t rows,
        uint32_t start, uint32_t ratio, uint32_t packed_rows,
        uint32_t offset) {
    (void)scores;
    (void)q;
    (void)weights;
    (void)keys;
    (void)packed;
    (void)source_rows;
    (void)rows;
    (void)start;
    (void)ratio;
    (void)packed_rows;
    (void)offset;
    return 0;
}

extern "C" int ds4_gpu_dsv41_projection_rows(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t width, uint32_t outputs,
        uint32_t rows, const ds4_gpu_tensor *in) {
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !v41_tensor_has_f32(in, (uint64_t)width * rows) ||
        !v41_tensor_has_f32(out, (uint64_t)outputs * rows))
        return 0;
    return ds4_gpu_matmul_f16_tensor(
        out, model_map, model_size, weight_offset,
        width, outputs, in, rows);
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    if (!n_tokens || !v41_tensor_has_f32(
            heads, (uint64_t)n_tokens * 32768u) ||
        !v41_tensor_has_f32(low, (uint64_t)n_tokens * 8192u) ||
        !v41_tensor_has_f32(out, (uint64_t)n_tokens * 5120u))
        return 0;
    return ds4_gpu_attention_output_low_q8_rows_exact_tensor(
               low, model_map, model_size, out_a_offset,
               4096u, 1024u, 8u, 0u, 8u, heads, n_tokens) &&
           ds4_gpu_dsv41_quantize(
               low, 8192u, n_tokens, DS4_V41_BF16) &&
           ds4_gpu_matmul_q8_0_tensor(
               out, model_map, model_size, out_b_offset,
               8192u, 5120u, low, n_tokens);
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens,
        uint32_t tp_rank) {
    (void)out;
    (void)low;
    (void)model_map;
    (void)model_size;
    (void)out_a_offset;
    (void)out_b_offset;
    (void)heads;
    (void)n_tokens;
    (void)tp_rank;
    return 0;
}

extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows, float eps) {
    if (!in_dim || !out_dim || !n_rows || !isfinite(eps) || eps <= 0.0f)
        return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(
               scale_scratch, x, in_dim, n_rows, eps) &&
           ds4_gpu_dsv41_projection_rows(
               out, model_map, model_size, weight_offset,
               in_dim, out_dim, n_rows, scale_scratch);
}

extern "C" int ds4_gpu_tp_failed(void) {
    return 0;
}
