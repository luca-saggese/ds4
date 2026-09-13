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
