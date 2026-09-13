#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;

static float random_value(void) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] :
            i < 8 ? ldexpf(i, -9) :
            ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error ||
            (error == best_error && !(i & 1) && (best & 1))) {
            best = i;
            best_value = v;
            best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    if (tensor && data && !ds4_gpu_tensor_write(tensor, 0, data, bytes)) {
        ds4_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, COUNT = WIDTH * ROWS };
    float *source = malloc(COUNT * sizeof(float));
    float *actual = malloc(COUNT * sizeof(float));
    ds4_gpu_tensor *tensor = upload(NULL, COUNT * sizeof(float));
    CHECK(source && actual && tensor);

    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < COUNT; i++)
            source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++)
            source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(
            tensor, 0, source, COUNT * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(
            tensor, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(
            tensor, 0, actual, COUNT * sizeof(float)));
        for (int start = 0; start < COUNT; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++)
                amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(
                    fmaxf(amax, 1.0e-4f) / 448.0f)));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(
                    fmaxf(amax, 0x1.8p-124f) / 6.0f)));
            if (mode == DS4_V41_FP4_E4M3)
                scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode)
                    expected = bf16(
                        nearest(expected / scale, mode != 1) * scale);
                CHECK(!memcmp(&expected, actual + start + i, sizeof(float)));
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(
        tensor, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(
        tensor, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(
        tensor, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(
        tensor, 32, 1, (ds4_v41_activation_format)4));

    ds4_gpu_tensor_free(tensor);
    free(actual);
    free(source);
    return 1;
}

static int check_rope(void) {
    enum {
        WIDTH = 512,
        HEADS = 2,
        ROWS = 129,
        COUNT = WIDTH * HEADS * ROWS
    };
    float *input = malloc(COUNT * sizeof(float));
    float *batch = malloc(COUNT * sizeof(float));
    float *rows = malloc(COUNT * sizeof(float));
    ds4_gpu_tensor *a = NULL, *b = NULL;
    CHECK(input && batch && rows);
    for (int i = 0; i < COUNT; i++) input[i] = bf16(random_value());
    a = upload(input, COUNT * sizeof(float));
    b = upload(input, COUNT * sizeof(float));
    CHECK(a && b);

    CHECK(ds4_gpu_dsv41_rope_stride(
        a, WIDTH, HEADS, ROWS, 32766, 2, true, false));
    for (uint32_t row = 0; row < ROWS; row++) {
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(
            b, (uint64_t)row * WIDTH * HEADS * sizeof(float),
            WIDTH * HEADS * sizeof(float));
        CHECK(view);
        CHECK(ds4_gpu_dsv41_rope(
            view, WIDTH, HEADS, 1, 32766 + row * 2u, true, false));
        ds4_gpu_tensor_free(view);
    }
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(a, 0, batch, COUNT * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(b, 0, rows, COUNT * sizeof(float)));
    CHECK(!memcmp(batch, rows, COUNT * sizeof(float)));
    CHECK(!ds4_gpu_dsv41_rope_stride(
        a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(
        a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));

    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    free(rows);
    free(batch);
    free(input);
    return 1;
}

int main(void) {
    CHECK(ds4_gpu_init());
    CHECK(check_quantization());
    CHECK(check_rope());
    ds4_gpu_cleanup();
    puts("CUDA V4.1 quantization and RoPE: PASS");
    return 0;
}
