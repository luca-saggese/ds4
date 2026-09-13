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

static int check_pool_and_candidates(void) {
    enum { WIDTH = 8, ROWS = 4, PAIRS = 2 };
    float kv[WIDTH * ROWS], scores[WIDTH * ROWS];
    float actual[WIDTH * PAIRS], expected[WIDTH * PAIRS];
    float previous_kv[WIDTH] = {0}, previous_scores[WIDTH] = {0};
    for (int i = 0; i < WIDTH * ROWS; i++) {
        kv[i] = (float)(i - 11) / 7.0f;
        scores[i] = (float)((i * 13) % 17) - 8.0f;
    }
    for (int pair = 0; pair < PAIRS; pair++) {
        for (int col = 0; col < WIDTH; col++) {
            const int a = (2 * pair) * WIDTH + col;
            const int b = a + WIDTH;
            const double peak = fmax(scores[a], scores[b]);
            const double ea = exp(scores[a] - peak);
            const double eb = exp(scores[b] - peak);
            expected[pair * WIDTH + col] =
                bf16((float)((kv[a] * ea + kv[b] * eb) / (ea + eb)));
        }
    }
    ds4_gpu_tensor *kt = upload(kv, sizeof(kv));
    ds4_gpu_tensor *st = upload(scores, sizeof(scores));
    ds4_gpu_tensor *out = upload(NULL, sizeof(actual));
    ds4_gpu_tensor *pk = upload(previous_kv, sizeof(previous_kv));
    ds4_gpu_tensor *ps = upload(previous_scores, sizeof(previous_scores));
    CHECK(kt && st && out && pk && ps);
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, WIDTH, ROWS, 0));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    for (int i = 0; i < WIDTH * PAIRS; i++)
        CHECK(fabsf(actual[i] - expected[i]) <=
              fmaxf(1e-6f, fabsf(expected[i]) / 128.0f));
    CHECK(ds4_gpu_tensor_read(pk, 0, previous_kv, sizeof(previous_kv)));
    CHECK(!memcmp(previous_kv, kv + 2 * WIDTH, sizeof(previous_kv)));

    enum { COLUMNS = 17, CANDIDATE_ROWS = 2, BLOCKS = 3 };
    float candidate_scores[COLUMNS * CANDIDATE_ROWS];
    float maxima[BLOCKS * CANDIDATE_ROWS];
    float mask[BLOCKS * CANDIDATE_ROWS];
    float filtered[COLUMNS * CANDIDATE_ROWS];
    for (int i = 0; i < COLUMNS * CANDIDATE_ROWS; i++)
        candidate_scores[i] = (float)(i + 1);
    for (int i = 0; i < BLOCKS * CANDIDATE_ROWS; i++)
        mask[i] = -INFINITY;
    mask[0] = mask[1] = mask[3] = 0.0f;
    ds4_gpu_tensor *cs = upload(candidate_scores, sizeof(candidate_scores));
    ds4_gpu_tensor *cb = upload(NULL, sizeof(maxima));
    ds4_gpu_tensor *cm = upload(mask, sizeof(mask));
    CHECK(cs && cb && cm);
    CHECK(ds4_gpu_dsv41_candidate_blocks(
        cb, cs, COLUMNS, CANDIDATE_ROWS, COLUMNS - 1, 1));
    CHECK(ds4_gpu_dsv41_candidate_filter(
        cs, cm, COLUMNS, CANDIDATE_ROWS, COLUMNS - 1, 1));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(cb, 0, maxima, sizeof(maxima)));
    CHECK(ds4_gpu_tensor_read(cs, 0, filtered, sizeof(filtered)));
    CHECK(isinf(maxima[2]) && maxima[2] > 0.0f);
    CHECK(filtered[0] == candidate_scores[0]);
    CHECK(isinf(filtered[16]) && filtered[16] < 0.0f);

    ds4_gpu_tensor_free(cm);
    ds4_gpu_tensor_free(cb);
    ds4_gpu_tensor_free(cs);
    ds4_gpu_tensor_free(ps);
    ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(kt);
    return 1;
}

static int check_gather_carry_and_indexer(void) {
    enum { SOURCE_ROWS = 4, KV_WIDTH = 512, SELECTED = 3 };
    float source[SOURCE_ROWS * KV_WIDTH];
    float gathered[SELECTED * KV_WIDTH];
    int32_t ids[SELECTED] = {3, 0, 2};
    for (int i = 0; i < SOURCE_ROWS * KV_WIDTH; i++)
        source[i] = (float)i;
    ds4_gpu_tensor *source_t = upload(source, sizeof(source));
    ds4_gpu_tensor *ids_t = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out_t = upload(NULL, sizeof(gathered));
    CHECK(source_t && ids_t && out_t);
    CHECK(ds4_gpu_dsv41_gather_kv(
        out_t, source_t, ids_t, SOURCE_ROWS, SELECTED));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(out_t, 0, gathered, sizeof(gathered)));
    for (int row = 0; row < SELECTED; row++)
        CHECK(!memcmp(
            gathered + row * KV_WIDTH,
            source + ids[row] * KV_WIDTH,
            KV_WIDTH * sizeof(float)));

    enum { CARRY_WIDTH = 35, CARRY_ROWS = 3 };
    float carry[CARRY_WIDTH * CARRY_ROWS];
    float roundtrip[CARRY_WIDTH * CARRY_ROWS];
    for (int i = 0; i < CARRY_WIDTH * CARRY_ROWS; i++)
        carry[i] = bf16(random_value());
    ds4_gpu_tensor *plain = upload(carry, sizeof(carry));
    ds4_gpu_tensor *packed = upload(
        NULL, CARRY_ROWS * ((CARRY_WIDTH + 1) / 2) * sizeof(uint32_t));
    CHECK(plain && packed);
    CHECK(ds4_gpu_dsv41_carry_copy(
        packed, 0, plain, CARRY_WIDTH, CARRY_ROWS,
        DS4_V41_CARRY_BF16, true));
    CHECK(ds4_gpu_tensor_fill_f32(
        plain, NAN, CARRY_WIDTH * CARRY_ROWS));
    CHECK(ds4_gpu_dsv41_carry_copy(
        packed, 0, plain, CARRY_WIDTH, CARRY_ROWS,
        DS4_V41_CARRY_BF16, false));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(plain, 0, roundtrip, sizeof(roundtrip)));
    CHECK(!memcmp(carry, roundtrip, sizeof(carry)));

    enum { INDEX_ROWS = 2, KEYS = 4, HEADS = 32, DIM = 128 };
    float q[INDEX_ROWS * HEADS * DIM];
    float weights[INDEX_ROWS * HEADS];
    float keys[KEYS * DIM];
    float index_scores[INDEX_ROWS * KEYS];
    for (int i = 0; i < INDEX_ROWS * HEADS * DIM; i++)
        q[i] = (float)((i % 7) - 3) / 4.0f;
    for (int i = 0; i < INDEX_ROWS * HEADS; i++)
        weights[i] = 1.0f / HEADS;
    for (int i = 0; i < KEYS * DIM; i++)
        keys[i] = (float)((i % 5) - 2) / 3.0f;
    ds4_gpu_tensor *q_t = upload(q, sizeof(q));
    ds4_gpu_tensor *weights_t = upload(weights, sizeof(weights));
    ds4_gpu_tensor *keys_t = upload(keys, sizeof(keys));
    ds4_gpu_tensor *scores_t = upload(NULL, sizeof(index_scores));
    CHECK(q_t && weights_t && keys_t && scores_t);
    CHECK(ds4_gpu_dsv41_indexer_scores_batch(
        scores_t, q_t, weights_t, keys_t, KEYS, INDEX_ROWS, 2, 1));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(
        scores_t, 0, index_scores, sizeof(index_scores)));
    CHECK(isinf(index_scores[3]) && index_scores[3] < 0.0f);
    for (int i = 0; i < INDEX_ROWS * KEYS; i++)
        CHECK(isfinite(index_scores[i]) || index_scores[i] == -INFINITY);

    ds4_gpu_tensor_free(scores_t);
    ds4_gpu_tensor_free(keys_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(packed);
    ds4_gpu_tensor_free(plain);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(ids_t);
    ds4_gpu_tensor_free(source_t);
    return 1;
}

int main(void) {
    CHECK(ds4_gpu_init());
    CHECK(check_quantization());
    CHECK(check_rope());
    CHECK(check_pool_and_candidates());
    CHECK(check_gather_carry_and_indexer());
    ds4_gpu_cleanup();
    puts("CUDA V4.1 layout, RoPE, CSA2 and KV helpers: PASS");
    return 0;
}
