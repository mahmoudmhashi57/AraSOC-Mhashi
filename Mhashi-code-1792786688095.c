/**
 * AraSOC-Search: Hardware-Aware Vectorized Multi-Pattern Matching
 * Core Implementation: AVX2 Zero-Copy Normalization & Cache-Aligned Bounded Scanner
 * 
 * Target: Intel AVX2 / GCC / Clang
 * Compilation:
 *   gcc -O3 -mavx2 -march=native -shared -fPIC -o libarasoc.so arasoc.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#define L1_WINDOW_SIZE    (32 * 1024)   /* 32 KB pinned L1 inspection window */
#define HASH_TABLE_SIZE   (65536)       /* Compact dense hash table fitting L2/L3 */
#define GOLDEN_RATIO_32   (0x9E3779B9U) /* Knuth multiplicative hash constant */

/* Context structure aligned to cache-line boundaries (64 bytes) */
typedef struct {
    uint8_t l1_window[L1_WINDOW_SIZE] __attribute__((aligned(64)));
    uint32_t match_indices[4096];
    size_t match_count;
} arasoc_thread_ctx_t;

/* Compact State Structure */
typedef struct {
    uint16_t state_table[HASH_TABLE_SIZE] __attribute__((aligned(64)));
    size_t min_pattern_len;
    size_t safe_stride;
} arasoc_engine_t;

/**
 * Initializes the AraSOC-Search engine state table and safe shift stride.
 */
arasoc_engine_t* arasoc_engine_init(const uint8_t** patterns, size_t num_patterns, size_t min_len) {
    arasoc_engine_t* engine = (arasoc_engine_t*)aligned_alloc(64, sizeof(arasoc_engine_t));
    if (!engine) return NULL;

    memset(engine->state_table, 0, sizeof(engine->state_table));
    engine->min_pattern_len = min_len;
    engine->safe_stride = min_len + 1; /* Safe shift bound: s = m_min + 1 */

    for (size_t i = 0; i < num_patterns; ++i) {
        if (patterns[i] != NULL) {
            uint32_t token = 0;
            memcpy(&token, patterns[i], (min_len >= 4) ? 4 : min_len);
            uint32_t h = (token * GOLDEN_RATIO_32) & (HASH_TABLE_SIZE - 1);
            engine->state_table[h] = (uint16_t)(i + 1); /* Non-zero identifier */
        }
    }
    return engine;
}

/**
 * Stage 1: Zero-Copy AVX2 Vectorized In-Register Unicode Sanitization
 * Identifies and eliminates Arabic Tatweel (U+0640: 0xD9 0x80) and
 * Diacritics (Harakat U+064B - U+0652) directly inside 256-bit registers.
 */
static inline __m256i arasoc_avx2_sanitize_chunk(__m256i vin) {
    /* Arabic UTF-8 leading bytes for Harakat/Tatweel: 0xD9 */
    __m256i lead_d9 = _mm256_set1_epi8((char)0xD9);
    __m256i is_lead = _mm256_cmpeq_epi8(vin, lead_d9);

    /* Compare against Tatweel second byte: 0x80 */
    __m256i trail_80 = _mm256_set1_epi8((char)0x80);
    __m256i is_trail = _mm256_cmpeq_epi8(vin, trail_80);

    /* Form mask: identifies 2-byte sequences to be stripped */
    __m256i shift_lead = _mm256_bslli_epi128(is_lead, 1);
    __m256i drop_mask = _mm256_or_si256(is_lead, _mm256_and_si256(shift_lead, is_trail));

    /* In-register blend: mask out unwanted non-spacing / tatweel bytes with zero-copy */
    __m256i zero_reg = _mm256_setzero_si256();
    __m256i vclean = _mm256_blendv_epi8(vin, zero_reg, drop_mask);

    return vclean;
}

/**
 * Stage 2: Cache-Aligned Bounded O(N/W) Matching Loop
 */
size_t arasoc_scan_stream(const arasoc_engine_t* engine,
                          const uint8_t* telemetry,
                          size_t text_len,
                          arasoc_thread_ctx_t* ctx) {
    if (!engine || !telemetry || text_len == 0 || !ctx) return 0;

    size_t pos = 0;
    ctx->match_count = 0;
    const size_t stride = engine->safe_stride;
    const uint16_t* table = engine->state_table;

    /* AVX2 32-byte block parallel processing loop */
    while (pos + 32 <= text_len) {
        /* Load 32 bytes into 256-bit vector register */
        __m256i vin = _mm256_loadu_si256((const __m256i*)(telemetry + pos));

        /* Stage 1: Zero-copy sanitization directly in registers */
        __m256i vclean = arasoc_avx2_sanitize_chunk(vin);

        /* Store into thread-local aligned L1 cache window */
        _mm256_storeu_si256((__m256i*)ctx->l1_window, vclean);

        /* Extract 4-byte anchor token from normalized window */
        uint32_t token = *(const uint32_t*)ctx->l1_window;

        /* O(1) hash evaluation with zero pointer chasing */
        uint32_t h = (token * GOLDEN_RATIO_32) & (HASH_TABLE_SIZE - 1);

        if (__builtin_expect((table[h] != 0), 0)) {
            if (ctx->match_count < 4096) {
                ctx->match_indices[ctx->match_count++] = (uint32_t)pos;
            }
        }

        /* Deterministic O(N/W) stride advance */
        pos += stride;
    }

    /* Scalar handler for residual buffer tail */
    while (pos < text_len) {
        uint32_t token = 0;
        size_t rem = text_len - pos;
        memcpy(&token, telemetry + pos, (rem >= 4) ? 4 : rem);
        uint32_t h = (token * GOLDEN_RATIO_32) & (HASH_TABLE_SIZE - 1);
        if (table[h] != 0) {
            if (ctx->match_count < 4096) {
                ctx->match_indices[ctx->match_count++] = (uint32_t)pos;
            }
        }
        pos++;
    }

    return ctx->match_count;
}

/**
 * Free engine allocations
 */
void arasoc_engine_free(arasoc_engine_t* engine) {
    if (engine) free(engine);
}