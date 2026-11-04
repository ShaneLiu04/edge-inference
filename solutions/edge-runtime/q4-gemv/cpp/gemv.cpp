// Matrix-vector products over block-quantized weights, the inner loop of LLM decoding on a CPU.
//
//   gemv --mode=bench [--threads=1] [--repeats=9]   time every format on a few layer shapes
//   gemv --mode=pack  --dir=D                      read w.f32 (rows x cols), write q4_0.bin and q8_0.bin
//   gemv --mode=check                              every kernel against a double reference
//
// Block layouts follow llama.cpp (ggml-common.h):
//   Q8_0: { fp16 d; int8  qs[32]; }            34 bytes per 32 weights, w = d * q
//   Q4_0: { fp16 d; uint8 qs[16]; }            18 bytes per 32 weights, w = d * (q - 8)
//         byte j holds element j in the low nibble and element j + 16 in the high nibble.
#include "edgebench.h"

#include <cmath>
#include <fstream>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

using namespace eb;

constexpr int QK = 32;

// ------------------------------------------------------------------ fp16 <-> fp32 (scalar, IEEE half)

static uint16_t f32_to_f16(float f) {
#if defined(__F16C__)
    return static_cast<uint16_t>(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT));
#else
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1) ++h;
    return static_cast<uint16_t>(h);
#endif
}

static float f16_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else { exp = 127 - 15 + 1; while (!(mant & 0x400)) { mant <<= 1; --exp; } mant &= 0x3ff; x = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 31) x = sign | 0x7f800000 | (mant << 13);
    else x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
#endif
}

// ------------------------------------------------------------------ packing (llama.cpp reference quantizers)

#pragma pack(push, 1)
struct BlockQ8 { uint16_t d; int8_t qs[QK]; };
struct BlockQ4 { uint16_t d; uint8_t qs[QK / 2]; };
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34, "Q8_0 block is 34 bytes");
static_assert(sizeof(BlockQ4) == 18, "Q4_0 block is 18 bytes");

static void quantize_q8_0(const float* x, BlockQ8* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        float amax = 0;
        for (int j = 0; j < QK; ++j) amax = std::max(amax, std::fabs(x[b * QK + j]));
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        y[b].d = f32_to_f16(d);
        for (int j = 0; j < QK; ++j) y[b].qs[j] = static_cast<int8_t>(std::nearbyintf(x[b * QK + j] * id));
    }
}

static void quantize_q4_0(const float* x, BlockQ4* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        float amax = 0, max = 0;  // max keeps the sign of the largest-magnitude element
        for (int j = 0; j < QK; ++j) {
            float v = x[b * QK + j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); max = v; }
        }
        float d = max / -8.0f, id = d ? 1.0f / d : 0.0f;
        y[b].d = f32_to_f16(d);
        for (int j = 0; j < QK / 2; ++j) {
            float x0 = x[b * QK + j] * id, x1 = x[b * QK + QK / 2 + j] * id;
            uint8_t q0 = static_cast<uint8_t>(std::min(15, static_cast<int>(x0 + 8.5f)));
            uint8_t q1 = static_cast<uint8_t>(std::min(15, static_cast<int>(x1 + 8.5f)));
            y[b].qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

static void dequant_q4_0_row(const BlockQ4* w, float* out, int cols) {
    for (int b = 0; b < cols / QK; ++b) {
        float d = f16_to_f32(w[b].d);
        for (int j = 0; j < QK / 2; ++j) {
            out[b * QK + j] = d * ((w[b].qs[j] & 0x0f) - 8);
            out[b * QK + QK / 2 + j] = d * ((w[b].qs[j] >> 4) - 8);
        }
    }
}

// ------------------------------------------------------------------ kernels: y[r] = sum_c W[r][c] * x[c]

#if defined(__AVX2__)
static inline __m256i unpack_q4(const uint8_t* qs) {
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0f));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0f));
    return _mm256_set_m128i(hi, lo);  // elements 0..15 then 16..31, values 0..15
}
#endif

#if defined(__AVX512F__)
static inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

static void gemv_f32(int rows, int cols, const float* W, const float* x, float* y, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float* w = W + static_cast<size_t>(r) * cols;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + c), _mm512_loadu_ps(x + c), a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + c + 16), _mm512_loadu_ps(x + c + 16), a1);
        }
        float s = hsum(_mm512_add_ps(a0, a1));
        for (; c < cols; ++c) s += w[c] * x[c];
        y[r] = s;
    }
    (void)rows;
}

static void gemv_f16(int rows, int cols, const uint16_t* W, const float* x, float* y, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t* w = W + static_cast<size_t>(r) * cols;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (int c = 0; c < cols; c += 32) {
            __m512 w0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c)));
            __m512 w1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 16)));
            a0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + c), a0);
            a1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + c + 16), a1);
        }
        y[r] = hsum(_mm512_add_ps(a0, a1));
    }
    (void)rows;
}

static void gemv_q8_f32(int rows, int cols, const BlockQ8* W, const float* x, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ8* w = W + static_cast<size_t>(r) * nb;
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[b].qs));
            __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[b].qs + 16));
            __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0));
            __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1));
            __m512 t = _mm512_fmadd_ps(f0, _mm512_loadu_ps(x + b * QK), _mm512_mul_ps(f1, _mm512_loadu_ps(x + b * QK + 16)));
            acc = _mm512_fmadd_ps(_mm512_set1_ps(f16_to_f32(w[b].d)), t, acc);
        }
        y[r] = hsum(acc);
    }
    (void)rows;
}

static void gemv_q4_f32(int rows, int cols, const BlockQ4* W, const float* x, float* y, int r0, int r1) {
    int nb = cols / QK;
    const __m512 eight = _mm512_set1_ps(8.0f);
    for (int r = r0; r < r1; ++r) {
        const BlockQ4* w = W + static_cast<size_t>(r) * nb;
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m256i q = unpack_q4(w[b].qs);
            __m512 f0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(q))), eight);
            __m512 f1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(q, 1))), eight);
            __m512 t = _mm512_fmadd_ps(f0, _mm512_loadu_ps(x + b * QK), _mm512_mul_ps(f1, _mm512_loadu_ps(x + b * QK + 16)));
            acc = _mm512_fmadd_ps(_mm512_set1_ps(f16_to_f32(w[b].d)), t, acc);
        }
        y[r] = hsum(acc);
    }
    (void)rows;
}
#else
// Portable versions with the same summation shape; the timing columns are only meaningful with AVX-512.
static void gemv_f32(int rows, int cols, const float* W, const float* x, float* y, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float* w = W + static_cast<size_t>(r) * cols;
        float s = 0;
        for (int c = 0; c < cols; ++c) s += w[c] * x[c];
        y[r] = s;
    }
    (void)rows;
}

static void gemv_f16(int rows, int cols, const uint16_t* W, const float* x, float* y, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t* w = W + static_cast<size_t>(r) * cols;
        float s = 0;
        for (int c = 0; c < cols; ++c) s += f16_to_f32(w[c]) * x[c];
        y[r] = s;
    }
    (void)rows;
}

static void gemv_q8_f32(int rows, int cols, const BlockQ8* W, const float* x, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ8* w = W + static_cast<size_t>(r) * nb;
        float s = 0;
        for (int b = 0; b < nb; ++b) {
            float t = 0;
            for (int j = 0; j < QK; ++j) t += w[b].qs[j] * x[b * QK + j];
            s += f16_to_f32(w[b].d) * t;
        }
        y[r] = s;
    }
    (void)rows;
}

static void gemv_q4_f32(int rows, int cols, const BlockQ4* W, const float* x, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ4* w = W + static_cast<size_t>(r) * nb;
        float s = 0;
        for (int b = 0; b < nb; ++b) {
            float t = 0;
            for (int j = 0; j < QK / 2; ++j)
                t += ((w[b].qs[j] & 0x0f) - 8) * x[b * QK + j] + ((w[b].qs[j] >> 4) - 8) * x[b * QK + QK / 2 + j];
            s += f16_to_f32(w[b].d) * t;
        }
        y[r] = s;
    }
    (void)rows;
}
#endif

// Two passes: write the dequantized row to a buffer, then an fp32 dot. What a naive port does.
static void gemv_q4_dequant_first(int rows, int cols, const BlockQ4* W, const float* x, float* y, int r0, int r1) {
    std::vector<float> row(cols);
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        dequant_q4_0_row(W + static_cast<size_t>(r) * nb, row.data(), cols);
        gemv_f32(1, cols, row.data(), x, y + r, 0, 1);
    }
    (void)rows;
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
// Activation quantized to Q8_0 once per call; the block dot is integer. llama.cpp's vec_dot_q4_0_q8_0.
// dpbusd wants u8 x s8: the nibble q in 0..15 is the u8 side, and sum((q - 8) x) = sum(q x) - 8 sum(x).
static void gemv_q4_q8_vnni(int rows, int cols, const BlockQ4* W, const BlockQ8* xq, const int32_t* xsum, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ4* w = W + static_cast<size_t>(r) * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m256i q = unpack_q4(w[b].qs);
            __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].qs));
            __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q, xv);
            dot = _mm256_sub_epi32(dot, _mm256_set1_epi32(xsum[b]));  // 8 * sum(x) spread as sum(x) over 8 lanes
            float d = f16_to_f32(w[b].d) * f16_to_f32(xq[b].d);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), acc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[r] = _mm_cvtss_f32(s);
    }
    (void)rows;
}

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(_mm_hadd_ps(s, s));
}

static inline __m256 q4q8_block(const BlockQ4& w, const BlockQ8& x, int32_t xsum, __m256 acc) {
    __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), unpack_q4(w.qs),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs)));
    dot = _mm256_sub_epi32(dot, _mm256_set1_epi32(xsum));
    return _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32(w.d) * f16_to_f32(x.d)), _mm256_cvtepi32_ps(dot), acc);
}

// Same block dot with four independent accumulators. With one, every block's FMA waits for the
// previous one (4-cycle latency), which caps a row at one block per 4 cycles regardless of bandwidth.
static void gemv_q4_q8_vnni_4acc(int rows, int cols, const BlockQ4* W, const BlockQ8* xq, const int32_t* xsum, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ4* w = W + static_cast<size_t>(r) * nb;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int b = 0;
        for (; b + 4 <= nb; b += 4) {
            a0 = q4q8_block(w[b], xq[b], xsum[b], a0);
            a1 = q4q8_block(w[b + 1], xq[b + 1], xsum[b + 1], a1);
            a2 = q4q8_block(w[b + 2], xq[b + 2], xsum[b + 2], a2);
            a3 = q4q8_block(w[b + 3], xq[b + 3], xsum[b + 3], a3);
        }
        for (; b < nb; ++b) a0 = q4q8_block(w[b], xq[b], xsum[b], a0);
        y[r] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    (void)rows;
}

static inline __m256 q8q8_block(const BlockQ8& w, const BlockQ8& x, __m256 acc) {
    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.qs));
    __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs));
    __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
    return _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32(w.d) * f16_to_f32(x.d)), _mm256_cvtepi32_ps(dot), acc);
}

static void gemv_q8_q8_vnni_4acc(int rows, int cols, const BlockQ8* W, const BlockQ8* xq, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ8* w = W + static_cast<size_t>(r) * nb;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int b = 0;
        for (; b + 4 <= nb; b += 4) {
            a0 = q8q8_block(w[b], xq[b], a0);
            a1 = q8q8_block(w[b + 1], xq[b + 1], a1);
            a2 = q8q8_block(w[b + 2], xq[b + 2], a2);
            a3 = q8q8_block(w[b + 3], xq[b + 3], a3);
        }
        for (; b < nb; ++b) a0 = q8q8_block(w[b], xq[b], a0);
        y[r] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    (void)rows;
}

// s8 x s8 through dpbusd: |w| as u8 and sign(x, w) as s8. llama.cpp's mul_sum_i8_pairs.
static void gemv_q8_q8_vnni(int rows, int cols, const BlockQ8* W, const BlockQ8* xq, float* y, int r0, int r1) {
    int nb = cols / QK;
    for (int r = r0; r < r1; ++r) {
        const BlockQ8* w = W + static_cast<size_t>(r) * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].qs));
            __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].qs));
            __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            float d = f16_to_f32(w[b].d) * f16_to_f32(xq[b].d);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), acc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[r] = _mm_cvtss_f32(s);
    }
    (void)rows;
}
#endif

// ------------------------------------------------------------------ driver

struct Shape { const char* name; int rows, cols; };

struct Weights {
    int rows, cols;
    std::vector<float> f32;
    std::vector<uint16_t> f16;
    std::vector<BlockQ8> q8;
    std::vector<BlockQ4> q4;
    std::vector<float> deq_q8, deq_q4;  // what the quantized kernels should compute exactly
};

static Weights make_weights(int rows, int cols, unsigned seed) {
    Weights w{rows, cols, {}, {}, {}, {}, {}, {}};
    size_t n = static_cast<size_t>(rows) * cols;
    w.f32.resize(n);
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (size_t i = 0; i < n; ++i) {  // Gaussian-ish with occasional outliers, like trained LLM weights
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u1 = ((s >> 11) + 0.5) / 9007199254740992.0;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u2 = ((s >> 11) + 0.5) / 9007199254740992.0;
        double g = std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        w.f32[i] = static_cast<float>(0.02 * g * ((i % 977) == 0 ? 8 : 1));
    }
    w.f16.resize(n);
    for (size_t i = 0; i < n; ++i) w.f16[i] = f32_to_f16(w.f32[i]);
    w.q8.resize(n / QK);
    w.q4.resize(n / QK);
    quantize_q8_0(w.f32.data(), w.q8.data(), static_cast<int64_t>(n));
    quantize_q4_0(w.f32.data(), w.q4.data(), static_cast<int64_t>(n));
    w.deq_q4.resize(n);
    w.deq_q8.resize(n);
    for (int r = 0; r < rows; ++r) dequant_q4_0_row(w.q4.data() + static_cast<size_t>(r) * (cols / QK), w.deq_q4.data() + static_cast<size_t>(r) * cols, cols);
    for (size_t b = 0; b < n / QK; ++b)
        for (int j = 0; j < QK; ++j) w.deq_q8[b * QK + j] = f16_to_f32(w.q8[b].d) * w.q8[b].qs[j];
    return w;
}

static std::vector<double> ref_gemv(const std::vector<float>& W, int rows, int cols, const std::vector<float>& x) {
    std::vector<double> y(rows);
    for (int r = 0; r < rows; ++r) {
        double s = 0;
        for (int c = 0; c < cols; ++c) s += static_cast<double>(W[static_cast<size_t>(r) * cols + c]) * x[c];
        y[r] = s;
    }
    return y;
}

static double rel_err(const std::vector<double>& ref, const float* y) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { num += (y[i] - ref[i]) * (y[i] - ref[i]); den += ref[i] * ref[i]; }
    return std::sqrt(num / std::max(den, 1e-300));
}

struct Method {
    const char* name;
    double weight_bytes;
    const std::vector<float>* exact;  // weights this kernel represents exactly (null: approximate activation)
    std::function<void(int, int)> run;  // rows [r0, r1)
};

static void run_methods(const Shape& s, int threads, int repeats, bool timing) {
    Weights w = make_weights(s.rows, s.cols, 1);
    std::vector<float> x(s.cols), y(s.rows);
    for (int c = 0; c < s.cols; ++c) x[c] = static_cast<float>(std::sin(0.37 * c) + 0.1 * std::cos(1.3 * c));
    int nb = s.cols / QK;
    std::vector<BlockQ8> xq(nb);
    std::vector<int32_t> xsum(nb);
    auto quant_x = [&] {
        quantize_q8_0(x.data(), xq.data(), s.cols);
        for (int b = 0; b < nb; ++b) {
            int32_t t = 0;
            for (int j = 0; j < QK; ++j) t += xq[b].qs[j];
            xsum[b] = t;  // 8 * sum spread evenly over the 8 int32 lanes = sum per lane
        }
    };
    quant_x();
    double n = static_cast<double>(s.rows) * s.cols;
    std::vector<Method> ms = {
        {"fp32", 4 * n, &w.f32, [&](int a, int b) { gemv_f32(s.rows, s.cols, w.f32.data(), x.data(), y.data(), a, b); }},
        {"fp16", 2 * n, nullptr, [&](int a, int b) { gemv_f16(s.rows, s.cols, w.f16.data(), x.data(), y.data(), a, b); }},
        {"q8_0_fused_f32", 34.0 / 32 * n, &w.deq_q8, [&](int a, int b) { gemv_q8_f32(s.rows, s.cols, w.q8.data(), x.data(), y.data(), a, b); }},
        {"q4_0_dequant_first", 18.0 / 32 * n, &w.deq_q4, [&](int a, int b) { gemv_q4_dequant_first(s.rows, s.cols, w.q4.data(), x.data(), y.data(), a, b); }},
        {"q4_0_fused_f32", 18.0 / 32 * n, &w.deq_q4, [&](int a, int b) { gemv_q4_f32(s.rows, s.cols, w.q4.data(), x.data(), y.data(), a, b); }},
    };
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    ms.push_back({"q8_0_x_q8_vnni", 34.0 / 32 * n, nullptr, [&](int a, int b) { gemv_q8_q8_vnni(s.rows, s.cols, w.q8.data(), xq.data(), y.data(), a, b); }});
    ms.push_back({"q4_0_x_q8_vnni", 18.0 / 32 * n, nullptr, [&](int a, int b) { gemv_q4_q8_vnni(s.rows, s.cols, w.q4.data(), xq.data(), xsum.data(), y.data(), a, b); }});
    ms.push_back({"q8_0_x_q8_vnni_4acc", 34.0 / 32 * n, nullptr, [&](int a, int b) { gemv_q8_q8_vnni_4acc(s.rows, s.cols, w.q8.data(), xq.data(), y.data(), a, b); }});
    ms.push_back({"q4_0_x_q8_vnni_4acc", 18.0 / 32 * n, nullptr, [&](int a, int b) { gemv_q4_q8_vnni_4acc(s.rows, s.cols, w.q4.data(), xq.data(), xsum.data(), y.data(), a, b); }});
#endif
    auto ref_true = ref_gemv(w.f32, s.rows, s.cols, x);
    std::vector<double> ref_q4 = ref_gemv(w.deq_q4, s.rows, s.cols, x), ref_q8 = ref_gemv(w.deq_q8, s.rows, s.cols, x);
    for (Method& m : ms) {
        auto call = [&] {
            if (std::string(m.name).find("x_q8") != std::string::npos) quant_x();
            parallel_for(threads, s.rows, [&](int64_t lo, int64_t hi, int) { m.run(static_cast<int>(lo), static_cast<int>(hi)); });
        };
        call();
        double err_true = rel_err(ref_true, y.data());
        double err_exact = -1;
        if (m.exact == &w.deq_q4) err_exact = rel_err(ref_q4, y.data());
        else if (m.exact == &w.deq_q8) err_exact = rel_err(ref_q8, y.data());
        else if (m.exact == &w.f32) err_exact = err_true;
        Json j;
        j.s("kind", timing ? "gemv" : "check").s("shape", s.name).i("rows", s.rows).i("cols", s.cols).s("method", m.name)
            .i("threads", threads).f("weight_bytes", m.weight_bytes).f("rel_err_vs_fp32", err_true).f("rel_err_vs_own_weights", err_exact);
        if (timing) {
            Stats st = time_fn(call, 3, repeats, 5e-2);
            j.f("weight_GBps", m.weight_bytes / st.p50 / 1e9).f("us", st.p50 * 1e6).s("isa", isa()).stats(st);
        }
        j.emit();
    }
}

static void pack_mode(const std::string& dir) {
    std::ifstream meta(dir + "/shape.txt");
    int rows, cols;
    meta >> rows >> cols;
    size_t n = static_cast<size_t>(rows) * cols;
    std::vector<float> w(n);
    std::ifstream(dir + "/w.f32", std::ios::binary).read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(n * 4));
    std::vector<BlockQ4> q4(n / QK);
    std::vector<BlockQ8> q8(n / QK);
    quantize_q4_0(w.data(), q4.data(), static_cast<int64_t>(n));
    quantize_q8_0(w.data(), q8.data(), static_cast<int64_t>(n));
    std::ofstream(dir + "/q4_0.bin", std::ios::binary).write(reinterpret_cast<const char*>(q4.data()), static_cast<std::streamsize>(q4.size() * sizeof(BlockQ4)));
    std::ofstream(dir + "/q8_0.bin", std::ios::binary).write(reinterpret_cast<const char*>(q8.data()), static_cast<std::streamsize>(q8.size() * sizeof(BlockQ8)));
    Json().s("kind", "pack").i("rows", rows).i("cols", cols).emit();
}

int main(int argc, char** argv) {
    std::string mode = arg_s(argc, argv, "mode", "check");
    // Decoder layers of a 1B-class and a 7B-class model, plus a GPT-2 small MLP.
    const Shape shapes[] = {{"gpt2_mlp_768x3072", 3072, 768}, {"llama1b_2048x2048", 2048, 2048},
                            {"llama1b_mlp_8192x2048", 8192, 2048}, {"llama7b_4096x4096", 4096, 4096},
                            // fp32 180 MB: larger than a 105 MB server L3, so fp32 / fp16 rows are DRAM-bound there.
                            {"llama7b_mlp_11008x4096", 11008, 4096}};
    if (mode == "bench") {
        int threads = static_cast<int>(arg_ll(argc, argv, "threads", 1));
        int repeats = static_cast<int>(arg_ll(argc, argv, "repeats", 9));
        for (const Shape& s : shapes) run_methods(s, threads, repeats, true);
    } else if (mode == "check") {
        run_methods({"check_96x256", 96, 256}, 1, 1, false);
        run_methods({"check_67x1024", 67, 1024}, 3, 1, false);
    } else if (mode == "pack") pack_mode(arg_s(argc, argv, "dir", "."));
    else { std::fprintf(stderr, "unknown mode\n"); return 1; }
    return 0;
}
