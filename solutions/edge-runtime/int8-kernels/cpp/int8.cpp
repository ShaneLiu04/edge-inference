// Integer convolution and GEMM, the way an int8 runtime computes them.
//
//   int8 --mode=conv  --dir=D    read a layer dumped by scripts/run_experiments.py, write y_<requant>.bin
//   int8 --mode=bench            time pointwise GEMMs: fp32, int8 scalar, int8 AVX2 maddubs, int8 VNNI
//   int8 --mode=saturate         count int16 saturation of vpmaddubsw on random u8 x s8 data
//
// Conventions (ONNX QLinearConv / TFLite): x is uint8 with zero point zx, w is int8 symmetric
// (zero point 0, per output channel scale), bias is int32 in units of sx*sw[oc], y is uint8.
#include "edgebench.h"

#include <cmath>
#include <fstream>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

using namespace eb;

// ------------------------------------------------------------------ requantization

// TFLite / gemmlowp: real multiplier m in (0, 1) as a Q31 integer and a right shift.
struct FixedMul {
    int32_t m0;
    int shift;  // right shift, >= 0 for m < 1
};

static FixedMul quantize_multiplier(double m) {
    if (m <= 0) return {0, 0};
    int exp;
    double q = std::frexp(m, &exp);  // m = q * 2^exp, q in [0.5, 1)
    int64_t q31 = static_cast<int64_t>(std::llround(q * (1ll << 31)));
    if (q31 == (1ll << 31)) { q31 /= 2; ++exp; }
    return {static_cast<int32_t>(q31), -exp};
}

static int32_t srdhm(int32_t a, int32_t b) {  // SaturatingRoundingDoublingHighMul
    if (a == b && a == INT32_MIN) return INT32_MAX;
    int64_t ab = static_cast<int64_t>(a) * b;
    int64_t nudge = ab >= 0 ? (1ll << 30) : (1 - (1ll << 30));
    return static_cast<int32_t>((ab + nudge) / (1ll << 31));
}

static int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static int32_t requant_fixed(int32_t acc, FixedMul fm) {
    int left = fm.shift < 0 ? -fm.shift : 0;
    int right = fm.shift > 0 ? fm.shift : 0;
    return rounding_divide_by_pot(srdhm(acc * (1 << left), fm.m0), right);
}

// ONNX Runtime / MLAS: float multiply, round half to even.
static int32_t requant_float(int32_t acc, float m) {
    return static_cast<int32_t>(std::nearbyintf(static_cast<float>(acc) * m));
}

// Exact real arithmetic, round half away from zero.
static int32_t requant_double(int32_t acc, double m) {
    return static_cast<int32_t>(std::llround(static_cast<double>(acc) * m));
}

static uint8_t clamp_u8(int32_t v) { return static_cast<uint8_t>(std::min(255, std::max(0, v))); }

// ------------------------------------------------------------------ direct int8 conv (NHWC)

struct ConvSpec {
    int h, w, ic, oc, k, stride, pad, groups;
    int oh() const { return (h + 2 * pad - k) / stride + 1; }
    int ow() const { return (w + 2 * pad - k) / stride + 1; }
};

// acc[oc] = sum (x - zx) * w + bias. Padding contributes zero real value, i.e. x == zx.
static void conv_int32(const ConvSpec& s, const uint8_t* x, int zx, const int8_t* w, const int32_t* bias,
                       std::vector<int32_t>& acc) {
    int OH = s.oh(), OW = s.ow(), icg = s.ic / s.groups, ocg = s.oc / s.groups;
    acc.assign(static_cast<size_t>(OH) * OW * s.oc, 0);
    for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
            for (int oc = 0; oc < s.oc; ++oc) {
                int g = oc / ocg;
                int32_t a = bias[oc];
                for (int kh = 0; kh < s.k; ++kh) {
                    int ih = oh * s.stride + kh - s.pad;
                    if (ih < 0 || ih >= s.h) continue;
                    for (int kw = 0; kw < s.k; ++kw) {
                        int iw = ow * s.stride + kw - s.pad;
                        if (iw < 0 || iw >= s.w) continue;
                        const uint8_t* xp = x + (static_cast<size_t>(ih) * s.w + iw) * s.ic + g * icg;
                        // weights [OC][KH][KW][ICg]
                        const int8_t* wp = w + ((static_cast<size_t>(oc) * s.k + kh) * s.k + kw) * icg;
                        for (int ci = 0; ci < icg; ++ci) a += (static_cast<int32_t>(xp[ci]) - zx) * wp[ci];
                    }
                }
                acc[(static_cast<size_t>(oh) * OW + ow) * s.oc + oc] = a;
            }
}

template <class T>
static std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (static_cast<size_t>(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "short read %s\n", path.c_str()); std::exit(2); }
    return v;
}

template <class T>
static void write_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

// meta.txt: h w ic oc k stride pad groups zx zy sx sy, then oc weight scales.
static void conv_mode(const std::string& dir) {
    std::ifstream meta(dir + "/meta.txt");
    ConvSpec s;
    int zx, zy;
    double sx, sy;
    meta >> s.h >> s.w >> s.ic >> s.oc >> s.k >> s.stride >> s.pad >> s.groups >> zx >> zy >> sx >> sy;
    std::vector<double> sw(s.oc);
    for (int i = 0; i < s.oc; ++i) meta >> sw[i];
    auto x = read_bin<uint8_t>(dir + "/x.bin", static_cast<size_t>(s.h) * s.w * s.ic);
    auto w = read_bin<int8_t>(dir + "/w.bin", static_cast<size_t>(s.oc) * s.k * s.k * (s.ic / s.groups));
    auto b = read_bin<int32_t>(dir + "/b.bin", s.oc);
    std::vector<int32_t> acc;
    conv_int32(s, x.data(), zx, w.data(), b.data(), acc);
    size_t n = acc.size();
    std::vector<uint8_t> yf(n), yfl(n), yd(n);
    std::vector<FixedMul> fm(s.oc);
    std::vector<float> mf(s.oc);
    std::vector<double> md(s.oc);
    for (int oc = 0; oc < s.oc; ++oc) {
        md[oc] = sx * sw[oc] / sy;
        // MLAS computes the multiplier in float: (x_scale * w_scale) / y_scale.
        mf[oc] = static_cast<float>(sx) * static_cast<float>(sw[oc]) / static_cast<float>(sy);
        fm[oc] = quantize_multiplier(md[oc]);
    }
    int64_t amax = 0;
    for (size_t i = 0; i < n; ++i) {
        int oc = static_cast<int>(i % s.oc);
        amax = std::max<int64_t>(amax, std::llabs(acc[i]));
        yf[i] = clamp_u8(requant_fixed(acc[i], fm[oc]) + zy);
        yfl[i] = clamp_u8(requant_float(acc[i], mf[oc]) + zy);
        yd[i] = clamp_u8(requant_double(acc[i], md[oc]) + zy);
    }
    write_bin(dir + "/acc.bin", acc);
    write_bin(dir + "/y_fixed.bin", yf);
    write_bin(dir + "/y_float.bin", yfl);
    write_bin(dir + "/y_double.bin", yd);
    Json().s("kind", "conv").i("outputs", static_cast<long long>(n)).i("acc_absmax", amax).emit();
}

// ------------------------------------------------------------------ pointwise GEMM kernels
// Y[M][N] = X[M][K] * W[K][N]. int8 paths: X uint8 (zero point zx), W int8 stored [N][K] (K contiguous).

#if defined(__AVX512F__)
template <int R>
static inline void f32_block(int i, int j, int N, int K, const float* X, const float* Wkn, float* Y) {
    int cols = std::min(32, N - j);
    __mmask16 m0 = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
    __mmask16 m1 = cols >= 32 ? 0xFFFF : cols > 16 ? static_cast<__mmask16>((1u << (cols - 16)) - 1) : 0;
    __m512 c[R][2];
    for (int r = 0; r < R; ++r) c[r][0] = c[r][1] = _mm512_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m512 b0 = _mm512_maskz_loadu_ps(m0, Wkn + static_cast<size_t>(k) * N + j);
        __m512 b1 = _mm512_maskz_loadu_ps(m1, Wkn + static_cast<size_t>(k) * N + j + 16);
        for (int r = 0; r < R; ++r) {
            __m512 a = _mm512_set1_ps(X[static_cast<size_t>(i + r) * K + k]);
            c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
            c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm512_mask_storeu_ps(Y + static_cast<size_t>(i + r) * N + j, m0, c[r][0]);
        _mm512_mask_storeu_ps(Y + static_cast<size_t>(i + r) * N + j + 16, m1, c[r][1]);
    }
}
#endif

static void gemm_f32(int M, int N, int K, const float* X, const float* Wkn, float* Y) {
#if defined(__AVX512F__)
    for (int i = 0; i < M; i += 6) {
        int rows = std::min(6, M - i);
        for (int j = 0; j < N; j += 32) {
            switch (rows) {
                case 6: f32_block<6>(i, j, N, K, X, Wkn, Y); break;
                case 5: f32_block<5>(i, j, N, K, X, Wkn, Y); break;
                case 4: f32_block<4>(i, j, N, K, X, Wkn, Y); break;
                case 3: f32_block<3>(i, j, N, K, X, Wkn, Y); break;
                case 2: f32_block<2>(i, j, N, K, X, Wkn, Y); break;
                default: f32_block<1>(i, j, N, K, X, Wkn, Y); break;
            }
        }
    }
#else
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float a = 0;
            for (int k = 0; k < K; ++k) a += X[static_cast<size_t>(i) * K + k] * Wkn[static_cast<size_t>(k) * N + j];
            Y[static_cast<size_t>(i) * N + j] = a;
        }
#endif
}

// Raw sum x*w over K. The zero-point term zx*sum(w) is removed by the caller (precomputed per column).
static void gemm_u8s8_scalar(int M, int N, int K, const uint8_t* X, const int8_t* Wnk, int32_t* Acc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t a = 0;
            const uint8_t* x = X + static_cast<size_t>(i) * K;
            const int8_t* w = Wnk + static_cast<size_t>(j) * K;
            for (int k = 0; k < K; ++k) a += static_cast<int32_t>(x[k]) * w[k];
            Acc[static_cast<size_t>(i) * N + j] = a;
        }
}

static int32_t hsum256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// AVX2: vpmaddubsw multiplies u8*s8 and adds adjacent pairs into SATURATED int16.
static void gemm_u8s8_maddubs(int M, int N, int K, const uint8_t* X, const int8_t* Wnk, int32_t* Acc) {
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < M; ++i) {
        const uint8_t* x = X + static_cast<size_t>(i) * K;
        for (int j = 0; j < N; ++j) {
            const int8_t* w = Wnk + static_cast<size_t>(j) * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
                __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, wv), ones));
            }
            int32_t a = hsum256(acc);
            for (; k < K; ++k) a += static_cast<int32_t>(x[k]) * w[k];
            Acc[static_cast<size_t>(i) * N + j] = a;
        }
    }
#else
    gemm_u8s8_scalar(M, N, K, X, Wnk, Acc);
#endif
}

// AVX-512 VNNI: vpdpbusd accumulates 4 u8*s8 products straight into int32. No int16 step.
// 4 rows x 4 columns register block; K must be a multiple of 64 for the vector part.
static void gemm_u8s8_vnni(int M, int N, int K, const uint8_t* X, const int8_t* Wnk, int32_t* Acc) {
#if defined(__AVX512VNNI__)
    for (int i = 0; i < M; i += 4) {
        int rows = std::min(4, M - i);
        for (int j = 0; j < N; j += 4) {
            int cols = std::min(4, N - j);
            __m512i c[4][4];
            for (int r = 0; r < 4; ++r) for (int q = 0; q < 4; ++q) c[r][q] = _mm512_setzero_si512();
            int k = 0;
            for (; k + 64 <= K; k += 64) {
                __m512i wv[4];
                for (int q = 0; q < cols; ++q) wv[q] = _mm512_loadu_si512(Wnk + static_cast<size_t>(j + q) * K + k);
                for (int r = 0; r < rows; ++r) {
                    __m512i xv = _mm512_loadu_si512(X + static_cast<size_t>(i + r) * K + k);
                    for (int q = 0; q < cols; ++q) c[r][q] = _mm512_dpbusd_epi32(c[r][q], xv, wv[q]);
                }
            }
            for (int r = 0; r < rows; ++r)
                for (int q = 0; q < cols; ++q) {
                    int32_t a = _mm512_reduce_add_epi32(c[r][q]);
                    for (int kk = k; kk < K; ++kk)
                        a += static_cast<int32_t>(X[static_cast<size_t>(i + r) * K + kk]) * Wnk[static_cast<size_t>(j + q) * K + kk];
                    Acc[static_cast<size_t>(i + r) * N + j + q] = a;
                }
        }
    }
#else
    gemm_u8s8_maddubs(M, N, K, X, Wnk, Acc);
#endif
}

// oneDNN-style packing for VNNI: [N/16][K/4][16 columns][4 k]. One 32-bit broadcast of x[k..k+3]
// times one 64-byte weight load gives 16 output columns, so there is no horizontal reduction.
static std::vector<int8_t> pack_vnni(int N, int K, const int8_t* Wnk) {
    int nb = (N + 15) / 16, kb = (K + 3) / 4;
    std::vector<int8_t> p(static_cast<size_t>(nb) * kb * 64, 0);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            p[((static_cast<size_t>(n / 16) * kb + k / 4) * 16 + n % 16) * 4 + k % 4] = Wnk[static_cast<size_t>(n) * K + k];
    return p;
}

// Per-column requantization parameters in structure-of-arrays form for 16-lane loads.
struct RequantSoA {
    std::vector<int32_t> col, m0, lshift, rshift;
    int zy = 0;
    RequantSoA(const std::vector<int32_t>& col_term, const std::vector<FixedMul>& fm, int zy_)
        : col(col_term), m0(fm.size()), lshift(fm.size()), rshift(fm.size()), zy(zy_) {
        for (size_t j = 0; j < fm.size(); ++j) {
            m0[j] = fm[j].m0;
            lshift[j] = fm[j].shift < 0 ? -fm[j].shift : 0;
            rshift[j] = fm[j].shift > 0 ? fm[j].shift : 0;
        }
    }
};

#if defined(__AVX512F__)
// (x * y + nudge) / 2^31 on the sign-extended low 32 bits of each 64-bit lane.
static inline __m512i srdhm_epi64(__m512i x, __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    __m512i p = _mm512_mul_epi32(x, y);
    __mmask8 neg = _mm512_cmplt_epi64_mask(p, zero);
    p = _mm512_add_epi64(p, _mm512_mask_blend_epi64(neg, _mm512_set1_epi64(1ll << 30), _mm512_set1_epi64(1 - (1ll << 30))));
    neg = _mm512_cmplt_epi64_mask(p, zero);
    p = _mm512_mask_add_epi64(p, neg, p, _mm512_set1_epi64((1ll << 31) - 1));  // C++ division truncates toward zero
    return _mm512_srai_epi64(p, 31);
}

// Bit-exact 16-lane version of clamp(requant_fixed(acc + col) + zy).
static inline __m512i requant16(__m512i acc, const RequantSoA& q, int j, __mmask16 m) {
    __m512i a = _mm512_add_epi32(acc, _mm512_maskz_loadu_epi32(m, q.col.data() + j));
    a = _mm512_sllv_epi32(a, _mm512_maskz_loadu_epi32(m, q.lshift.data() + j));
    __m512i mul = _mm512_maskz_loadu_epi32(m, q.m0.data() + j);
    const __m512i zero = _mm512_setzero_si512();
    __m512i even = srdhm_epi64(a, mul);
    __m512i odd = srdhm_epi64(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(mul, 32));
    __m512i h = _mm512_mask_blend_epi32(0xAAAA, even, _mm512_slli_epi64(odd, 32));
    __m512i rs = _mm512_maskz_loadu_epi32(m, q.rshift.data() + j);
    __m512i mask = _mm512_sub_epi32(_mm512_sllv_epi32(_mm512_set1_epi32(1), rs), _mm512_set1_epi32(1));
    __m512i rem = _mm512_and_si512(h, mask);
    __m512i thr = _mm512_add_epi32(_mm512_srai_epi32(mask, 1), _mm512_srli_epi32(h, 31));
    __m512i y = _mm512_srav_epi32(h, rs);
    y = _mm512_mask_add_epi32(y, _mm512_cmpgt_epi32_mask(rem, thr), y, _mm512_set1_epi32(1));
    y = _mm512_add_epi32(y, _mm512_set1_epi32(q.zy));
    return _mm512_min_epi32(_mm512_max_epi32(y, zero), _mm512_set1_epi32(255));
}

static void epilogue_avx512(int M, int N, const int32_t* Acc, const RequantSoA& q, uint8_t* Y) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; j += 16) {
            int cols = std::min(16, N - j);
            __mmask16 m = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
            __m512i a = _mm512_maskz_loadu_epi32(m, Acc + static_cast<size_t>(i) * N + j);
            _mm512_mask_cvtepi32_storeu_epi8(Y + static_cast<size_t>(i) * N + j, m, requant16(a, q, j, m));
        }
}
#endif

#if defined(__AVX512VNNI__)
// R rows x 32 columns register block; R is a template parameter so the accumulators stay in zmm.
template <int R, bool Fused>
static inline void vnni_block(int i, int jb, int nb, int N, int kb, const uint8_t* X, int ldx, const int8_t* Wp,
                              int32_t* Acc, const RequantSoA* q, uint8_t* Y) {
    int blocks = std::min(2, nb - jb);
    __m512i c[R][2];
    for (int r = 0; r < R; ++r) c[r][0] = c[r][1] = _mm512_setzero_si512();
    const int8_t* w0 = Wp + static_cast<size_t>(jb) * kb * 64;
    const int8_t* w1 = blocks > 1 ? w0 + static_cast<size_t>(kb) * 64 : w0;
    const uint8_t* xr[R];
    for (int r = 0; r < R; ++r) xr[r] = X + static_cast<size_t>(i + r) * ldx;
    for (int qk = 0; qk < kb; ++qk) {
        __m512i b0 = _mm512_loadu_si512(w0 + static_cast<size_t>(qk) * 64);
        __m512i b1 = _mm512_loadu_si512(w1 + static_cast<size_t>(qk) * 64);
        for (int r = 0; r < R; ++r) {
            int32_t x4;
            std::memcpy(&x4, xr[r] + qk * 4, 4);
            __m512i a = _mm512_set1_epi32(x4);
            c[r][0] = _mm512_dpbusd_epi32(c[r][0], a, b0);
            c[r][1] = _mm512_dpbusd_epi32(c[r][1], a, b1);
        }
    }
    for (int b = 0; b < blocks; ++b) {
        int j = (jb + b) * 16;
        int cols = std::min(16, N - j);
        __mmask16 m = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
        for (int r = 0; r < R; ++r) {
            if constexpr (Fused)
                _mm512_mask_cvtepi32_storeu_epi8(Y + static_cast<size_t>(i + r) * N + j, m, requant16(c[r][b], *q, j, m));
            else
                _mm512_mask_storeu_epi32(Acc + static_cast<size_t>(i + r) * N + j, m, c[r][b]);
        }
    }
}

// X rows must be padded to a multiple of 4 bytes (the caller passes ldx >= K, zero padded).
// Fused: requantize straight from registers and write uint8, so the int32 tile never touches memory.
template <bool Fused>
static void gemm_u8s8_vnni_packed(int M, int N, int K, const uint8_t* X, int ldx, const int8_t* Wp, int32_t* Acc,
                                  const RequantSoA* q = nullptr, uint8_t* Y = nullptr) {
    int nb = (N + 15) / 16, kb = (K + 3) / 4;
    for (int i = 0; i < M; i += 4) {
        int rows = std::min(4, M - i);
        for (int jb = 0; jb < nb; jb += 2) {
            switch (rows) {
                case 4: vnni_block<4, Fused>(i, jb, nb, N, kb, X, ldx, Wp, Acc, q, Y); break;
                case 3: vnni_block<3, Fused>(i, jb, nb, N, kb, X, ldx, Wp, Acc, q, Y); break;
                case 2: vnni_block<2, Fused>(i, jb, nb, N, kb, X, ldx, Wp, Acc, q, Y); break;
                default: vnni_block<1, Fused>(i, jb, nb, N, kb, X, ldx, Wp, Acc, q, Y); break;
            }
        }
    }
}
#endif

// Zero-point correction and requantization, the epilogue every int8 GEMM needs.
static void epilogue(int M, int N, const int32_t* Acc, const int32_t* col_term, const FixedMul* fm, int zy, uint8_t* Y) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Y[static_cast<size_t>(i) * N + j] = clamp_u8(requant_fixed(Acc[static_cast<size_t>(i) * N + j] + col_term[j], fm[j]) + zy);
}

struct Shape { const char* name; int M, N, K; };

static void bench(int repeats) {
    // MobileNetV2 pointwise layers at 224x224 input: M = H*W, K = IC, N = OC.
    const Shape shapes[] = {
        {"mbv2_pw_expand_56", 56 * 56, 144, 24},
        {"mbv2_pw_project_56", 56 * 56, 24, 144},
        {"mbv2_pw_expand_14", 14 * 14, 576, 96},
        {"mbv2_pw_project_14", 14 * 14, 96, 576},
        {"mbv2_pw_head_7", 7 * 7, 1280, 320},
        {"square_512", 512, 512, 512},
    };
    for (const Shape& s : shapes) {
        std::srand(5);
        Buffer<float> xf(static_cast<size_t>(s.M) * s.K), wf(static_cast<size_t>(s.K) * s.N), yf(static_cast<size_t>(s.M) * s.N);
        Buffer<uint8_t> xq(static_cast<size_t>(s.M) * s.K), yq(static_cast<size_t>(s.M) * s.N);
        Buffer<int8_t> wq(static_cast<size_t>(s.K) * s.N);
        Buffer<int32_t> acc(static_cast<size_t>(s.M) * s.N), ref(static_cast<size_t>(s.M) * s.N);
        for (size_t i = 0; i < xf.n; ++i) { xq[i] = static_cast<uint8_t>(std::rand() % 256); xf[i] = xq[i] / 255.0f; }
        // Weights in [-63, 63] so the AVX2 path cannot saturate here; saturation is measured separately.
        for (size_t i = 0; i < wf.n; ++i) { wq[i] = static_cast<int8_t>(std::rand() % 127 - 63); }
        for (int k = 0; k < s.K; ++k) for (int n = 0; n < s.N; ++n) wf[static_cast<size_t>(k) * s.N + n] = wq[static_cast<size_t>(n) * s.K + k] / 63.0f;
        std::vector<int32_t> col(s.N);
        std::vector<FixedMul> fm(s.N);
        for (int n = 0; n < s.N; ++n) fm[n] = quantize_multiplier(2e-4 * (1 + n % 7));
        int zx = 128, zy = 3;
        for (int n = 0; n < s.N; ++n) {
            int32_t sum = 0;
            for (int k = 0; k < s.K; ++k) sum += wq[static_cast<size_t>(n) * s.K + k];
            col[n] = -zx * sum;
        }
        gemm_u8s8_scalar(s.M, s.N, s.K, xq.data(), wq.data(), ref.data());
        double macs = static_cast<double>(s.M) * s.N * s.K;
        int ldx = (s.K + 3) / 4 * 4;
        Buffer<uint8_t> xpad(static_cast<size_t>(s.M) * ldx);
        for (int i = 0; i < s.M; ++i)
            for (int k = 0; k < ldx; ++k) xpad[static_cast<size_t>(i) * ldx + k] = k < s.K ? xq[static_cast<size_t>(i) * s.K + k] : 0;
        Buffer<uint8_t> yref(static_cast<size_t>(s.M) * s.N);
        epilogue(s.M, s.N, ref.data(), col.data(), fm.data(), zy, yref.data());
        RequantSoA soa(col, fm, zy);
        std::vector<int8_t> wp;
#if defined(__AVX512VNNI__)
        wp = pack_vnni(s.N, s.K, wq.data());
#endif
        // check: 0 = none, 1 = int32 accumulators, 2 = uint8 outputs
        struct Run { const char* name; std::function<void()> fn; int check; };
        std::vector<Run> runs = {
            {"fp32_avx512", [&] { gemm_f32(s.M, s.N, s.K, xf.data(), wf.data(), yf.data()); }, 0},
            {"int8_scalar_autovec", [&] { gemm_u8s8_scalar(s.M, s.N, s.K, xq.data(), wq.data(), acc.data()); }, 1},
            {"int8_avx2_maddubs_dot", [&] { gemm_u8s8_maddubs(s.M, s.N, s.K, xq.data(), wq.data(), acc.data()); }, 1},
            {"int8_vnni_dot", [&] { gemm_u8s8_vnni(s.M, s.N, s.K, xq.data(), wq.data(), acc.data()); }, 1},
        };
#if defined(__AVX512VNNI__)
        runs.push_back({"int8_vnni_packed", [&] { gemm_u8s8_vnni_packed<false>(s.M, s.N, s.K, xpad.data(), ldx, wp.data(), acc.data()); }, 1});
        runs.push_back({"int8_vnni_packed+scalar_requant", [&] {
            gemm_u8s8_vnni_packed<false>(s.M, s.N, s.K, xpad.data(), ldx, wp.data(), acc.data());
            epilogue(s.M, s.N, acc.data(), col.data(), fm.data(), zy, yq.data()); }, 2});
        runs.push_back({"int8_vnni_packed+simd_requant", [&] {
            gemm_u8s8_vnni_packed<false>(s.M, s.N, s.K, xpad.data(), ldx, wp.data(), acc.data());
            epilogue_avx512(s.M, s.N, acc.data(), soa, yq.data()); }, 2});
        runs.push_back({"int8_vnni_packed_fused_requant", [&] {
            gemm_u8s8_vnni_packed<true>(s.M, s.N, s.K, xpad.data(), ldx, wp.data(), nullptr, &soa, yq.data()); }, 2});
#endif
        for (Run& r : runs) {
            Stats st = time_fn(r.fn, 2, repeats, 2e-2);
            bool exact = true;
            if (r.check) {
                std::fill(acc.ptr, acc.ptr + acc.n, 0);
                std::fill(yq.ptr, yq.ptr + yq.n, 0);
                r.fn();
                if (r.check == 1) { for (size_t i = 0; i < acc.n; ++i) if (acc[i] != ref[i]) { exact = false; break; } }
                else { for (size_t i = 0; i < yq.n; ++i) if (yq[i] != yref[i]) { exact = false; break; } }
            }
            Json().s("kind", "gemm").s("shape", s.name).i("M", s.M).i("N", s.N).i("K", s.K).s("method", r.name)
                .f("gops", 2 * macs / st.p50 / 1e9).b("exact", exact).s("isa", isa()).stats(st).emit();
        }
    }
}

// vpmaddubsw computes x[2i]*w[2i] + x[2i+1]*w[2i+1] in int16 with saturation.
// The pair sum reaches 2*255*127 = 64770 > 32767, so full-range int8 weights overflow.
static void saturate() {
    const int K = 256, N = 4096;
    for (int wmax : {63, 64, 100, 127}) {
        std::srand(9);
        std::vector<uint8_t> x(K);
        std::vector<int8_t> w(static_cast<size_t>(N) * K);
        for (auto& v : x) v = static_cast<uint8_t>(128 + std::rand() % 128);  // ReLU-like large activations
        for (auto& v : w) v = static_cast<int8_t>(std::rand() % (2 * wmax + 1) - wmax);
        std::vector<int32_t> a(N), b(N);
        gemm_u8s8_scalar(1, N, K, x.data(), w.data(), a.data());
        gemm_u8s8_maddubs(1, N, K, x.data(), w.data(), b.data());
        int wrong = 0;
        long long worst = 0;
        for (int j = 0; j < N; ++j) {
            if (a[j] != b[j]) ++wrong;
            worst = std::max<long long>(worst, std::llabs(static_cast<long long>(a[j]) - b[j]));
        }
        Json().s("kind", "saturate").i("wmax", wmax).i("columns", N).i("K", K).i("wrong", wrong)
            .f("wrong_frac", wrong / static_cast<double>(N)).i("worst_abs_err", worst).emit();
    }
}

int main(int argc, char** argv) {
    std::string mode = arg_s(argc, argv, "mode", "bench");
    if (mode == "conv") conv_mode(arg_s(argc, argv, "dir", "."));
    else if (mode == "bench") bench(static_cast<int>(arg_ll(argc, argv, "repeats", 9)));
    else if (mode == "saturate") saturate();
    else if (mode == "multiplier") {
        double m = std::atof(arg_s(argc, argv, "m", "0.5").c_str());
        FixedMul fm = quantize_multiplier(m);
        Json().f("m", m).i("m0", fm.m0).i("shift", fm.shift).i("acc1000", requant_fixed(1000, fm)).emit();
    } else { std::fprintf(stderr, "unknown mode\n"); return 1; }
    return 0;
}
