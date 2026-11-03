// fp32 convolution on one CPU core, five ways, on real MobileNetV2 / ResNet layer shapes.
//
//   conv --mode=check                      small random shapes, max error of every kernel vs fp64
//   conv --mode=bench --layer=NAME         time every applicable kernel on one layer
//   conv --mode=list                       print the layer table
//
// Layout: NHWC activations, weights [KH][KW][IC][OC] for dense conv and [KH][KW][C] for
// depthwise. The naive reference is NCHW with OIHW weights, so the check also covers layout.
#include "edgebench.h"

#include <cmath>
#include <map>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

using namespace eb;

struct Layer {
    std::string name;
    int h, w, ic, oc, k, stride;
    bool depthwise;
    int oh() const { return (h + 2 * (k / 2) - k) / stride + 1; }
    int ow() const { return (w + 2 * (k / 2) - k) / stride + 1; }
    double macs() const {
        double per = depthwise ? k * k : static_cast<double>(k) * k * ic;
        return per * oh() * ow() * (depthwise ? ic : oc);
    }
};

static std::vector<Layer> layers() {
    return {
        {"mbv2_stem_224", 224, 224, 3, 32, 3, 2, false},
        {"mbv2_pw_expand_56", 56, 56, 24, 144, 1, 1, false},
        {"mbv2_dw_56", 56, 56, 144, 144, 3, 1, true},
        {"mbv2_pw_project_56", 56, 56, 144, 24, 1, 1, false},
        {"mbv2_dw_s2_56", 56, 56, 144, 144, 3, 2, true},
        {"mbv2_dw_14", 14, 14, 576, 576, 3, 1, true},
        {"mbv2_pw_expand_14", 14, 14, 96, 576, 1, 1, false},
        {"resnet_3x3_56", 56, 56, 64, 64, 3, 1, false},
        {"resnet_3x3_14", 14, 14, 256, 256, 3, 1, false},
    };
}

// ------------------------------------------------------------------ reference (fp64, NCHW)

static void conv_ref(const Layer& L, const std::vector<double>& x_nchw, const std::vector<double>& w_oihw,
                     std::vector<double>& y_nchw) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow();
    int groups = L.depthwise ? L.ic : 1;
    int icg = L.ic / groups, ocg = (L.depthwise ? L.ic : L.oc) / groups;
    int OC = L.depthwise ? L.ic : L.oc;
    y_nchw.assign(static_cast<size_t>(OC) * OH * OW, 0.0);
    for (int oc = 0; oc < OC; ++oc) {
        int g = oc / ocg;
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                double acc = 0;
                for (int ci = 0; ci < icg; ++ci) {
                    int ic = g * icg + ci;
                    for (int kh = 0; kh < L.k; ++kh)
                        for (int kw = 0; kw < L.k; ++kw) {
                            int ih = oh * L.stride + kh - pad, iw = ow * L.stride + kw - pad;
                            if (ih < 0 || ih >= L.h || iw < 0 || iw >= L.w) continue;
                            acc += x_nchw[(static_cast<size_t>(ic) * L.h + ih) * L.w + iw] *
                                   w_oihw[((static_cast<size_t>(oc) * icg + ci) * L.k + kh) * L.k + kw];
                        }
                }
                y_nchw[(static_cast<size_t>(oc) * OH + oh) * OW + ow] = acc;
            }
    }
}

// ------------------------------------------------------------------ naive fp32, NCHW, scalar

static void conv_naive_nchw(const Layer& L, const float* x, const float* w, float* y) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow();
    int groups = L.depthwise ? L.ic : 1;
    int icg = L.ic / groups;
    int OC = L.depthwise ? L.ic : L.oc;
    int ocg = OC / groups;
    for (int oc = 0; oc < OC; ++oc) {
        int g = oc / ocg;
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                float acc = 0;
                for (int ci = 0; ci < icg; ++ci) {
                    int ic = g * icg + ci;
                    for (int kh = 0; kh < L.k; ++kh) {
                        int ih = oh * L.stride + kh - pad;
                        if (ih < 0 || ih >= L.h) continue;
                        for (int kw = 0; kw < L.k; ++kw) {
                            int iw = ow * L.stride + kw - pad;
                            if (iw < 0 || iw >= L.w) continue;
                            acc += x[(static_cast<size_t>(ic) * L.h + ih) * L.w + iw] *
                                   w[((static_cast<size_t>(oc) * icg + ci) * L.k + kh) * L.k + kw];
                        }
                    }
                }
                y[(static_cast<size_t>(oc) * OH + oh) * OW + ow] = acc;
            }
    }
}

// ------------------------------------------------------------------ GEMM micro-kernel
// C[M][N] = A[M][K] * B[K][N], row-major. R rows x 32 columns of accumulators; R is a template
// parameter so the 2R accumulators stay in zmm registers instead of spilling.

#if defined(__AVX512F__)
template <int R>
static inline void gemm_block(int i, int j, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    int cols = std::min(32, N - j);
    __mmask16 m0 = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
    __mmask16 m1 = cols >= 32 ? 0xFFFF : cols > 16 ? static_cast<__mmask16>((1u << (cols - 16)) - 1) : 0;
    __m512 c[R][2];
    for (int r = 0; r < R; ++r) c[r][0] = c[r][1] = _mm512_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m512 b0 = _mm512_maskz_loadu_ps(m0, B + static_cast<size_t>(k) * ldb + j);
        __m512 b1 = _mm512_maskz_loadu_ps(m1, B + static_cast<size_t>(k) * ldb + j + 16);
        for (int r = 0; r < R; ++r) {
            __m512 a = _mm512_set1_ps(A[static_cast<size_t>(i + r) * lda + k]);
            c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
            c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm512_mask_storeu_ps(C + static_cast<size_t>(i + r) * ldc + j, m0, c[r][0]);
        _mm512_mask_storeu_ps(C + static_cast<size_t>(i + r) * ldc + j + 16, m1, c[r][1]);
    }
}
#endif

static void gemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
#if defined(__AVX512F__)
    for (int i = 0; i < M; i += 6) {
        int rows = std::min(6, M - i);
        for (int j = 0; j < N; j += 32) {
            switch (rows) {
                case 6: gemm_block<6>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
                case 5: gemm_block<5>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
                case 4: gemm_block<4>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
                case 3: gemm_block<3>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
                case 2: gemm_block<2>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
                default: gemm_block<1>(i, j, N, K, A, lda, B, ldb, C, ldc); break;
            }
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        float* c = C + static_cast<size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) c[j] = 0;
        for (int k = 0; k < K; ++k) {
            float a = A[static_cast<size_t>(i) * lda + k];
            const float* b = B + static_cast<size_t>(k) * ldb;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    }
#endif
}

// ------------------------------------------------------------------ im2col + GEMM (NHWC)

static size_t im2col_bytes(const Layer& L) {
    return static_cast<size_t>(L.oh()) * L.ow() * L.k * L.k * L.ic * sizeof(float);
}

static void conv_im2col(const Layer& L, const float* x, const float* w, float* y, float* col) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow(), K = L.k * L.k * L.ic;
    for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            float* row = col + (static_cast<size_t>(oh) * OW + ow) * K;
            for (int kh = 0; kh < L.k; ++kh)
                for (int kw = 0; kw < L.k; ++kw) {
                    int ih = oh * L.stride + kh - pad, iw = ow * L.stride + kw - pad;
                    float* dst = row + (kh * L.k + kw) * L.ic;
                    if (ih < 0 || ih >= L.h || iw < 0 || iw >= L.w) {
                        std::memset(dst, 0, sizeof(float) * L.ic);
                    } else {
                        std::memcpy(dst, x + (static_cast<size_t>(ih) * L.w + iw) * L.ic, sizeof(float) * L.ic);
                    }
                }
        }
    gemm(OH * OW, L.oc, K, col, K, w, L.oc, y, L.oc);
}

// 1x1 stride 1: the NHWC input already is the im2col matrix.
static void conv_pointwise(const Layer& L, const float* x, const float* w, float* y) {
    gemm(L.h * L.w, L.oc, L.ic, x, L.ic, w, L.oc, y, L.oc);
}

// ------------------------------------------------------------------ implicit GEMM, no workspace

static void conv_implicit(const Layer& L, const float* x, const float* w, float* y) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow(), N = L.oc;
    int P = OH * OW;
#if defined(__AVX512F__)
    // Padding taps read a zero row instead of branching, so the 4-row block is always full;
    // tail rows duplicate the last pixel and are simply not stored.
    std::vector<float> zero_row(static_cast<size_t>(L.ic), 0.0f);
    for (int p0 = 0; p0 < P; p0 += 4) {
        int rows = std::min(4, P - p0);
        int ohs[4], ows[4];
        for (int r = 0; r < 4; ++r) { int p = std::min(p0 + r, P - 1); ohs[r] = p / OW; ows[r] = p % OW; }
        for (int j = 0; j < N; j += 32) {
            int cols = std::min(32, N - j);
            __mmask16 m0 = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
            __mmask16 m1 = cols >= 32 ? 0xFFFF : cols > 16 ? static_cast<__mmask16>((1u << (cols - 16)) - 1) : 0;
            __m512 c[4][2];
            for (int r = 0; r < 4; ++r) c[r][0] = c[r][1] = _mm512_setzero_ps();
            for (int kh = 0; kh < L.k; ++kh)
                for (int kw = 0; kw < L.k; ++kw) {
                    const float* src[4];
                    for (int r = 0; r < 4; ++r) {
                        int ih = ohs[r] * L.stride + kh - pad, iw = ows[r] * L.stride + kw - pad;
                        src[r] = (ih < 0 || ih >= L.h || iw < 0 || iw >= L.w) ? zero_row.data()
                                 : x + (static_cast<size_t>(ih) * L.w + iw) * L.ic;
                    }
                    const float* wk = w + static_cast<size_t>((kh * L.k + kw) * L.ic) * N;
                    for (int ic = 0; ic < L.ic; ++ic) {
                        __m512 b0 = _mm512_maskz_loadu_ps(m0, wk + static_cast<size_t>(ic) * N + j);
                        __m512 b1 = _mm512_maskz_loadu_ps(m1, wk + static_cast<size_t>(ic) * N + j + 16);
                        for (int r = 0; r < 4; ++r) {
                            __m512 a = _mm512_set1_ps(src[r][ic]);
                            c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
                            c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
                        }
                    }
                }
            for (int r = 0; r < rows; ++r) {
                _mm512_mask_storeu_ps(y + static_cast<size_t>(p0 + r) * N + j, m0, c[r][0]);
                _mm512_mask_storeu_ps(y + static_cast<size_t>(p0 + r) * N + j + 16, m1, c[r][1]);
            }
        }
    }
#else
    for (int p = 0; p < P; ++p) {
        int oh = p / OW, ow = p % OW;
        float* out = y + static_cast<size_t>(p) * N;
        for (int j = 0; j < N; ++j) out[j] = 0;
        for (int kh = 0; kh < L.k; ++kh)
            for (int kw = 0; kw < L.k; ++kw) {
                int ih = oh * L.stride + kh - pad, iw = ow * L.stride + kw - pad;
                if (ih < 0 || ih >= L.h || iw < 0 || iw >= L.w) continue;
                const float* in = x + (static_cast<size_t>(ih) * L.w + iw) * L.ic;
                const float* wk = w + static_cast<size_t>((kh * L.k + kw) * L.ic) * N;
                for (int ic = 0; ic < L.ic; ++ic)
                    for (int j = 0; j < N; ++j) out[j] += in[ic] * wk[static_cast<size_t>(ic) * N + j];
            }
    }
#endif
}

// ------------------------------------------------------------------ depthwise

// What a framework without a depthwise kernel does: one channel at a time, CHW, scalar.
static void dw_generic_chw(const Layer& L, const float* x_chw, const float* w_ckk, float* y_chw) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow();
    for (int c = 0; c < L.ic; ++c)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                float acc = 0;
                for (int kh = 0; kh < L.k; ++kh) {
                    int ih = oh * L.stride + kh - pad;
                    if (ih < 0 || ih >= L.h) continue;
                    for (int kw = 0; kw < L.k; ++kw) {
                        int iw = ow * L.stride + kw - pad;
                        if (iw < 0 || iw >= L.w) continue;
                        acc += x_chw[(static_cast<size_t>(c) * L.h + ih) * L.w + iw] * w_ckk[(c * L.k + kh) * L.k + kw];
                    }
                }
                y_chw[(static_cast<size_t>(c) * OH + oh) * OW + ow] = acc;
            }
}

// NHWC, vectorized across channels: every tap is one contiguous load of C floats.
static void dw_nhwc(const Layer& L, const float* x, const float* w_kkc, float* y) {
    int pad = L.k / 2, OH = L.oh(), OW = L.ow(), C = L.ic;
    for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            float* out = y + (static_cast<size_t>(oh) * OW + ow) * C;
#if defined(__AVX512F__)
            for (int c = 0; c < C; c += 16) {
                __mmask16 m = C - c >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << (C - c)) - 1);
                __m512 acc = _mm512_setzero_ps();
                for (int kh = 0; kh < L.k; ++kh) {
                    int ih = oh * L.stride + kh - pad;
                    if (ih < 0 || ih >= L.h) continue;
                    for (int kw = 0; kw < L.k; ++kw) {
                        int iw = ow * L.stride + kw - pad;
                        if (iw < 0 || iw >= L.w) continue;
                        __m512 xv = _mm512_maskz_loadu_ps(m, x + (static_cast<size_t>(ih) * L.w + iw) * C + c);
                        __m512 wv = _mm512_maskz_loadu_ps(m, w_kkc + (kh * L.k + kw) * C + c);
                        acc = _mm512_fmadd_ps(xv, wv, acc);
                    }
                }
                _mm512_mask_storeu_ps(out + c, m, acc);
            }
#else
            for (int c = 0; c < C; ++c) out[c] = 0;
            for (int kh = 0; kh < L.k; ++kh) {
                int ih = oh * L.stride + kh - pad;
                if (ih < 0 || ih >= L.h) continue;
                for (int kw = 0; kw < L.k; ++kw) {
                    int iw = ow * L.stride + kw - pad;
                    if (iw < 0 || iw >= L.w) continue;
                    const float* in = x + (static_cast<size_t>(ih) * L.w + iw) * C;
                    const float* wk = w_kkc + (kh * L.k + kw) * C;
                    for (int c = 0; c < C; ++c) out[c] += in[c] * wk[c];
                }
            }
#endif
        }
}

// ------------------------------------------------------------------ Winograd F(2x2, 3x3), stride 1

struct WinogradPlan {
    int tiles_h, tiles_w, tiles;
    size_t bytes_U, bytes_V, bytes_M;
};

static WinogradPlan wino_plan(const Layer& L) {
    WinogradPlan p;
    p.tiles_h = (L.oh() + 1) / 2;
    p.tiles_w = (L.ow() + 1) / 2;
    p.tiles = p.tiles_h * p.tiles_w;
    p.bytes_U = 16ull * L.ic * L.oc * sizeof(float);
    p.bytes_V = 16ull * p.tiles * L.ic * sizeof(float);
    p.bytes_M = 16ull * p.tiles * L.oc * sizeof(float);
    return p;
}

// U[16][IC][OC] = G g G^T, done once per weight set (offline in a real runtime).
static void wino_weights(const Layer& L, const float* w_kkio, float* U) {
    const float G[4][3] = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
    for (int ic = 0; ic < L.ic; ++ic)
        for (int oc = 0; oc < L.oc; ++oc) {
            float g[3][3], tmp[4][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) g[a][b] = w_kkio[((a * 3 + b) * L.ic + ic) * L.oc + oc];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 3; ++j) tmp[i][j] = G[i][0] * g[0][j] + G[i][1] * g[1][j] + G[i][2] * g[2][j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    U[(static_cast<size_t>(i * 4 + j) * L.ic + ic) * L.oc + oc] =
                        tmp[i][0] * G[j][0] + tmp[i][1] * G[j][1] + tmp[i][2] * G[j][2];
        }
}

static void conv_winograd(const Layer& L, const float* x, const float* U, float* y, float* V, float* Mbuf) {
    WinogradPlan p = wino_plan(L);
    int OH = L.oh(), OW = L.ow();
    // V[16][tiles][IC] = B^T d B
    for (int th = 0; th < p.tiles_h; ++th)
        for (int tw = 0; tw < p.tiles_w; ++tw) {
            int t = th * p.tiles_w + tw;
            for (int ic = 0; ic < L.ic; ++ic) {
                float d[4][4];
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) {
                        int ih = th * 2 + i - 1, iw = tw * 2 + j - 1;
                        d[i][j] = (ih < 0 || ih >= L.h || iw < 0 || iw >= L.w) ? 0.0f
                                  : x[(static_cast<size_t>(ih) * L.w + iw) * L.ic + ic];
                    }
                float t1[4][4];
                for (int j = 0; j < 4; ++j) {
                    t1[0][j] = d[0][j] - d[2][j];
                    t1[1][j] = d[1][j] + d[2][j];
                    t1[2][j] = d[2][j] - d[1][j];
                    t1[3][j] = d[1][j] - d[3][j];
                }
                for (int i = 0; i < 4; ++i) {
                    float v0 = t1[i][0] - t1[i][2], v1 = t1[i][1] + t1[i][2];
                    float v2 = t1[i][2] - t1[i][1], v3 = t1[i][1] - t1[i][3];
                    float vs[4] = {v0, v1, v2, v3};
                    for (int j = 0; j < 4; ++j) V[(static_cast<size_t>(i * 4 + j) * p.tiles + t) * L.ic + ic] = vs[j];
                }
            }
        }
    // 16 GEMMs: M[e][tiles][OC] = V[e][tiles][IC] * U[e][IC][OC]
    for (int e = 0; e < 16; ++e)
        gemm(p.tiles, L.oc, L.ic, V + static_cast<size_t>(e) * p.tiles * L.ic, L.ic,
             U + static_cast<size_t>(e) * L.ic * L.oc, L.oc, Mbuf + static_cast<size_t>(e) * p.tiles * L.oc, L.oc);
    // y = A^T m A
    for (int th = 0; th < p.tiles_h; ++th)
        for (int tw = 0; tw < p.tiles_w; ++tw) {
            int t = th * p.tiles_w + tw;
            for (int oc = 0; oc < L.oc; ++oc) {
                float m[4][4];
                for (int e = 0; e < 16; ++e) m[e / 4][e % 4] = Mbuf[(static_cast<size_t>(e) * p.tiles + t) * L.oc + oc];
                float t1[2][4];
                for (int j = 0; j < 4; ++j) {
                    t1[0][j] = m[0][j] + m[1][j] + m[2][j];
                    t1[1][j] = m[1][j] - m[2][j] - m[3][j];
                }
                for (int i = 0; i < 2; ++i) {
                    float o0 = t1[i][0] + t1[i][1] + t1[i][2];
                    float o1 = t1[i][1] - t1[i][2] - t1[i][3];
                    int oh = th * 2 + i;
                    if (oh >= OH) continue;
                    if (tw * 2 < OW) y[(static_cast<size_t>(oh) * OW + tw * 2) * L.oc + oc] = o0;
                    if (tw * 2 + 1 < OW) y[(static_cast<size_t>(oh) * OW + tw * 2 + 1) * L.oc + oc] = o1;
                }
            }
        }
}

// ------------------------------------------------------------------ driver

struct Data {
    std::vector<double> x_nchw, w_oihw, y_ref;  // fp64 reference, NCHW / OIHW
    std::vector<float> x_nchw_f, w_oihw_f;      // fp32 copies for the naive kernel
    std::vector<float> x_nhwc, w_kkio, w_kkc;   // fp32 NHWC and runtime weight layouts
};

static Data make_data(const Layer& L, unsigned seed, bool with_ref) {
    Data d;
    std::srand(seed);
    auto rnd = [] { return (std::rand() / static_cast<double>(RAND_MAX)) * 2.0 - 1.0; };
    int OC = L.depthwise ? L.ic : L.oc;
    int icg = L.depthwise ? 1 : L.ic;
    d.x_nchw.resize(static_cast<size_t>(L.ic) * L.h * L.w);
    for (auto& v : d.x_nchw) v = static_cast<float>(rnd());
    d.w_oihw.resize(static_cast<size_t>(OC) * icg * L.k * L.k);
    double ws = 1.0 / std::sqrt(static_cast<double>(icg * L.k * L.k));
    for (auto& v : d.w_oihw) v = static_cast<float>(rnd() * ws);
    d.x_nchw_f.assign(d.x_nchw.begin(), d.x_nchw.end());
    d.w_oihw_f.assign(d.w_oihw.begin(), d.w_oihw.end());
    d.x_nhwc.resize(d.x_nchw.size());
    for (int c = 0; c < L.ic; ++c)
        for (int h = 0; h < L.h; ++h)
            for (int w = 0; w < L.w; ++w)
                d.x_nhwc[(static_cast<size_t>(h) * L.w + w) * L.ic + c] = static_cast<float>(d.x_nchw[(static_cast<size_t>(c) * L.h + h) * L.w + w]);
    if (L.depthwise) {
        d.w_kkc.resize(static_cast<size_t>(L.k) * L.k * L.ic);
        for (int c = 0; c < L.ic; ++c)
            for (int kh = 0; kh < L.k; ++kh)
                for (int kw = 0; kw < L.k; ++kw)
                    d.w_kkc[(kh * L.k + kw) * L.ic + c] = static_cast<float>(d.w_oihw[(c * L.k + kh) * L.k + kw]);
    } else {
        d.w_kkio.resize(static_cast<size_t>(L.k) * L.k * L.ic * L.oc);
        for (int oc = 0; oc < L.oc; ++oc)
            for (int ic = 0; ic < L.ic; ++ic)
                for (int kh = 0; kh < L.k; ++kh)
                    for (int kw = 0; kw < L.k; ++kw)
                        d.w_kkio[((static_cast<size_t>(kh) * L.k + kw) * L.ic + ic) * L.oc + oc] =
                            static_cast<float>(d.w_oihw[((static_cast<size_t>(oc) * L.ic + ic) * L.k + kh) * L.k + kw]);
    }
    if (with_ref) conv_ref(L, d.x_nchw, d.w_oihw, d.y_ref);
    return d;
}

// Max |y - ref| / max |ref|, where y is NHWC (nhwc=true) or NCHW.
static double rel_err(const Layer& L, const std::vector<double>& ref, const float* y, bool nhwc) {
    int OC = L.depthwise ? L.ic : L.oc, OH = L.oh(), OW = L.ow();
    double emax = 0, rmax = 0;
    for (int c = 0; c < OC; ++c)
        for (int h = 0; h < OH; ++h)
            for (int w = 0; w < OW; ++w) {
                double r = ref[(static_cast<size_t>(c) * OH + h) * OW + w];
                double v = nhwc ? y[(static_cast<size_t>(h) * OW + w) * OC + c] : y[(static_cast<size_t>(c) * OH + h) * OW + w];
                emax = std::max(emax, std::abs(v - r));
                rmax = std::max(rmax, std::abs(r));
            }
    return emax / std::max(rmax, 1e-30);
}

using Kernel = std::function<void()>;

struct Method {
    std::string name;
    size_t workspace;
    Kernel run;
    const float* out;
    bool nhwc;
};

static std::vector<Method> methods(const Layer& L, Data& d, std::vector<std::vector<float>>& bufs) {
    std::vector<Method> ms;
    int OC = L.depthwise ? L.ic : L.oc;
    size_t ysz = static_cast<size_t>(OC) * L.oh() * L.ow();
    auto out = [&](size_t n) -> float* { bufs.emplace_back(n); return bufs.back().data(); };
    float* y0 = out(ysz);
    ms.push_back({"naive_nchw", 0, [&L, &d, y0] { conv_naive_nchw(L, d.x_nchw_f.data(), d.w_oihw_f.data(), y0); }, y0, false});
    if (L.depthwise) {
        float* y1 = out(ysz);
        ms.push_back({"dw_generic_chw", 0, [&L, &d, y1] { dw_generic_chw(L, d.x_nchw_f.data(), d.w_oihw_f.data(), y1); }, y1, false});
        float* y2 = out(ysz);
        ms.push_back({"dw_nhwc_simd", 0, [&L, &d, y2] { dw_nhwc(L, d.x_nhwc.data(), d.w_kkc.data(), y2); }, y2, true});
        return ms;
    }
    float* col = out(im2col_bytes(L) / sizeof(float));
    float* y1 = out(ysz);
    ms.push_back({"im2col_gemm", im2col_bytes(L), [&L, &d, y1, col] { conv_im2col(L, d.x_nhwc.data(), d.w_kkio.data(), y1, col); }, y1, true});
    float* y2 = out(ysz);
    ms.push_back({"implicit_gemm", 0, [&L, &d, y2] { conv_implicit(L, d.x_nhwc.data(), d.w_kkio.data(), y2); }, y2, true});
    if (L.k == 1 && L.stride == 1) {
        float* y3 = out(ysz);
        ms.push_back({"pointwise_gemm", 0, [&L, &d, y3] { conv_pointwise(L, d.x_nhwc.data(), d.w_kkio.data(), y3); }, y3, true});
    }
    if (L.k == 3 && L.stride == 1) {
        WinogradPlan p = wino_plan(L);
        float* U = out(p.bytes_U / sizeof(float));
        wino_weights(L, d.w_kkio.data(), U);
        float* V = out(p.bytes_V / sizeof(float));
        float* M = out(p.bytes_M / sizeof(float));
        float* y4 = out(ysz);
        ms.push_back({"winograd_f2x3", p.bytes_V + p.bytes_M + p.bytes_U, [&L, U, V, M, y4, &d] { conv_winograd(L, d.x_nhwc.data(), U, y4, V, M); }, y4, true});
    }
    return ms;
}

static void check() {
    std::vector<Layer> small = {
        {"dense3x3_s1", 9, 7, 5, 37, 3, 1, false}, {"dense3x3_s2", 10, 9, 6, 17, 3, 2, false},
        {"pw", 6, 5, 7, 33, 1, 1, false},          {"dense5x5", 8, 8, 3, 8, 5, 1, false},
        {"dw3x3_s1", 9, 8, 21, 21, 3, 1, true},    {"dw3x3_s2", 9, 9, 35, 35, 3, 2, true},
        {"wino_odd", 7, 5, 4, 3, 3, 1, false},
    };
    for (const Layer& L : small) {
        Data d = make_data(L, 7, true);
        std::vector<std::vector<float>> bufs;
        for (Method& m : methods(L, d, bufs)) {
            m.run();
            Json().s("kind", "check").s("layer", L.name).s("method", m.name).f("rel_err", rel_err(L, d.y_ref, m.out, m.nhwc)).emit();
        }
    }
}

static void bench(const std::string& name, int repeats) {
    for (const Layer& L : layers()) {
        if (name != "all" && L.name != name) continue;
        Data d = make_data(L, 11, true);
        std::vector<std::vector<float>> bufs;
        for (Method& m : methods(L, d, bufs)) {
            Stats st = time_fn(m.run, 2, repeats, 2e-2);
            double err = rel_err(L, d.y_ref, m.out, m.nhwc);
            int OC = L.depthwise ? L.ic : L.oc;
            double act_bytes = 4.0 * (static_cast<double>(L.h) * L.w * L.ic + static_cast<double>(L.oh()) * L.ow() * OC);
            Json().s("kind", "conv").s("layer", L.name).s("method", m.name).i("H", L.h).i("W", L.w).i("IC", L.ic)
                .i("OC", OC).i("K", L.k).i("stride", L.stride).b("depthwise", L.depthwise)
                .f("macs", L.macs()).f("gflops", 2 * L.macs() / st.p50 / 1e9).i("workspace_bytes", static_cast<long long>(m.workspace))
                .f("act_bytes", act_bytes).f("rel_err", err).s("isa", isa()).stats(st).emit();
        }
    }
}

int main(int argc, char** argv) {
    std::string mode = arg_s(argc, argv, "mode", "check");
    if (mode == "check") check();
    else if (mode == "bench") bench(arg_s(argc, argv, "layer", "all"), static_cast<int>(arg_ll(argc, argv, "repeats", 9)));
    else if (mode == "list") {
        for (const Layer& L : layers())
            Json().s("kind", "layer").s("layer", L.name).i("H", L.h).i("IC", L.ic).i("OC", L.oc).i("K", L.k)
                .i("stride", L.stride).b("depthwise", L.depthwise).f("macs", L.macs()).emit();
    } else { std::fprintf(stderr, "unknown mode\n"); return 1; }
    return 0;
}
