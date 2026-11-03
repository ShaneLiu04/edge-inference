// Host roofline: bandwidth by working-set size, fp32 FMA peak, int8 dot-product peak.
//
//   roofline --mode=bw   --threads=1 --max-mib=512
//   roofline --mode=fma  --threads=4
//   roofline --mode=int8 --threads=4
//
// Bandwidth kernels are read-only sum (one stream) and triad a = b + s*c
// (two reads, one write; bytes counted as 3 streams, write-allocate not counted).
#include "edgebench.h"

#include <cmath>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace eb;

static float sum_read(const float* x, int64_t n) {
#if defined(__AVX512F__)
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_add_ps(a0, _mm512_load_ps(x + i));
        a1 = _mm512_add_ps(a1, _mm512_load_ps(x + i + 16));
        a2 = _mm512_add_ps(a2, _mm512_load_ps(x + i + 32));
        a3 = _mm512_add_ps(a3, _mm512_load_ps(x + i + 48));
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; i < n; ++i) s += x[i];
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) { s0 += x[i]; s1 += x[i + 1]; s2 += x[i + 2]; s3 += x[i + 3]; }
    for (; i < n; ++i) s0 += x[i];
    return s0 + s1 + s2 + s3;
#endif
}

static void triad(float* a, const float* b, const float* c, float s, int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) a[i] = b[i] + s * c[i];
}

static void bandwidth(int threads, int64_t max_bytes) {
    for (int64_t bytes = 16 << 10; bytes <= max_bytes; bytes *= 2) {
        int64_t n = bytes / sizeof(float);
        n = n / (64 * threads) * (64 * threads);
        Buffer<float> x(n), b(n), c(n), a(n);
        for (int64_t i = 0; i < n; ++i) { x[i] = 1.0f; b[i] = 1.0f; c[i] = 2.0f; a[i] = 0.0f; }
        std::vector<float> partial(threads * 16);
        // Even a pool wake-up is longer than one pass over an L1-sized buffer, so every thread sweeps
        // its own slice `reps` times inside a single launch; the working set stays n.
        int reps = static_cast<int>(std::max<int64_t>(1, (64 << 20) / std::max<int64_t>(bytes, 1)));
        auto read_rep = [&] {
            parallel_for(threads, n / 64, [&](int64_t lo, int64_t hi, int t) {
                float s = 0;
                for (int r = 0; r < reps; ++r) {
                    s += sum_read(x.data() + lo * 64, (hi - lo) * 64);
                    clobber();  // otherwise the repeated pass over unchanged memory can be hoisted
                }
                partial[t * 16] = s;
            });
            keep(partial[0]);
        };
        auto triad_rep = [&] {
            parallel_for(threads, n, [&](int64_t lo, int64_t hi, int) {
                for (int r = 0; r < reps; ++r) {
                    triad(a.data(), b.data(), c.data(), 3.0f, lo, hi);
                    clobber();
                }
            });
        };
        Stats sr = time_fn(read_rep, 2, 11, 5e-3);
        Stats st = time_fn(triad_rep, 2, 11, 5e-3);
        double rb = static_cast<double>(n) * sizeof(float) * reps;
        Json().s("kind", "bw").s("kernel", "read").i("threads", threads).i("bytes", n * 4)
            .f("GBps", rb / sr.p50 / 1e9).f("GBps_best", rb / sr.min / 1e9).stats(sr).emit();
        Json().s("kind", "bw").s("kernel", "triad").i("threads", threads).i("bytes", n * 4 * 3)
            .f("GBps", 3 * rb / st.p50 / 1e9).f("GBps_best", 3 * rb / st.min / 1e9).stats(st).emit();
    }
}

// 16 independent FMA chains per thread hide the 4-cycle latency on two FMA ports.
static double fma_kernel(int64_t iters) {
#if defined(__AVX512F__)
    __m512 acc[16];
    __m512 m = _mm512_set1_ps(0.999999f), add = _mm512_set1_ps(1e-7f);
    for (int k = 0; k < 16; ++k) acc[k] = _mm512_set1_ps(static_cast<float>(k));
    for (int64_t i = 0; i < iters; ++i) {
        for (int k = 0; k < 16; ++k) acc[k] = _mm512_fmadd_ps(acc[k], m, add);
    }
    __m512 s = acc[0];
    for (int k = 1; k < 16; ++k) s = _mm512_add_ps(s, acc[k]);
    keep(_mm512_reduce_add_ps(s));
    return 16.0 * 16 * 2 * iters;  // chains * lanes * (mul+add)
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc[12];
    __m256 m = _mm256_set1_ps(0.999999f), add = _mm256_set1_ps(1e-7f);
    for (int k = 0; k < 12; ++k) acc[k] = _mm256_set1_ps(static_cast<float>(k));
    for (int64_t i = 0; i < iters; ++i) {
        for (int k = 0; k < 12; ++k) acc[k] = _mm256_fmadd_ps(acc[k], m, add);
    }
    float out[8];
    __m256 s = acc[0];
    for (int k = 1; k < 12; ++k) s = _mm256_add_ps(s, acc[k]);
    _mm256_storeu_ps(out, s);
    keep(out[0]);
    return 12.0 * 8 * 2 * iters;
#else
    float acc[16];
    for (int k = 0; k < 16; ++k) acc[k] = static_cast<float>(k);
    for (int64_t i = 0; i < iters; ++i) for (int k = 0; k < 16; ++k) acc[k] = acc[k] * 0.999999f + 1e-7f;
    keep(acc[3]);
    return 16.0 * 2 * iters;
#endif
}

// VNNI: vpdpbusd does 64 u8*s8 products summed into 16 int32 lanes = 128 int ops.
static double int8_kernel(int64_t iters) {
#if defined(__AVX512VNNI__)
    __m512i acc[16];
    __m512i a = _mm512_set1_epi8(3), b = _mm512_set1_epi8(-2);
    for (int k = 0; k < 16; ++k) acc[k] = _mm512_set1_epi32(k);
    for (int64_t i = 0; i < iters; ++i) {
        for (int k = 0; k < 16; ++k) acc[k] = _mm512_dpbusd_epi32(acc[k], a, b);
    }
    __m512i s = acc[0];
    for (int k = 1; k < 16; ++k) s = _mm512_add_epi32(s, acc[k]);
    keep(_mm512_reduce_add_epi32(s));
    return 16.0 * 64 * 2 * iters;
#elif defined(__AVX2__)
    // vpmaddubsw + vpmaddwd: 32 u8*s8 products per pair of instructions.
    __m256i acc[8];
    __m256i a = _mm256_set1_epi8(3), b = _mm256_set1_epi8(-2), ones = _mm256_set1_epi16(1);
    for (int k = 0; k < 8; ++k) acc[k] = _mm256_set1_epi32(k);
    for (int64_t i = 0; i < iters; ++i) {
        for (int k = 0; k < 8; ++k) acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
    }
    int out[8];
    __m256i s = acc[0];
    for (int k = 1; k < 8; ++k) s = _mm256_add_epi32(s, acc[k]);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), s);
    keep(out[0]);
    return 8.0 * 32 * 2 * iters;
#else
    return 0.0;
#endif
}

static void peak(const char* kind, int threads, double (*kernel)(int64_t)) {
    const int64_t iters = 20'000'000;
    std::vector<double> ops(threads);
    auto fn = [&] {
        parallel_for(threads, threads, [&](int64_t lo, int64_t, int t) { ops[t] = kernel(iters); (void)lo; });
    };
    Stats st = time_fn(fn, 1, 7, 0.0);
    double total = 0;
    for (double o : ops) total += o;
    Json().s("kind", kind).i("threads", threads).s("isa", isa()).f("Gops", total / st.p50 / 1e9)
        .f("Gops_best", total / st.min / 1e9).stats(st).emit();
}

int main(int argc, char** argv) {
    std::string mode = arg_s(argc, argv, "mode", "bw");
    int threads = static_cast<int>(arg_ll(argc, argv, "threads", 1));
    if (mode == "bw") bandwidth(threads, arg_ll(argc, argv, "max-mib", 512) << 20);
    else if (mode == "fma") peak("fma", threads, fma_kernel);
    else if (mode == "int8") peak("int8", threads, int8_kernel);
    else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }
    return 0;
}
