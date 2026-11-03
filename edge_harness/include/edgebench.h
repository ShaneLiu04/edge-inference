// Shared helpers for the edge-deploy C++ benchmarks.
// Every benchmark prints one JSON object per line on stdout; edge_harness.cpp.run_json parses them.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace eb {

inline double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline void* aligned_alloc64(size_t bytes) {
    bytes = (bytes + 63) / 64 * 64;
#if defined(_WIN32)
    void* p = _aligned_malloc(bytes, 64);
#else
    void* p = std::aligned_alloc(64, bytes);
#endif
    if (!p) { std::fprintf(stderr, "allocation of %zu bytes failed\n", bytes); std::exit(2); }
    return p;
}

inline void aligned_free64(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

template <class T>
struct Buffer {
    T* ptr = nullptr;
    size_t n = 0;
    explicit Buffer(size_t count) : ptr(static_cast<T*>(aligned_alloc64(count * sizeof(T)))), n(count) {}
    ~Buffer() { aligned_free64(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
};

// Keeps a value alive so the compiler cannot delete the computation that produced it.
template <class T>
inline void keep(T const& value) {
#if defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    static volatile T sink; sink = value;
#endif
}

inline void clobber() {
#if defined(__GNUC__)
    asm volatile("" : : : "memory");
#endif
}

struct Stats {
    double p10, p50, p90, min;
    int samples;
};

inline double percentile(std::vector<double> xs, double q) {
    std::sort(xs.begin(), xs.end());
    if (xs.size() == 1) return xs[0];
    double pos = (xs.size() - 1) * q;
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, xs.size() - 1);
    double frac = pos - lo;
    return xs[lo] * (1 - frac) + xs[hi] * frac;
}

// Times fn() `repeats` times after `warmup` calls. If one call is shorter than
// min_s, each sample loops fn() enough times to reach min_s and divides.
inline Stats time_fn(const std::function<void()>& fn, int warmup = 3, int repeats = 15, double min_s = 2e-3) {
    for (int i = 0; i < warmup; ++i) fn();
    double t0 = now_s();
    fn();
    double one = now_s() - t0;
    int inner = one >= min_s ? 1 : std::max(1, static_cast<int>(min_s / std::max(one, 1e-9)));
    std::vector<double> xs;
    for (int r = 0; r < repeats; ++r) {
        double a = now_s();
        for (int i = 0; i < inner; ++i) fn();
        xs.push_back((now_s() - a) / inner);
    }
    return {percentile(xs, 0.1), percentile(xs, 0.5), percentile(xs, 0.9), *std::min_element(xs.begin(), xs.end()),
            repeats};
}

inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// Persistent workers that spin on a generation counter. Creating std::threads per call costs
// ~0.2-1 ms on Windows, more than a whole GEMV, so a fresh-thread parallel_for measures the OS.
class ThreadPool {
public:
    explicit ThreadPool(int threads) : threads_(threads) {
        for (int t = 1; t < threads; ++t) workers_.emplace_back([this, t] { loop(t); });
    }
    ~ThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& w : workers_) w.join();
    }
    int size() const { return threads_; }

    void run(int64_t n, const std::function<void(int64_t, int64_t, int)>& body) {
        body_ = &body;
        n_ = n;
        pending_.store(threads_ - 1, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        body(0, n / threads_, 0);
        while (pending_.load(std::memory_order_acquire) != 0) spin_pause();
    }

private:
    void loop(int t) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            long spins = 0;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                // spin ~ms for back-to-back calls, then sleep so idle workers do not steal cores from
                // single-threaded phases (yield alone keeps a core busy when nothing else is runnable)
                if (++spins < (1 << 16)) spin_pause();
                else std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;
            (*body_)(n_ * t / threads_, n_ * (t + 1) / threads_, t);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }

    int threads_;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int64_t, int64_t, int)>* body_ = nullptr;
    int64_t n_ = 0;
};

// Static block partition of [0, n) over `threads` workers; the caller runs block 0.
inline void parallel_for(int threads, int64_t n, const std::function<void(int64_t, int64_t, int)>& body) {
    if (threads <= 1) { body(0, n, 0); return; }
    static std::unique_ptr<ThreadPool> pool;
    if (!pool || pool->size() != threads) pool = std::make_unique<ThreadPool>(threads);
    pool->run(n, body);
}

// Minimal JSON line writer: eb::Json j; j.s("kind","x").i("n",3).f("t",1.5).emit();
class Json {
public:
    Json& s(const char* k, const std::string& v) { sep(); buf_ += "\"" + std::string(k) + "\":\"" + v + "\""; return *this; }
    Json& i(const char* k, long long v) { sep(); buf_ += "\"" + std::string(k) + "\":" + std::to_string(v); return *this; }
    Json& f(const char* k, double v) {
        sep();
        char tmp[64];
        std::snprintf(tmp, sizeof tmp, "%.9g", v);
        buf_ += "\"" + std::string(k) + "\":" + tmp;
        return *this;
    }
    Json& b(const char* k, bool v) { sep(); buf_ += "\"" + std::string(k) + "\":" + (v ? "true" : "false"); return *this; }
    Json& stats(const Stats& st) {
        return f("p10_s", st.p10).f("p50_s", st.p50).f("p90_s", st.p90).f("min_s", st.min).i("samples", st.samples);
    }
    void emit() { std::printf("{%s}\n", buf_.c_str()); std::fflush(stdout); buf_.clear(); }

private:
    void sep() { if (!buf_.empty()) buf_ += ","; }
    std::string buf_;
};

inline std::string isa() {
    std::string s;
#if defined(__AVX512F__)
    s += "avx512f ";
#endif
#if defined(__AVX512VNNI__)
    s += "avx512vnni ";
#endif
#if defined(__AVX2__)
    s += "avx2 ";
#endif
#if defined(__FMA__)
    s += "fma ";
#endif
#if defined(__ARM_NEON)
    s += "neon ";
#endif
    if (!s.empty()) s.pop_back();
    return s;
}

inline long long arg_ll(int argc, char** argv, const char* name, long long fallback) {
    std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0) return std::atoll(argv[i] + key.size());
    }
    return fallback;
}

inline std::string arg_s(int argc, char** argv, const char* name, const std::string& fallback) {
    std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0) return std::string(argv[i] + key.size());
    }
    return fallback;
}

}  // namespace eb
