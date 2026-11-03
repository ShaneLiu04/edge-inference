// Copy/compute overlap on a CPU: a "DMA" thread streams tiles from a cold DRAM buffer into small
// SRAM-sized buffers while the compute thread works on the previous tile.
//
//   dma --mode=sweep [--tile-kib=64] [--tiles=256]   serial vs double buffering over a range of compute work
//   dma --mode=tiles [--work=4]                      the same at fixed work, over tile sizes
//
// Each row reports the measured stage times (copy alone, compute alone, per tile), the two models
//   serial   n * (c + m)
//   double   n * max(c, m) + min(c, m)
// and the measured totals. On a CPU the two threads share caches and DRAM bandwidth, so the measured
// overlap is worse than on an MCU whose DMA engine has its own bus port; the gap is the result.
#include "edgebench.h"

#include <atomic>
#include <thread>

using namespace eb;

struct Rig {
    size_t tile_floats;
    int tiles;
    Buffer<float> dram;
    Buffer<float> sram[2];
    Buffer<float> acc;

    Rig(size_t tile_bytes, int n)
        : tile_floats(tile_bytes / sizeof(float)), tiles(n), dram(tile_floats * n),
          sram{Buffer<float>(tile_floats), Buffer<float>(tile_floats)}, acc(tile_floats) {
        for (size_t i = 0; i < dram.n; ++i) dram[i] = static_cast<float>(i % 97) * 0.01f;
        for (size_t i = 0; i < tile_floats; ++i) { sram[0][i] = sram[1][i] = 0; acc[i] = 0; }
    }

    void copy(int t, int b) {
        std::memcpy(sram[b].data(), dram.data() + static_cast<size_t>(t) * tile_floats, tile_floats * sizeof(float));
    }

    // `work` passes of a dependent FMA chain per element: compute time scales linearly with work.
    void compute(int b, int work) {
        float* a = acc.data();
        const float* s = sram[b].data();
        for (int w = 0; w < work; ++w)
            for (size_t i = 0; i < tile_floats; ++i) a[i] = a[i] * 0.999f + s[i];
        clobber();
    }

    // Evict the DRAM buffer from cache by touching a buffer larger than the LLC.
    void flush() {
        static Buffer<char> junk(64ull << 20);
        for (size_t i = 0; i < junk.n; i += 64) junk[i] = static_cast<char>(i);
        clobber();
    }
};

static double time_serial(Rig& r, int work) {
    r.flush();
    double t0 = now_s();
    for (int t = 0; t < r.tiles; ++t) {
        r.copy(t, 0);
        r.compute(0, work);
    }
    return now_s() - t0;
}

// Two buffers, two threads, one flag per buffer: 0 = free for the copier, 1 = full for compute.
static double time_double(Rig& r, int work) {
    r.flush();
    std::atomic<int> state[2];
    state[0] = 0;
    state[1] = 0;
    double t0 = now_s();
    std::thread dma([&] {
        for (int t = 0; t < r.tiles; ++t) {
            int b = t & 1;
            while (state[b].load(std::memory_order_acquire) != 0) {}
            r.copy(t, b);
            state[b].store(1, std::memory_order_release);
        }
    });
    for (int t = 0; t < r.tiles; ++t) {
        int b = t & 1;
        while (state[b].load(std::memory_order_acquire) != 1) {}
        r.compute(b, work);
        state[b].store(0, std::memory_order_release);
    }
    dma.join();
    return now_s() - t0;
}

static void stage_times(Rig& r, int work, double& m, double& c) {
    r.flush();
    double t0 = now_s();
    for (int t = 0; t < r.tiles; ++t) r.copy(t, t & 1);
    m = (now_s() - t0) / r.tiles;
    t0 = now_s();
    for (int t = 0; t < r.tiles; ++t) r.compute(t & 1, work);
    c = (now_s() - t0) / r.tiles;
}

static double median_of(int reps, const std::function<double()>& fn) {
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) v.push_back(fn());
    return percentile(v, 0.5);
}

static void row(Rig& r, int work, size_t tile_bytes, int reps) {
    std::vector<double> ms, cs;
    for (int i = 0; i < reps; ++i) {
        double m, c;
        stage_times(r, work, m, c);
        ms.push_back(m);
        cs.push_back(c);
    }
    double m = percentile(ms, 0.5), c = percentile(cs, 0.5);
    double ser = median_of(reps, [&] { return time_serial(r, work); });
    double dbl = median_of(reps, [&] { return time_double(r, work); });
    double n = r.tiles;
    Json().s("kind", "dma").i("tile_bytes", static_cast<long long>(tile_bytes)).i("tiles", r.tiles).i("work", work)
        .f("copy_s", m).f("compute_s", c).f("ratio_c_over_m", c / m)
        .f("model_serial_s", n * (c + m)).f("model_double_s", n * std::max(c, m) + std::min(c, m))
        .f("serial_s", ser).f("double_s", dbl).f("speedup", ser / dbl)
        .f("model_speedup", (c + m) / (std::max(c, m) + std::min(c, m) / n)).emit();
}

int main(int argc, char** argv) {
    std::string mode = arg_s(argc, argv, "mode", "sweep");
    int reps = static_cast<int>(arg_ll(argc, argv, "repeats", 5));
    if (mode == "sweep") {
        size_t tile = static_cast<size_t>(arg_ll(argc, argv, "tile-kib", 64)) << 10;
        Rig r(tile, static_cast<int>(arg_ll(argc, argv, "tiles", 256)));
        for (int work : {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) row(r, work, tile, reps);
    } else if (mode == "tiles") {
        int work = static_cast<int>(arg_ll(argc, argv, "work", 4));
        for (long long kib : {4, 16, 64, 256, 1024}) {
            size_t tile = static_cast<size_t>(kib) << 10;
            Rig r(tile, static_cast<int>(std::max<long long>(16, (16ll << 20) / static_cast<long long>(tile))));
            row(r, work, tile, reps);
        }
    } else { std::fprintf(stderr, "unknown mode\n"); return 1; }
    return 0;
}
