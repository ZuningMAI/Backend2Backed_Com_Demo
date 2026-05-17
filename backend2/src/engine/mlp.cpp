#include "engine/mlp.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <QDebug>

namespace engine {

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using RowVector = Eigen::RowVectorXf;

// ── Xavier init ──
static float xavierScale(int fanIn, int fanOut) {
    return std::sqrt(6.0f / (fanIn + fanOut));
}

static Matrix xavierInit(int rows, int cols) {
    float scale = xavierScale(cols, rows);
    return Matrix::Random(rows, cols) * scale;
}

static RowVector xavierInitRow(int size) {
    float scale = xavierScale(size, 1);
    return RowVector::Random(size) * scale;
}

// ── ReLU ──
static RowVector relu(const RowVector &x) { return x.cwiseMax(0.0f); }
static RowVector reluDeriv(const RowVector &x) {
    return (x.array() > 0.0f).cast<float>();
}

// ── PImpl ──

struct MLP::Impl {
    // Layer 1: 30→128
    Matrix W1; RowVector b1;
    // Layer 2: 128→64
    Matrix W2; RowVector b2;
    // Layer 3: 64→200
    Matrix W3; RowVector b3;

    // Cached forward values for backward pass
    RowVector z1, a1, z2, a2, z3, a3;
    RowVector input;

    Impl() {
        std::mt19937 rng(42);
        // Xavier init
        W1 = xavierInit(128, 30);
        b1 = RowVector::Zero(128);
        W2 = xavierInit(64, 128);
        b2 = RowVector::Zero(64);
        W3 = xavierInit(200, 64);
        b3 = RowVector::Zero(200);
    }
};

MLP::MLP() : d(std::make_unique<Impl>()) {}
MLP::~MLP() = default;

// ── Forward ──

std::vector<double> MLP::forward(const std::vector<double> &input) {
    if (input.size() != 30) return std::vector<double>(200, 0.0);

    // Convert to Eigen
    d->input = RowVector(30);
    for (int i = 0; i < 30; ++i) d->input(i) = (float)input[i];

    // Layer 1: 30→128
    d->z1 = d->input * d->W1.transpose() + d->b1;
    d->a1 = relu(d->z1);

    // Layer 2: 128→64
    d->z2 = d->a1 * d->W2.transpose() + d->b2;
    d->a2 = relu(d->z2);

    // Layer 3: 64→200
    d->z3 = d->a2 * d->W3.transpose() + d->b3;
    d->a3 = d->z3;  // no activation on output

    std::vector<double> result(200);
    for (int i = 0; i < 200; ++i) {
        float v = d->a3(i);
        if (std::isnan(v) || std::isinf(v)) v = 0.0f;
        result[i] = v;
    }
    return result;
}

// ── Train step (forward + backward + SGD) ──

double MLP::trainStep(const std::vector<double> &input,
                      const std::vector<double> &target,
                      double lr) {
    if (input.size() != 30 || target.size() != 200) return 1e9;

    // ── Forward ──
    d->input = RowVector(30);
    for (int i = 0; i < 30; ++i) d->input(i) = (float)input[i];

    d->z1 = d->input * d->W1.transpose() + d->b1;
    d->a1 = relu(d->z1);
    d->z2 = d->a1 * d->W2.transpose() + d->b2;
    d->a2 = relu(d->z2);
    d->z3 = d->a2 * d->W3.transpose() + d->b3;
    d->a3 = d->z3;

    // ── Loss ──
    RowVector pred = d->a3;
    RowVector tgt(200);
    for (int i = 0; i < 200; ++i) tgt(i) = (float)target[i];

    RowVector diff = pred - tgt;
    double mse = diff.squaredNorm() / 200.0;

    // ── Backward (MSE gradient) ──
    RowVector dL = diff * (2.0f / 200.0f);  // (1, 200)

    // Layer 3 gradient: d3 = dL (no activation)
    Matrix dW3 = dL.transpose() * d->a2;     // (200, 64)
    RowVector db3 = dL;                       // (1, 200)

    // Layer 2: d2 = dL @ W3 * relu'(z2)
    RowVector d2 = (dL * d->W3).cwiseProduct(reluDeriv(d->z2));  // (1, 64)
    Matrix dW2 = d2.transpose() * d->a1;     // (64, 128)
    RowVector db2 = d2;                       // (1, 64)

    // Layer 1: d1 = d2 @ W2 * relu'(z1)
    RowVector d1 = (d2 * d->W2).cwiseProduct(reluDeriv(d->z1));  // (1, 128)
    Matrix dW1 = d1.transpose() * d->input;  // (128, 30)
    RowVector db1 = d1;                       // (1, 128)

    // ── SGD update ──
    if (!std::isnan(mse) && !std::isinf(mse)) {
        float lrf = (float)lr;
        d->W1 -= lrf * dW1; d->b1 -= lrf * db1;
        d->W2 -= lrf * dW2; d->b2 -= lrf * db2;
        d->W3 -= lrf * dW3; d->b3 -= lrf * db3;
    }

    return std::isnan(mse) ? 1e9 : mse;
}

// ── Feature extraction ──

std::vector<double> extractFeatures(const std::vector<TelemetryPoint> &history) {
    int n = (int)history.size();
    if (n < 10) return std::vector<double>(30, 0.0);

    std::vector<double> spd, frc, rte, soc, bpw;
    for (auto &p : history) {
        spd.push_back(p.speed);
        frc.push_back(p.force);
        rte.push_back(p.real_time_energy);
        soc.push_back(p.soc);
        bpw.push_back(p.battery_power);
    }

    auto mean = [](const std::vector<double> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stdev = [&](const std::vector<double> &v, double m) {
        double sq = 0;
        for (auto x : v) sq += (x - m) * (x - m);
        return std::sqrt(sq / v.size());
    };
    auto slope = [](const std::vector<double> &v) {
        int nv = (int)v.size();
        if (nv < 2) return 0.0;
        return (v.back() - v.front()) / (nv - 1);
    };
    auto ratio = [&](const std::vector<TelemetryPoint> &pts, int modeVal) {
        int cnt = 0;
        for (auto &p : pts) if ((int)p.mode == modeVal) cnt++;
        return (double)cnt / pts.size();
    };

    double spd_m = mean(spd), spd_s = stdev(spd, spd_m);
    double frc_m = mean(frc), frc_s = stdev(frc, frc_m);
    double rte_m = mean(rte), rte_s = stdev(rte, rte_m);
    double soc_m = mean(soc), soc_s = stdev(soc, soc_m);
    double bpw_m = mean(bpw), bpw_s = stdev(bpw, bpw_m);

    std::vector<double> f(30);
    f[0] = spd_m;   f[1] = spd_s;   f[2] = spd.back();  f[3] = spd.front();
    f[4] = slope(spd);
    f[5] = frc_m;   f[6] = frc_s;   f[7] = frc.back();
    f[8] = *std::max_element(frc.begin(), frc.end());
    f[9] = *std::min_element(frc.begin(), frc.end());
    f[10] = slope(frc);
    f[11] = rte_m;  f[12] = rte_s;  f[13] = rte.back();
    f[14] = soc_m;  f[15] = soc_s;  f[16] = soc.back();
    f[17] = bpw_m;  f[18] = bpw_s;
    f[19] = *std::max_element(bpw.begin(), bpw.end());
    f[20] = *std::min_element(bpw.begin(), bpw.end());
    f[21] = ratio(history, 0);  // traction ratio
    f[22] = ratio(history, 2);  // coast ratio
    f[23] = ratio(history, 3);  // braking ratio
    f[24] = ratio(history, 1);  // cruise ratio
    f[25] = history.front().speed;
    f[26] = history.back().speed;
    f[27] = history.front().real_time_energy;
    f[28] = history.back().real_time_energy;
    f[29] = slope(rte);

    return f;
}

} // namespace engine
