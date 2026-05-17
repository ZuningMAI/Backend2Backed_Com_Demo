#ifndef ENGINE_MLP_H
#define ENGINE_MLP_H

#include <vector>
#include <random>
#include <memory>

namespace engine {

/**
 * Tiny MLP with forward/backward/SGD, using Eigen for matrix ops.
 * Architecture: 30 → 128 → 64 → 200
 */
class MLP {
public:
    MLP();
    ~MLP();

    /** Forward pass. Input: [1, 30], Output: [1, 200] */
    std::vector<double> forward(const std::vector<double> &input);

    /** Train one step: forward + backward + SGD update.
     *  input: [30] features, target: [200] labels, lr: learning rate.
     *  Returns MSE loss. */
    double trainStep(const std::vector<double> &input,
                     const std::vector<double> &target,
                     double lr = 0.001);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

/**
 * Extract 30 statistical features from 800ms history of telemetry data.
 * Each data point has: time_ms, speed, force, mode, tractive_force,
 *                      electric_brake_force, battery_power, soc, real_time_energy
 */
struct TelemetryPoint {
    double time_ms, speed, force, mode;
    double tractive_force, electric_brake_force;
    double battery_power, soc, real_time_energy;
    double position_m;  // absolute position in meters
};

std::vector<double> extractFeatures(const std::vector<TelemetryPoint> &history);

} // namespace engine

#endif
