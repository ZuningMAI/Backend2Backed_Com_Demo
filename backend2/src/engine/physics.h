#ifndef ENGINE_PHYSICS_H
#define ENGINE_PHYSICS_H

#include <cstdint>
#include <vector>
#include <deque>

namespace engine {

struct DataPoint {
    int64_t time;             // ms timestamp
    double tractive_force;    // kN
    double electric_brake_force; // kN
    double speed;             // m/s
    double battery_power;     // kW, positive=discharging, negative=charging
    double soc;               // 0~100%
};

struct EnergyResult {
    double real_time_energy;       // kWh/km instantaneous
    double total_traction_energy;  // kWh cumulative
    double regenerative_energy;    // kWh cumulative
    double net_energy;             // kWh cumulative
    double battery_energy;         // kWh cumulative (discharge only)
};

/**
 * Compute max wheel power limit based on SOC (traction direction).
 * Returns max power in kW.
 */
double maxTractionPowerBySoc(double soc);

/**
 * Compute max regenerative braking power limit based on SOC.
 * Returns max regen power in kW (positive value representing limit).
 */
double maxRegenPowerBySoc(double soc);

/**
 * Compute mechanical (wheel-rim) power.
 * Positive = consuming, Negative = regenerating.
 * @param tractive_force  kN, positive in traction
 * @param brake_force     kN, positive in electric braking
 * @param speed           m/s
 * @return mechanical power in kW
 */
double calcMechanicalPower(double tractive_force, double brake_force, double speed);

/**
 * Compute DC bus side electrical power from mechanical power.
 * @param mech_power  kW, positive=traction, negative=braking
 * @param eta         efficiency (0~1), default 0.85
 * @return DC bus power in kW
 */
double calcDcPower(double mech_power, double eta = 0.85);

/**
 * Derive catenary (overhead line) power from DC bus and battery.
 * @param dc_power       kW
 * @param battery_power  kW, positive=discharging
 * @return catenary power in kW
 */
double calcCatenaryPower(double dc_power, double battery_power);

/**
 * Apply SOC-based validation and clamping to battery power.
 * Returns corrected battery_power.
 */
double validateBatteryPower(double battery_power, double soc);

/**
 * Compute all energy metrics for a buffer of data points within [start_time, end_time].
 * @param data_points    chronologically ordered buffer
 * @param sample_interval  seconds between samples
 * @param start_time        ms timestamp
 * @param end_time          ms timestamp
 * @param eta               efficiency
 * @return EnergyResult with cumulative values
 */
EnergyResult computeEnergy(const std::deque<DataPoint> &data_points,
                           double sample_interval,
                           int64_t start_time,
                           int64_t end_time,
                           double eta = 0.85);

} // namespace engine

#endif // ENGINE_PHYSICS_H
