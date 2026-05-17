#include "engine/physics.h"
#include <cmath>
#include <algorithm>

namespace engine {

// ---------- SOC-based power limits ----------

double maxTractionPowerBySoc(double soc)
{
    if (soc >= 25.0) {
        return 2000.0;  // kW, full power
    } else if (soc >= 10.0) {
        return 700.0;   // kW, reduced power
    } else if (soc > 0.0) {
        // Linear decline from 700 kW at 10% to 0 kW at 0%
        return 700.0 * (soc / 10.0);
    }
    return 0.0;
}

double maxRegenPowerBySoc(double soc)
{
    if (soc <= 95.0) {
        return 1500.0;  // kW, full regen
    } else if (soc < 100.0) {
        // Linear decline from 1500 kW at 95% to 0 kW at 100%
        return 1500.0 * ((100.0 - soc) / 5.0);
    }
    return 0.0;
}

// ---------- Power calculations ----------

double calcMechanicalPower(double tractive_force, double brake_force, double speed)
{
    if (speed <= 0.0)
        return 0.0;

    if (tractive_force > 0.0) {
        // Traction: positive mechanical power (consuming)
        return tractive_force * speed;
    } else if (brake_force > 0.0) {
        // Electric braking: negative mechanical power (regenerating)
        return -brake_force * speed;
    }
    // Coasting: zero
    return 0.0;
}

double calcDcPower(double mech_power, double eta)
{
    if (mech_power > 0.0) {
        // Traction: DC side draws more due to losses
        return mech_power / eta;
    } else if (mech_power < 0.0) {
        // Regeneration: DC side receives less due to losses
        return mech_power * eta;
    }
    return 0.0;
}

double calcCatenaryPower(double dc_power, double battery_power)
{
    // P_cat = P_dc - P_bat
    return dc_power - battery_power;
}

double validateBatteryPower(double battery_power, double soc)
{
    // Charging limit: battery nearly full, cannot absorb more
    if (battery_power < 0.0 && soc >= 99.0) {
        return 0.0;
    }
    // Discharging limit: battery nearly empty, cannot discharge
    if (battery_power > 0.0 && soc <= 10.0) {
        return 0.0;
    }
    return battery_power;
}

// ---------- Cumulative energy computation ----------

EnergyResult computeEnergy(const std::deque<DataPoint> &data_points,
                           double sample_interval,
                           int64_t start_time,
                           int64_t end_time,
                           double eta)
{
    EnergyResult result{};
    result.real_time_energy = 0.0;
    result.total_traction_energy = 0.0;
    result.regenerative_energy = 0.0;
    result.net_energy = 0.0;
    result.battery_energy = 0.0;

    // Filter points within [start_time, end_time]
    for (const auto &dp : data_points) {
        if (dp.time < start_time || dp.time > end_time)
            continue;

        // Speed from TDengine is km/h, convert to m/s for physics calculations
        double speed_ms = dp.speed / 3.6;
        if (speed_ms <= 0.0)
            continue;

        // Validate battery power against SOC
        double bat_power = validateBatteryPower(dp.battery_power, dp.soc);

        // Step 1: Mechanical power
        double mech_power = calcMechanicalPower(dp.tractive_force,
                                                 dp.electric_brake_force,
                                                 speed_ms);

        // Step 2: DC bus power
        double dc_power = calcDcPower(mech_power, eta);

        // Step 3: Catenary power
        double cat_power = calcCatenaryPower(dc_power, bat_power);

        // Step 4: Real-time consumption power
        double p_consume = std::max(cat_power, 0.0) + std::max(bat_power, 0.0);

        // Step 5: Real-time regeneration power
        double p_regen = std::abs(std::min(cat_power, 0.0)) +
                         std::abs(std::min(bat_power, 0.0));

        // Step 6: Accumulate (P * Δt in hours → kWh)
        double dt_hours = sample_interval / 3600.0;
        result.total_traction_energy += p_consume * dt_hours;
        result.regenerative_energy += p_regen * dt_hours;
        result.battery_energy += std::max(bat_power, 0.0) * dt_hours;
    }

    // Net energy = total consumed - regenerated
    result.net_energy = result.total_traction_energy - result.regenerative_energy;

    // Real-time energy: use the last data point for instantaneous value
    if (!data_points.empty()) {
        const auto &last = data_points.back();
        if (last.speed > 0.0) {
            double bat_power = validateBatteryPower(last.battery_power, last.soc);
            double mech_power = calcMechanicalPower(last.tractive_force,
                                                     last.electric_brake_force,
                                                     last.speed);
            double dc_power = calcDcPower(mech_power, eta);
            double cat_power = calcCatenaryPower(dc_power, bat_power);
            double p_consume = std::max(cat_power, 0.0) + std::max(bat_power, 0.0);
            // RTE in kWh/km: p_consume(kW) / speed(km/h)
            // speed from TDengine is already in km/h
            result.real_time_energy = p_consume / last.speed;
        }
    }

    return result;
}

} // namespace engine
