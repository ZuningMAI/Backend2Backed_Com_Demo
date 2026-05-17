-- TDengine schema for energy management system
-- Applied via REST API during Step 2

CREATE DATABASE IF NOT EXISTS energy_mgmt KEEP 30 DURATION 1;

USE energy_mgmt;

CREATE STABLE IF NOT EXISTS vehicle_telemetry (
    ts                      TIMESTAMP,
    speed                   FLOAT,
    tractive_force          FLOAT,
    electric_brake_force    FLOAT,
    battery_power           FLOAT,
    soc                     FLOAT,
    energy                  FLOAT
) TAGS (
    session_id              NCHAR(64),
    vehicle_id              NCHAR(32)
);
