<template>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-icon">⚡</div>
      <div class="metric-body">
        <div class="metric-label">实时能耗</div>
        <div class="metric-value">{{ fmt(metrics.real_time_energy) }}</div>
        <div class="metric-unit">kWh/km</div>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">🔺</div>
      <div class="metric-body">
        <div class="metric-label">总牵引能耗</div>
        <div class="metric-value">{{ fmt(metrics.total_traction_energy) }}</div>
        <div class="metric-unit">kWh</div>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">♻️</div>
      <div class="metric-body">
        <div class="metric-label">再生能量</div>
        <div class="metric-value regen">{{ fmt(metrics.regenerative_energy) }}</div>
        <div class="metric-unit">kWh</div>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">📊</div>
      <div class="metric-body">
        <div class="metric-label">净能耗</div>
        <div class="metric-value primary">{{ fmt(metrics.net_energy) }}</div>
        <div class="metric-unit">kWh</div>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">🔋</div>
      <div class="metric-body">
        <div class="metric-label">电池能耗</div>
        <div class="metric-value">{{ fmt(metrics.battery_energy) }}</div>
        <div class="metric-unit">kWh</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EnergyResultData } from "../types/api";

defineProps<{
  metrics: EnergyResultData;
}>();

function fmt(val: number): string {
  return val.toFixed(3);
}
</script>

<style scoped>
.metrics-row {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  padding: 8px 0;
}
.metric-card {
  flex: 1;
  min-width: 175px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  transition: box-shadow 0.2s;
}
.metric-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
}
.metric-icon {
  font-size: 26px;
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
  border-radius: 10px;
}
.metric-body {
  display: flex;
  flex-direction: column;
}
.metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 2px;
}
.metric-value {
  font-size: 26px;
  font-weight: 700;
  color: #303133;
  line-height: 1.2;
}
.metric-value.regen {
  color: #67c23a;
}
.metric-value.primary {
  color: #409eff;
}
.metric-unit {
  font-size: 11px;
  color: #c0c4cc;
}
</style>
