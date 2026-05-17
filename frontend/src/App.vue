<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="header-left">
        <div class="logo">🚂</div>
        <div><h1>列车能耗监控系统</h1><span class="subtitle">Energy Consumption Monitor</span></div>
      </div>
      <div class="header-right">
        <div class="status-indicator">
          <span class="status-dot" :class="{ active: state === 'running', idle: state === 'idle', done: state === 'completed' }"></span>
          <span class="status-text">{{ stateText }}</span>
        </div>
        <span class="session-badge" v-if="sessionId">会话: {{ sessionId.slice(0, 8) }}...</span>
      </div>
    </header>

    <main>
      <TimeRangePicker @confirm="onConfirm" :disabled="state === 'running'" />

      <!-- Progress -->
      <div class="progress-bar-wrap" v-if="state === 'running' || state === 'completed'">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
        <span class="progress-text">{{ progressPercent }}% ({{ currentMs }}/{{ totalMs }}ms)</span>
      </div>

      <EnergyMetrics :metrics="energyData" />
      <div class="chart-card">
        <div class="chart-header"><h3>能耗曲线</h3></div>
        <div ref="chartDiv" style="height: 420px; width: 100%"></div>
      </div>

      <div class="completion-msg" v-if="state === 'completed'">✅ 计算完成</div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted, onMounted } from "vue";
import TimeRangePicker from "./components/TimeRangePicker.vue";
import EnergyMetrics from "./components/EnergyMetrics.vue";
import { healthCheck, postEnergyResult, postTimePredict } from "./api";
import type { EnergyResultData, CurvePoint } from "./types/api";

const MAX_TOTAL_TIME = 2813000; // ms

const state = ref<"idle" | "running" | "completed" | "error">("idle");
const sessionId = ref<string | null>(null);
const energyData = ref<EnergyResultData>({
  real_time_energy: 0, total_traction_energy: 0,
  regenerative_energy: 0, net_energy: 0, battery_energy: 0,
});
const actualCurve = ref<CurvePoint[]>([]);
const predictedCurve = ref<CurvePoint[]>([]);
const currentMs = ref(0);
const totalMs = ref(0);
const cumulativePos = ref(0);

// Chart (direct DOM, CDN echarts)
const chartDiv = ref<HTMLElement | null>(null);
let chartInst: any = null;
declare function echarts_init(el: HTMLElement): any;

function initChart() {
  const ec = (window as any).echarts;
  if (!chartDiv.value || !ec) { setTimeout(initChart, 200); return; }
  try {
    // Dispose existing instance to avoid duplicate chart warning
    const existing = ec.getInstanceByDom(chartDiv.value);
    if (existing) existing.dispose();
    chartInst = ec.init(chartDiv.value);
    chartInst.setOption({
      tooltip: { trigger: "axis" },
      animation: true,
      animationDuration: 300,
      grid: { left: "4%", right: "8%", bottom: "12%", top: "5%", containLabel: true },
      xAxis: { type: "value", name: "位置 (km)", nameLocation: "middle", nameGap: 30 },
      yAxis: { type: "value", name: "累计净能耗 (kWh)" },
      dataZoom: [
        { type: "slider", xAxisIndex: 0, start: 0, end: 100, height: 20, bottom: 4 },
        { type: "inside", xAxisIndex: 0 },
      ],
      series: [
        { name: "实际能耗", type: "line", data: [], smooth: true, symbol: "none",
          lineStyle: { color: "#409EFF", width: 2.5 }, connectNulls: true },
        { name: "预测能耗", type: "line", data: [], smooth: true, symbol: "none",
          lineStyle: { color: "#E6A23C", width: 2.5, type: "dashed" }, connectNulls: true },
      ],
    });
  } catch (e) { console.error("chart init failed:", e); }
}

let pollTimer: ReturnType<typeof setInterval> | null = null;

const stateText = computed(() => {
  switch (state.value) {
    case "idle": return "等待确认";
    case "running": return "系统运行中";
    case "completed": return "计算完成";
    case "error": return "连接断开";
  }
});
const progressPercent = computed(() =>
  totalMs.value > 0 ? Math.min(100, Math.round((currentMs.value / totalMs.value) * 100)) : 0
);

function onConfirm(_startMs: number, _endMs: number, durationSec: number) {
  if (durationSec * 1000 > MAX_TOTAL_TIME) {
    alert(`所选时段超过全线路时间 ${MAX_TOTAL_TIME / 1000}s`);
    return;
  }
  if (durationSec <= 0) { alert("结束时间必须晚于开始时间"); return; }
  startPolling(durationSec * 1000);
}

function startPolling(total: number) {
  stopPolling();
  state.value = "running";
  currentMs.value = 0;
  totalMs.value = total;
  actualCurve.value = [];
  predictedCurve.value = [];
  cumulativePos.value = 0;
  energyData.value = { real_time_energy: 0, total_traction_energy: 0, regenerative_energy: 0, net_energy: 0, battery_energy: 0 };

  pollTimer = setInterval(pollData, 1000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollData() {
  try {
    // Always include session_id (null on first call)
    const reqBody: any = {
      session_id: sessionId.value,  // null on first call
      tractive_force: 0, electric_brake_force: 0,
      speed: 0, battery_power: 0, soc: 0, sample_interval: 1.0,
      start_time: 0, end_time: totalMs.value,
    };
    const resp = await postEnergyResult(reqBody);

    // Extract session_id from first response
    if (!sessionId.value && resp.message) {
      const m = resp.message.match(/session[=:]\s*(\S+)/);
      if (m) sessionId.value = m[1];
    }

    energyData.value = resp.data;

    const p = resp.progress;
    if (p) {
      currentMs.value = p.current_ms;
      if (p.state === "completed") { state.value = "completed"; stopPolling(); }
    }

    // Update chart actual series (blue solid line)
    const ac = resp.actual_curve;
    if (ac && ac.length > 0 && chartInst) {
      const actualData = ac.map((p: CurvePoint) => [p.position, p.energy]);
      chartInst.setOption({
        series: [
          { data: actualData },
        ],
      });
    }

    // Prediction
    if (sessionId.value && currentMs.value > 800) {
      try {
        const pred = await postTimePredict({
          session_id: sessionId.value,
          lookback_window: 800, forecast_horizon: 200, model_type: "math_only",
        });
        const pc = pred.data.predicted_curve;
        if (pc && pc.length > 1 && chartInst) {
          predictedCurve.value = pc;
          const predData = pc.map((p: CurvePoint) => [p.position, p.energy]);
          // Extend x-axis max so prediction is visible; dataZoom slider handles zoom
          chartInst.setOption({
            xAxis: { max: pc[pc.length - 1].position + 0.001 },
            series: [
              {},
              { data: predData },
            ],
          });
        }
      } catch { /* predict may not be ready */ }
    }
  } catch (err) {
    console.error("Poll error:", err);
    state.value = "error";
  }
}

onMounted(async () => {
  try { await healthCheck(); } catch { /* ok */ }
  // Delay ensures DOM + echarts module fully loaded
  setTimeout(initChart, 300);
});
onUnmounted(() => { stopPolling(); chartInst?.dispose(); });
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #f0f2f5; color: #303133; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.app-shell { max-width: 1280px; margin: 0 auto; min-height: 100vh; }
.app-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 28px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #fff; }
.header-left { display: flex; align-items: center; gap: 14px; }
.logo { font-size: 32px; }
.header-left h1 { font-size: 20px; font-weight: 600; margin: 0; color: #fff; }
.subtitle { font-size: 12px; color: rgba(255,255,255,0.6); }
.header-right { display: flex; align-items: center; gap: 16px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #909399; }
.status-dot.active { background: #67c23a; box-shadow: 0 0 6px rgba(103,194,58,0.6); }
.status-dot.idle { background: #e6a23c; }
.status-dot.done { background: #409eff; }
.status-text { font-size: 13px; color: rgba(255,255,255,0.85); }
.session-badge { font-size: 11px; color: rgba(255,255,255,0.5); background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 10px; }
main { padding: 20px 28px; }
.progress-bar-wrap {
  height: 28px; background: #e4e7ed; border-radius: 14px; position: relative;
  margin: 8px 0 16px; overflow: hidden;
}
.progress-fill {
  height: 100%; background: linear-gradient(90deg, #67c23a, #409eff);
  border-radius: 14px; transition: width 0.3s; min-width: 0;
}
.progress-text { position: absolute; top: 4px; left: 50%; transform: translateX(-50%); font-size: 12px; color: #303133; font-weight: 600; }
.completion-msg { text-align: center; font-size: 18px; font-weight: 600; color: #67c23a; padding: 16px; }
</style>
