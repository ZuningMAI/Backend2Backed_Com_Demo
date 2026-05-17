<template>
  <div class="chart-card">
    <div class="chart-header">
      <h3>能耗曲线</h3>
      <div class="chart-legend">
        <span class="legend-item"><i class="dot actual"></i>实际能耗</span>
        <span class="legend-item"><i class="dot predicted"></i>预测能耗</span>
      </div>
    </div>
    <v-chart :option="opt" autoresize style="height: 420px" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import VChart from "vue-echarts";
import "echarts";
import type { CurvePoint } from "../types/api";

const props = defineProps<{ actualCurve: CurvePoint[]; predictedCurve: CurvePoint[] }>();

const opt = ref<any>({
  tooltip: { trigger: "axis" },
  grid: { left: "4%", right: "4%", bottom: "8%", top: "5%", containLabel: true },
  xAxis: { type: "value", name: "位置 (km)", nameLocation: "middle", nameGap: 30 },
  yAxis: { type: "value", name: "累计净能耗 (kWh)" },
  series: [
    { name: "实际能耗", type: "line", data: [], smooth: true, symbol: "none",
      lineStyle: { color: "#409EFF", width: 2.5 } },
    { name: "预测能耗", type: "line", data: [], smooth: true, symbol: "none",
      lineStyle: { color: "#E6A23C", width: 2.5, type: "dashed" } },
  ],
});

watch([() => props.actualCurve, () => props.predictedCurve], ([ac, pc]) => {
  opt.value.series[0].data = ac.map(p => [p.position, p.energy]);
  opt.value.series[1].data = pc.map(p => [p.position, p.energy]);
});
</script>

<style scoped>
.chart-card { background:#fff; border-radius:12px; padding:20px 24px; box-shadow:0 2px 10px rgba(0,0,0,0.05); margin:16px 0; }
.chart-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.chart-header h3 { font-size:16px; font-weight:600; color:#303133; margin:0; }
.chart-legend { display:flex; gap:20px; }
.legend-item { font-size:13px; color:#606266; display:flex; align-items:center; gap:6px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot.actual { background:#409EFF; }
.dot.predicted { background:#E6A23C; }
</style>
