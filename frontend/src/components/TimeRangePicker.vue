<template>
  <div class="time-range-picker">
    <div class="picker-group">
      <span class="picker-icon">&#x1F552;</span>
      <span class="picker-label">计算时段选择</span>
    </div>
    <div class="picker-controls">
      <!-- Start: HH:MM:SS -->
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('start', 'hour', 1)">▲</button>
        <div class="time-display">{{ pad(startHour) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('start', 'hour', -1)">▼</button>
        <div class="time-unit">时</div>
      </div>
      <span class="time-sep">:</span>
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('start', 'min', 1)">▲</button>
        <div class="time-display">{{ pad(startMin) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('start', 'min', -1)">▼</button>
        <div class="time-unit">分</div>
      </div>
      <span class="time-sep">:</span>
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('start', 'sec', 1)">▲</button>
        <div class="time-display">{{ pad(startSec) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('start', 'sec', -1)">▼</button>
        <div class="time-unit">秒</div>
      </div>
      <span class="range-sep">至</span>
      <!-- End: HH:MM:SS -->
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('end', 'hour', 1)">▲</button>
        <div class="time-display">{{ pad(endHour) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('end', 'hour', -1)">▼</button>
        <div class="time-unit">时</div>
      </div>
      <span class="time-sep">:</span>
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('end', 'min', 1)">▲</button>
        <div class="time-display">{{ pad(endMin) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('end', 'min', -1)">▼</button>
        <div class="time-unit">分</div>
      </div>
      <span class="time-sep">:</span>
      <div class="time-col">
        <button class="scroll-btn" @mousedown="startScroll('end', 'sec', 1)">▲</button>
        <div class="time-display">{{ pad(endSec) }}</div>
        <button class="scroll-btn" @mousedown="startScroll('end', 'sec', -1)">▼</button>
        <div class="time-unit">秒</div>
      </div>
    </div>
    <button class="confirm-btn" @click="confirm">确认计算</button>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from "vue";

const emit = defineEmits<{
  (e: "confirm", startMs: number, endMs: number, durationSec: number): void;
}>();

const startHour = ref(8);
const startMin = ref(0);
const startSec = ref(0);
const endHour = ref(8);
const endMin = ref(1);
const endSec = ref(26);

function pad(n: number): string {
  return String(Math.floor(n)).padStart(2, "0");
}

// ── Scroll with hold-repeat ──
const scrollTimers = new Map<string, ReturnType<typeof setInterval>>();

function startScroll(target: "start" | "end", field: "hour" | "min" | "sec", delta: number) {
  applyScroll(target, field, delta);
  const key = `${target}-${field}`;
  stopScroll(key);
  scrollTimers.set(key, setInterval(() => applyScroll(target, field, delta), 120));
  const stop = () => {
    stopScroll(key);
    document.removeEventListener("mouseup", stop);
  };
  document.addEventListener("mouseup", stop);
}

function stopScroll(key: string) {
  const t = scrollTimers.get(key);
  if (t) { clearInterval(t); scrollTimers.delete(key); }
}

onUnmounted(() => scrollTimers.forEach(t => clearInterval(t)));

function applyScroll(target: "start" | "end", field: "hour" | "min" | "sec", delta: number) {
  if (target === "start") {
    if (field === "hour") startHour.value = (startHour.value + delta + 24) % 24;
    if (field === "min") startMin.value = (startMin.value + delta + 60) % 60;
    if (field === "sec") startSec.value = (startSec.value + delta + 60) % 60;
  } else {
    if (field === "hour") endHour.value = (endHour.value + delta + 24) % 24;
    if (field === "min") endMin.value = (endMin.value + delta + 60) % 60;
    if (field === "sec") endSec.value = (endSec.value + delta + 60) % 60;
  }
}

function confirm() {
  const now = new Date();
  const s = new Date(now.getFullYear(), now.getMonth(), now.getDate(),
    startHour.value, startMin.value, startSec.value, 0);
  const e = new Date(now.getFullYear(), now.getMonth(), now.getDate(),
    endHour.value, endMin.value, endSec.value, 0);
  const dur = Math.round((e.getTime() - s.getTime()) / 1000);
  emit("confirm", s.getTime(), e.getTime(), dur);
}
</script>

<style scoped>
.time-range-picker {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 18px;
  padding: 20px 24px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin: 16px 0;
  flex-wrap: wrap;
}
.picker-group { display: flex; align-items: center; gap: 8px; }
.picker-icon { font-size: 22px; }
.picker-label { font-size: 15px; font-weight: 600; color: #303133; }
.picker-controls { display: flex; align-items: center; gap: 4px; }
.time-col { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.time-display {
  width: 52px; height: 42px; display: flex; align-items: center; justify-content: center;
  font-size: 20px; font-weight: 700; color: #303133; background: #f5f7fa;
  border-radius: 8px; font-family: "Courier New", monospace;
}
.time-unit { font-size: 11px; color: #909399; }
.time-sep { font-size: 20px; font-weight: 700; color: #909399; margin-bottom: 12px; }
.range-sep { font-size: 14px; color: #909399; padding: 0 8px; margin-bottom: 12px; }
.scroll-btn {
  width: 28px; height: 16px; display: flex; align-items: center; justify-content: center;
  border: none; background: transparent; color: #909399; cursor: pointer;
  font-size: 10px; border-radius: 4px; user-select: none;
}
.scroll-btn:hover { background: #ecf5ff; color: #409eff; }
.confirm-btn {
  padding: 10px 28px; background: #409eff; color: #fff; border: none;
  border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer;
  transition: background 0.15s;
}
.confirm-btn:hover { background: #337ecc; }
</style>
