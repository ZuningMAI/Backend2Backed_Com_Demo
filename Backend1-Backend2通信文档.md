# Backend1 ↔ Backend2 通信文档

## 概述

Backend1 (FastAPI/Python) 通过 HTTP JSON 调用 Backend2 (Qt6/C++) 的内部端点。
通信协议为 **HTTP POST**，Content-Type 为 **application/json**，超时 10 秒。

Backend2 监听 **端口 9000**，由 `QHttpServer` 处理请求。

## 端点总览

| 方法 | 路径 | 调用方 | 用途 |
|------|------|--------|------|
| GET | `/health` | Backend1 `health.py` | 健康检查 |
| POST | `/internal/calc/energy` | `scheduler.py` calc 线程 | 批量能耗积分 |
| POST | `/internal/predict/train` | `scheduler.py` predict 线程 | 多项式拟合预测 |
| POST | `/internal/session/reset` | `backend2_client.py` | 会话重置 |

---

## 1. 健康检查

### 调用方

`backend1/app/routers/health.py:18` — 聚合健康检查中调用
`backend1/app/services/backend2_client.py:30` — `Backend2Client.health_check()`

### 请求

```
GET /health
```

无请求体。

### 响应

**Backend2** (`server.cpp:81`)：

```json
{
  "status": 0,
  "message": "Backend2 healthy",
  "version": "0.5.0"
}
```

**Backend1** 聚合后返回：

```json
{
  "status": "ok",
  "service": "Backend1 v0.1.0",
  "dependencies": {
    "tdengine": "ok",
    "backend2": "ok",
    "detail": {
      "status": 0,
      "message": "Backend2 healthy",
      "version": "0.5.0"
    }
  },
  "active_sessions": 0
}
```

### 代码路径

```
health.py:health_check()
  → tdengine_client.health_check()
  → backend2_client.health_check()
    → GET localhost:9000/health
      → server.cpp:81 lambda
```

---

## 2. 能耗计算 — `/internal/calc/energy`

### 调用方

`scheduler.py:96-100` — `Session._run_calc()` 线程，每 100ms 调用一次

### 请求

```
POST /internal/calc/energy
Content-Type: application/json
```

```json
{
  "session_id": "4f199012-f0b9-4b3d-b0f9-b214c454530f",
  "data_points": [
    {
      "time": 0,
      "tractive_force": 0.0,
      "electric_brake_force": 0.0,
      "speed": 2.99,
      "battery_power": 120.0,
      "soc": 80.0
    }
  ],
  "sample_interval": 0.001
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| session_id | string | — | UUID 会话标识 |
| data_points | array | — | 最多 100 个数据点（1ms 间隔） |
| data_points[].time | int | ms | 全局时间戳 |
| data_points[].tractive_force | float | kN | 牵引力 |
| data_points[].electric_brake_force | float | kN | 电制动力 |
| data_points[].speed | float | km/h | 列车速度（注意：不是 m/s） |
| data_points[].battery_power | float | kW | 电池功率（正=放电，负=充电） |
| data_points[].soc | float | % | 荷电状态 (0~100) |
| sample_interval | float | s | 采样间隔（0.001 = 1ms） |

### 数据来源

数据由调度器从 TDengine 查询得到 (`scheduler.py:73-77`)：

```python
rows = _td_query(
    f"SELECT gtm,pos,spd,frc,opr,trf,ebf,bpw,soc,rte "
    f"FROM {TABLE} WHERE gtm >= {t} AND gtm < {t + 100} "
    f"ORDER BY gtm ASC")
```

查询返回的行被转换为 `data_points` 数组 (`scheduler.py:89-93`)：

```python
data_points = [{
    "time": r["time"], "tractive_force": r["tractive_force"],
    "electric_brake_force": r["electric_brake_force"],
    "speed": r["speed"], "battery_power": r["battery_power"],
    "soc": r["soc"],
} for r in rows]
```

### Backend2 处理

**`server.cpp:89-118`** — `/internal/calc/energy` 路由处理器：

```cpp
m_server->route("/internal/calc/energy", [this](const QHttpServerRequest &req) {
    QJsonDocument body = QJsonDocument::fromJson(req.body());
    QJsonObject obj = body.object();
    QString sid = obj.value("session_id").toString();
    QJsonArray pts = obj.value("data_points").toArray();
    double dt = obj.value("sample_interval").toDouble(0.001);

    // 1. 将数据点追加到会话缓冲区
    for (const auto &v : pts) {
        QJsonObject p = v.toObject();
        engine::DataPoint dp;
        dp.time = (int64_t)p.value("time").toDouble();
        dp.tractive_force = p.value("tractive_force").toDouble();
        dp.electric_brake_force = p.value("electric_brake_force").toDouble();
        dp.speed = p.value("speed").toDouble();
        dp.battery_power = p.value("battery_power").toDouble();
        dp.soc = p.value("soc").toDouble();
        m_sessionMgr->append(sid, dp);        // → 写入 deque<DataPoint>
    }

    // 2. 获取完整缓冲区并计算累计能耗
    auto buf = m_sessionMgr->getBuffer(sid);
    auto res = engine::computeEnergy(buf, dt, 0, INT64_MAX);

    // 3. 返回结果
    QJsonObject data;
    data["real_time_energy"] = res.real_time_energy;
    data["total_traction_energy"] = res.total_traction_energy;
    data["regenerative_energy"] = res.regenerative_energy;
    data["net_energy"] = res.net_energy;
    data["battery_energy"] = res.battery_energy;
    ...
});
```

物理计算在 `physics.cpp:84` 的 `computeEnergy()` 中进行：

```
对每个数据点:
  1. 速度转换: km/h → m/s (除以 3.6)
  2. 校验电池功率 (SOC 限制)
  3. 计算机械功率: P_mech = F_traction × v − F_brake × v
  4. 计算直流功率: P_dc = P_mech/η (牵引), P_mech×η (制动)
  5. 计算接触网功率: P_cat = P_dc − P_bat
  6. 消耗功率: P_consume = max(P_cat, 0) + max(P_bat, 0)
  7. 再生功率: P_regen = |min(P_cat, 0)| + |min(P_bat, 0)|
  8. 累加能量: E += P × Δt (Δt = 1ms / 3600 s/h → kWh)
  9. 净能耗 = 总牵引 − 再生
```

### 响应

```json
{
  "status": 0,
  "data": {
    "real_time_energy": 86.27,
    "total_traction_energy": 0.21,
    "regenerative_energy": 0.0,
    "net_energy": 0.21,
    "battery_energy": 0.02
  },
  "message": "session=4f199012..., pts=100"
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| data.real_time_energy | float | kWh/km | 瞬时能耗率（最后一个数据点） |
| data.total_traction_energy | float | kWh | 累计牵引能耗 |
| data.regenerative_energy | float | kWh | 累计再生能量 |
| data.net_energy | float | kWh | 净能耗 (traction − regen) |
| data.battery_energy | float | kWh | 累计电池放电能量 |

### 调用后处理

Backend1 `scheduler.py:103-121` 处理响应：

```python
with self._lock:
    self.energy_data = resp.json().get("data", {})

# 构建 actual_curve: 插值 net_energy 到采样位置
net_e = self.energy_data.get("net_energy", 0)
if net_e > 0:
    batch_samples = list(range(0, len(rows), 10))  # 每 10 行采样 1 个
    n_samples = len(batch_samples)
    for j, i in enumerate(batch_samples):
        pos_km = rows[i]["position"] / 1000.0
        frac = (j + 1) / n_samples
        interp_e = self._last_net_e + (net_e - self._last_net_e) * frac
        self.actual_curve.append({"position": pos_km, "energy": interp_e})
    self._last_net_e = net_e
```

`energy_data` 通过 `/vehicle/energy/result` API 返回给前端。

---

## 3. 能耗预测 — `/internal/predict/train`

### 调用方

`scheduler.py:156-159` — `Session._run_predict()` 线程，延迟 800ms 后每 100ms 调用一次

### 请求

```
POST /internal/predict/train
Content-Type: application/json
```

```json
{
  "session_id": "4f199012-f0b9-4b3d-b0f9-b214c454530f",
  "history_data": [
    {
      "time": 0,
      "speed": 2.99,
      "force": 50.0,
      "mode": 0,
      "tractive_force": 50.0,
      "electric_brake_force": 0.0,
      "battery_power": 120.0,
      "soc": 80.0,
      "real_time_energy": 49.53,
      "position": 242.0
    }
  ],
  "cumulative_energy": 0.21
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| session_id | string | — | UUID 会话标识 |
| history_data | array | — | 最近 800ms 的历史数据（最多 800 行） |
| history_data[].time | int | ms | 全局时间戳 |
| history_data[].speed | float | km/h | 列车速度 |
| history_data[].force | float | kN | 合力 |
| history_data[].mode | int | — | 运行模式 |
| history_data[].tractive_force | float | kN | 牵引力 |
| history_data[].electric_brake_force | float | kN | 电制动力 |
| history_data[].battery_power | float | kW | 电池功率 |
| history_data[].soc | float | % | 荷电状态 |
| history_data[].real_time_energy | float | kWh/km | 实时能耗率 |
| history_data[].position | float | m | 位置（用于预测起点） |
| cumulative_energy | float | kWh | 当前累计净能耗（预测能量起点） |

### 数据来源

调度器从 TDengine 查询最近 800ms 窗口 (`scheduler.py:132-133`)：

```python
start_t = max(0, t - 800)
rows = _td_query(
    f"SELECT gtm,pos,spd,frc,opr,trf,ebf,bpw,soc,rte "
    f"FROM {TABLE} WHERE gtm >= {start_t} AND gtm < {t} "
    f"ORDER BY gtm ASC")
```

构造 `history_data` (`scheduler.py:140-147`)：

```python
history_data = [{
    "time": r["time"], "speed": r["speed"], "force": r["force"],
    "mode": r["mode"], "tractive_force": r["tractive_force"],
    "electric_brake_force": r["electric_brake_force"],
    "battery_power": r["battery_power"], "soc": r["soc"],
    "real_time_energy": r["real_time_energy"],
    "position": r["position"],
} for r in rows]
```

`cumulative_energy` 来自 `self.energy_data["net_energy"]` (`scheduler.py:154`)：

```python
with self._lock:
    net_e = self.energy_data.get("net_energy", 0)
```

### Backend2 处理

**`server.cpp:125-183`** — `/internal/predict/train` 路由处理器：

**步骤 1: 验证历史数据量** (`server.cpp:131`)

```cpp
if (history.size() < 200) {
    // 历史数据不足 200 条，返回空预测
    QJsonObject r; r["status"] = 1;
    r["message"] = "need >= 200 history points";
    r["data"] = QJsonObject{{"predicted_curve", QJsonArray()}};
    return ...;
}
```

**步骤 2: 提取最近 200ms 用于拟合** (`server.cpp:137-143`)

```cpp
int fit_len = 200;
int fit_start = n - 200;   // 取最后 200 个点
std::vector<double> t_fit(200), spd(200), rte(200);
for (int i = 0; i < 200; ++i) {
    QJsonObject p = history[fit_start + i].toObject();
    t_fit[i] = (double)i;                     // x 轴: [0, 1, 2, ..., 199]
    spd[i] = p.value("speed").toDouble();     // y 轴: 速度
    rte[i] = p.value("real_time_energy").toDouble();  // y 轴: RTE
}
```

**步骤 3: 多项式最小二乘拟合** (`server.cpp:146-147`)

```cpp
auto spd_coef = polyfit(t_fit, spd, 3);   // 速度: 3 次多项式
auto rte_coef = polyfit(t_fit, rte, 2);   // RTE:  2 次多项式
```

`polyfit()` 函数 (`server.cpp:24-59`) 通过高斯消元求解正规方程 `(XᵀX)β = Xᵀy`：

```
1. 构建 Vandermonde 矩阵 X (n × m), m = degree + 1
2. 计算 XᵀX (m × m) 和 Xᵀy (m × 1)
3. 高斯消元 + 回代求解系数 β = [a₀, a₁, ..., a_d]
```

**步骤 4: 外推 200 步** (`server.cpp:155-177`)

```cpp
QJsonObject lastP = history.last().toObject();
double cumPos = lastP.value("position").toDouble(0.0) / 1000.0;  // m → km
double cumEnergy = cumulative_energy;  // 从当前累计净能耗开始

for (int i = 0; i < 200; ++i) {
    double t = (double)(200 + i);               // x = [200, 201, ..., 399]
    double s = polyval(spd_coef, t);            // 外推速度
    double r = polyval(rte_coef, t);            // 外推 RTE

    s = std::max(0.1, std::min(s, 200.0));      // 限幅
    r = std::max(0.0, std::min(r, 200.0));

    cumPos += s / 3.6 * 0.001 / 1000.0;         // 位置积分 (km)
    cumEnergy += r * s * 0.001 / 3600.0;         // 能量积分 (kWh)

    QJsonObject pt;
    pt["position"] = cumPos;
    pt["energy"] = cumEnergy;
    predicted.append(pt);
}
```

**步骤 5: 返回预测曲线**

```cpp
QJsonObject data;
data["predicted_curve"] = predicted;  // 200 个点

QJsonObject r;
r["status"] = 0;
r["data"] = data;
r["message"] = "polyfit prediction";
```

### 响应

```json
{
  "status": 0,
  "data": {
    "predicted_curve": [
      { "position": 0.244091, "energy": 0.151081 },
      { "position": 0.244091, "energy": 0.151123 },
      ...
    ]
  },
  "message": "polyfit prediction"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| data.predicted_curve | array | 200 个预测点 |
| predicted_curve[].position | float | 累计位置 (km) |
| predicted_curve[].energy | float | 累计能量 (kWh)，从 cumulative_energy 起算 |

### 调用后处理

Backend1 `scheduler.py:159-161` 存储原始预测曲线。
`get_predict_result()` (`scheduler.py:196-216`) 返回结果时，在预测曲线前插入锚点（最后一个实际曲线点）以保证视觉连续性：

```python
with sess._lock:
    predicted = list(sess.predicted_curve)
    if sess.actual_curve and predicted:
        anchor = dict(sess.actual_curve[-1])
        if predicted[0].get("position", 0) != anchor.get("position", -1):
            predicted.insert(0, anchor)  # 前置锚点
    return {
        "status": 0, "message": "success",
        "data": {
            "actual_curve": sess.actual_curve[-1:],
            "predicted_curve": predicted,  # 201 个点 (1 锚点 + 200 预测)
        },
        "progress": sess.progress(),
    }
```

---

## 4. 会话重置 — `/internal/session/reset`

### 调用方

`backend2_client.py:43` — `Backend2Client.reset_session()`

### 请求与响应

```
POST /internal/session/reset
{"session_id": "uuid-string"}
→ {"status": 0, "message": "session uuid-string reset"}
```

**Backend2** (`server.cpp:189-195`)：

```cpp
m_server->route("/internal/session/reset", [this](const QHttpServerRequest &req) {
    QString sid = QJsonDocument::fromJson(req.body()).object()
                      .value("session_id").toString();
    m_sessionMgr->reset(sid);  // 从 hash map 中删除会话
    ...
});
```

---

## 5. 调度器双线程模型

`backend1/app/services/scheduler.py` — 完整线程模型：

### Calc 线程 (`_run_calc`, line 70)

```
┌─────────────────────────────────────────┐
│ while state == "running":               │
│   t = self.t_ms                         │
│   rows = TDengine.query(t, t+100)       │
│   data_points = rows → B2 格式          │
│   POST /internal/calc/energy            │
│   energy_data = response.data           │
│   build actual_curve (插值)             │
│   t_ms += 100                           │
│   sleep(0.1)                            │
│   if t_ms >= total_ms: state=completed  │
└─────────────────────────────────────────┘
```

### Predict 线程 (`_run_predict`, line 133)

```
┌─────────────────────────────────────────┐
│ sleep(0.8)  # 800ms 冷启动延迟          │
│ while state == "running":               │
│   t = self.t_ms                         │
│   start_t = max(0, t - 800)             │
│   rows = TDengine.query(start_t, t)     │
│   if len(rows) < 100: continue          │
│   history_data = rows → B2 格式         │
│   net_e = energy_data["net_energy"]     │
│   POST /internal/predict/train          │
│     {history_data, cumulative_energy}   │
│   predicted_curve = response.data       │
│   sleep(0.1)                            │
└─────────────────────────────────────────┘
```

### 时序图

```
时间 (ms)     Calc 线程              Predict 线程
─────────────────────────────────────────────────
0            查询 [0, 100)           (sleep 800ms)
100          查询 [100, 200)         ↓
200          查询 [200, 300)         ↓
...          ...                     ↓
800          查询 [800, 900)         → 查询 [0, 800)
                                     → POST /internal/predict/train
900          查询 [900, 1000)        查询 [100, 900)
                                     → POST /internal/predict/train
1000         查询 [1000, 1100)       查询 [200, 1000)
                                     → ...
```

---

## 6. 数据格式汇总

### Backend1 → Backend2 数据点格式

**Calc 端点** — 5 个字段:

```json
{ "time": int, "tractive_force": float, "electric_brake_force": float,
  "speed": float, "battery_power": float, "soc": float }
```

**Predict 端点** — 11 个字段:

```json
{ "time": int, "speed": float, "force": float, "mode": int,
  "tractive_force": float, "electric_brake_force": float,
  "battery_power": float, "soc": float,
  "real_time_energy": float, "position": float }
```

### Backend2 → Backend1 响应格式

**Calc 响应**: 5 项能耗指标 (flat object)
**Predict 响应**: `predicted_curve` 数组 (200 个 position/energy 点)

### 前端接收格式

**EnergyResult**: 5 项能耗 + `actual_curve` (位置-能量对)
**TimePredict**: `actual_curve[-1:]` (1 个锚点) + `predicted_curve` (201 个点)

---

## 7. 错误处理

| 错误场景 | Backend2 响应 | Backend1 处理 |
|----------|---------------|---------------|
| 历史数据 < 200 条 | `status=1, message="need >= 200 history points"` | 预测曲线保持为空 |
| TDengine 查询无数据 | — | 继续循环，不调用 Backend2 |
| HTTP 超时/连接失败 | — | `logger.error`, 继续循环 |
| Session 不存在 | — | `get_*_result` 返回 `status=1` |

---

## 8. 通信模式对比

| | Calc 端点 | Predict 端点 |
|---|---|---|
| 间隔 | 100ms | 100ms (延迟 800ms) |
| 数据量 | 100 行 × 5 字段 | 800 行 × 11 字段 |
| 是否累积 | 是（追加到 session buffer） | 否（无状态） |
| 返回类型 | 标量（5 个能量值） | 数组（200 个位置-能量对） |
| 物理引擎 | `computeEnergy()` 积分 | `polyfit()` + `polyval()` 外推 |
| 能量起点 | 从 session 开始 | 从 `cumulative_energy` 开始 |
