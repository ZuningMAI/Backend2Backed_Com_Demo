# Backend1 ↔ Backend2 集成通信方案

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      对方团队（甲方）                              │
│                                                                 │
│  Frontend (readdy.cc)                                           │
│       │                                                         │
│       ▼                                                         │
│  Backend1 (FastAPI / 甲方服务器)                                  │
│       │                                                         │
│       │  HTTP JSON (内部 API)                                    │
│       │                                                         │
├───────┼─────────────────────────────────────────────────────────┤
│       │                    我方团队（乙方）                        │
│       ▼                                                         │
│  Backend2 (Qt6/C++ / 乙方服务器，端口 9000)                       │
│                                                                 │
│  提供: 能耗物理计算引擎 + 多项式拟合预测                            │
└─────────────────────────────────────────────────────────────────┘
```

**说明**：
- Frontend 由甲方开发和部署（readdy.cc 平台）
- Backend1 由甲方开发和部署（FastAPI 服务）
- Backend2 由我方开发和部署（Qt6/C++ 引擎）
- Backend1 通过 HTTP JSON 调用 Backend2 的内部端点

## 2. 外部 API（Frontend → Backend1）

前端通过以下接口调用 Backend1（详见《API接口.md》）：

### 2.1 能耗计算 `POST /vehicle/energy/result`

**请求**（前端 → Backend1）：

```json
{
  "tractive_force": 50.0,
  "electric_brake_force": 0.0,
  "speed": 60.0,
  "battery_power": 120.0,
  "soc": 80.0,
  "sample_interval": 1.0,
  "start_time": 1713123456000,
  "end_time": 1713123516000
}
```

**响应**（Backend1 → 前端）：

```json
{
  "status": 0,
  "message": "success",
  "data": {
    "real_time_energy": 0.01,
    "total_traction_energy": 0.01,
    "regenerative_energy": 0.01,
    "net_energy": 0.01,
    "battery_energy": 0.01
  },
  "timestamp": 1713123456890
}
```

### 2.2 能耗预测 `POST /vehicle/energy/time_predict`

**请求**（前端 → Backend1）：

```json
{
  "history_data": [
    {
      "time": 1713123456000,
      "speed": 55.0,
      "energy": 100.2,
      "tractive_force": 48.0,
      "electric_brake_force": 0.0,
      "battery_power": 110.0,
      "soc": 82.0
    }
  ],
  "actual_data_point": [
    {
      "time": 1713123456000,
      "speed": 25.0,
      "energy": 110.2,
      "tractive_force": 0.0,
      "electric_brake_force": 0.0,
      "battery_power": 90.0,
      "soc": 50.0
    }
  ],
  "lookback_window": 1000,
  "forecast_horizon": 200,
  "model_type": "dl_math_hybrid"
}
```

**响应**（Backend1 → 前端）：

```json
{
  "status": 0,
  "message": "success",
  "data": {
    "actual_curve": [
      { "position": 1.5, "energy": 125.3 }
    ],
    "predicted_curve": [
      { "position": 1.5, "energy": 125.3 },
      { "position": 2.0, "energy": 210.8 },
      { "position": 3.2, "energy": 380.1 }
    ]
  },
  "timestamp": 1713123456890
}
```

---

## 3. 内部 API（Backend1 → Backend2）

Backend1 对 Backend2 的调用为 **HTTP POST**，Content-Type 为 **application/json**。

Backend2 监听 **端口 9000**（可配置）。

### 3.1 健康检查

```
GET /health
→ {"status": 0, "message": "Backend2 healthy", "version": "0.5.0"}
```

### 3.2 能耗计算 `POST /internal/calc/energy`

Backend1 收到前端的能耗计算请求后，将原始数据转发给 Backend2 进行物理积分。

**请求**（Backend1 → Backend2）：

```json
{
  "session_id": "uuid-string",
  "data_points": [
    {
      "time": 1713123456000,
      "tractive_force": 50.0,
      "electric_brake_force": 0.0,
      "speed": 60.0,
      "battery_power": 120.0,
      "soc": 80.0
    }
  ],
  "sample_interval": 0.001
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| session_id | string | — | 会话标识 |
| data_points | array | — | 数据点数组 |
| data_points[].time | int | ms | 时间戳 |
| data_points[].tractive_force | float | kN | 牵引力（正=牵引） |
| data_points[].electric_brake_force | float | kN | 电制动力（正=制动） |
| data_points[].speed | float | km/h | 列车速度 |
| data_points[].battery_power | float | kW | 电池功率（正=放电，负=充电） |
| data_points[].soc | float | % | 电池荷电状态 0~100 |
| sample_interval | float | s | 采样间隔 |

**Backend2 处理流程**（`server.cpp:89-118`）：

1. 将 data_points 追加到 session buffer（`session_mgr.cpp`）
2. 调用 `computeEnergy()` 进行物理积分（`physics.cpp:84`）：

```
对每个数据点:
  速度转换: km/h → m/s (除以 3.6)
  机械功率: P_mech = F_traction × v − F_brake × v
  直流功率: P_dc = P_mech / η (牵引), P_mech × η (制动), η = 0.85
  接触网功率: P_cat = P_dc − P_bat
  消耗功率: P_consume = max(P_cat, 0) + max(P_bat, 0)
  再生功率: P_regen = |min(P_cat, 0)| + |min(P_bat, 0)|
  能量累加: E += P × Δt  (Δt = sample_interval / 3600 hour)
```

3. 返回累计能耗结果

**响应**（Backend2 → Backend1）：

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
  "message": "session=xxx, pts=100"
}
```

| 字段 | 单位 | 说明 |
|------|------|------|
| real_time_energy | kWh/km | 实时能耗率 |
| total_traction_energy | kWh | 累计牵引能耗 |
| regenerative_energy | kWh | 累计再生能量 |
| net_energy | kWh | 净能耗 (traction − regen) |
| battery_energy | kWh | 累计电池放电 |

### 3.3 能耗预测 `POST /internal/predict/train`

Backend1 将历史数据发给 Backend2，Backend2 通过多项式拟合外推未来能耗曲线。

**请求**（Backend1 → Backend2）：

```json
{
  "session_id": "uuid-string",
  "history_data": [
    {
      "time": 1713123456000,
      "speed": 55.0,
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
| session_id | string | — | 会话标识 |
| history_data | array | — | 历史数据（至少 200 条） |
| history_data[].time | int | ms | 时间戳 |
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

**Backend2 处理流程**（`server.cpp:125-183`）：

```
1. 验证历史数据 ≥ 200 条
2. 取最近 200 条用于拟合
3. 多项式最小二乘拟合:
   - 速度: 3 次多项式 (t_fit=[0,1,...,199] → speed[last 200])
   - RTE:  2 次多项式 (t_fit=[0,1,...,199] → rte[last 200])
4. 外推 200 步 (t=[200,...,399]):
   - speed_pred = polyval(spd_coef, t)
   - rte_pred = polyval(rte_coef, t)
   - cumPos += speed_pred / 3.6 × 0.001 / 1000  (km 累加)
   - cumEnergy += rte_pred × speed_pred × 0.001 / 3600 (kWh 累加)
5. 返回 200 个预测点 (position, energy)
```

**响应**（Backend2 → Backend1）：

```json
{
  "status": 0,
  "data": {
    "predicted_curve": [
      { "position": 0.244091, "energy": 0.151081 },
      { "position": 0.244092, "energy": 0.151123 }
    ]
  },
  "message": "polyfit prediction"
}
```

200 个预测点，每个点包含累计位置 (km) 和累计能耗 (kWh)。
能量从 `cumulative_energy` 起算，确保预测曲线与实际曲线连续。

### 3.4 会话重置 `POST /internal/session/reset`

```
请求: {"session_id": "uuid-string"}
响应: {"status": 0, "message": "session xxx reset"}
```

---

## 4. Backend1 集成代码

甲方在 Backend1 中需要实现以下逻辑来调用 Backend2。

### 4.1 能耗计算调用

```python
import httpx

BACKEND2_URL = "http://<乙方服务器IP>:9000"

def calc_energy(session_id: str, data_points: list[dict], 
                sample_interval: float = 0.001) -> dict:
    """将前端数据转发给 Backend2 进行能耗积分"""
    resp = httpx.post(
        f"{BACKEND2_URL}/internal/calc/energy",
        json={
            "session_id": session_id,
            "data_points": data_points,
            "sample_interval": sample_interval,
        },
        timeout=10.0
    )
    resp.raise_for_status()
    return resp.json()["data"]  # 5 项能耗指标
```

### 4.2 能耗预测调用

```python
def predict_energy(session_id: str, history_data: list[dict],
                   cumulative_energy: float, 
                   forecast_len: int = 200) -> list[dict]:
    """将历史数据发给 Backend2 进行多项式拟合预测"""
    resp = httpx.post(
        f"{BACKEND2_URL}/internal/predict/train",
        json={
            "session_id": session_id,
            "history_data": history_data,
            "cumulative_energy": cumulative_energy,
            "forecast_len": forecast_len,
        },
        timeout=10.0
    )
    resp.raise_for_status()
    return resp.json()["data"]["predicted_curve"]  # 200 个预测点
```

### 4.3 健康检查

```python
def health_check() -> dict:
    resp = httpx.get(f"{BACKEND2_URL}/health", timeout=5.0)
    return resp.json()
```

---

## 5. 数据格式映射

### 5.1 前端请求 → Backend2 请求

前端传给 Backend1 的数据字段到 Backend2 的映射：

| 前端字段 | Backend2 字段 | 单位 | 说明 |
|----------|---------------|------|------|
| tractive_force | tractive_force | kN | 直接映射 |
| electric_brake_force | electric_brake_force | kN | 直接映射 |
| speed | speed | km/h | 直接映射（Backend2 内部会转换为 m/s） |
| battery_power | battery_power | kW | 直接映射 |
| soc | soc | % | 直接映射 |
| sample_interval | sample_interval | s | 直接映射 |
| start_time / end_time | — | ms | Backend1 用于分片，Backend2 不需要 |

### 5.2 Backend2 响应 → 前端响应

Backend2 返回的能耗指标直接作为 Backend1 对外 API 的 `data` 字段返回：

| Backend2 字段 | 前端需要字段 | 单位 |
|---------------|-------------|------|
| real_time_energy | real_time_energy | kWh/km |
| total_traction_energy | total_traction_energy | kWh |
| regenerative_energy | regenerative_energy | kWh |
| net_energy | net_energy | kWh |
| battery_energy | battery_energy | kWh |

### 5.3 预测曲线映射

Backend2 返回 200 个 `(position, energy)` 点，Backend1 需要：
- 取前端发送的 `actual_data_point`（最后 1 个实际点）作为 `actual_curve`
- 将 Backend2 的 `predicted_curve` 作为 `predicted_curve` 返回

---

## 6. 部署方案

### 方案 A：Backend2 独立部署（当前方案）

```
甲方服务器                      乙方服务器
┌──────────────┐              ┌──────────────┐
│  Frontend    │              │              │
│  Backend1    │── HTTP ────→ │  Backend2    │
│  TDengine    │              │  (端口 9000)  │
└──────────────┘              └──────────────┘
```

- Backend1 和 TDengine 在甲方服务器
- Backend2 在乙方服务器
- Backend1 通过 HTTP 调用 Backend2
- **需要**：甲方开放防火墙规则，允许访问乙方 Backend2 的 9000 端口

### 方案 B：Backend2 移植到甲方

```
甲方服务器
┌──────────────────────────────┐
│  Frontend                    │
│  Backend1 ──→ Backend2:9000 │
│  TDengine                    │
└──────────────────────────────┘
```

- Backend2 部署到甲方服务器（同机或内网）
- 通信延迟最低
- **需要**：甲方服务器有 Qt6 运行环境 + ONNX Runtime 库
- Backend2 构建依赖：Qt 6.8.1, CMake 3.16+, C++17

### 方案 C：Backend1 直连 + Backend2 云部署

```
甲方服务器          乙方云服务器
┌──────────┐      ┌──────────────┐
│ Frontend │      │              │
│ Backend1 │──→   │  Backend2    │
└──────────┘      │  (云部署)     │
                  └──────────────┘
```

- Backend2 部署在云服务器上，暴露 9000 端口
- Backend1 通过公网/专线访问

---

## 7. 通信参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 协议 | HTTP/1.1 | |
| 数据格式 | JSON | Content-Type: application/json |
| 端口 | 9000 | 可修改（见 `config.h`） |
| 超时 | 10 秒 | |
| 会话超时 | 3600 秒 | 会话在 Backend2 内存中，超时自动清理 |

## 8. 错误处理

| 情况 | Backend2 响应 | Backend1 应如何处理 |
|------|---------------|-------------------|
| 历史数据 < 200 条 | `status=1, "need >= 200 history points"` | 等待更多数据或返回空预测 |
| Backend2 不可达 | 连接超时 | 返回 `status=1` 给前端 |
| 正常 | `status=0` | 正常返回 |

## 9. 物理模型参数

双方需要统一以下物理参数，确保计算结果一致：

| 参数 | 值 | 说明 |
|------|-----|------|
| 传动效率 η | 0.85 | 牵引和制动效率 |
| 电池牵引比 | 0.30 | 牵引时电池提供 30% 功率 |
| 电池制动比 | 0.50 | 制动时电池吸收 50% 功率 |
| 辅助功耗 | 120 kW | 惰行/停站时的辅助功率 |
| 电池容量 | 200 kWh | |
| SOC 充电截止 | 99% | SOC ≥ 99%，禁止充电 |
| SOC 放电截止 | 10% | SOC ≤ 10%，禁止放电 |
