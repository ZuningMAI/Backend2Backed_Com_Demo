"""
Pydantic models for all API request/response schemas.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


# ---------- Energy Result ----------

class EnergyResultRequest(BaseModel):
    session_id: Optional[str] = None
    tractive_force: float = 0.0
    electric_brake_force: float = 0.0
    speed: float = 0.0
    battery_power: float = 0.0
    soc: float = Field(default=0.0, ge=0.0, le=100.0)
    sample_interval: float = 1.0
    start_time: int
    end_time: int


class EnergyResultData(BaseModel):
    real_time_energy: float = 0.0
    total_traction_energy: float = 0.0
    regenerative_energy: float = 0.0
    net_energy: float = 0.0
    battery_energy: float = 0.0


class EnergyResultResponse(BaseModel):
    status: int = 0
    message: str = "success"
    data: EnergyResultData
    timestamp: int = 0
    progress: Optional[dict] = None
    actual_curve: list[dict] = []


# ---------- Time Predict ----------

class TimePredictRequest(BaseModel):
    session_id: str
    lookback_window: int = 1000
    forecast_horizon: int = 200
    model_type: str = "math_only"


class CurvePoint(BaseModel):
    position: float
    energy: float


class TimePredictData(BaseModel):
    actual_curve: list[CurvePoint] = []
    predicted_curve: list[CurvePoint] = []


class TimePredictResponse(BaseModel):
    status: int = 0
    message: str = "success"
    data: TimePredictData
    timestamp: int = 0
    progress: Optional[dict] = None


# ---------- Internal (Backend 1 -> Backend 2) ----------

class DataPoint(BaseModel):
    time: int
    tractive_force: float = 0.0
    electric_brake_force: float = 0.0
    speed: float = 0.0
    battery_power: float = 0.0
    soc: float = 0.0


class InternalCalcRequest(BaseModel):
    session_id: str
    data_point: DataPoint
    sample_interval: float = 1.0
    start_time: int = 0
    end_time: int = 0


class InternalMathPredictRequest(BaseModel):
    history_data: list[dict[str, Any]] = []
    forecast_horizon: int = 200
