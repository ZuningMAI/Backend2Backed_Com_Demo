"""
External API routes for energy calculation and prediction.
Phase 3: data from TDengine via scheduler, not from frontend telemetry.
"""
import time
import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    EnergyResultResponse, EnergyResultData,
    TimePredictResponse, TimePredictData, CurvePoint,
)
from app.services.scheduler import (
    create_session, get_energy_result, get_predict_result, _sessions
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vehicle/energy", tags=["energy"])


@router.post("/result", response_model=EnergyResultResponse)
async def energy_result(req: dict):
    """
    Get current energy metrics for a session.
    If session_id is null or not found, create a new session.
    Frontend passes: {session_id, start_time_ms, end_time_ms}
    """
    sid = req.get("session_id")
    start_ms = req.get("start_time", 0)
    end_ms = req.get("end_time", 0)
    total_ms = end_ms - start_ms if end_ms > start_ms else 2813000

    if not sid or sid not in _sessions:
        sid = create_session(total_ms)

    result = get_energy_result(sid)
    return EnergyResultResponse(
        status=result["status"],
        message=f"session={sid}",
        data=EnergyResultData(**result.get("data", {})),
        timestamp=int(time.time() * 1000),
        progress=result.get("progress"),
        actual_curve=result.get("actual_curve", []),
    )


@router.post("/time_predict", response_model=TimePredictResponse)
async def time_predict(req: dict):
    sid = req.get("session_id")
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")

    result = get_predict_result(sid)
    data = result.get("data", {})
    return TimePredictResponse(
        status=result["status"],
        message="success",
        data=TimePredictData(
            actual_curve=[CurvePoint(position=p.get("position",0) or 0, energy=p.get("energy") or 0) for p in data.get("actual_curve", [])],
            predicted_curve=[CurvePoint(position=p.get("position",0) or 0, energy=p.get("energy") or 0) for p in data.get("predicted_curve", [])],
        ),
        timestamp=int(time.time() * 1000),
        progress=result.get("progress"),
    )
