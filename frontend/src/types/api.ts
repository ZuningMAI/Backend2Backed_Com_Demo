// Energy calculation request
export interface EnergyResultRequest {
  session_id?: string;
  tractive_force: number;
  electric_brake_force: number;
  speed: number;
  battery_power: number;
  soc: number;
  sample_interval: number;
  start_time: number;
  end_time: number;
}

// Energy calculation response
export interface EnergyResultData {
  real_time_energy: number;
  total_traction_energy: number;
  regenerative_energy: number;
  net_energy: number;
  battery_energy: number;
}

export interface EnergyResultResponse {
  status: number;
  message: string;
  data: EnergyResultData;
  timestamp: number;
  progress?: { current_ms: number; total_ms: number; percent: number; state: string };
  actual_curve?: CurvePoint[];
}

// Time predict request
export interface TimePredictRequest {
  session_id: string;
  lookback_window: number;
  forecast_horizon: number;
  model_type: string;
}

// Curve point
export interface CurvePoint {
  position: number;
  energy: number;
}

// Time predict response
export interface TimePredictData {
  actual_curve: CurvePoint[];
  predicted_curve: CurvePoint[];
}

export interface TimePredictResponse {
  status: number;
  message: string;
  data: TimePredictData;
  timestamp: number;
}

// Health check
export interface HealthResponse {
  status: string;
  service: string;
  timestamp_ms: number;
  dependencies: Record<string, unknown>;
  active_sessions: number;
}
