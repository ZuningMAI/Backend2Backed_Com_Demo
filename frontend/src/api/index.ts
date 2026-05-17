import axios from "axios";
import type {
  EnergyResultRequest,
  EnergyResultResponse,
  TimePredictRequest,
  TimePredictResponse,
} from "../types/api";

const BASE_URL = "http://localhost:8000";

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

export async function healthCheck(): Promise<unknown> {
  const resp = await apiClient.get("/health");
  return resp.data;
}

export async function postEnergyResult(
  data: EnergyResultRequest
): Promise<EnergyResultResponse> {
  const resp = await apiClient.post("/vehicle/energy/result", data);
  return resp.data;
}

export async function postTimePredict(
  data: TimePredictRequest
): Promise<TimePredictResponse> {
  const resp = await apiClient.post("/vehicle/energy/time_predict", data);
  return resp.data;
}
