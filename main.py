import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import RiskClassifier

ARTIFACTS = Path(__file__).parent / "artifacts"
LABELS = {0: "low", 1: "medium", 2: "high"}

app = FastAPI(
    title="Wildfire Risk Classifier",
    description="4-layer MLP trained on NASA FIRMS + Open-Meteo data. Classifies wildfire risk into low, medium, or high.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts once at startup
_checkpoint = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True)
_model = RiskClassifier(_checkpoint["input_size"])
_model.load_state_dict(_checkpoint["state_dict"])
_model.eval()

with open(ARTIFACTS / "scaler.pkl", "rb") as _f:
    _scaler = pickle.load(_f)


class PredictRequest(BaseModel):
    daynight_N: float
    lat: float
    lon: float
    fire_weather_index: float
    pressure_mean: float
    wind_direction_mean: float
    wind_direction_std: float
    solar_radiation_mean: float
    dewpoint_mean: float
    cloud_cover_mean: float
    evapotranspiration_total: float
    humidity_min: float
    temp_mean: float
    temp_range: float
    wind_speed_max: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "daynight_N": 0.0,
                "lat": -15.2,
                "lon": 38.5,
                "fire_weather_index": 5.6,
                "pressure_mean": 955.6,
                "wind_direction_mean": 136.1,
                "wind_direction_std": 43.6,
                "solar_radiation_mean": 250.3,
                "dewpoint_mean": 15.9,
                "cloud_cover_mean": 18.4,
                "evapotranspiration_total": 4.9,
                "humidity_min": 35.0,
                "temp_mean": 23.2,
                "temp_range": 12.9,
                "wind_speed_max": 13.5,
            }
        }
    }


class PredictResponse(BaseModel):
    risk_level: int
    label: str
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Mirror the preprocessing from train_and_export.py exactly
    fwi = float(np.clip(req.fire_weather_index, 0, 100))
    temp_humidity_interaction = req.temp_mean * (1 - req.humidity_min / 100)
    wind_fwi_interaction = req.wind_speed_max * fwi

    features = np.array([[
        req.daynight_N, req.lat, req.lon, fwi,
        req.pressure_mean, req.wind_direction_mean, req.wind_direction_std,
        req.solar_radiation_mean, req.dewpoint_mean, req.cloud_cover_mean,
        req.evapotranspiration_total, req.humidity_min,
        req.temp_mean, req.temp_range, req.wind_speed_max,
        temp_humidity_interaction, wind_fwi_interaction,
    ]], dtype=np.float64)

    scaled = _scaler.transform(features).astype(np.float32)
    tensor = torch.from_numpy(scaled)

    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1).squeeze().tolist()

    risk_level = int(np.argmax(probs))
    return PredictResponse(
        risk_level=risk_level,
        label=LABELS[risk_level],
        probabilities={
            "low": round(probs[0], 4),
            "medium": round(probs[1], 4),
            "high": round(probs[2], 4),
        },
    )
