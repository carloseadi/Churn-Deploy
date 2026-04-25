import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Carrega artefato ──────────────────────────────────────────────────────────
with open("churn_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model   = artifact["model"]
scaler  = artifact["scaler"]
FEATURES = artifact["features"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Predictor API", version="1.0")

class ClienteInput(BaseModel):
    tenure_months:      int   = Field(..., ge=1,  le=120,  example=8)
    monthly_charges:    float = Field(..., ge=0,  le=500,  example=95.0)
    num_products:       int   = Field(..., ge=1,  le=10,   example=1)
    support_calls:      int   = Field(..., ge=0,  le=50,   example=5)
    has_contract:       int   = Field(..., ge=0,  le=1,    example=0)
    payment_delay_days: int   = Field(..., ge=0,  le=365,  example=15)
    satisfaction_score: int   = Field(..., ge=1,  le=5,    example=2)

class PredictResponse(BaseModel):
    churn: bool
    probabilidade_churn: float
    risco: str

@app.get("/")
def root():
    return {"status": "ok", "modelo": "XGBoost Churn", "features": FEATURES}

@app.post("/predict", response_model=PredictResponse)
def predict(cliente: ClienteInput):
    X = np.array([[getattr(cliente, f) for f in FEATURES]])
    X_scaled = scaler.transform(X)

    churn_flag = bool(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])

    if prob >= 0.7:
        risco = "Alto"
    elif prob >= 0.4:
        risco = "Médio"
    else:
        risco = "Baixo"

    return PredictResponse(churn=churn_flag, probabilidade_churn=round(prob, 4), risco=risco)

@app.post("/predict/batch")
def predict_batch(clientes: list[ClienteInput]):
    if len(clientes) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 clientes por batch")

    X = np.array([[getattr(c, f) for f in FEATURES] for c in clientes])
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    return [
        {"id": i, "probabilidade_churn": round(float(p), 4), "churn": p >= 0.5}
        for i, p in enumerate(probs)
    ]
