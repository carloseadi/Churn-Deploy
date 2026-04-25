import requests

BASE = "http://localhost:8000"

# ── Teste 1: health check ─────────────────────────────────────────────────────
r = requests.get(f"{BASE}/")
print("Health:", r.json())

# ── Teste 2: cliente de alto risco ────────────────────────────────────────────
cliente_alto_risco = {
    "tenure_months": 8,
    "monthly_charges": 95.0,
    "num_products": 1,
    "support_calls": 5,
    "has_contract": 0,
    "payment_delay_days": 15,
    "satisfaction_score": 2
}

r = requests.post(f"{BASE}/predict", json=cliente_alto_risco)
print("\nCliente alto risco:", r.json())

# ── Teste 3: cliente fiel ─────────────────────────────────────────────────────
cliente_fiel = {
    "tenure_months": 48,
    "monthly_charges": 35.0,
    "num_products": 3,
    "support_calls": 0,
    "has_contract": 1,
    "payment_delay_days": 0,
    "satisfaction_score": 5
}

r = requests.post(f"{BASE}/predict", json=cliente_fiel)
print("Cliente fiel:      ", r.json())

# ── Teste 4: batch ────────────────────────────────────────────────────────────
batch = [cliente_alto_risco, cliente_fiel]
r = requests.post(f"{BASE}/predict/batch", json=batch)
print("\nBatch:", r.json())
