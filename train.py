import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# ── 1. Dataset sintético de churn ────────────────────────────────────────────
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "tenure_months":      np.random.randint(1, 72, n),
    "monthly_charges":    np.random.uniform(20, 120, n),
    "num_products":       np.random.randint(1, 5, n),
    "support_calls":      np.random.poisson(2, n),
    "has_contract":       np.random.binomial(1, 0.5, n),
    "payment_delay_days": np.random.exponential(5, n).astype(int),
    "satisfaction_score": np.random.randint(1, 6, n),
})

# Regra de churn com ruído
churn_prob = (
    0.3 * (df["tenure_months"] < 12) +
    0.2 * (df["monthly_charges"] > 80) +
    0.2 * (df["support_calls"] > 3) +
    0.15 * (df["satisfaction_score"] <= 2) +
    0.15 * (df["has_contract"] == 0) +
    0.1 * (df["payment_delay_days"] > 10)
)
churn_prob = churn_prob / churn_prob.max()
df["churn"] = (np.random.rand(n) < churn_prob).astype(int)

print(f"Dataset: {n} clientes | Churn rate: {df['churn'].mean():.1%}")

# ── 2. Split ──────────────────────────────────────────────────────────────────
FEATURES = ["tenure_months", "monthly_charges", "num_products",
            "support_calls", "has_contract", "payment_delay_days", "satisfaction_score"]

X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 3. XGBoost ───────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42,
)

model.fit(
    X_train_s, y_train,
    eval_set=[(X_test_s, y_test)],
    verbose=False,
)

# ── 4. Avaliação ─────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Ativo", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\n── Feature Importances ────────────────────────────────")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ── 5. Salvar pkl ─────────────────────────────────────────────────────────────
artifact = {"model": model, "scaler": scaler, "features": FEATURES}
with open("churn_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n✓ Modelo salvo em churn_model.pkl")
