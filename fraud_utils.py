# fraud_utils.py (enhanced)
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

# Optional dependencies (safe fallbacks)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    _HAS_IMB = True
except Exception:
    _HAS_IMB = False

# =============================
# Global holders for supervised model
# =============================
trained_model = None
scaler = None
feature_cols = None
label_col_global = "Class"


# ========= Supervised training =========
def train_model(
    df: pd.DataFrame,
    label_col: str = "Class",
    use_smote: bool = False,
    use_balanced_rf: bool = True,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Tuple[object, StandardScaler]:
    """Train a classifier with stronger imbalance handling."""

    global trained_model, scaler, feature_cols, label_col_global

    assert label_col in df.columns, f"Label column '{label_col}' not found."
    label_col_global = label_col

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    # Keep numeric
    X_num = X.select_dtypes(include=["int64", "float64"]).copy()
    non_numeric_cols = [c for c in X.columns if c not in X_num.columns]
    if len(non_numeric_cols) > 0:
        X_cat = pd.get_dummies(X[non_numeric_cols].astype(str), drop_first=True)
        X_proc = pd.concat([X_num, X_cat], axis=1)
    else:
        X_proc = X_num

    feature_cols = X_proc.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    # Choose model
    if _HAS_IMB and use_balanced_rf:
        model = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            sampling_strategy="auto",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    # Optional SMOTE
    if _HAS_IMB and use_smote:
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_scaled, y)
        model.fit(X_res, y_res)
    else:
        model.fit(X_scaled, y)

    trained_model = model
    return model, scaler


# ========= Threshold tuning =========
def suggest_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "max_f1",
    min_precision: float = 0.5,
) -> Dict[str, float]:
    """Return suggested threshold according to strategy."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thr = np.r_[0.0, thresholds]
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    if strategy == "target_precision":
        mask = precision >= float(min_precision)
        if np.any(mask):
            idx = np.argmax(recall[mask])
            best_thr = float(thr[mask][idx])
        else:
            idx = int(np.nanargmax(f1))
            best_thr = float(thr[idx])
    else:  # max_f1
        idx = int(np.nanargmax(f1))
        best_thr = float(thr[idx])

    return {
        "threshold": best_thr,
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


# ========= Supervised inference =========
def predict_supervised(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict probabilities and labels using the trained supervised model."""
    global trained_model, scaler, feature_cols, label_col_global
    assert trained_model is not None, "Model not trained. Call train_model first."

    X = df.drop(columns=[label_col_global], errors="ignore")

    X_num = X.select_dtypes(include=["int64", "float64"]).copy()
    non_numeric_cols = [c for c in X.columns if c not in X_num.columns]
    if len(non_numeric_cols) > 0:
        X_cat = pd.get_dummies(X[non_numeric_cols].astype(str), drop_first=True)
        X_proc = pd.concat([X_num, X_cat], axis=1)
    else:
        X_proc = X_num

    for c in feature_cols:
        if c not in X_proc.columns:
            X_proc[c] = 0
    X_proc = X_proc[feature_cols]

    X_scaled = scaler.transform(X_proc)
    y_proba = trained_model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= float(threshold)).astype(int)

    out = df.copy()
    out["Fraud_Probability"] = y_proba
    out["Fraud_Prediction"] = y_pred
    return out


# ========= Unsupervised anomaly detection =========
def anomaly_detect(
    df: pd.DataFrame,
    contamination: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Isolation Forest with dynamic contamination if not provided."""
    global label_col_global

    X = df.drop(columns=[label_col_global], errors="ignore")
    X_num = X.select_dtypes(include=["int64", "float64"]).copy()
    if X_num.shape[1] == 0:
        raise ValueError("No numeric columns available for anomaly detection.")

    if contamination is None:
        if label_col_global in df.columns:
            contamination = max(0.005, min(0.2, float(df[label_col_global].mean())))
        else:
            contamination = 0.02

    iso = IsolationForest(
        contamination=float(contamination),
        random_state=random_state,
        n_estimators=300,
        n_jobs=-1,
    )
    y_if = iso.fit_predict(X_num)
    scores = -iso.score_samples(X_num)

    out = df.copy()
    out["Anomaly_Score"] = scores
    out["Anomaly"] = (y_if == -1).astype(int)
    return out


# ========= Hybrid fusion =========
def hybrid_predict(
    df: pd.DataFrame,
    sup_threshold: float = 0.5,
    contamination: Optional[float] = None,
    strategy: str = "or",  # 'or', 'and', 'weighted'
    weight_supervised: float = 0.7,
    weight_anomaly: float = 0.3,
) -> pd.DataFrame:
    """Combine supervised probability and anomaly signal."""
    sup = predict_supervised(df, threshold=sup_threshold)
    ano = anomaly_detect(df, contamination=contamination)

    out = sup.copy()
    out["Anomaly_Score"] = ano["Anomaly_Score"]
    out["Anomaly"] = ano["Anomaly"]

    if strategy == "and":
        out["Hybrid_Prediction"] = (
            (out["Fraud_Probability"] >= sup_threshold) & (out["Anomaly"] == 1)
        ).astype(int)
    elif strategy == "weighted":
        s = out["Anomaly_Score"].values
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        combo = weight_supervised * out["Fraud_Probability"].values + weight_anomaly * s_norm
        out["Hybrid_Score"] = combo
        out["Hybrid_Prediction"] = (combo >= sup_threshold).astype(int)
    else:  # 'or'
        out["Hybrid_Prediction"] = (
            (out["Fraud_Probability"] >= sup_threshold) | (out["Anomaly"] == 1)
        ).astype(int)

    return out


# ========= Utilities =========
def get_pr_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
    p, r, t = precision_recall_curve(y_true, y_proba)
    return {"precision": p, "recall": r, "thresholds": t}
