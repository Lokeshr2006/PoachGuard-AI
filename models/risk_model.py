"""
models/risk_model.py - Random Forest Risk Model
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib, os

FEATURE_COLS = ["incident_count","high_severity_count","patrol_count",
                "avg_patrol_hours","sensor_triggers","suspicious_triggers","night_incidents"]

class PoachingRiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8,
                                            random_state=42, class_weight="balanced")
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, df):
        X = df[FEATURE_COLS].fillna(0)
        y = df["risk_label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)
        self.is_trained = True
        y_pred = self.model.predict(X_test_s)
        print("\n📊 Model Evaluation:")
        print(classification_report(y_test, y_pred, target_names=["Low Risk","High Risk"]))
        return {"accuracy": self.model.score(X_test_s, y_test),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred)}

    def predict(self, df):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = df[FEATURE_COLS].fillna(0)
        X_s = self.scaler.transform(X)
        df = df.copy()
        df["risk_score"] = self.model.predict_proba(X_s)[:,1]
        df["risk_label"]  = self.model.predict(X_s)
        df["risk_level"]  = df["risk_score"].apply(
            lambda s: "🔴 HIGH" if s>=0.7 else "🟡 MEDIUM" if s>=0.4 else "🟢 LOW")
        return df

    def feature_importance(self):
        return pd.DataFrame({"feature": FEATURE_COLS,
                              "importance": self.model.feature_importances_}
                             ).sort_values("importance", ascending=False)

    def save(self, path="models/poachguard_model.pkl"):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"✅ Model saved to {path}")

    def load(self, path="models/poachguard_model.pkl"):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]; self.scaler = data["scaler"]
            self.is_trained = True
        else:
            raise FileNotFoundError(f"No model at {path}")