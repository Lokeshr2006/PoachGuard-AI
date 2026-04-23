"""
main.py - PoachGuard AI Main Pipeline
"""
import os
import sys

from data.sample_data import (
    generate_incident_data, generate_patrol_data,
    generate_sensor_data, build_feature_dataset,
)
from models.risk_model import PoachingRiskModel
from utils.geo_analysis import generate_alerts, build_risk_map, compute_hotspot_summary


def run_pipeline():
    print("=" * 60)
    print("🐆  PoachGuard AI — Poaching Detection Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating data...")
    incidents = generate_incident_data(n=300)
    patrols = generate_patrol_data(n=500)
    sensors = generate_sensor_data(n=200)
    features = build_feature_dataset(incidents, patrols, sensors)
    print(f"      Grid cells: {len(features)} | High-risk: {features.risk_label.sum()}")

    print("\n[2/4] Training Random Forest model...")
    model = PoachingRiskModel()
    results = model.train(features)
    print(f"      Accuracy: {results['accuracy']:.2%}")

    os.makedirs("models", exist_ok=True)
    model.save("models/poachguard_model.pkl")

    print("\n[3/4] Generating predictions...")
    predictions = model.predict(features)
    alerts = generate_alerts(predictions, threshold=0.70)
    summary = compute_hotspot_summary(predictions)

    print(f"\n📊 RISK SUMMARY")
    print(f"   🔴 High Risk Zones  : {summary['high_risk']}")
    print(f"   🟡 Medium Risk Zones: {summary['medium_risk']}")
    print(f"   🟢 Low Risk Zones   : {summary['low_risk']}")
    print(f"   🚨 Active Alerts    : {len(alerts)}")

    print("\n[4/4] Generating map...")
    os.makedirs("outputs", exist_ok=True)
    risk_map = build_risk_map(predictions, incidents)
    risk_map.save("outputs/poachguard_map.html")
    alerts.to_csv("outputs/active_alerts.csv", index=False)
    print("      ✅ Map saved to outputs/poachguard_map.html")
    print("      ✅ Alerts saved to outputs/active_alerts.csv")

    print("\n✅  Pipeline complete!")
    print("👉  Run dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    run_pipeline()