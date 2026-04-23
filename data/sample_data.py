"""
data/sample_data.py - Synthetic data generator
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_incident_data(n=300, seed=42):
    np.random.seed(seed); random.seed(seed)
    c1_lat = np.random.normal(-2.5, 0.3, n//2)
    c1_lon = np.random.normal(37.5, 0.3, n//2)
    c2_lat = np.random.normal(-2.0, 0.2, n//2)
    c2_lon = np.random.normal(38.1, 0.2, n//2)
    lats = np.concatenate([c1_lat, c2_lat])
    lons = np.concatenate([c1_lon, c2_lon])
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=random.randint(0, 730)) for _ in range(n)]
    hours = [random.randint(0, 23) for _ in range(n)]
    types = random.choices(["snare","gunshot","vehicle","camp","carcass"], weights=[35,25,20,10,10], k=n)
    severity = ["high" if t in ["gunshot","carcass"] else "medium" if t=="vehicle" else "low" for t in types]
    return pd.DataFrame({
        "incident_id": [f"INC{1000+i}" for i in range(n)],
        "latitude": lats, "longitude": lons,
        "date": dates, "hour": hours,
        "incident_type": types, "severity": severity,
        "reported_by": random.choices(["ranger","sensor","informant"], k=n),
    })


def generate_patrol_data(n=500, seed=42):
    np.random.seed(seed); random.seed(seed)
    lats = np.random.uniform(-3.0, -1.5, n)
    lons = np.random.uniform(37.0, 38.5, n)
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=random.randint(0, 730)) for _ in range(n)]
    return pd.DataFrame({
        "patrol_id": [f"PAT{2000+i}" for i in range(n)],
        "latitude": lats, "longitude": lons,
        "date": dates,
        "hour": [random.randint(5, 22) for _ in range(n)],
        "duration_hours": np.random.randint(1, 8, n),
        "rangers_count": np.random.randint(1, 6, n),
        "vehicle_used": random.choices([True, False], weights=[40,60], k=n),
    })


def generate_sensor_data(n=200, seed=42):
    np.random.seed(seed); random.seed(seed)
    lats = np.random.uniform(-3.0, -1.5, n)
    lons = np.random.uniform(37.0, 38.5, n)
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(days=random.randint(0, 365)) for _ in range(n)]
    types = random.choices(["motion","acoustic","thermal","camera_trap"], k=n)
    suspicious = [1 if (t in ["acoustic","thermal"] and random.random()>0.4) else 0 for t in types]
    return pd.DataFrame({
        "sensor_id": [f"SEN{3000+i}" for i in range(n)],
        "latitude": lats, "longitude": lons,
        "date": dates, "trigger_type": types,
        "suspicious_flag": suspicious,
        "confidence_score": np.round(np.random.uniform(0.4, 1.0, n), 2),
    })


def build_feature_dataset(incidents, patrols, sensors):
    lat_bins = np.arange(-3.0, -1.5, 0.1)
    lon_bins = np.arange(37.0, 38.5, 0.1)
    records = []
    for lat in lat_bins:
        for lon in lon_bins:
            inc = incidents[(incidents.latitude>=lat)&(incidents.latitude<lat+0.1)&
                            (incidents.longitude>=lon)&(incidents.longitude<lon+0.1)]
            pat = patrols[(patrols.latitude>=lat)&(patrols.latitude<lat+0.1)&
                          (patrols.longitude>=lon)&(patrols.longitude<lon+0.1)]
            sen = sensors[(sensors.latitude>=lat)&(sensors.latitude<lat+0.1)&
                          (sensors.longitude>=lon)&(sensors.longitude<lon+0.1)]
            incident_count = len(inc)
            high_sev = len(inc[inc.severity=="high"]) if len(inc)>0 else 0
            records.append({
                "lat": lat+0.05, "lon": lon+0.05,
                "incident_count": incident_count,
                "high_severity_count": high_sev,
                "patrol_count": len(pat),
                "avg_patrol_hours": round(pat.duration_hours.mean(),2) if len(pat)>0 else 0,
                "sensor_triggers": len(sen),
                "suspicious_triggers": int(sen.suspicious_flag.sum()) if len(sen)>0 else 0,
                "night_incidents": len(inc[inc.hour.between(20,23)]) if len(inc)>0 else 0,
                "risk_label": 1 if (incident_count>=2 or high_sev>=1) else 0,
            })
    return pd.DataFrame(records)