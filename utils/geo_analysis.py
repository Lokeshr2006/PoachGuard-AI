"""
utils/geo_analysis.py - Geospatial Analysis & Alert Generation
"""
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime


def generate_alerts(predictions, threshold=0.7):
    alerts = predictions[predictions["risk_score"]>=threshold].copy()
    alerts = alerts.sort_values("risk_score", ascending=False).reset_index(drop=True)
    alerts["alert_id"]  = [f"ALERT-{1000+i}" for i in range(len(alerts))]
    alerts["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts["priority"]  = alerts["risk_score"].apply(
        lambda s: "CRITICAL" if s>=0.85 else "HIGH" if s>=0.7 else "MEDIUM")
    return alerts[["alert_id","lat","lon","risk_score","priority","timestamp",
                   "incident_count","suspicious_triggers"]]


def build_risk_map(predictions, incidents):
    m = folium.Map(location=[predictions["lat"].mean(), predictions["lon"].mean()],
                   zoom_start=9, tiles="CartoDB dark_matter")

    heat_data = [[r["lat"],r["lon"],r["risk_score"]]
                 for _,r in predictions.iterrows() if r["risk_score"]>0.1]
    HeatMap(heat_data, name="Risk Heatmap", radius=18, blur=15,
            gradient={"0.3":"blue","0.5":"yellow","0.7":"orange","1.0":"red"}).add_to(m)

    for _,r in predictions[predictions["risk_score"]>=0.7].iterrows():
        folium.Circle(location=[r["lat"],r["lon"]], radius=5000,
                      color="red", fill=True, fill_opacity=0.25,
                      tooltip=f"⚠️ Risk: {r['risk_score']:.0%} | Incidents: {r['incident_count']}"
                      ).add_to(m)

    cluster = MarkerCluster(name="Incidents").add_to(m)
    color_map = {"high":"red","medium":"orange","low":"green"}
    for _,r in incidents.iterrows():
        folium.Marker(
            location=[r["latitude"],r["longitude"]],
            popup=f"{r['incident_type'].upper()} | {r['severity']} | Hour:{r['hour']}",
            icon=folium.Icon(color=color_map.get(r["severity"],"gray"),
                             icon="exclamation-sign"),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


def compute_hotspot_summary(predictions):
    high   = predictions[predictions["risk_score"]>=0.7]
    medium = predictions[(predictions["risk_score"]>=0.4)&(predictions["risk_score"]<0.7)]
    low    = predictions[predictions["risk_score"]<0.4]
    return {
        "total_cells": len(predictions),
        "high_risk": len(high), "medium_risk": len(medium), "low_risk": len(low),
        "max_risk_score": round(predictions["risk_score"].max(), 3),
        "avg_risk_score": round(predictions["risk_score"].mean(), 3),
    }