"""
dashboard/app.py - Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from data.sample_data import (generate_incident_data, generate_patrol_data,
                               generate_sensor_data, build_feature_dataset)
from models.risk_model import PoachingRiskModel
from utils.geo_analysis import generate_alerts, build_risk_map, compute_hotspot_summary

st.set_page_config(page_title="PoachGuard AI", page_icon="🐆", layout="wide")
st.markdown("""<style>
.alert-high{background:#3d0c0c;border-left:4px solid #e63946;padding:8px;border-radius:6px;margin:4px 0}
.alert-critical{background:#5c0a0a;border-left:4px solid #ff0000;padding:8px;border-radius:6px;margin:4px 0}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    incidents = generate_incident_data()
    patrols   = generate_patrol_data()
    sensors   = generate_sensor_data()
    features  = build_feature_dataset(incidents, patrols, sensors)
    model     = PoachingRiskModel()
    model.train(features)
    predictions = model.predict(features)
    alerts      = generate_alerts(predictions)
    summary     = compute_hotspot_summary(predictions)
    return incidents, patrols, sensors, predictions, alerts, summary, model

incidents, patrols, sensors, predictions, alerts, summary, model = load_pipeline()

# Sidebar
with st.sidebar:
    st.markdown("## 🐆 PoachGuard AI")
    st.markdown("**Wildlife Protection System for Panthera**")
    st.divider()
    risk_threshold = st.slider("Alert Threshold", 0.5, 0.99, 0.70, 0.05)
    st.divider()
    st.write(f"📍 Incidents: `{len(incidents)}`")
    st.write(f"🚶 Patrols: `{len(patrols)}`")
    st.write(f"📡 Sensors: `{len(sensors)}`")

# Header
st.markdown("# 🐆 PoachGuard AI — Real-Time Threat Dashboard")
st.caption("Developed for Panthera | ML-powered poaching detection")
st.divider()

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("🔴 High Risk",    summary["high_risk"])
c2.metric("🟡 Medium Risk",  summary["medium_risk"])
c3.metric("🟢 Low Risk",     summary["low_risk"])
c4.metric("🚨 Alerts",       len(alerts[alerts["risk_score"]>=risk_threshold]))
c5.metric("📈 Max Score",    f"{summary['max_risk_score']:.0%}")
st.divider()

# Map + Alerts
col_map, col_alerts = st.columns([3,1])
with col_map:
    st.subheader("🗺️ Geospatial Risk Map")
    st_folium(build_risk_map(predictions, incidents), width=750, height=480)

with col_alerts:
    st.subheader("🚨 Active Alerts")
    filtered = alerts[alerts["risk_score"]>=risk_threshold].head(15)
    for _,row in filtered.iterrows():
        css = "alert-critical" if row["priority"]=="CRITICAL" else "alert-high"
        st.markdown(f"""<div class="{css}">
        <b>{row['alert_id']}</b> — {row['priority']}<br>
        📍 ({row['lat']:.2f}, {row['lon']:.2f})<br>
        Risk: <b>{row['risk_score']:.0%}</b>
        </div>""", unsafe_allow_html=True)

st.divider()

# Charts
st.subheader("📊 Analytics")
ch1,ch2,ch3 = st.columns(3)

with ch1:
    st.markdown("**Incident Types**")
    fig,ax = plt.subplots(figsize=(4,3), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    counts = incidents["incident_type"].value_counts()
    ax.barh(counts.index, counts.values,
            color=["#e63946","#f4a261","#2a9d8f","#e9c46a","#a8dadc"])
    ax.tick_params(colors="white")
    [s.set_visible(False) for s in ax.spines.values()]
    st.pyplot(fig)

with ch2:
    st.markdown("**Incidents by Hour**")
    fig2,ax2 = plt.subplots(figsize=(4,3), facecolor="#1e2130")
    ax2.set_facecolor("#1e2130")
    hc = incidents["hour"].value_counts().sort_index()
    ax2.plot(hc.index, hc.values, color="#e63946", linewidth=2)
    ax2.fill_between(hc.index, hc.values, alpha=0.3, color="#e63946")
    ax2.tick_params(colors="white")
    [s.set_visible(False) for s in ax2.spines.values()]
    st.pyplot(fig2)

with ch3:
    st.markdown("**Feature Importance**")
    fi = model.feature_importance()
    fig3,ax3 = plt.subplots(figsize=(4,3), facecolor="#1e2130")
    ax3.set_facecolor("#1e2130")
    colors = ["#e63946" if i==0 else "#a8dadc" for i in range(len(fi))]
    ax3.barh(fi["feature"], fi["importance"], color=colors)
    ax3.tick_params(colors="white", labelsize=7)
    [s.set_visible(False) for s in ax3.spines.values()]
    st.pyplot(fig3)

st.divider()
st.subheader("📋 Risk Predictions")
show = predictions[["lat","lon","risk_score","risk_level","incident_count",
                     "suspicious_triggers","patrol_count"]
                   ].sort_values("risk_score", ascending=False)
st.dataframe(show.head(50), use_container_width=True)
st.caption("PoachGuard AI © 2024 | Panthera Wildlife Conservation")