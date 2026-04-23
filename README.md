# 🐆 PoachGuard AI — Wildlife Poaching Detection System

> Developed for **Panthera** | Python · Machine Learning · Geospatial Analysis

PoachGuard AI analyzes patrol data, sensor inputs, and historical incident records to identify high-risk zones and suspicious patterns — generating real-time alerts and visual dashboards to assist conservation teams in proactive wildlife protection.

---

## ✨ Key Features

- Risk Zone Detection — Random Forest model predicts high-risk poaching zones
- Geospatial Heatmaps — Interactive Folium maps with incident clustering
- Real-Time Alert System — CRITICAL / HIGH / MEDIUM priority alerts
- Pattern Analysis — Time-series analysis of patrol and incident data
- Interactive Dashboard — Streamlit UI with live filters and charts
- Exportable Outputs — Risk maps as HTML, alerts as CSV

---

## 📁 Project Structure
poachguard-ai/
├── data/
│   └── sample_data.py       # Generates patrol, sensor, incident data
├── models/
│   └── risk_model.py        # Random Forest ML model
├── utils/
│   └── geo_analysis.py      # Geospatial analysis and alert generation
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── tests/
│   └── test_model.py        # Unit tests
├── main.py                  # Main pipeline entry point
└── requirements.txt

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/poachguard-ai.git
cd poachguard-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Input Features:** incident count, severity, patrol frequency, sensor triggers, night incidents
- **Output:** Risk score (0.0 to 1.0) + Risk level (HIGH / MEDIUM / LOW)
- **Alert Priority:** CRITICAL (≥85%) · HIGH (≥70%) · MEDIUM (≥50%)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Scikit-learn | ML model training |
| Pandas / NumPy | Data processing |
| Folium | Geospatial maps |
| Streamlit | Dashboard UI |
| Matplotlib | Charts |
| Joblib | Save/load model |

---

## 📊 Data Sources

- **Incident Records** — Historical poaching events (type, severity, location, time)
- **Patrol Logs** — Ranger patrol coverage and duration
- **Sensor Events** — Camera traps, acoustic and thermal sensor triggers

---

## 🌍 Project Impact

PoachGuard AI moves conservation teams from reactive to proactive wildlife protection. Rangers receive predicted risk zones and alerts in advance — enabling smarter deployment of limited resources to protect endangered animals.

---

*PoachGuard AI © 2024 | Built for Panthera Wildlife Conservation*
