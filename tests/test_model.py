"""
tests/test_model.py - Unit Tests
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import pandas as pd
from data.sample_data import (generate_incident_data, generate_patrol_data,
                               generate_sensor_data, build_feature_dataset)
from models.risk_model import PoachingRiskModel
from utils.geo_analysis import generate_alerts, compute_hotspot_summary

class TestData(unittest.TestCase):
    def test_incidents(self):
        df = generate_incident_data(n=50)
        self.assertEqual(len(df), 50)
        self.assertIn("severity", df.columns)

    def test_features(self):
        features = build_feature_dataset(
            generate_incident_data(100), generate_patrol_data(100), generate_sensor_data(50))
        self.assertIn("risk_label", features.columns)

class TestModel(unittest.TestCase):
    def setUp(self):
        features = build_feature_dataset(
            generate_incident_data(200), generate_patrol_data(200), generate_sensor_data(100))
        self.model = PoachingRiskModel()
        self.model.train(features)
        self.preds = self.model.predict(features)

    def test_trained(self):    self.assertTrue(self.model.is_trained)
    def test_scores(self):     self.assertTrue((self.preds["risk_score"].between(0,1)).all())
    def test_risk_levels(self):self.assertIn("risk_level", self.preds.columns)

class TestAlerts(unittest.TestCase):
    def test_summary(self):
        dummy = pd.DataFrame({"risk_score":[0.9,0.6,0.3,0.8,0.1],"incident_count":[5,2,0,3,1]})
        s = compute_hotspot_summary(dummy)
        self.assertEqual(s["high_risk"], 2)

if __name__ == "__main__": unittest.main(verbosity=2)