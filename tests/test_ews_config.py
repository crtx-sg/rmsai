#!/usr/bin/env python3
"""
Test cases for Early Warning System (EWS) configuration and scoring functionality.
Tests the NEWS2-based EWS implementation in RMSAI system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from config import (
    get_ews_scoring_template,
    get_ews_risk_categories,
    get_ews_score_for_vital,
    get_ews_risk_category,
    EWS_SCORING_TEMPLATE,
    EWS_RISK_CATEGORIES
)


class TestEWSConfiguration(unittest.TestCase):
    """Test EWS configuration functions and constants"""

    def test_ews_scoring_template_structure(self):
        """Test that EWS scoring template has correct structure"""
        template = get_ews_scoring_template()

        # Check required vital signs are present
        required_vitals = ['heart_rate', 'respiratory_rate', 'systolic_bp', 'temperature', 'oxygen_saturation', 'consciousness']
        for vital in required_vitals:
            self.assertIn(vital, template, f"Missing vital sign: {vital}")

        # Check structure of each vital
        for vital_name, vital_config in template.items():
            self.assertIn('ranges', vital_config)
            self.assertIn('units', vital_config)
            self.assertIn('display_name', vital_config)

            # Check ranges structure
            for range_config in vital_config['ranges']:
                self.assertIn('range', range_config)
                self.assertIn('score', range_config)
                # Should have either min/max or value
                self.assertTrue(
                    ('min' in range_config and 'max' in range_config) or 'value' in range_config,
                    f"Range config missing min/max or value for {vital_name}"
                )

    def test_ews_risk_categories_structure(self):
        """Test that EWS risk categories have correct structure"""
        categories = get_ews_risk_categories()

        # Check required categories
        required_categories = ['low', 'medium', 'high']
        for category in required_categories:
            self.assertIn(category, categories)

            config = categories[category]
            self.assertIn('score_range', config)
            self.assertIn('category', config)
            self.assertIn('color', config)
            self.assertIn('monitoring_frequency', config)
            self.assertIn('clinical_response', config)

            # Check score_range is tuple
            self.assertIsInstance(config['score_range'], tuple)
            self.assertEqual(len(config['score_range']), 2)

    def test_heart_rate_scoring(self):
        """Test heart rate EWS scoring"""
        # Test different heart rate values and expected scores
        test_cases = [
            (30, 3),   # ≤40 -> score 3
            (45, 1),   # 41-50 -> score 1
            (75, 0),   # 51-90 -> score 0
            (100, 1),  # 91-110 -> score 1
            (120, 2),  # 111-130 -> score 2
            (150, 3),  # ≥131 -> score 3
        ]

        for hr_value, expected_score in test_cases:
            with self.subTest(heart_rate=hr_value):
                actual_score = get_ews_score_for_vital('heart_rate', hr_value)
                self.assertEqual(actual_score, expected_score,
                               f"Heart rate {hr_value} should score {expected_score}, got {actual_score}")

    def test_respiratory_rate_scoring(self):
        """Test respiratory rate EWS scoring"""
        test_cases = [
            (7, 3),    # ≤8 -> score 3
            (10, 1),   # 9-11 -> score 1
            (16, 0),   # 12-20 -> score 0
            (22, 2),   # 21-24 -> score 2
            (30, 3),   # ≥25 -> score 3
        ]

        for rr_value, expected_score in test_cases:
            with self.subTest(respiratory_rate=rr_value):
                actual_score = get_ews_score_for_vital('respiratory_rate', rr_value)
                self.assertEqual(actual_score, expected_score,
                               f"Respiratory rate {rr_value} should score {expected_score}, got {actual_score}")

    def test_systolic_bp_scoring(self):
        """Test systolic blood pressure EWS scoring"""
        test_cases = [
            (85, 3),   # ≤90 -> score 3
            (95, 2),   # 91-100 -> score 2
            (105, 1),  # 101-110 -> score 1
            (150, 0),  # 111-219 -> score 0
            (230, 3),  # ≥220 -> score 3
        ]

        for sbp_value, expected_score in test_cases:
            with self.subTest(systolic_bp=sbp_value):
                actual_score = get_ews_score_for_vital('systolic_bp', sbp_value)
                self.assertEqual(actual_score, expected_score,
                               f"Systolic BP {sbp_value} should score {expected_score}, got {actual_score}")

    def test_temperature_scoring(self):
        """Test temperature EWS scoring"""
        test_cases = [
            (34.5, 3),  # ≤35.0 -> score 3
            (35.5, 1),  # 35.1-36.0 -> score 1
            (37.0, 0),  # 36.1-38.0 -> score 0
            (38.5, 1),  # 38.1-39.0 -> score 1
            (40.0, 2),  # ≥39.1 -> score 2
        ]

        for temp_value, expected_score in test_cases:
            with self.subTest(temperature=temp_value):
                actual_score = get_ews_score_for_vital('temperature', temp_value)
                self.assertEqual(actual_score, expected_score,
                               f"Temperature {temp_value} should score {expected_score}, got {actual_score}")

    def test_oxygen_saturation_scoring(self):
        """Test oxygen saturation EWS scoring"""
        test_cases = [
            (88, 3),   # ≤91 -> score 3
            (92, 2),   # 92-93 -> score 2
            (94, 1),   # 94-95 -> score 1
            (98, 0),   # ≥96 -> score 0
        ]

        for spo2_value, expected_score in test_cases:
            with self.subTest(oxygen_saturation=spo2_value):
                actual_score = get_ews_score_for_vital('oxygen_saturation', spo2_value)
                self.assertEqual(actual_score, expected_score,
                               f"SpO2 {spo2_value}% should score {expected_score}, got {actual_score}")

    def test_consciousness_scoring(self):
        """Test consciousness level EWS scoring"""
        test_cases = [
            ('alert', 0),
            ('Alert', 0),  # Case insensitive
            ('cvpu', 3),
            ('CVPU', 3),   # Case insensitive
        ]

        for consciousness_value, expected_score in test_cases:
            with self.subTest(consciousness=consciousness_value):
                actual_score = get_ews_score_for_vital('consciousness', consciousness_value)
                self.assertEqual(actual_score, expected_score,
                               f"Consciousness '{consciousness_value}' should score {expected_score}, got {actual_score}")

    def test_risk_category_classification(self):
        """Test EWS risk category classification"""
        test_cases = [
            (0, 'Low Risk'),
            (2, 'Low Risk'),
            (4, 'Low Risk'),
            (5, 'Medium Risk'),
            (6, 'Medium Risk'),
            (7, 'High Risk'),
            (10, 'High Risk'),
            (15, 'High Risk'),
        ]

        for total_score, expected_category in test_cases:
            with self.subTest(total_score=total_score):
                risk_info = get_ews_risk_category(total_score)
                self.assertEqual(risk_info['category'], expected_category,
                               f"EWS score {total_score} should be {expected_category}, got {risk_info['category']}")

                # Check all required fields are present
                required_fields = ['category', 'color', 'monitoring_frequency', 'clinical_response', 'level']
                for field in required_fields:
                    self.assertIn(field, risk_info, f"Missing field '{field}' in risk info for score {total_score}")

    def test_invalid_vital_scoring(self):
        """Test scoring for non-existent vital signs"""
        invalid_score = get_ews_score_for_vital('invalid_vital', 100)
        self.assertEqual(invalid_score, 0, "Invalid vital signs should return score 0")

    def test_edge_case_values(self):
        """Test edge case values for EWS scoring"""
        # Test boundary values
        edge_cases = [
            ('heart_rate', 40, 3),   # Exactly at boundary
            ('heart_rate', 41, 1),   # Just above boundary
            ('heart_rate', 90, 0),   # Upper boundary
            ('heart_rate', 91, 1),   # Just above boundary
        ]

        for vital_name, value, expected_score in edge_cases:
            with self.subTest(vital=vital_name, value=value):
                actual_score = get_ews_score_for_vital(vital_name, value)
                self.assertEqual(actual_score, expected_score,
                               f"{vital_name} {value} should score {expected_score}, got {actual_score}")


class TestEWSIntegration(unittest.TestCase):
    """Test EWS integration with vitals analysis"""

    def test_vitals_analyzer_ews_calculation(self):
        """Test EWS calculation in vitals analyzer"""
        try:
            from rmsai_vitals_analysis import RMSAIVitalsAnalyzer

            analyzer = RMSAIVitalsAnalyzer()

            # Test vitals data
            test_vitals = {
                'heart_rate': 120,      # Score: 2
                'respiratory_rate': 22, # Score: 2
                'systolic_bp': 95,      # Score: 2
                'temperature': 38.5,    # Score: 1
                'oxygen_saturation': 94 # Score: 1
            }

            # Calculate EWS
            ews_result = analyzer.calculate_ews_score(test_vitals)

            # Check structure
            required_fields = ['total_score', 'risk_category', 'risk_color', 'clinical_response',
                             'monitoring_frequency', 'risk_level', 'score_breakdown', 'timestamp']
            for field in required_fields:
                self.assertIn(field, ews_result, f"Missing field '{field}' in EWS result")

            # Check total score calculation (2+2+2+1+1 = 8)
            expected_total = 8
            self.assertEqual(ews_result['total_score'], expected_total,
                           f"Expected total score {expected_total}, got {ews_result['total_score']}")

            # Check risk category (score 8 should be High Risk)
            self.assertEqual(ews_result['risk_category'], 'High Risk')

        except ImportError:
            self.skipTest("rmsai_vitals_analysis module not available")

    def test_vital_mapping_in_analyzer(self):
        """Test vital sign key mapping in analyzer"""
        try:
            from rmsai_vitals_analysis import RMSAIVitalsAnalyzer

            analyzer = RMSAIVitalsAnalyzer()

            # Test with different key formats
            test_vitals_formats = [
                {'HR': 100, 'RespRate': 18, 'Systolic': 120, 'Temp': 37.0, 'SpO2': 98},
                {'heart_rate': 100, 'respiratory_rate': 18, 'systolic_bp': 120, 'temperature': 37.0, 'oxygen_saturation': 98}
            ]

            for vitals in test_vitals_formats:
                with self.subTest(vitals_format=vitals):
                    ews_result = analyzer.calculate_ews_score(vitals)
                    self.assertIsInstance(ews_result['total_score'], int)
                    self.assertIn('score_breakdown', ews_result)

        except ImportError:
            self.skipTest("rmsai_vitals_analysis module not available")

    def test_linear_regression_trend_analysis(self):
        """Test linear regression trend analysis functionality"""
        try:
            from rmsai_vitals_analysis import RMSAIVitalsAnalyzer
            import pandas as pd

            analyzer = RMSAIVitalsAnalyzer()

            # Test with trending up data (deteriorating)
            df_deteriorating = pd.DataFrame({
                'time_hours': [0, 1, 2, 3, 4],
                'ews': [2, 3, 4, 6, 8]  # Clear upward trend
            })

            result = analyzer._linear_regression_trend(df_deteriorating)

            # Check structure
            required_fields = ['slope', 'r_squared', 'p_value', 'trend', 'significance']
            for field in required_fields:
                self.assertIn(field, result)

            # Check trend detection
            self.assertGreater(result['slope'], 0, "Slope should be positive for deteriorating trend")
            self.assertEqual(result['trend'], 'deteriorating', "Should detect deteriorating trend")

            # Test with stable data
            df_stable = pd.DataFrame({
                'time_hours': [0, 1, 2, 3, 4],
                'ews': [3, 3, 3, 3, 3]  # Stable trend
            })

            result_stable = analyzer._linear_regression_trend(df_stable)
            self.assertEqual(result_stable['trend'], 'stable', "Should detect stable trend for flat data")

            # Test with insufficient data
            df_insufficient = pd.DataFrame({
                'time_hours': [0],
                'ews': [3]
            })

            result_insufficient = analyzer._linear_regression_trend(df_insufficient)
            self.assertEqual(result_insufficient['trend'], 'insufficient_data', "Should handle insufficient data")

        except ImportError:
            self.skipTest("rmsai_vitals_analysis module not available")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)