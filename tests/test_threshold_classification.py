#!/usr/bin/env python3
"""
Test suite for threshold values and heart rate-based anomaly classification
===========================================================================

Tests the new optimized threshold system and heart rate-based classification
logic for distinguishing between Tachycardia and Bradycardia.

Usage:
    python test_threshold_classification.py
"""

import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from rmsai_lstm_autoencoder_proc import RMSAIConfig, ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThresholdClassificationTester:
    """Test suite for threshold values and heart rate-based classification"""

    def __init__(self):
        self.config = RMSAIConfig()
        self.model_manager = ModelManager(self.config)
        self.test_results = {}

    def test_threshold_values(self) -> bool:
        """Test that threshold values match expected optimized values"""
        logger.info("Testing optimized threshold values...")

        expected_thresholds = {
            'Normal': 0.8,
            'Tachycardia': 0.85,
            'Bradycardia': 0.85,
            'Atrial Fibrillation (PTB-XL)': 0.9,
            'Ventricular Tachycardia (MIT-BIH)': 1.0
        }

        success = True
        for condition, expected_threshold in expected_thresholds.items():
            actual_threshold = self.config.condition_thresholds.get(condition)
            if actual_threshold != expected_threshold:
                logger.error(f"Threshold mismatch for {condition}: expected {expected_threshold}, got {actual_threshold}")
                success = False
            else:
                logger.info(f"✓ {condition}: {actual_threshold}")

        self.test_results['threshold_values'] = success
        return success

    def test_heart_rate_ranges(self) -> bool:
        """Test heart rate range configuration"""
        logger.info("Testing heart rate range configuration...")

        expected_brady_max = 60
        expected_tachy_min = 100

        success = True
        if self.config.bradycardia_max_hr != expected_brady_max:
            logger.error(f"Bradycardia max HR: expected {expected_brady_max}, got {self.config.bradycardia_max_hr}")
            success = False
        else:
            logger.info(f"✓ Bradycardia max HR: {self.config.bradycardia_max_hr}")

        if self.config.tachycardia_min_hr != expected_tachy_min:
            logger.error(f"Tachycardia min HR: expected {expected_tachy_min}, got {self.config.tachycardia_min_hr}")
            success = False
        else:
            logger.info(f"✓ Tachycardia min HR: {self.config.tachycardia_min_hr}")

        self.test_results['heart_rate_ranges'] = success
        return success

    def test_heart_rate_classification(self) -> bool:
        """Test heart rate-based anomaly classification logic"""
        logger.info("Testing heart rate-based classification...")

        test_cases = [
            # (condition, heart_rate, is_anomaly, error_score, expected_classification)
            ('Normal', 45, True, 0.9, 'Bradycardia'),  # Slow HR → Brady
            ('Normal', 120, True, 0.9, 'Tachycardia'),  # Fast HR → Tachy
            ('Tachycardia', 45, True, 0.9, 'Bradycardia'),  # HR overrides condition
            ('Bradycardia', 120, True, 0.9, 'Tachycardia'),  # HR overrides condition
            ('Atrial Fibrillation (PTB-XL)', 120, True, 0.9, 'Atrial Fibrillation (PTB-XL)'),  # Morphological condition preserved
            ('Ventricular Tachycardia (MIT-BIH)', 45, True, 0.9, 'Ventricular Tachycardia (MIT-BIH)'),  # Morphological condition preserved
            ('Normal', 75, False, 0.3, 'normal'),  # Not anomaly → normal
            ('Normal', 75, True, 0.95, 'Unknown Arrhythmia'),  # Normal HR but high error
        ]

        success = True
        for i, (condition, heart_rate, is_anomaly, error_score, expected) in enumerate(test_cases):
            result = self.model_manager.classify_anomaly_by_heart_rate(
                condition, heart_rate, is_anomaly, error_score
            )

            if result != expected:
                logger.error(f"Test case {i+1} failed: expected '{expected}', got '{result}'")
                logger.error(f"  Input: condition={condition}, HR={heart_rate}, anomaly={is_anomaly}, score={error_score}")
                success = False
            else:
                logger.info(f"✓ Test case {i+1}: {condition} (HR:{heart_rate}) → {result}")

        self.test_results['heart_rate_classification'] = success
        return success

    def test_threshold_hierarchy(self) -> bool:
        """Test that threshold hierarchy makes clinical sense"""
        logger.info("Testing threshold hierarchy...")

        thresholds = self.config.condition_thresholds

        # Expected hierarchy: Normal < (Tachy = Brady) < A-Fib < V-Tac
        success = True

        # Normal should be lowest
        if not (thresholds['Normal'] < thresholds['Tachycardia']):
            logger.error("Normal threshold should be < Tachycardia threshold")
            success = False

        # Tachy and Brady should be equal
        if thresholds['Tachycardia'] != thresholds['Bradycardia']:
            logger.error("Tachycardia and Bradycardia thresholds should be equal")
            success = False

        # A-Fib should be higher than Tachy/Brady
        if not (thresholds['Atrial Fibrillation (PTB-XL)'] > thresholds['Tachycardia']):
            logger.error("A-Fib threshold should be > Tachycardia/Bradycardia threshold")
            success = False

        # V-Tac should be highest
        if not (thresholds['Ventricular Tachycardia (MIT-BIH)'] > thresholds['Atrial Fibrillation (PTB-XL)']):
            logger.error("V-Tac threshold should be > A-Fib threshold")
            success = False

        if success:
            logger.info("✓ Threshold hierarchy is clinically logical")
            logger.info(f"  Normal ({thresholds['Normal']}) < Tachy/Brady ({thresholds['Tachycardia']}) < A-Fib ({thresholds['Atrial Fibrillation (PTB-XL)']}) < V-Tac ({thresholds['Ventricular Tachycardia (MIT-BIH)']})")

        self.test_results['threshold_hierarchy'] = success
        return success

    def test_clinical_accuracy_simulation(self) -> bool:
        """Simulate clinical scenarios to test accuracy"""
        logger.info("Testing clinical accuracy simulation...")

        # Based on observed data from README
        clinical_scenarios = [
            # (condition, avg_score, heart_rate, should_be_anomaly, expected_classification)
            ('Normal', 0.6970, 75, False, 'normal'),  # Normal case
            ('Tachycardia', 0.6378, 120, False, 'normal'),  # Good reconstruction, but we'd classify by HR if anomaly
            ('Atrial Fibrillation (PTB-XL)', 0.6970, 130, False, 'normal'),  # Good reconstruction
            ('Ventricular Tachycardia (MIT-BIH)', 0.8742, 170, False, 'normal'),  # Close to threshold but not over

            # Simulated anomaly cases (scores above thresholds)
            ('Normal', 0.9, 75, True, 'Unknown Arrhythmia'),  # Normal HR but anomalous pattern
            ('Tachycardia', 0.9, 120, True, 'Tachycardia'),  # Fast HR with anomaly
            ('Bradycardia', 0.9, 45, True, 'Bradycardia'),  # Slow HR with anomaly
            ('Atrial Fibrillation (PTB-XL)', 0.95, 130, True, 'Atrial Fibrillation (PTB-XL)'),  # Morphological condition
            ('Ventricular Tachycardia (MIT-BIH)', 1.1, 170, True, 'Ventricular Tachycardia (MIT-BIH)'),  # Morphological condition
        ]

        success = True
        for i, (condition, score, heart_rate, should_be_anomaly, expected_class) in enumerate(clinical_scenarios):
            # Determine if it should be anomaly based on threshold
            threshold = self.config.condition_thresholds[condition]
            is_anomaly = score > threshold

            if is_anomaly != should_be_anomaly:
                logger.warning(f"Scenario {i+1}: Expected anomaly={should_be_anomaly}, but score {score} vs threshold {threshold} gives {is_anomaly}")

            # Test classification
            if is_anomaly:
                classification = self.model_manager.classify_anomaly_by_heart_rate(
                    condition, heart_rate, is_anomaly, score
                )
                if classification != expected_class:
                    logger.error(f"Scenario {i+1} failed: expected '{expected_class}', got '{classification}'")
                    success = False
                else:
                    logger.info(f"✓ Scenario {i+1}: {condition} (HR:{heart_rate}, score:{score}) → {classification}")
            else:
                logger.info(f"✓ Scenario {i+1}: {condition} (HR:{heart_rate}, score:{score}) → normal (below threshold)")

        self.test_results['clinical_accuracy'] = success
        return success

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all threshold and classification tests"""
        logger.info("=" * 60)
        logger.info("RMSAI Threshold & Classification Test Suite")
        logger.info("=" * 60)

        test_methods = [
            self.test_threshold_values,
            self.test_heart_rate_ranges,
            self.test_heart_rate_classification,
            self.test_threshold_hierarchy,
            self.test_clinical_accuracy_simulation
        ]

        all_passed = True
        for test_method in test_methods:
            try:
                logger.info(f"\n{'-' * 40}")
                passed = test_method()
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
                self.test_results[test_method.__name__] = False
                all_passed = False

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'=' * 60}")

        passed_count = sum(1 for result in self.test_results.values() if result)
        total_count = len(self.test_results)

        for test_name, passed in self.test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\nOverall: {passed_count}/{total_count} tests passed")

        if all_passed:
            logger.info("🎉 All threshold and classification tests PASSED!")
        else:
            logger.error("❌ Some tests FAILED. Please review the issues above.")

        return self.test_results

def main():
    """Main test runner"""
    tester = ThresholdClassificationTester()
    results = tester.run_all_tests()

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()