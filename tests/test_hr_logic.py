#!/usr/bin/env python3
"""
Test script to verify HR-based anomaly detection logic for ambiguity
"""

from config import CLINICAL_SEVERITY_ORDER, HR_THRESHOLDS, sort_by_severity

def get_event_ai_verdict(anomaly_types_list, heart_rate=None, anomaly_status_count=0):
    """Enhanced AI verdict with HR-based anomaly detection"""
    # Use shared configuration
    bradycardia_max_hr = HR_THRESHOLDS['bradycardia_max']
    tachycardia_min_hr = HR_THRESHOLDS['tachycardia_min']

    # Start with existing anomaly types
    unique_anomalies = list(set([t for t in anomaly_types_list if t and t != 'None']))

    # If no LSTM-detected anomalies but abnormal HR, add HR-based anomaly
    if anomaly_status_count == 0 and heart_rate is not None:
        try:
            hr_value = float(heart_rate)
            if hr_value <= bradycardia_max_hr:
                unique_anomalies.append('Bradycardia')
            elif hr_value >= tachycardia_min_hr:
                unique_anomalies.append('Tachycardia')
        except (ValueError, TypeError):
            pass  # Skip if heart rate is not a valid number

    # If still no anomalies, return Normal
    if not unique_anomalies:
        return "Normal"

    # Sort by severity and return
    unique_anomalies = list(set(unique_anomalies))  # Remove duplicates
    return ", ".join(sort_by_severity(unique_anomalies))

def test_scenarios():
    """Test various scenarios for potential ambiguity"""
    print("🧪 Testing HR-based anomaly detection logic")
    print("=" * 60)

    # Test scenarios
    scenarios = [
        # (anomaly_types, heart_rate, anomaly_count, expected_behavior, description)
        ([], 50, 0, "Bradycardia", "Normal LSTM + Brady HR → Brady"),
        ([], 120, 0, "Tachycardia", "Normal LSTM + Tachy HR → Tachy"),
        ([], 80, 0, "Normal", "Normal LSTM + Normal HR → Normal"),
        (['Tachycardia'], 50, 1, "Tachycardia", "LSTM Tachy + Brady HR → LSTM wins"),
        (['Bradycardia'], 120, 1, "Bradycardia", "LSTM Brady + Tachy HR → LSTM wins"),
        (['Atrial Fibrillation (PTB-XL)'], 45, 1, "Atrial Fibrillation (PTB-XL)", "A-Fib + Brady HR → A-Fib wins"),
        (['Ventricular Tachycardia (MIT-BIH)'], 50, 1, "Ventricular Tachycardia (MIT-BIH)", "V-Tac + Brady HR → V-Tac wins"),
        (['Unknown Arrhythmia'], 130, 1, "Unknown Arrhythmia", "Unknown + Tachy HR → Unknown wins"),
        ([], None, 0, "Normal", "Normal LSTM + No HR → Normal"),
        ([], "invalid", 0, "Normal", "Normal LSTM + Invalid HR → Normal"),
        # Edge cases and realistic clinical scenarios
        ([], 60, 0, "Bradycardia", "Boundary case: exactly 60 BPM → Brady"),
        ([], 61, 0, "Normal", "Boundary case: exactly 61 BPM → Normal"),
        ([], 99, 0, "Normal", "Boundary case: exactly 99 BPM → Normal"),
        ([], 100, 0, "Tachycardia", "Boundary case: exactly 100 BPM → Tachy"),
        ([], 35, 0, "Bradycardia", "Severe bradycardia (35 BPM)"),
        ([], 180, 0, "Tachycardia", "Severe tachycardia (180 BPM)"),
        (['Tachycardia', 'Bradycardia'], 80, 2, "Tachycardia, Bradycardia", "Multiple LSTM anomalies → all shown, sorted by severity"),
        (['Ventricular Tachycardia (MIT-BIH)', 'Tachycardia'], 50, 2, "Ventricular Tachycardia (MIT-BIH), Tachycardia", "V-Tac + Tachy → both shown, V-Tac first"),
    ]

    print("Test Results:")
    print("-" * 60)

    all_passed = True
    for i, (anomaly_types, hr, count, expected, description) in enumerate(scenarios, 1):
        result = get_event_ai_verdict(anomaly_types, hr, count)
        passed = result == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i:2d}. {status} | {description}")
        print(f"    Input: anomaly_types={anomaly_types}, HR={hr}, count={count}")
        print(f"    Expected: {expected} | Got: {result}")
        if not passed:
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Logic appears to be unambiguous.")
    else:
        print("⚠️  Some tests failed. Review logic for potential ambiguity.")

    print("\n📋 Clinical Logic Summary:")
    print("1. LSTM anomaly detection takes precedence over HR-based detection")
    print("2. HR-based detection only applies when LSTM says 'normal' (count=0)")
    print("3. Severity hierarchy prevents conflicting interpretations")
    print("4. Bradycardia: ≤60 BPM, Tachycardia: ≥100 BPM")
    print("5. Normal range: 61-99 BPM")

if __name__ == "__main__":
    test_scenarios()