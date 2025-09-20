#!/usr/bin/env python3
"""
Shared configuration constants for RMSAI system.
Centralizes all configuration to avoid duplication and maintenance issues.
"""

# Clinical severity hierarchy for anomaly prioritization
# Higher numbers indicate more severe conditions
CLINICAL_SEVERITY_ORDER = {
    'Ventricular Tachycardia (MIT-BIH)': 5,  # Most severe
    'Atrial Fibrillation (PTB-XL)': 4,
    'Unknown Arrhythmia': 3,
    'Tachycardia': 2,
    'Bradycardia': 1                         # Least severe
}

# Heart rate thresholds for classification (BPM)
HR_THRESHOLDS = {
    'bradycardia_max': 60,      # ≤60 BPM = Bradycardia
    'tachycardia_min': 100      # ≥100 BPM = Tachycardia
    # Normal range: 61-99 BPM (implicit)
}

# Standard condition names (for consistency across modules)
CONDITION_NAMES = {
    'V_TAC': 'Ventricular Tachycardia (MIT-BIH)',
    'A_FIB': 'Atrial Fibrillation (PTB-XL)',
    'UNKNOWN': 'Unknown Arrhythmia',
    'TACHY': 'Tachycardia',
    'BRADY': 'Bradycardia',
    'NORMAL': 'Normal'
}

# Default reconstruction error thresholds per condition
DEFAULT_CONDITION_THRESHOLDS = {
    'Normal': 0.8,
    'Atrial Fibrillation (PTB-XL)': 0.9,
    'Tachycardia': 0.85,
    'Bradycardia': 0.85,
    'Unknown Arrhythmia': 0.9,
    'Ventricular Tachycardia (MIT-BIH)': 1.0
}

# Adaptive threshold ranges per condition (min, max)
ADAPTIVE_THRESHOLD_RANGES = {
    'Normal': (0.6, 1.0),
    'Atrial Fibrillation (PTB-XL)': (0.75, 1.05),
    'Tachycardia': (0.7, 1.0),
    'Bradycardia': (0.7, 1.0),
    'Unknown Arrhythmia': (0.75, 1.05),
    'Ventricular Tachycardia (MIT-BIH)': (0.85, 1.15)
}

def get_severity_score(condition_name: str) -> int:
    """Get clinical severity score for a condition (higher = more severe)"""
    return CLINICAL_SEVERITY_ORDER.get(condition_name, 0)

def is_bradycardia(heart_rate: float) -> bool:
    """Check if heart rate indicates bradycardia"""
    return heart_rate <= HR_THRESHOLDS['bradycardia_max']

def is_tachycardia(heart_rate: float) -> bool:
    """Check if heart rate indicates tachycardia"""
    return heart_rate >= HR_THRESHOLDS['tachycardia_min']

def is_normal_heart_rate(heart_rate: float) -> bool:
    """Check if heart rate is in normal range"""
    return HR_THRESHOLDS['bradycardia_max'] < heart_rate < HR_THRESHOLDS['tachycardia_min']

# Adaptive threshold configuration
ENABLE_ADAPTIVE_THRESHOLDS = False  # Default: disabled

def sort_by_severity(anomaly_list: list) -> list:
    """Sort anomaly types by clinical severity (most severe first)"""
    return sorted(anomaly_list, key=lambda x: get_severity_score(x), reverse=True)