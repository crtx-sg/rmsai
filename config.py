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

# ECG Leads Configuration
# Complete list of all possible ECG leads in the system
ALL_AVAILABLE_ECG_LEADS = [
    'ECG1',   # Lead I
    'ECG2',   # Lead II
    'ECG3',   # Lead III
    'aVR',    # Augmented Vector Right
    'aVL',    # Augmented Vector Left
    'aVF',    # Augmented Vector Foot
    'vVX'     # Precordial lead (V1-V6 equivalent)
]

# Default selected leads for AI processing
# Note: Currently limited to ECG2 for performance/compatibility
DEFAULT_SELECTED_ECG_LEADS = ['ECG2']

# ECG Processing Parameters
ECG_PROCESSING_CONFIG = {
    'chunk_size': 140,           # Samples per chunk (0.7 seconds at 200Hz)
    'step_size': 70,             # Step between chunks (0.35 seconds at 200Hz)
    'max_chunks_per_lead': 33,   # Maximum chunks per lead per event
    'sampling_rate': 200,        # Hz
    'event_duration': 12,        # seconds
    'ecg_samples_per_event': 2400  # 12 seconds × 200 Hz
}

# Performance configuration based on lead selection
LEAD_PERFORMANCE_ESTIMATES = {
    1: {'chunks_per_event': 33, 'performance_gain_percent': 85.7},    # Single lead
    2: {'chunks_per_event': 66, 'performance_gain_percent': 71.4},    # Two leads
    3: {'chunks_per_event': 99, 'performance_gain_percent': 57.1},    # Three leads
    4: {'chunks_per_event': 132, 'performance_gain_percent': 42.8},   # Four leads
    5: {'chunks_per_event': 165, 'performance_gain_percent': 28.6},   # Five leads
    6: {'chunks_per_event': 198, 'performance_gain_percent': 14.3},   # Six leads
    7: {'chunks_per_event': 231, 'performance_gain_percent': 0.0}     # All leads (baseline)
}

def get_performance_estimates(selected_leads_count: int) -> dict:
    """Get performance estimates based on number of selected leads"""
    if selected_leads_count not in LEAD_PERFORMANCE_ESTIMATES:
        # Default to proportional calculation
        baseline_chunks = LEAD_PERFORMANCE_ESTIMATES[7]['chunks_per_event']
        chunks_per_event = selected_leads_count * ECG_PROCESSING_CONFIG['max_chunks_per_lead']
        performance_gain = ((baseline_chunks - chunks_per_event) / baseline_chunks) * 100

        return {
            'chunks_per_event': chunks_per_event,
            'performance_gain_percent': round(performance_gain, 1)
        }

    return LEAD_PERFORMANCE_ESTIMATES[selected_leads_count]

def validate_ecg_leads(leads: list) -> tuple[bool, list]:
    """
    Validate ECG leads selection

    Args:
        leads: List of lead names to validate

    Returns:
        tuple: (is_valid, invalid_leads)
    """
    invalid_leads = [lead for lead in leads if lead not in ALL_AVAILABLE_ECG_LEADS]
    return len(invalid_leads) == 0, invalid_leads

def get_default_ecg_config() -> dict:
    """Get default ECG processing configuration"""
    return {
        'available_leads': ALL_AVAILABLE_ECG_LEADS.copy(),
        'selected_leads': DEFAULT_SELECTED_ECG_LEADS.copy(),
        'processing_config': ECG_PROCESSING_CONFIG.copy(),
        'performance_estimates': get_performance_estimates(len(DEFAULT_SELECTED_ECG_LEADS))
    }

def sort_by_severity(anomaly_list: list) -> list:
    """Sort anomaly types by clinical severity (most severe first)"""
    return sorted(anomaly_list, key=lambda x: get_severity_score(x), reverse=True)