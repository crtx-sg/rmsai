#!/usr/bin/env python3
"""
Shared configuration constants for RMSAI system.
Centralizes all configuration to avoid duplication and maintenance issues.
"""

# Clinical severity hierarchy for anomaly prioritization
# Higher numbers indicate more severe conditions
CLINICAL_SEVERITY_ORDER = {
    'Ventricular Tachycardia (MIT-BIH)': 4,  # Most severe
    'Atrial Fibrillation (PTB-XL)': 3,
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
    'TACHY': 'Tachycardia',
    'BRADY': 'Bradycardia',
    'NORMAL': 'Normal'
}

# Default reconstruction error thresholds per condition
DEFAULT_CONDITION_THRESHOLDS = {
    'Normal': 0.75,   # 0.8
    'Atrial Fibrillation (PTB-XL)': 0.85,   # 0.9
    'Tachycardia': 0.80,  # 0.85
    'Bradycardia': 0.80,  # 0.85
    'Ventricular Tachycardia (MIT-BIH)': 0.93  # 1.0
}

# Adaptive threshold ranges per condition (min, max)
ADAPTIVE_THRESHOLD_RANGES = {
    'Normal': (0.6, 1.0),
    'Atrial Fibrillation (PTB-XL)': (0.75, 1.05),
    'Tachycardia': (0.7, 1.0),
    'Bradycardia': (0.7, 1.0),
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

# Default values for various processing parameters
DEFAULT_PROCESSING_CONFIG = {
    'base_anomaly_threshold': 0.1,      # Base MSE threshold for anomaly detection
    'adaptation_rate': 0.1,             # How quickly to adapt thresholds (0.0-1.0)
    'min_samples_for_adaptation': 10,   # Minimum samples before adapting
    'default_heart_rate': 60.0,         # Default HR when None provided
    'default_error_score': 0.0,         # Default error score when None/calculation fails
    'default_adaptation_ratio': 1.0,    # Default ratio when base_threshold is 0
    'max_queue_size': 100,              # Maximum processing queue size
    'sliding_window_size': 100,         # Sliding window for adaptive thresholds
    'stats_report_interval': 60         # Seconds between stats reports
}

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

# Early Warning System (EWS) Configuration
# Based on NEWS2 (National Early Warning Score 2) scoring system
EWS_SCORING_TEMPLATE = {
    'heart_rate': {
        'ranges': [
            {'range': '≤40', 'score': 3, 'min': 0, 'max': 40},
            {'range': '41-50', 'score': 1, 'min': 41, 'max': 50},
            {'range': '51-90', 'score': 0, 'min': 51, 'max': 90},
            {'range': '91-110', 'score': 1, 'min': 91, 'max': 110},
            {'range': '111-130', 'score': 2, 'min': 111, 'max': 130},
            {'range': '≥131', 'score': 3, 'min': 131, 'max': 999}
        ],
        'units': 'bpm',
        'display_name': 'Heart Rate'
    },
    'respiratory_rate': {
        'ranges': [
            {'range': '≤8', 'score': 3, 'min': 0, 'max': 8},
            {'range': '9-11', 'score': 1, 'min': 9, 'max': 11},
            {'range': '12-20', 'score': 0, 'min': 12, 'max': 20},
            {'range': '21-24', 'score': 2, 'min': 21, 'max': 24},
            {'range': '≥25', 'score': 3, 'min': 25, 'max': 999}
        ],
        'units': 'breaths/min',
        'display_name': 'Respiratory Rate'
    },
    'systolic_bp': {
        'ranges': [
            {'range': '≤90', 'score': 3, 'min': 0, 'max': 90},
            {'range': '91-100', 'score': 2, 'min': 91, 'max': 100},
            {'range': '101-110', 'score': 1, 'min': 101, 'max': 110},
            {'range': '111-219', 'score': 0, 'min': 111, 'max': 219},
            {'range': '≥220', 'score': 3, 'min': 220, 'max': 999}
        ],
        'units': 'mmHg',
        'display_name': 'Systolic Blood Pressure'
    },
    'temperature': {
        'ranges': [
            {'range': '≤35.0', 'score': 3, 'min': 0, 'max': 35.0},
            {'range': '35.1-36.0', 'score': 1, 'min': 35.1, 'max': 36.0},
            {'range': '36.1-38.0', 'score': 0, 'min': 36.1, 'max': 38.0},
            {'range': '38.1-39.0', 'score': 1, 'min': 38.1, 'max': 39.0},
            {'range': '≥39.1', 'score': 2, 'min': 39.1, 'max': 999}
        ],
        'units': '°C',
        'display_name': 'Temperature'
    },
    'oxygen_saturation': {
        'ranges': [
            {'range': '≤91', 'score': 3, 'min': 0, 'max': 91},
            {'range': '92-93', 'score': 2, 'min': 92, 'max': 93},
            {'range': '94-95', 'score': 1, 'min': 94, 'max': 95},
            {'range': '≥96', 'score': 0, 'min': 96, 'max': 100}
        ],
        'units': '%',
        'display_name': 'Oxygen Saturation (SpO2)'
    },
    'consciousness': {
        'ranges': [
            {'range': 'Alert', 'score': 0, 'value': 'alert'},
            {'range': 'CVPU', 'score': 3, 'value': 'cvpu'}  # Confusion, Voice, Pain, Unresponsive
        ],
        'units': '',
        'display_name': 'Level of Consciousness'
    }
}

# EWS Risk Categories based on total score
EWS_RISK_CATEGORIES = {
    'low': {
        'score_range': (0, 4),
        'category': 'Low Risk',
        'color': '#28a745',  # Green
        'monitoring_frequency': 'every 12 hours',
        'clinical_response': 'Continue routine monitoring'
    },
    'medium': {
        'score_range': (5, 6),
        'category': 'Medium Risk',
        'color': '#ffc107',  # Yellow/Orange
        'monitoring_frequency': 'every 4-6 hours',
        'clinical_response': 'Increase monitoring frequency and consider medical review'
    },
    'high': {
        'score_range': (7, float('inf')),
        'category': 'High Risk',
        'color': '#dc3545',  # Red
        'monitoring_frequency': 'continuous or every hour',
        'clinical_response': 'Urgent medical review required and consider intensive monitoring'
    }
}

def get_ews_scoring_template() -> dict:
    """Get the EWS scoring template configuration"""
    return EWS_SCORING_TEMPLATE.copy()

def get_ews_risk_categories() -> dict:
    """Get the EWS risk categories configuration"""
    return EWS_RISK_CATEGORIES.copy()

def get_ews_score_for_vital(vital_name: str, value: float) -> int:
    """Get EWS score for a specific vital sign value"""
    if vital_name not in EWS_SCORING_TEMPLATE:
        return 0

    vital_config = EWS_SCORING_TEMPLATE[vital_name]

    for range_config in vital_config['ranges']:
        if 'min' in range_config and 'max' in range_config:
            if range_config['min'] <= value <= range_config['max']:
                return range_config['score']
        elif 'value' in range_config:
            # For categorical values like consciousness
            if str(value).lower() == range_config['value']:
                return range_config['score']

    return 0  # Default score if no range matches

def get_ews_risk_category(total_score: int) -> dict:
    """Get risk category information for a given EWS total score"""
    for category, config in EWS_RISK_CATEGORIES.items():
        score_min, score_max = config['score_range']
        if score_min <= total_score <= score_max:
            return {
                'category': config['category'],
                'color': config['color'],
                'monitoring_frequency': config['monitoring_frequency'],
                'clinical_response': config['clinical_response'],
                'level': category
            }

    # Default to high risk if score is very high
    return {
        'category': 'High Risk',
        'color': '#dc3545',
        'monitoring_frequency': 'continuous',
        'clinical_response': 'Immediate medical attention required',
        'level': 'high'
    }
