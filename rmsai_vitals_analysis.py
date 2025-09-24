#!/usr/bin/env python3
"""
RMSAI Vitals Analysis Module
============================

Comprehensive vitals analysis system integrating:
- Early Warning System (EWS) scoring
- ML-based trend analysis
- Patient deterioration/improvement assessment
- Clinical decision support
- Historical data analysis from HDF5 extras

Author: RMSAI Team
"""

import numpy as np
import pandas as pd
import json
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RMSAIVitalsAnalyzer:
    """
    Comprehensive vitals analysis system for RMSAI HDF5 data
    combining EWS scoring with advanced trend analysis
    """

    def __init__(self, window_size=6, recent_values_count=3, confidence_level=0.95):
        self.window_size = window_size
        self.recent_values_count = recent_values_count
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        # EWS scoring weights (based on NEWS2 and modified for available vitals)
        self.ews_weights = {
            'HR': self._get_hr_score,
            'Systolic': self._get_bp_systolic_score,
            'SpO2': self._get_spo2_score,
            'RespRate': self._get_resprate_score,
            'Temp': self._get_temp_score,
            'Pulse': self._get_pulse_score,
        }

    def _get_hr_score(self, hr):
        """Heart Rate EWS scoring (NEWS2-based)"""
        if hr is None:
            return 0
        if hr <= 40:
            return 3
        elif hr <= 50:
            return 1
        elif hr <= 90:
            return 0
        elif hr <= 110:
            return 1
        elif hr <= 130:
            return 2
        else:
            return 3

    def _get_pulse_score(self, pulse):
        """Pulse rate scoring (similar to HR)"""
        return self._get_hr_score(pulse)

    def _get_bp_systolic_score(self, systolic):
        """Systolic BP EWS scoring"""
        if systolic is None:
            return 0
        if systolic <= 90:
            return 3
        elif systolic <= 100:
            return 2
        elif systolic <= 110:
            return 1
        elif systolic <= 219:
            return 0
        else:
            return 2

    def _get_spo2_score(self, spo2):
        """SpO2 EWS scoring"""
        if spo2 is None:
            return 0
        if spo2 <= 91:
            return 3
        elif spo2 <= 93:
            return 2
        elif spo2 <= 95:
            return 1
        else:
            return 0

    def _get_resprate_score(self, resprate):
        """Respiratory Rate EWS scoring"""
        if resprate is None:
            return 0
        if resprate <= 8:
            return 3
        elif resprate <= 11:
            return 1
        elif resprate <= 20:
            return 0
        elif resprate <= 24:
            return 2
        else:
            return 3

    def _get_temp_score(self, temp_f):
        """Temperature EWS scoring (input in Fahrenheit)"""
        if temp_f is None:
            return 0
        # Convert to Celsius
        temp_c = (temp_f - 32) * 5/9
        if temp_c <= 35:
            return 3
        elif temp_c <= 36:
            return 1
        elif temp_c <= 38:
            return 0
        elif temp_c <= 39:
            return 1
        else:
            return 2

    def calculate_ews_score(self, vitals_dict):
        """
        Calculate Early Warning System score from vitals using configurable scoring template

        Args:
            vitals_dict: Dictionary with vital signs values

        Returns:
            Dictionary with EWS score and breakdown
        """
        # Import from config
        from config import get_ews_score_for_vital, get_ews_risk_category, get_ews_scoring_template

        total_score = 0
        score_breakdown = {}

        # Map vitals_dict keys to config keys
        vital_mapping = {
            'heart_rate': 'heart_rate',
            'HR': 'heart_rate',
            'respiratory_rate': 'respiratory_rate',
            'RespRate': 'respiratory_rate',
            'systolic_bp': 'systolic_bp',
            'Systolic': 'systolic_bp',
            'temperature': 'temperature',
            'Temp': 'temperature',
            'oxygen_saturation': 'oxygen_saturation',
            'SpO2': 'oxygen_saturation',
            'consciousness': 'consciousness'
        }

        # Get template for validation
        ews_template = get_ews_scoring_template()

        # Calculate scores for each vital
        for vital_key, vital_value in vitals_dict.items():
            config_key = vital_mapping.get(vital_key, vital_key.lower())

            if config_key in ews_template and vital_value is not None:
                score = get_ews_score_for_vital(config_key, vital_value)
                total_score += score
                score_breakdown[ews_template[config_key]['display_name']] = {
                    'value': vital_value,
                    'score': score,
                    'units': ews_template[config_key]['units']
                }
            elif vital_value is not None:
                # Handle vitals not in template
                score_breakdown[vital_key] = {
                    'value': vital_value,
                    'score': 0,
                    'note': 'Not included in EWS calculation'
                }

        # Add missing vitals from template
        for config_key, config_data in ews_template.items():
            display_name = config_data['display_name']
            if display_name not in score_breakdown:
                score_breakdown[display_name] = {
                    'value': None,
                    'score': 0,
                    'note': 'Missing data',
                    'units': config_data['units']
                }

        # Get risk category from config
        risk_info = get_ews_risk_category(total_score)

        return {
            'total_score': total_score,
            'risk_category': risk_info['category'],
            'risk_color': risk_info['color'],
            'clinical_response': risk_info['clinical_response'],
            'monitoring_frequency': risk_info['monitoring_frequency'],
            'risk_level': risk_info['level'],
            'score_breakdown': score_breakdown,
            'timestamp': datetime.now().isoformat()
        }

    def extract_vitals_history_from_hdf5(self, hdf5_file, event_id):
        """
        Extract vitals history from RMSAI HDF5 file

        Args:
            hdf5_file: Path to HDF5 file
            event_id: Event ID (e.g., 'event_1001')

        Returns:
            Dictionary with vitals history and current values
        """
        vitals_data = {}

        try:
            with h5py.File(hdf5_file, 'r') as f:
                if event_id not in f:
                    raise ValueError(f"Event {event_id} not found in HDF5 file")

                event_group = f[event_id]
                if 'vitals' not in event_group:
                    raise ValueError(f"No vitals data found for {event_id}")

                vitals_group = event_group['vitals']

                for vital_name in vitals_group.keys():
                    vital_group = vitals_group[vital_name]

                    # Get current values
                    current_value = vital_group['value'][()]
                    current_timestamp = vital_group['timestamp'][()]
                    units = vital_group['units'][()].decode('utf-8')

                    # Extract history from extras JSON
                    history = []
                    if 'extras' in vital_group:
                        try:
                            extras_json = json.loads(vital_group['extras'][()].decode('utf-8'))
                            history = extras_json.get('history', [])
                        except (json.JSONDecodeError, KeyError):
                            pass

                    vitals_data[vital_name] = {
                        'current_value': current_value,
                        'current_timestamp': current_timestamp,
                        'units': units,
                        'history': history,
                        'total_samples': len(history) + 1  # Include current value
                    }

        except Exception as e:
            raise Exception(f"Error extracting vitals from HDF5: {str(e)}")

        return vitals_data

    def analyze_vital_trends(self, vitals_data):
        """
        Analyze trends for all vitals using integrated EWS analysis

        Args:
            vitals_data: Dictionary from extract_vitals_history_from_hdf5

        Returns:
            Comprehensive analysis results
        """
        analysis_results = {}
        overall_ews_scores = []
        overall_timestamps = []

        for vital_name, vital_info in vitals_data.items():
            if vital_name not in self.ews_weights:
                continue  # Skip vitals not used in EWS scoring

            history = vital_info['history']
            current_value = vital_info['current_value']
            current_timestamp = vital_info['current_timestamp']

            if len(history) < 2:
                analysis_results[vital_name] = {
                    'status': 'insufficient_data',
                    'trend': 'unknown',
                    'message': 'Need at least 2 historical data points'
                }
                continue

            # Combine historical and current data
            all_values = [h['value'] for h in history] + [current_value]
            all_timestamps = [datetime.fromtimestamp(h['timestamp']) for h in history] + [datetime.fromtimestamp(current_timestamp)]

            # Perform trend analysis using adapted EWS methods
            trend_analysis = self._analyze_single_vital_trend(all_timestamps, all_values, vital_name)
            analysis_results[vital_name] = trend_analysis

        # Calculate EWS scores over time using historical data
        if len(vitals_data) > 0:
            # Create a comprehensive time series by merging all vital histories
            all_timepoints = set()

            # Collect all unique timestamps from all vitals
            for vital_name, vital_info in vitals_data.items():
                if vital_name not in self.ews_weights:
                    continue

                # Add historical timestamps
                for hist_point in vital_info['history']:
                    all_timepoints.add(hist_point['timestamp'])

                # Add current timestamp
                all_timepoints.add(vital_info['current_timestamp'])

            # Sort timestamps chronologically
            sorted_timestamps = sorted(all_timepoints)

            # Calculate EWS score at each timepoint
            for timestamp in sorted_timestamps:
                vitals_at_timestamp = {}

                # For each vital, find the most recent value at or before this timestamp
                for vital_name, vital_info in vitals_data.items():
                    if vital_name not in self.ews_weights:
                        continue

                    # Find the most recent value for this vital at this timestamp
                    most_recent_value = None

                    # Check historical values
                    for hist_point in vital_info['history']:
                        if hist_point['timestamp'] <= timestamp:
                            if most_recent_value is None or hist_point['timestamp'] > most_recent_value['timestamp']:
                                most_recent_value = hist_point

                    # Check current value if timestamp matches or no historical value found
                    if vital_info['current_timestamp'] <= timestamp:
                        if most_recent_value is None or vital_info['current_timestamp'] > most_recent_value['timestamp']:
                            most_recent_value = {
                                'value': vital_info['current_value'],
                                'timestamp': vital_info['current_timestamp']
                            }

                    # Add to vitals at this timestamp
                    if most_recent_value is not None:
                        vitals_at_timestamp[vital_name] = most_recent_value['value']

                # Calculate EWS score if we have any vitals data
                if vitals_at_timestamp:
                    ews_result = self.calculate_ews_score(vitals_at_timestamp)
                    overall_ews_scores.append(ews_result['total_score'])
                    overall_timestamps.append(datetime.fromtimestamp(timestamp))

                    # Store detailed breakdown for this timepoint (useful for debugging)
                    if not hasattr(self, '_ews_detailed_history'):
                        self._ews_detailed_history = []

                    self._ews_detailed_history.append({
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp),
                        'vitals_used': vitals_at_timestamp,
                        'ews_breakdown': ews_result
                    })

        # Analyze overall EWS trend
        overall_trend_analysis = None
        if len(overall_ews_scores) >= 2:
            overall_trend_analysis = self._analyze_ews_trend(overall_timestamps, overall_ews_scores)

        return {
            'individual_vitals': analysis_results,
            'ews_trend_analysis': overall_trend_analysis,
            'current_ews': self.calculate_ews_score({vn: vd['current_value'] for vn, vd in vitals_data.items() if vn in self.ews_weights}),
            'ews_history': {
                'scores': overall_ews_scores,
                'timestamps': [ts.isoformat() for ts in overall_timestamps],
                'detailed_history': getattr(self, '_ews_detailed_history', [])
            }
        }

    def get_ews_detailed_history(self):
        """
        Get detailed EWS calculation history with breakdown by timepoint

        Returns:
            List of detailed EWS calculations at each timepoint
        """
        return getattr(self, '_ews_detailed_history', [])

    def _analyze_single_vital_trend(self, timestamps, values, vital_name):
        """Analyze trend for a single vital sign"""
        if len(values) < 2:
            return {'status': 'insufficient_data', 'trend': 'unknown'}

        # Convert to DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'time_hours': [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        })

        # Linear regression analysis
        X = df['time_hours'].values.reshape(-1, 1)
        y = df['value'].values

        reg = LinearRegression()
        reg.fit(X, y)

        slope = reg.coef_[0]
        r_squared = reg.score(X, y)

        # Statistical significance test
        n = len(df)
        if n > 2:
            y_pred = reg.predict(X)
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if t_stat != 0 else 1.0
        else:
            p_value = 1.0

        # Determine trend direction and significance
        is_significant = p_value < self.alpha

        if is_significant:
            if slope > 0:
                trend = 'increasing'
                clinical_interpretation = self._interpret_increasing_trend(vital_name)
            else:
                trend = 'decreasing'
                clinical_interpretation = self._interpret_decreasing_trend(vital_name)
        else:
            trend = 'stable'
            clinical_interpretation = 'Stable trend, continue routine monitoring'

        # Recent values assessment
        recent_values = values[-self.recent_values_count:] if len(values) >= self.recent_values_count else values
        recent_change = recent_values[-1] - recent_values[0] if len(recent_values) > 1 else 0

        return {
            'status': 'analyzed',
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared,
            'p_value': p_value,
            'significance': 'significant' if is_significant else 'not_significant',
            'recent_change': recent_change,
            'clinical_interpretation': clinical_interpretation,
            'data_points': len(values),
            'time_span_hours': df['time_hours'].iloc[-1] if len(df) > 1 else 0
        }

    def _interpret_increasing_trend(self, vital_name):
        """Provide clinical interpretation for increasing vital trends"""
        interpretations = {
            'HR': 'Increasing heart rate may indicate stress, pain, or developing sepsis',
            'Pulse': 'Rising pulse rate suggests physiological stress or cardiac issues',
            'Systolic': 'Rising blood pressure may indicate hypertension or stress response',
            'RespRate': 'Increasing respiratory rate suggests respiratory distress or metabolic issues',
            'Temp': 'Rising temperature indicates possible infection or inflammatory response',
            'SpO2': 'Improving oxygen saturation is positive (if previously low)'
        }
        return interpretations.get(vital_name, 'Increasing trend detected, clinical review recommended')

    def _interpret_decreasing_trend(self, vital_name):
        """Provide clinical interpretation for decreasing vital trends"""
        interpretations = {
            'HR': 'Decreasing heart rate may indicate improvement or potential cardiac issues',
            'Pulse': 'Declining pulse rate could suggest improvement or bradycardia',
            'Systolic': 'Decreasing blood pressure may indicate hypotension or shock',
            'RespRate': 'Decreasing respiratory rate suggests improvement or respiratory depression',
            'Temp': 'Declining temperature indicates fever resolution or hypothermia risk',
            'SpO2': 'Declining oxygen saturation requires immediate attention'
        }
        return interpretations.get(vital_name, 'Decreasing trend detected, clinical assessment needed')

    def _analyze_ews_trend(self, timestamps, ews_scores):
        """Analyze overall EWS score trends using ensemble methods from original tool"""
        # Adapt the original EWS trend analyzer methods
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ews': ews_scores,
            'time_hours': [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        })

        # Linear regression
        linear_result = self._linear_regression_trend(df)

        # Mann-Kendall test
        mk_result = self._mann_kendall_test(ews_scores)

        # CUSUM analysis
        cusum_result = self._cusum_analysis(ews_scores)

        # Recent values assessment
        recent_result = self._recent_values_assessment(ews_scores)

        # Ensemble decision
        votes = []
        weights = []

        if linear_result['significance'] == 'significant':
            votes.append(linear_result['trend'])
            weights.append(linear_result['r_squared'])

        if mk_result['significance'] == 'significant':
            votes.append(mk_result['trend'])
            weights.append(abs(mk_result['tau']))

        if cusum_result['trend'] != 'stable':
            votes.append(cusum_result['trend'])
            weights.append(0.8)

        # Determine consensus
        if len(votes) == 0:
            overall_trend = 'stable'
            confidence = 'low'
        else:
            improving_weight = sum(w for v, w in zip(votes, weights) if v == 'improving')
            deteriorating_weight = sum(w for v, w in zip(votes, weights) if v == 'deteriorating')

            if improving_weight > deteriorating_weight:
                overall_trend = 'improving'
            elif deteriorating_weight > improving_weight:
                overall_trend = 'deteriorating'
            else:
                overall_trend = 'stable'

            # Calculate confidence
            total_weight = improving_weight + deteriorating_weight
            max_weight = max(improving_weight, deteriorating_weight)
            confidence_ratio = max_weight / total_weight if total_weight > 0 else 0

            if confidence_ratio > 0.7:
                confidence = 'high'
            elif confidence_ratio > 0.5:
                confidence = 'moderate'
            else:
                confidence = 'low'

        return {
            'overall_trend': overall_trend,
            'confidence': confidence,
            'improvement': recent_result.get('improvement'),
            'clinical_interpretation': self._generate_ews_clinical_interpretation(overall_trend, confidence),
            'individual_methods': {
                'linear_regression': linear_result,
                'mann_kendall': mk_result,
                'cusum': cusum_result,
                'recent_assessment': recent_result
            }
        }

    def _linear_regression_trend(self, df):
        """Linear regression trend analysis (adapted from original)"""
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1.0, 'trend': 'insufficient_data', 'significance': 'not_significant'}

        X = df['time_hours'].values.reshape(-1, 1)
        y = df['ews'].values

        reg = LinearRegression()
        reg.fit(X, y)

        slope = reg.coef_[0]
        r_squared = reg.score(X, y)

        # Calculate p-value
        n = len(df)
        if n > 2:
            y_pred = reg.predict(X)
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if t_stat != 0 else 1.0
        else:
            p_value = 1.0

        if p_value < self.alpha:
            trend = 'deteriorating' if slope > 0 else 'improving'
            significance = 'significant'
        else:
            trend = 'stable'
            significance = 'not_significant'

        return {
            'slope': slope,
            'r_squared': r_squared,
            'p_value': p_value,
            'trend': trend,
            'significance': significance
        }

    def _mann_kendall_test(self, ews_values):
        """Mann-Kendall test (adapted from original)"""
        n = len(ews_values)
        if n < 3:
            return {'tau': 0, 'p_value': 1.0, 'trend': 'insufficient_data', 'significance': 'not_significant'}

        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if ews_values[j] > ews_values[i]:
                    S += 1
                elif ews_values[j] < ews_values[i]:
                    S -= 1

        var_S = n * (n - 1) * (2 * n + 5) / 18

        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0

        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        tau = S / (0.5 * n * (n - 1))

        if p_value < self.alpha:
            trend = 'deteriorating' if tau > 0 else 'improving'
            significance = 'significant'
        else:
            trend = 'stable'
            significance = 'not_significant'

        return {
            'tau': tau,
            'p_value': p_value,
            'trend': trend,
            'significance': significance
        }

    def _cusum_analysis(self, ews_values):
        """CUSUM analysis (adapted from original)"""
        if len(ews_values) < 2:
            return {'trend': 'insufficient_data', 'alerts': []}

        target_mean = np.mean(ews_values)
        std_dev = np.std(ews_values)

        if std_dev == 0:
            return {'trend': 'stable', 'alerts': []}

        k = 0.5 * std_dev
        h = 4 * std_dev

        cusum_pos = [0]
        cusum_neg = [0]
        alerts = []

        for i, value in enumerate(ews_values[1:], 1):
            cusum_pos_new = max(0, cusum_pos[-1] + (value - target_mean) - k)
            cusum_pos.append(cusum_pos_new)

            cusum_neg_new = min(0, cusum_neg[-1] + (value - target_mean) + k)
            cusum_neg.append(cusum_neg_new)

            if cusum_pos_new > h:
                alerts.append({'index': i, 'type': 'deterioration'})
            elif cusum_neg_new < -h:
                alerts.append({'index': i, 'type': 'improvement'})

        recent_cusum_pos = cusum_pos[-min(self.window_size, len(cusum_pos)):]
        recent_cusum_neg = cusum_neg[-min(self.window_size, len(cusum_neg)):]

        if any(cp > h for cp in recent_cusum_pos):
            trend = 'deteriorating'
        elif any(cn < -h for cn in recent_cusum_neg):
            trend = 'improving'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'alerts': alerts,
            'cusum_pos': cusum_pos,
            'cusum_neg': cusum_neg
        }

    def _recent_values_assessment(self, ews_values):
        """Recent values assessment (adapted from original)"""
        if len(ews_values) < self.recent_values_count:
            return {'improvement': None, 'reason': 'insufficient_recent_data'}

        recent_values = ews_values[-self.recent_values_count:]
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)

        improvement = slope < 0

        return {
            'improvement': improvement,
            'recent_slope': slope,
            'r_value': r_value,
            'p_value': p_value
        }

    def _generate_ews_clinical_interpretation(self, trend, confidence):
        """Generate clinical interpretation for EWS trends"""
        interpretations = {
            'improving': {
                'high': 'Patient showing strong improvement trend - continue current care plan with regular monitoring',
                'moderate': 'Patient trending toward improvement - maintain current interventions',
                'low': 'Possible improvement trend - continue close monitoring'
            },
            'deteriorating': {
                'high': 'Patient showing clear deterioration - immediate medical review and intervention required',
                'moderate': 'Patient may be deteriorating - increase monitoring frequency and consider medical review',
                'low': 'Possible deterioration trend - maintain vigilant monitoring'
            },
            'stable': {
                'high': 'Patient condition stable - continue routine care',
                'moderate': 'Patient appears stable - regular monitoring sufficient',
                'low': 'Patient status unclear - maintain current monitoring'
            }
        }

        return interpretations.get(trend, {}).get(confidence, 'Clinical assessment recommended')

    def create_vitals_dashboard_plots(self, vitals_data, analysis_results):
        """
        Create comprehensive dashboard plots for Patient Analysis

        Returns:
            Dictionary of Plotly figure objects for dashboard integration
        """
        plots = {}

        # 1. EWS Score Trend Plot
        if analysis_results['ews_trend_analysis']:
            ews_history = analysis_results['ews_history']

            fig = go.Figure()

            timestamps = [datetime.fromisoformat(ts) for ts in ews_history['timestamps']]
            scores = ews_history['scores']

            # Create hover text with detailed breakdown
            hover_texts = []
            detailed_history = ews_history.get('detailed_history', [])

            for i, (timestamp, score) in enumerate(zip(timestamps, scores)):
                if i < len(detailed_history):
                    detail = detailed_history[i]
                    vitals_text = "<br>".join([f"{vital}: {value}" for vital, value in detail['vitals_used'].items()])
                    hover_text = f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>EWS Score: {score}<br>Vitals:<br>{vitals_text}"
                else:
                    hover_text = f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>EWS Score: {score}"
                hover_texts.append(hover_text)

            # Add EWS score line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name='EWS Score',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts
            ))

            # Add risk level zones
            max_score = max(scores) if scores else 10
            fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1, annotation_text="Low Risk", line_width=0)
            fig.add_hrect(y0=1, y1=4, fillcolor="yellow", opacity=0.1, annotation_text="Low-Medium Risk", line_width=0)
            fig.add_hrect(y0=4, y1=6, fillcolor="orange", opacity=0.1, annotation_text="Medium Risk", line_width=0)
            fig.add_hrect(y0=6, y1=10, fillcolor="red", opacity=0.1, annotation_text="High Risk", line_width=0)
            fig.add_hrect(y0=10, y1=max(max_score + 1, 15), fillcolor="darkred", opacity=0.1, annotation_text="Critical", line_width=0)

            fig.update_layout(
                title='Early Warning System (EWS) Score Trend',
                xaxis_title='Time',
                yaxis_title='EWS Score',
                height=400,
                showlegend=True
            )

            plots['ews_trend'] = fig

        # 2. Individual Vitals Trends
        vitals_subplot_data = []

        for vital_name, vital_info in vitals_data.items():
            if vital_name in self.ews_weights and len(vital_info['history']) > 0:
                # Combine historical and current data
                all_values = [h['value'] for h in vital_info['history']] + [vital_info['current_value']]
                all_timestamps = [datetime.fromtimestamp(h['timestamp']) for h in vital_info['history']] + [datetime.fromtimestamp(vital_info['current_timestamp'])]

                vitals_subplot_data.append({
                    'name': vital_name,
                    'timestamps': all_timestamps,
                    'values': all_values,
                    'units': vital_info['units']
                })

        if vitals_subplot_data:
            # Create subplots for individual vitals
            n_vitals = len(vitals_subplot_data)
            cols = 2
            rows = (n_vitals + cols - 1) // cols

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"{v['name']} ({v['units']})" for v in vitals_subplot_data],
                vertical_spacing=0.08
            )

            for i, vital_data in enumerate(vitals_subplot_data):
                row = (i // cols) + 1
                col = (i % cols) + 1

                fig.add_trace(
                    go.Scatter(
                        x=vital_data['timestamps'],
                        y=vital_data['values'],
                        mode='lines+markers',
                        name=vital_data['name'],
                        showlegend=False
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                title='Individual Vitals Trends',
                height=200 * rows,
                showlegend=False
            )

            plots['individual_vitals'] = fig

        # 3. Current Risk Assessment Gauge
        current_ews = analysis_results['current_ews']

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_ews['total_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current EWS Risk Level"},
            delta = {'reference': 5},
            gauge = {
                'axis': {'range': [None, 15]},
                'bar': {'color': current_ews['risk_color']},
                'steps': [
                    {'range': [0, 1], 'color': "lightgreen"},
                    {'range': [1, 4], 'color': "yellow"},
                    {'range': [4, 6], 'color': "orange"},
                    {'range': [6, 10], 'color': "red"},
                    {'range': [10, 15], 'color': "darkred"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))

        fig.update_layout(height=400, title=f"Risk Category: {current_ews['risk_category']}")
        plots['current_risk_gauge'] = fig

        return plots

    def generate_clinical_summary(self, analysis_results):
        """
        Generate a comprehensive clinical summary for dashboard display

        Returns:
            Dictionary with structured clinical information
        """
        current_ews = analysis_results['current_ews']
        ews_trend = analysis_results['ews_trend_analysis']
        individual_vitals = analysis_results['individual_vitals']

        summary = {
            'current_status': {
                'ews_score': current_ews['total_score'],
                'risk_category': current_ews['risk_category'],
                'risk_color': current_ews['risk_color'],
                'clinical_response': current_ews['clinical_response']
            },
            'trend_assessment': {},
            'individual_vital_alerts': [],
            'recommendations': [],
            'monitoring_frequency': 'routine'
        }

        # Add trend assessment if available
        if ews_trend:
            summary['trend_assessment'] = {
                'overall_trend': ews_trend['overall_trend'],
                'confidence': ews_trend['confidence'],
                'interpretation': ews_trend['clinical_interpretation']
            }

            # Update monitoring frequency based on trend
            if ews_trend['overall_trend'] == 'deteriorating':
                if ews_trend['confidence'] == 'high':
                    summary['monitoring_frequency'] = 'continuous'
                else:
                    summary['monitoring_frequency'] = 'frequent'
            elif ews_trend['overall_trend'] == 'improving':
                summary['monitoring_frequency'] = 'regular'

        # Process individual vital alerts
        for vital_name, vital_analysis in individual_vitals.items():
            if vital_analysis.get('status') == 'analyzed' and vital_analysis.get('significance') == 'significant':
                alert = {
                    'vital': vital_name,
                    'trend': vital_analysis['trend'],
                    'interpretation': vital_analysis['clinical_interpretation'],
                    'slope': vital_analysis['slope'],
                    'confidence': vital_analysis['r_squared']
                }
                summary['individual_vital_alerts'].append(alert)

        # Generate recommendations
        if current_ews['total_score'] >= 7:
            summary['recommendations'].append('Immediate medical review required')
            summary['recommendations'].append('Consider ICU consultation')
        elif current_ews['total_score'] >= 5:
            summary['recommendations'].append('Medical review within 1 hour')
            summary['recommendations'].append('Increase vital signs monitoring frequency')
        elif current_ews['total_score'] >= 3:
            summary['recommendations'].append('Clinical assessment recommended')
            summary['recommendations'].append('Monitor for changes')

        if ews_trend and ews_trend['overall_trend'] == 'deteriorating':
            summary['recommendations'].append('Trending toward deterioration - proactive intervention may be needed')

        return summary

# Example usage function for testing
def test_vitals_analyzer():
    """Test the RMSAI Vitals Analyzer with sample data"""
    analyzer = RMSAIVitalsAnalyzer()

    # Test EWS scoring
    sample_vitals = {
        'HR': 95,
        'Systolic': 110,
        'SpO2': 96,
        'RespRate': 18,
        'Temp': 99.2,  # Fahrenheit
        'Pulse': 93
    }

    ews_result = analyzer.calculate_ews_score(sample_vitals)
    print("Sample EWS Analysis:")
    print(f"Total Score: {ews_result['total_score']}")
    print(f"Risk Category: {ews_result['risk_category']}")
    print(f"Clinical Response: {ews_result['clinical_response']}")

    return analyzer

if __name__ == "__main__":
    test_vitals_analyzer()