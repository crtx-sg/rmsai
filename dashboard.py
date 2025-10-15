#!/usr/bin/env python3
"""
RMSAI Monitoring Dashboard
==========================

Comprehensive web dashboard for real-time monitoring of the ECG processing pipeline,
anomaly detection results, and system performance.

Features:
- Real-time system overview with key metrics
- Interactive anomaly detection timeline
- Condition-based analysis with visualizations
- ECG lead performance analysis
- Recent anomalies table with filtering
- System health monitoring
- Similarity search interface
- Patient Events view with AI anomaly verdicts and processing stats
- Patient-specific analysis with anomaly patterns and event history
- Auto-refresh functionality

Usage:
    streamlit run dashboard.py

    # Access at http://localhost:8501
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
import time
import requests
import json
import re
import h5py
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from config import CLINICAL_SEVERITY_ORDER, HR_THRESHOLDS, sort_by_severity, get_severity_score

# Configure Streamlit page
st.set_page_config(
    page_title="RMS.AI ECG Anomaly Detection Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
    }
    .success-alert {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
    }
    .stMetric > label {
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

class RMSAIDashboard:
    """Real-time monitoring dashboard for RMSAI system"""

    def __init__(self):
        self.db_path = "rmsai_metadata.db"
        self.api_base = "http://localhost:8000/api/v1"

        # Initialize session state
        self.initialize_session_state()

        # Load processor configuration to get selected leads
        self.selected_leads = self._get_selected_leads()

    def _extract_patient_id(self, source_file):
        """Extract patient ID from source file path"""
        if 'PT' in str(source_file):
            # Find PT followed by digits
            match = re.search(r'PT\d+', str(source_file))
            if match:
                return match.group()
        return 'Unknown'

    def get_unified_ai_verdict(self, anomaly_types_list, heart_rate=None, anomaly_status_count=0):
        """
        Simplified AI verdict calculation: Pick the condition with highest severity from all chunks.

        Logic:
        1. Collect all anomaly types from chunks (AF, VT only from LSTM)
        2. If rhythm abnormalities found (AF/VT), return highest severity
        3. If only Normal Sinus Rhythm variations, simplify based on HR patterns:
           - If only NSR + Tachycardia → return "Tachycardia"
           - If only NSR + Bradycardia → return "Bradycardia"
           - Otherwise → return HR-based verdict

        Args:
            anomaly_types_list: List of anomaly types detected by LSTM for all chunks
            heart_rate: Heart rate value (optional, for HR-based detection)
            anomaly_status_count: Number of chunks marked as anomalous

        Returns:
            str: AI verdict - highest severity condition found
        """
        # Use shared configuration
        bradycardia_max_hr = HR_THRESHOLDS['bradycardia_max']
        tachycardia_min_hr = HR_THRESHOLDS['tachycardia_min']

        # Collect all unique anomaly types from chunks (only AF and VT from LSTM)
        unique_anomalies = list(set([t for t in anomaly_types_list if t and t != 'None' and pd.notna(t)]))

        # If rhythm abnormalities found (AF or VT), return the highest severity one
        if unique_anomalies:
            sorted_anomalies = sort_by_severity(unique_anomalies)
            return sorted_anomalies[0] if sorted_anomalies else "Normal"

        # No rhythm abnormalities detected - all chunks are Normal Sinus Rhythm
        # Check HR to determine if it's consistently Tachycardia or Bradycardia
        if heart_rate is not None:
            try:
                hr_value = float(heart_rate)
                if hr_value <= bradycardia_max_hr:
                    # If event HR is consistently in brady range, report as Bradycardia
                    return 'Bradycardia'
                elif hr_value >= tachycardia_min_hr:
                    # If event HR is consistently in tachy range, report as Tachycardia
                    return 'Tachycardia'
                else:
                    # Normal HR range
                    return 'Normal'
            except (ValueError, TypeError):
                pass

        return "Normal"

    def calculate_performance_metrics(self, ai_predictions, ground_truth, positive_class='anomaly'):
        """
        Calculate Precision, Recall, F1 score for binary or multiclass classification

        Args:
            ai_predictions: List of AI predictions
            ground_truth: List of ground truth labels
            positive_class: Class to treat as positive for binary metrics ('anomaly' or specific condition)

        Returns:
            dict: Dictionary containing precision, recall, f1, accuracy, and detailed counts
        """
        if len(ai_predictions) != len(ground_truth):
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'support': 0}

        # Convert to binary classification if needed
        if positive_class == 'anomaly':
            # Binary: Normal vs Any Anomaly
            ai_binary = ['anomaly' if pred != 'Normal' and pred != 'Unknown' else 'normal' for pred in ai_predictions]
            gt_binary = ['anomaly' if gt != 'Normal' and gt != 'Unknown' else 'normal' for gt in ground_truth]
        else:
            # Specific condition vs everything else
            ai_binary = [positive_class if pred == positive_class else 'other' for pred in ai_predictions]
            gt_binary = [positive_class if gt == positive_class else 'other' for gt in ground_truth]
            positive_class = positive_class  # Keep original for calculations

        # Calculate confusion matrix components
        tp = sum(1 for ai, gt in zip(ai_binary, gt_binary) if ai != 'normal' and ai != 'other' and gt != 'normal' and gt != 'other')
        fp = sum(1 for ai, gt in zip(ai_binary, gt_binary) if ai != 'normal' and ai != 'other' and (gt == 'normal' or gt == 'other'))
        fn = sum(1 for ai, gt in zip(ai_binary, gt_binary) if (ai == 'normal' or ai == 'other') and gt != 'normal' and gt != 'other')
        tn = sum(1 for ai, gt in zip(ai_binary, gt_binary) if (ai == 'normal' or ai == 'other') and (gt == 'normal' or gt == 'other'))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': tp + fn,  # Total positive cases
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

    def calculate_exact_condition_metrics(self, ai_predictions, ground_truth):
        """
        Calculate metrics using exact condition matching (not binary anomaly vs normal)

        Args:
            ai_predictions: List of AI predictions (exact conditions)
            ground_truth: List of ground truth labels (exact conditions)

        Returns:
            dict: Dictionary containing precision, recall, f1, accuracy for exact matching
        """
        if len(ai_predictions) != len(ground_truth):
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'support': 0}

        # Exact condition matching
        total_predictions = len(ai_predictions)
        correct_predictions = sum(1 for ai, gt in zip(ai_predictions, ground_truth) if ai == gt)

        # For multiclass exact matching, we calculate micro-averaged metrics
        # Get all unique conditions
        all_conditions = list(set(ai_predictions + ground_truth))

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        # Calculate per-condition metrics and sum them (micro-averaging)
        for condition in all_conditions:
            tp = sum(1 for ai, gt in zip(ai_predictions, ground_truth)
                    if ai == condition and gt == condition)
            fp = sum(1 for ai, gt in zip(ai_predictions, ground_truth)
                    if ai == condition and gt != condition)
            fn = sum(1 for ai, gt in zip(ai_predictions, ground_truth)
                    if ai != condition and gt == condition)
            tn = sum(1 for ai, gt in zip(ai_predictions, ground_truth)
                    if ai != condition and gt != condition)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        # Calculate micro-averaged metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': total_predictions,
            'correct': correct_predictions,
            'total': total_predictions,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn
        }

    def get_clinical_group(self, condition: str) -> str:
        """Group conditions into clinical categories for evaluation"""
        # Normal group includes all Normal Sinus Rhythm variations
        if condition in ['Normal', 'Unknown']:
            return 'Normal'
        # Rhythm abnormalities (AF, VT)
        elif condition in ['Atrial Fibrillation (PTB-XL)', 'Ventricular Tachycardia (MIT-BIH)']:
            return 'Rhythm_Abnormality'
        # Rate abnormalities (Tachycardia, Bradycardia)
        elif condition in ['Tachycardia', 'Bradycardia']:
            return 'Rate_Abnormality'
        else:
            return 'Other_Abnormality'

    def calculate_clinical_grouped_metrics(self, ai_predictions, ground_truth):
        """
        Calculate metrics using clinical grouping instead of exact matching
        Groups: Normal, Rhythm_Abnormality, Rate_Abnormality, Other_Abnormality
        """
        if len(ai_predictions) != len(ground_truth):
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'support': 0}

        # Convert to clinical groups
        ai_groups = [self.get_clinical_group(pred) for pred in ai_predictions]
        gt_groups = [self.get_clinical_group(gt) for gt in ground_truth]

        # Calculate exact group matching
        total_predictions = len(ai_groups)
        correct_predictions = sum(1 for ai, gt in zip(ai_groups, gt_groups) if ai == gt)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate per-group metrics
        unique_groups = list(set(ai_groups + gt_groups))
        total_tp = total_fp = total_fn = 0

        for group in unique_groups:
            tp = sum(1 for ai, gt in zip(ai_groups, gt_groups) if ai == group and gt == group)
            fp = sum(1 for ai, gt in zip(ai_groups, gt_groups) if ai == group and gt != group)
            fn = sum(1 for ai, gt in zip(ai_groups, gt_groups) if ai != group and gt == group)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calculate micro-averaged metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': total_predictions,
            'clinical_groups': True
        }

    def calculate_condition_specific_metrics(self, ai_predictions, ground_truth):
        """
        Calculate metrics for each specific condition using exact matching

        Returns:
            dict: Metrics for each condition type
        """
        # Get unique conditions (excluding Unknown)
        all_conditions = set(ground_truth + ai_predictions)
        conditions = [c for c in all_conditions if c != 'Unknown']

        results = {}

        # Overall exact condition matching metrics
        results['Overall_Anomaly_Detection'] = self.calculate_exact_condition_metrics(
            ai_predictions, ground_truth
        )

        # Condition-specific metrics
        for condition in conditions:
            results[condition] = self.calculate_performance_metrics(
                ai_predictions, ground_truth, positive_class=condition
            )

        return results

    def initialize_session_state(self):
        """Initialize session state variables"""
        # Cache for expensive operations
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
            st.session_state.cache_time = {}

    def _get_selected_leads(self) -> List[str]:
        """Get currently selected leads from processor configuration"""
        try:
            # Try to get from API first
            response = requests.get(f"{self.api_base}/leads", timeout=3)
            if response.status_code == 200:
                leads_data = response.json()
                return [lead['lead_name'] for lead in leads_data['leads']]
        except:
            pass

        # Fallback: query database for available leads
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT lead_name FROM chunks ORDER BY lead_name")
                leads = [row[0] for row in cursor.fetchall()]
                return leads
        except:
            # Default leads if all else fails
            return ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']

    def load_data(self, use_cache: bool = True, cache_duration: int = 30):
        """Load data from SQLite database with caching"""
        cache_key = "main_data"
        current_time = time.time()

        # Check cache
        if (use_cache and cache_key in st.session_state.data_cache and
            cache_key in st.session_state.cache_time):
            if current_time - st.session_state.cache_time[cache_key] < cache_duration:
                return st.session_state.data_cache[cache_key]

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Main chunks data - expanded for patient views
                chunks_df = pd.read_sql_query("""
                    SELECT chunk_id, event_id, source_file, lead_name,
                           anomaly_status, anomaly_type, error_score,
                           processing_timestamp, vector_id, metadata
                    FROM chunks
                    ORDER BY processing_timestamp DESC
                    LIMIT 5000
                """, conn)

                # Files data
                files_df = pd.read_sql_query("""
                    SELECT filename, file_hash, status, processing_timestamp, error_message
                    FROM processed_files
                    ORDER BY processing_timestamp DESC
                    LIMIT 1000
                """, conn)

                # Process chunks data for patient views
                if not chunks_df.empty:
                    # Add timestamp column
                    chunks_df['timestamp'] = pd.to_datetime(chunks_df['processing_timestamp'])
                    # Add processed_time (for now, same as processing_timestamp)
                    chunks_df['processed_time'] = chunks_df['processing_timestamp']

                    # Extract patient ID from source_file
                    def extract_patient_id(source_file):
                        if 'PT' in str(source_file):
                            # Find PT followed by digits
                            match = re.search(r'PT\d+', str(source_file))
                            if match:
                                return match.group()
                        return 'Unknown'

                    chunks_df['patient_id'] = chunks_df['source_file'].apply(extract_patient_id)

                    # Extract heart rate from metadata JSON
                    def extract_heart_rate(metadata_json):
                        if pd.isna(metadata_json) or not metadata_json:
                            return None
                        try:
                            import json
                            metadata = json.loads(metadata_json)
                            return metadata.get('heart_rate', None)
                        except (json.JSONDecodeError, TypeError):
                            return None

                    chunks_df['heart_rate'] = chunks_df['metadata'].apply(extract_heart_rate)

                    # DEBUG: Check heart rate extraction for event_1006
                    event_1006_sample = chunks_df[chunks_df['event_id'] == 'event_1006'].head(1)
                    if not event_1006_sample.empty:
                        hr = event_1006_sample.iloc[0]['heart_rate']
                        print(f"🔍 DEBUG LOAD: event_1006 heart_rate = {hr} (type: {type(hr)})")

                    # Add condition column - map from anomaly types or infer from pattern
                    def infer_condition(row):
                        if pd.isna(row['anomaly_type']) or row['anomaly_type'] == 'None':
                            return 'Normal'
                        elif row['anomaly_status'] == 'normal':
                            return 'Normal'
                        else:
                            # Use the anomaly_type as the condition if available
                            return row['anomaly_type']

                    chunks_df['condition'] = chunks_df.apply(infer_condition, axis=1)

                # Cache the results
                data = (chunks_df, files_df)
                st.session_state.data_cache[cache_key] = data
                st.session_state.cache_time[cache_key] = current_time

                return data

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_api_stats(self) -> Optional[Dict]:
        """Get statistics from API if available"""
        try:
            response = requests.get(f"{self.api_base}/stats", timeout=3)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def render_header(self):
        """Render dashboard header with controls"""
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.title("🫀 RMSAI ECG Anomaly Detection Dashboard")
            st.markdown("Real-time monitoring of ECG processing and anomaly detection")

        with col2:
            # Auto-refresh control
            auto_refresh = st.selectbox(
                "Auto-refresh",
                ["Off", "10s", "30s", "60s"],
                index=0
            )

        with col3:
            # Manual refresh button
            if st.button("🔄 Refresh Now", width='stretch'):
                # Clear cache
                st.session_state.data_cache = {}
                st.session_state.cache_time = {}
                st.rerun()

        # Handle auto-refresh with proper timing control
        if auto_refresh != "Off":
            refresh_seconds = int(auto_refresh[:-1])

            # Display refresh status
            st.sidebar.info(f"🔄 Auto-refresh: {auto_refresh}")

            # Simple approach - let the user know auto-refresh is on
            # The actual refresh will happen on next user interaction or manual refresh
            # This prevents the infinite loop while still providing refresh capability

        else:
            st.sidebar.info("🔄 Auto-refresh: Off")

        return auto_refresh

    def render_overview_metrics(self, chunks_df: pd.DataFrame, files_df: pd.DataFrame):
        """Render overview metrics cards"""
        st.header("📊 System Overview")

        # Calculate metrics
        total_chunks = len(chunks_df)
        total_anomalies = len(chunks_df[chunks_df['anomaly_status'] == 'anomaly'])
        anomaly_rate = (total_anomalies / total_chunks * 100) if total_chunks > 0 else 0
        files_processed = len(files_df[files_df['status'] == 'completed'])
        avg_error = chunks_df['error_score'].mean() if len(chunks_df) > 0 else 0

        # System status
        if len(chunks_df) > 0:
            latest_processing = pd.to_datetime(chunks_df['processing_timestamp'].max())
            time_since_last = datetime.now() - latest_processing

            if time_since_last.total_seconds() < 300:  # 5 minutes
                status = "🟢 Active"
                status_color = "success"
            elif time_since_last.total_seconds() < 1800:  # 30 minutes
                status = "🟡 Recent"
                status_color = "warning"
            else:
                status = "🔴 Inactive"
                status_color = "error"
        else:
            status = "⚫ No Data"
            status_color = "error"

        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Chunks", f"{total_chunks:,}")

        with col2:
            st.metric("Anomalies", f"{total_anomalies:,}", f"{anomaly_rate:.1f}%")

        with col3:
            st.metric("Files Processed", f"{files_processed:,}")

        with col4:
            st.metric("Avg Error Score", f"{avg_error:.4f}")

        with col5:
            st.metric("System Status", status)

        with col6:
            # Processing rate (chunks per hour)
            if len(chunks_df) > 0:
                chunks_df['timestamp'] = pd.to_datetime(chunks_df['processing_timestamp'])
                time_span = (chunks_df['timestamp'].max() - chunks_df['timestamp'].min()).total_seconds() / 3600
                processing_rate = len(chunks_df) / time_span if time_span > 0 else 0
                st.metric("Rate", f"{processing_rate:.0f}/hr")
            else:
                st.metric("Rate", "0/hr")

        # Alert for high anomaly rate
        if anomaly_rate > 20:
            # Use Streamlit's built-in error component for better visibility
            st.error(f"⚠️ **High Anomaly Rate Alert:** {anomaly_rate:.1f}% of recent chunks detected as anomalies. This may indicate a system issue or unusual patient conditions.")

            # Also add the custom styled version
            st.markdown(f"""
            <div class="anomaly-alert">
                ⚠️ <strong>High Anomaly Rate Alert:</strong> {anomaly_rate:.1f}% of recent chunks detected as anomalies.
                This may indicate a system issue or unusual patient conditions.
            </div>
            """, unsafe_allow_html=True)

        # System-wide Performance Metrics
        self.render_system_performance_metrics(chunks_df)

    def render_system_performance_metrics(self, chunks_df: pd.DataFrame):
        """Render system-wide precision, recall, and F1 metrics"""
        st.subheader("🎯 System-Wide Performance Metrics")

        if len(chunks_df) == 0:
            st.info("No data available for performance analysis")
            return

        # Get ground truth for all chunks
        try:
            ground_truth_conditions = self._get_ground_truth_conditions(chunks_df)

            # Filter to events with known ground truth
            valid_chunks = chunks_df[chunks_df['event_id'].isin(ground_truth_conditions.keys())].copy()

            if len(valid_chunks) == 0:
                st.info("No events with ground truth available for performance analysis")
                return

            # Group by event to get AI verdicts
            events_grouped = valid_chunks.groupby('event_id', as_index=False).agg({
                'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
                'anomaly_status': lambda x: (x == 'anomaly').sum(),
                'heart_rate': 'first'
            })

            # Calculate AI verdicts for each event
            ai_predictions = []
            ground_truth = []

            for _, row in events_grouped.iterrows():
                event_id = row['event_id']

                # Get AI verdict using the same logic as Patient Analysis
                ai_verdict = self.get_unified_ai_verdict(
                    row['anomaly_type'],
                    row['heart_rate'],
                    row['anomaly_status']
                )

                # Get ground truth
                gt_condition = ground_truth_conditions.get(event_id, 'Unknown')

                # Only include events with known ground truth
                if gt_condition != 'Unknown':
                    ai_predictions.append(ai_verdict)
                    ground_truth.append(gt_condition)

            if len(ai_predictions) == 0:
                st.info("No valid events for performance calculation")
                return

            # Calculate system-wide metrics
            system_metrics = self.calculate_condition_specific_metrics(ai_predictions, ground_truth)

            # Display overall system performance
            overall = system_metrics['Overall_Anomaly_Detection']

            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

            with perf_col1:
                st.metric("System Precision", f"{overall['precision']*100:.1f}%",
                         help="Overall reliability of anomaly predictions across all patients")

            with perf_col2:
                st.metric("System Recall", f"{overall['recall']*100:.1f}%",
                         help="Overall sensitivity in detecting anomalies across all patients")

            with perf_col3:
                st.metric("System F1 Score", f"{overall['f1']*100:.1f}%",
                         help="Balanced performance measure across all patients")

            with perf_col4:
                st.metric("Events Analyzed", f"{len(ai_predictions):,}",
                         help="Total events with ground truth used for performance calculation")

            # Detailed system performance breakdown
            with st.expander("📊 Detailed System Performance Breakdown", expanded=False):
                sys_col1, sys_col2 = st.columns(2)

                with sys_col1:
                    st.write("**System-Wide Confusion Matrix**")

                    confusion_data = [
                        ['', 'Predicted Normal', 'Predicted Anomaly', 'Total'],
                        ['Actual Normal', f"{overall['tn']}", f"{overall['fp']}", f"{overall['tn'] + overall['fp']}"],
                        ['Actual Anomaly', f"{overall['fn']}", f"{overall['tp']}", f"{overall['fn'] + overall['tp']}"],
                        ['Total', f"{overall['tn'] + overall['fn']}", f"{overall['fp'] + overall['tp']}", f"{len(ai_predictions)}"]
                    ]

                    st.table(pd.DataFrame(confusion_data[1:], columns=confusion_data[0]))

                with sys_col2:
                    st.write("**Performance by Condition Type**")

                    # Show performance for each condition
                    condition_perf_data = []

                    for condition, metrics in system_metrics.items():
                        if condition != 'Overall_Anomaly_Detection' and metrics['support'] > 0:
                            condition_perf_data.append({
                                'Condition': condition,
                                'Events': metrics['support'],
                                'Precision': f"{metrics['precision']*100:.1f}%",
                                'Recall': f"{metrics['recall']*100:.1f}%",
                                'F1': f"{metrics['f1']*100:.1f}%"
                            })

                    if condition_perf_data:
                        st.dataframe(pd.DataFrame(condition_perf_data), use_container_width=True)
                    else:
                        st.info("No specific conditions found for detailed analysis")

                # Clinical insights
                st.write("**Clinical Performance Insights:**")

                if overall['precision'] > 0.8:
                    precision_insight = "✅ High precision - Low false alarm rate"
                elif overall['precision'] > 0.6:
                    precision_insight = "⚠️ Moderate precision - Some false alarms expected"
                else:
                    precision_insight = "❌ Low precision - High false alarm rate may cause alert fatigue"

                if overall['recall'] > 0.8:
                    recall_insight = "✅ High recall - Most anomalies detected"
                elif overall['recall'] > 0.6:
                    recall_insight = "⚠️ Moderate recall - Some anomalies may be missed"
                else:
                    recall_insight = "❌ Low recall - Significant anomalies being missed"

                st.write(f"• {precision_insight}")
                st.write(f"• {recall_insight}")

                if overall['f1'] > 0.8:
                    st.success("🎯 Excellent overall performance - System is performing well clinically")
                elif overall['f1'] > 0.6:
                    st.warning("⚖️ Good overall performance - Consider threshold tuning for improvement")
                else:
                    st.error("🔧 Performance needs improvement - Review model parameters and thresholds")

        except Exception as e:
            st.error(f"Error calculating system performance metrics: {str(e)}")

    def render_anomaly_timeline(self, chunks_df: pd.DataFrame):
        """Render anomaly detection timeline with interactive features"""
        st.header("📈 Anomaly Detection Timeline")

        if len(chunks_df) == 0:
            st.warning("No data available for timeline")
            return

        # Time aggregation selection
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            time_window = st.selectbox(
                "Time Window",
                ["1 hour", "6 hours", "24 hours", "7 days"],
                index=2
            )

        with col2:
            aggregation = st.selectbox(
                "Aggregation",
                ["Hour", "Day"],
                index=0
            )

        with col3:
            show_leads = st.checkbox("Show by Lead", value=False)

        # Filter data based on time window
        chunks_df['timestamp'] = pd.to_datetime(chunks_df['processing_timestamp'])
        now = datetime.now()

        if time_window == "1 hour":
            cutoff = now - timedelta(hours=1)
        elif time_window == "6 hours":
            cutoff = now - timedelta(hours=6)
        elif time_window == "24 hours":
            cutoff = now - timedelta(hours=24)
        else:  # 7 days
            cutoff = now - timedelta(days=7)

        filtered_df = chunks_df[chunks_df['timestamp'] >= cutoff]

        if len(filtered_df) == 0:
            st.warning(f"No data available for the selected time window ({time_window})")
            return

        # Prepare timeline data
        if aggregation == "Hour":
            filtered_df['time_bucket'] = filtered_df['timestamp'].dt.floor('h')
        else:
            filtered_df['time_bucket'] = filtered_df['timestamp'].dt.floor('D')

        if show_leads:
            # Group by time and lead
            timeline_data = filtered_df.groupby(['time_bucket', 'lead_name', 'anomaly_status']).size().unstack(fill_value=0)

            # Create subplot for each lead
            leads = filtered_df['lead_name'].unique()
            fig = make_subplots(
                rows=len(leads), cols=1,
                subplot_titles=[f"Lead {lead}" for lead in leads],
                shared_xaxes=True,
                vertical_spacing=0.02
            )

            for i, lead in enumerate(leads, 1):
                lead_data = timeline_data.xs(lead, level=1)

                if 'normal' in lead_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=lead_data.index,
                            y=lead_data['normal'],
                            mode='lines+markers',
                            name=f'{lead} Normal',
                            line=dict(color='green'),
                            showlegend=(i == 1)
                        ),
                        row=i, col=1
                    )

                if 'anomaly' in lead_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=lead_data.index,
                            y=lead_data['anomaly'],
                            mode='lines+markers',
                            name=f'{lead} Anomaly',
                            line=dict(color='red'),
                            fill='tonexty',
                            showlegend=(i == 1)
                        ),
                        row=i, col=1
                    )

        else:
            # Standard timeline
            timeline_data = filtered_df.groupby(['time_bucket', 'anomaly_status']).size().unstack(fill_value=0)

            fig = go.Figure()

            if 'normal' in timeline_data.columns:
                fig.add_trace(go.Scatter(
                    x=timeline_data.index,
                    y=timeline_data['normal'],
                    mode='lines+markers',
                    name='Normal',
                    line=dict(color='green'),
                    fill='tozeroy'
                ))

            if 'anomaly' in timeline_data.columns:
                fig.add_trace(go.Scatter(
                    x=timeline_data.index,
                    y=timeline_data['anomaly'],
                    mode='lines+markers',
                    name='Anomalies',
                    line=dict(color='red'),
                    fill='tonexty'
                ))

        fig.update_layout(
            title=f"Processing Activity - {time_window} window",
            xaxis_title="Time",
            yaxis_title="Number of Chunks",
            hovermode='x unified',
            height=400 if not show_leads else 200 * len(leads)
        )

        st.plotly_chart(fig, width='stretch')

    def render_condition_analysis(self, chunks_df: pd.DataFrame):
        """Render condition-based analysis with enhanced visualizations"""
        st.header("🔍 Condition Analysis")

        if len(chunks_df) == 0:
            st.warning("No data available for condition analysis")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Condition distribution pie chart
            condition_counts = chunks_df['anomaly_type'].fillna('Unknown').value_counts()

            fig = px.pie(
                values=condition_counts.values,
                names=condition_counts.index,
                title="Condition Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Anomaly rate by condition
            condition_stats = chunks_df.groupby('anomaly_type').agg({
                'anomaly_status': lambda x: (x == 'anomaly').mean() * 100,
                'error_score': 'mean',
                'chunk_id': 'count'
            }).round(2)

            condition_stats.columns = ['Anomaly Rate (%)', 'Avg Error Score', 'Count']
            condition_stats = condition_stats.sort_values('Anomaly Rate (%)', ascending=True)

            fig = px.bar(
                x=condition_stats['Anomaly Rate (%)'],
                y=condition_stats.index,
                orientation='h',
                title="Anomaly Rate by Condition",
                color=condition_stats['Anomaly Rate (%)'],
                color_continuous_scale='Reds',
                hover_data={'Count': condition_stats['Count']}
            )
            fig.update_layout(yaxis_title="Condition")
            st.plotly_chart(fig, width='stretch')

        # Detailed condition table
        st.subheader("Condition Statistics")

        # Add sample counts and other metrics
        detailed_stats = chunks_df.groupby('anomaly_type').agg({
            'chunk_id': 'count',
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'error_score': ['mean', 'std', 'min', 'max']
        }).round(4)

        # Flatten column names
        detailed_stats.columns = ['Total Chunks', 'Anomalies', 'Mean Error', 'Std Error', 'Min Error', 'Max Error']
        detailed_stats['Anomaly Rate (%)'] = (detailed_stats['Anomalies'] / detailed_stats['Total Chunks'] * 100).round(2)
        detailed_stats = detailed_stats[['Total Chunks', 'Anomalies', 'Anomaly Rate (%)', 'Mean Error', 'Std Error', 'Min Error', 'Max Error']]

        st.dataframe(
            detailed_stats,
            use_container_width=True,
            column_config={
                "Total Chunks": st.column_config.NumberColumn(format="%d"),
                "Anomalies": st.column_config.NumberColumn(format="%d"),
                "Anomaly Rate (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Mean Error": st.column_config.NumberColumn(format="%.4f"),
                "Std Error": st.column_config.NumberColumn(format="%.4f"),
                "Min Error": st.column_config.NumberColumn(format="%.4f"),
                "Max Error": st.column_config.NumberColumn(format="%.4f")
            }
        )

    def render_lead_analysis(self, chunks_df: pd.DataFrame):
        """Render ECG lead analysis with comprehensive metrics"""
        st.header("🔬 ECG Lead Analysis")

        if len(chunks_df) == 0:
            st.warning("No data available for lead analysis")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Error scores by lead with error bars
            lead_stats = chunks_df.groupby('lead_name')['error_score'].agg(['mean', 'std', 'count'])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=lead_stats.index,
                y=lead_stats['mean'],
                error_y=dict(type='data', array=lead_stats['std']),
                name='Mean Error Score',
                marker_color='lightblue',
                hovertemplate='Lead: %{x}<br>Mean: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>'
            ))

            fig.update_layout(
                title="Average Error Score by ECG Lead",
                xaxis_title="ECG Lead",
                yaxis_title="Error Score",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Lead processing volume - filter to selected leads only
            selected_chunks = chunks_df[chunks_df['lead_name'].isin(self.selected_leads)] if len(chunks_df) > 0 else chunks_df
            lead_counts = selected_chunks['lead_name'].value_counts()

            fig = px.bar(
                x=lead_counts.index,
                y=lead_counts.values,
                title="Processing Volume by Lead",
                color=lead_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                xaxis_title="ECG Lead",
                yaxis_title="Number of Chunks"
            )
            st.plotly_chart(fig, width='stretch')

        # Lead anomaly heatmap
        st.subheader("Lead vs Condition Anomaly Heatmap")

        # Create heatmap data - filter to selected leads
        selected_chunks = chunks_df[chunks_df['lead_name'].isin(self.selected_leads)] if len(chunks_df) > 0 else chunks_df
        heatmap_data = selected_chunks.groupby(['lead_name', 'anomaly_type']).apply(
            lambda x: (x['anomaly_status'] == 'anomaly').mean() * 100, include_groups=False
        ).unstack(fill_value=0)

        if not heatmap_data.empty:
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                title="Anomaly Rate (%) by Lead and Condition",
                color_continuous_scale='Reds',
                aspect='auto',
                text_auto=True
            )
            fig.update_layout(
                xaxis_title="Condition",
                yaxis_title="ECG Lead"
            )
            st.plotly_chart(fig, width='stretch')

    def render_recent_anomalies(self, chunks_df: pd.DataFrame):
        """Render recent anomalies table with advanced filtering"""
        st.header("⚠️ Recent Anomalies")

        anomalies_df = chunks_df[chunks_df['anomaly_status'] == 'anomaly'].copy()

        if len(anomalies_df) == 0:
            st.info("No recent anomalies detected")
            return

        # Filtering controls
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Condition filter
            conditions = ['All'] + list(anomalies_df['anomaly_type'].fillna('Unknown').unique())
            selected_condition = st.selectbox("Filter by Condition", conditions)

        with col2:
            # Lead filter
            leads = ['All'] + list(anomalies_df['lead_name'].unique())
            selected_lead = st.selectbox("Filter by Lead", leads)

        with col3:
            # Error score threshold
            min_error = st.number_input(
                "Min Error Score",
                min_value=0.0,
                max_value=anomalies_df['error_score'].max(),
                value=0.0,
                step=0.01,
                format="%.4f"
            )

        with col4:
            # Number of records
            max_records = st.selectbox("Show Records", [20, 50, 100, 200], index=0)

        # Apply filters
        filtered_anomalies = anomalies_df.copy()

        if selected_condition != 'All':
            if selected_condition == 'Unknown':
                filtered_anomalies = filtered_anomalies[filtered_anomalies['anomaly_type'].isna()]
            else:
                filtered_anomalies = filtered_anomalies[filtered_anomalies['anomaly_type'] == selected_condition]

        if selected_lead != 'All':
            filtered_anomalies = filtered_anomalies[filtered_anomalies['lead_name'] == selected_lead]

        filtered_anomalies = filtered_anomalies[filtered_anomalies['error_score'] >= min_error]
        filtered_anomalies = filtered_anomalies.head(max_records)

        # Display summary
        st.write(f"Showing {len(filtered_anomalies)} of {len(anomalies_df)} total anomalies")

        if len(filtered_anomalies) == 0:
            st.warning("No anomalies match the current filters")
            return

        # Format data for display
        display_data = filtered_anomalies[[
            'chunk_id', 'event_id', 'lead_name', 'anomaly_type',
            'error_score', 'processing_timestamp'
        ]].copy()

        display_data['anomaly_type'] = display_data['anomaly_type'].fillna('Unknown')
        display_data['error_score'] = display_data['error_score'].round(4)
        display_data['processing_timestamp'] = pd.to_datetime(
            display_data['processing_timestamp']
        ).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Color code by error score
        def color_error_score(val):
            if val > 0.2:
                return 'background-color: #ffcccc'  # Light red for high error
            elif val > 0.1:
                return 'background-color: #fff2cc'  # Light yellow for medium error
            else:
                return ''

        styled_data = display_data.style.applymap(
            color_error_score,
            subset=['error_score']
        )

        st.dataframe(
            styled_data,
            column_config={
                "error_score": st.column_config.NumberColumn(
                    "Error Score",
                    help="Reconstruction error score",
                    format="%.4f"
                ),
                "chunk_id": st.column_config.TextColumn("Chunk ID"),
                "event_id": st.column_config.TextColumn("Event ID"),
                "lead_name": st.column_config.TextColumn("Lead"),
                "anomaly_type": st.column_config.TextColumn("Condition"),
                "processing_timestamp": st.column_config.TextColumn("Timestamp")
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Export functionality
        if st.button("📥 Export Filtered Anomalies"):
            csv = filtered_anomalies.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"rmsai_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    def render_system_health(self, files_df: pd.DataFrame):
        """Render system health metrics with comprehensive monitoring"""
        st.header("💊 System Health")

        col1, col2 = st.columns(2)

        with col1:
            # File processing status
            if len(files_df) > 0:
                status_counts = files_df['status'].value_counts()

                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="File Processing Status",
                    color_discrete_map={
                        'completed': '#28a745',
                        'processing': '#ffc107',
                        'failed': '#dc3545'
                    }
                )
                st.plotly_chart(fig, width='stretch')

                # Show error details if any
                failed_files = files_df[files_df['status'] == 'failed']
                if len(failed_files) > 0:
                    st.error(f"⚠️ {len(failed_files)} files failed to process")
                    with st.expander("View Failed Files"):
                        st.dataframe(
                            failed_files[['filename', 'error_message', 'processing_timestamp']],
                            use_container_width=True
                        )
            else:
                st.info("No file processing data available")

        with col2:
            # Processing timeline
            if len(files_df) > 0:
                files_df['timestamp'] = pd.to_datetime(files_df['processing_timestamp'])
                files_df['date'] = files_df['timestamp'].dt.date

                daily_counts = files_df.groupby(['date', 'status']).size().unstack(fill_value=0)

                fig = go.Figure()

                for status in daily_counts.columns:
                    color_map = {
                        'completed': '#28a745',
                        'processing': '#ffc107',
                        'failed': '#dc3545'
                    }
                    fig.add_trace(go.Scatter(
                        x=daily_counts.index,
                        y=daily_counts[status],
                        mode='lines+markers',
                        name=status.title(),
                        line=dict(color=color_map.get(status, '#007bff'))
                    ))

                fig.update_layout(
                    title="Daily File Processing",
                    xaxis_title="Date",
                    yaxis_title="Number of Files"
                )
                st.plotly_chart(fig, width='stretch')

        # System performance metrics
        st.subheader("Performance Metrics")

        if len(files_df) > 0:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                success_rate = (files_df['status'] == 'completed').mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")

            with col2:
                total_files = len(files_df)
                st.metric("Total Files", f"{total_files:,}")

            with col3:
                # Calculate processing time span
                if len(files_df) > 1:
                    time_span = (files_df['timestamp'].max() - files_df['timestamp'].min())
                    days = time_span.days
                    st.metric("Active Days", f"{days}")
                else:
                    st.metric("Active Days", "1")

            with col4:
                # Average files per day
                if len(files_df) > 1:
                    avg_per_day = len(files_df) / max(1, (files_df['timestamp'].max() - files_df['timestamp'].min()).days)
                    st.metric("Avg Files/Day", f"{avg_per_day:.1f}")
                else:
                    st.metric("Avg Files/Day", "1.0")

    def render_similarity_search(self, chunks_df: pd.DataFrame):
        """Render similarity search interface"""
        st.header("🔎 Similarity Search")

        if len(chunks_df) == 0:
            st.warning("No chunks available for similarity search")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            chunk_ids = chunks_df['chunk_id'].unique()
            selected_chunk = st.selectbox("Select chunk for similarity search:", chunk_ids)

        with col2:
            n_results = st.number_input("Number of results", min_value=1, max_value=20, value=5)

        if st.button("🔍 Find Similar Patterns", use_container_width=True):
            try:
                with st.spinner("Searching for similar patterns..."):
                    response = requests.post(
                        f"{self.api_base}/search/similar",
                        json={"chunk_id": selected_chunk, "n_results": n_results, "include_metadata": True},
                        timeout=10
                    )

                if response.status_code == 200:
                    results = response.json()

                    st.success(f"Found {len(results['similar_chunks'])} similar patterns")

                    if results['similar_chunks']:
                        # Display results in a nice table
                        similar_data = []
                        for similar in results['similar_chunks']:
                            similar_data.append({
                                'Chunk ID': similar['chunk_id'],
                                'Similarity': f"{similar['similarity_score']:.4f}",
                                'Distance': f"{similar['distance']:.4f}",
                                'Condition': similar.get('metadata', {}).get('condition', 'Unknown'),
                                'Lead': similar.get('metadata', {}).get('lead_name', 'Unknown'),
                                'Anomaly Status': similar.get('metadata', {}).get('anomaly_status', 'Unknown')
                            })

                        similar_df = pd.DataFrame(similar_data)
                        st.dataframe(similar_df, use_container_width=True)

                        # Visualization of similarity scores
                        fig = px.bar(
                            similar_df,
                            x='Chunk ID',
                            y='Similarity',
                            color='Condition',
                            title=f"Similarity Scores for {selected_chunk}",
                            hover_data=['Distance', 'Lead', 'Anomaly Status']
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, width='stretch')

                    else:
                        st.info("No similar patterns found above the similarity threshold")

                elif response.status_code == 404:
                    st.error(f"Chunk {selected_chunk} not found in vector database")
                elif response.status_code == 503:
                    st.error("Vector search service not available")
                elif response.status_code == 500:
                    try:
                        error_detail = response.json().get('detail', 'Internal server error')
                        st.error(f"Internal server error: {error_detail}")
                        st.info("💡 Try refreshing the page or selecting a different chunk ID")
                    except:
                        st.error("Internal server error (500). Try refreshing the page or selecting a different chunk ID.")
                else:
                    try:
                        error_detail = response.json().get('detail', f'Status {response.status_code}')
                        st.error(f"Search failed: {error_detail}")
                    except:
                        st.error(f"Search failed with status {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("Search request timed out. The vector database may be busy.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server. Make sure it's running on port 8000.")
            except Exception as e:
                st.error(f"Error performing similarity search: {e}")

    def _aggregate_patient_events(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate chunks by Patient ID, Event ID, and Lead for Patient Events view"""
        if len(chunks_df) == 0:
            return pd.DataFrame()

        # Group by Patient ID, Event ID, and Lead
        grouped = chunks_df.groupby(['patient_id', 'event_id', 'lead_name'], as_index=False).agg({
            'condition': 'first',  # Original condition from data
            'anomaly_status': lambda x: (x == 'anomaly').sum(),  # Count of anomalous chunks
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],  # Collect anomaly types
            'error_score': 'mean',  # Average error score
            'timestamp': 'first',  # Event timestamp
            'processed_time': 'max',  # Last chunk processing time (approximation)
            'chunk_id': 'count'  # Total chunks for this lead
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Process AI Anomaly Verdict - unique anomalies with most severe first

        def process_ai_verdict(anomaly_types_list, total_chunks, anomaly_count):
            """
            Process AI verdict using the new 90% majority rule logic.
            This function is used for grouped events in the dashboard.
            """
            if total_chunks == 0:
                return "Normal"

            # Count frequency of each condition (including "Normal" for non-anomalous chunks)
            condition_counts = {}
            normal_count = total_chunks - anomaly_count

            # Count Normal chunks
            if normal_count > 0:
                condition_counts['Normal'] = normal_count

            # Count anomaly types from anomalous chunks only
            valid_anomaly_types = [t for t in anomaly_types_list if t and t != 'None' and pd.notna(t)]
            for anomaly_type in valid_anomaly_types:
                condition_counts[anomaly_type] = condition_counts.get(anomaly_type, 0) + 1

            # Rule 1: Dominant Condition Rule (90% Majority)
            for condition, count in condition_counts.items():
                percentage = (count / total_chunks) * 100
                if percentage >= 90:
                    return condition

            # Rule 2: Majority Anomaly Rule - look only at abnormal chunks
            anomaly_condition_counts = {k: v for k, v in condition_counts.items() if k != 'Normal'}

            # Rule 4: Edge Case - no abnormal chunks
            if not anomaly_condition_counts:
                return "Normal"

            # Find the most frequent anomaly type(s)
            max_count = max(anomaly_condition_counts.values())
            most_frequent_anomalies = [condition for condition, count in anomaly_condition_counts.items()
                                     if count == max_count]

            # Rule 3: Severity Tie-Breaker
            if len(most_frequent_anomalies) == 1:
                anomaly = most_frequent_anomalies[0]
                percentage = (max_count / total_chunks) * 100
                # Show percentage for low confidence cases
                if percentage < 30:
                    return f"{anomaly} ({percentage:.0f}%)"
                else:
                    return anomaly
            else:
                # Multiple anomalies tied - return most severe
                sorted_anomalies = sort_by_severity(most_frequent_anomalies)
                most_severe = sorted_anomalies[0] if sorted_anomalies else "Normal"
                percentage = (max_count / total_chunks) * 100
                if percentage < 30:
                    return f"{most_severe} ({percentage:.0f}%)"
                else:
                    return most_severe

        grouped['ai_anomaly_verdict'] = grouped.apply(
            lambda row: process_ai_verdict(
                row['anomaly_type'],
                row['total_chunks'],
                row['anomaly_status']
            ),
            axis=1
        )

        # Calculate processing duration (mock - would need actual timestamps)
        grouped['processing_duration_ms'] = grouped['total_chunks'] * 10  # Estimate: 10ms per chunk

        # Format final columns
        result_df = grouped[[
            'patient_id', 'event_id', 'lead_name', 'condition',
            'ai_anomaly_verdict', 'error_score', 'timestamp',
            'processing_duration_ms'
        ]].copy()

        result_df.columns = [
            'Patient ID', 'Event ID', 'Lead', 'Condition',
            'AI Anomaly Verdict', 'Avg Error Score', 'Event Timestamp',
            'Processing Duration (ms)'
        ]

        return result_df

    def render_patient_events(self, chunks_df: pd.DataFrame):
        """Render Patient Events view"""
        st.header("🏥 Patient Events")
        st.markdown("*Event-level view showing AI anomaly detection results per lead*")

        if len(chunks_df) == 0:
            st.warning("No patient event data available")
            return

        # Aggregate data
        events_df = self._aggregate_patient_events(chunks_df)

        if len(events_df) == 0:
            st.warning("No aggregated patient events found")
            return

        # Stats for this view
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_events = len(events_df['Event ID'].unique())
            st.metric("Total Events", f"{total_events:,}")

        with col2:
            # Time to process single event (all configured leads)
            avg_processing_time = events_df.groupby('Event ID')['Processing Duration (ms)'].sum().mean()
            st.metric("Avg Event Processing Time", f"{avg_processing_time:.0f}ms")

        with col3:
            # Throughput calculation
            if len(events_df) > 1:
                time_span = (pd.to_datetime(events_df['Event Timestamp']).max() -
                           pd.to_datetime(events_df['Event Timestamp']).min()).total_seconds()
                if time_span > 0:
                    throughput = (total_events * 60) / time_span  # events/min
                    st.metric("Throughput", f"{throughput:.1f} events/min")
                else:
                    st.metric("Throughput", "N/A")
            else:
                st.metric("Throughput", "N/A")

        with col4:
            # Anomaly detection rate
            anomaly_rate = (events_df['AI Anomaly Verdict'] != 'Normal').mean() * 100
            st.metric("Anomaly Detection Rate", f"{anomaly_rate:.1f}%")

        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            patients = ['All'] + list(events_df['Patient ID'].unique())
            selected_patient = st.selectbox("Patient ID", patients)

        with col2:
            conditions = ['All'] + list(events_df['Condition'].unique())
            selected_condition = st.selectbox("Condition", conditions)

        with col3:
            verdicts = ['All'] + list(events_df['AI Anomaly Verdict'].unique())
            selected_verdict = st.selectbox("AI Verdict", verdicts)

        # Apply filters
        filtered_df = events_df.copy()
        if selected_patient != 'All':
            filtered_df = filtered_df[filtered_df['Patient ID'] == selected_patient]
        if selected_condition != 'All':
            filtered_df = filtered_df[filtered_df['Condition'] == selected_condition]
        if selected_verdict != 'All':
            filtered_df = filtered_df[filtered_df['AI Anomaly Verdict'] == selected_verdict]

        # Pagination
        st.subheader("Patient Events")
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)

        if len(filtered_df) > 0:
            total_pages = (len(filtered_df) - 1) // rows_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1))

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            # Display table
            display_df = filtered_df.iloc[start_idx:end_idx].copy()

            # Format columns for display
            display_df['Avg Error Score'] = display_df['Avg Error Score'].round(4)
            display_df['Event Timestamp'] = pd.to_datetime(display_df['Event Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.info(f"Showing {len(display_df)} of {len(filtered_df)} events (Page {page} of {total_pages})")
        else:
            st.warning("No events match the selected filters")


    def _get_ground_truth_conditions(self, chunks_df: pd.DataFrame) -> Dict[str, str]:
        """Extract ground truth conditions from HDF5 files"""
        ground_truth = {}

        try:
            # Get unique source files
            unique_files = chunks_df['source_file'].unique()

            for source_file in unique_files:
                file_path = source_file  # source_file already includes the data/ prefix

                if os.path.exists(file_path):
                    try:
                        with h5py.File(file_path, 'r') as f:
                            # Get all events in this file
                            events = [k for k in f.keys() if k.startswith('event_')]

                            for event_key in events:
                                event_group = f[event_key]

                                # Extract ground truth condition from event attributes
                                if hasattr(event_group, 'attrs'):
                                    condition = None

                                    # Try different possible attribute names
                                    for attr_name in ['condition', 'Condition', 'reported_condition']:
                                        if attr_name in event_group.attrs:
                                            condition = event_group.attrs[attr_name]
                                            if isinstance(condition, bytes):
                                                condition = condition.decode('utf-8')
                                            break

                                    if condition:
                                        ground_truth[event_key] = condition
                                    else:
                                        # Fallback: look in metadata if available
                                        if 'metadata' in f and hasattr(f['metadata'], 'attrs'):
                                            for attr_name in ['condition', 'default_condition']:
                                                if attr_name in f['metadata'].attrs:
                                                    condition = f['metadata'].attrs[attr_name]
                                                    if isinstance(condition, bytes):
                                                        condition = condition.decode('utf-8')
                                                    ground_truth[event_key] = condition
                                                    break

                                # If still no condition found, mark as unknown
                                if event_key not in ground_truth:
                                    ground_truth[event_key] = 'Unknown'

                    except Exception as e:
                        st.warning(f"Could not read HDF5 file {file_path}: {e}")

        except Exception as e:
            st.warning(f"Error extracting ground truth conditions: {e}")

        return ground_truth

    def _get_lead_configuration(self) -> list:
        """Get ECG lead configuration from API"""
        try:
            import requests

            # Try to get lead configuration from API
            response = requests.get(f"{self.api_url}/api/v1/config/leads", timeout=5)
            if response.status_code == 200:
                config_data = response.json()
                return config_data.get('selected_leads', [])
            else:
                # Fallback to default leads if API call fails
                return self.selected_leads
        except Exception as e:
            # Fallback to default leads if API is not available
            return self.selected_leads

    def _calculate_data_quality_score(self, signal_data: np.ndarray, sampling_rate: int = 500) -> float:
        """
        Calculate data quality score for ECG signal based on multiple factors
        Returns score from 0-100 (higher is better)
        """
        import numpy as np

        if len(signal_data) == 0:
            return 0.0

        try:
            # Convert to numpy array if not already
            signal = np.array(signal_data)

            # 1. Signal completeness (no NaN/inf values)
            completeness_score = (np.sum(np.isfinite(signal)) / len(signal)) * 100

            # 2. Signal range and saturation check
            signal_range = np.ptp(signal[np.isfinite(signal)]) if np.any(np.isfinite(signal)) else 0
            # Typical ECG range is -5mV to +5mV, but varies by lead
            range_score = min(100, (signal_range / 10.0) * 100) if signal_range > 0 else 0

            # 3. Noise estimation using high-frequency components
            if len(signal) > 1:
                # Simple noise estimation using differences
                diff_signal = np.diff(signal[np.isfinite(signal)])
                noise_level = np.std(diff_signal) if len(diff_signal) > 0 else 0
                # Lower noise = higher score (inverse relationship)
                noise_score = max(0, 100 - (noise_level * 20))
            else:
                noise_score = 0

            # 4. Artifact detection (sudden spikes)
            if len(signal) > 10:
                # Detect outliers using z-score
                z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-10))
                artifact_ratio = np.sum(z_scores > 4) / len(signal)  # Samples with z-score > 4
                artifact_score = max(0, 100 - (artifact_ratio * 200))
            else:
                artifact_score = 100

            # 5. Signal stability (check for lead-off or disconnection)
            if len(signal) > sampling_rate:  # At least 1 second of data
                # Check for flat-line segments
                segment_size = sampling_rate // 10  # 0.1 second segments
                flat_segments = 0
                for i in range(0, len(signal) - segment_size, segment_size):
                    segment = signal[i:i + segment_size]
                    if np.std(segment) < 0.01:  # Very low variance = flat line
                        flat_segments += 1

                total_segments = (len(signal) // segment_size)
                flatline_ratio = flat_segments / total_segments if total_segments > 0 else 0
                stability_score = max(0, 100 - (flatline_ratio * 100))
            else:
                stability_score = 50  # Neutral score for short signals

            # Weighted average of all quality metrics
            weights = {
                'completeness': 0.3,
                'range': 0.2,
                'noise': 0.2,
                'artifact': 0.15,
                'stability': 0.15
            }

            overall_score = (
                completeness_score * weights['completeness'] +
                range_score * weights['range'] +
                noise_score * weights['noise'] +
                artifact_score * weights['artifact'] +
                stability_score * weights['stability']
            )

            return max(0, min(100, overall_score))

        except Exception as e:
            # Return neutral score if calculation fails
            return 50.0

    def _extract_event_report_data(self, patient_id: str, event_id: str, source_file: str) -> Dict:
        """Extract comprehensive event data from HDF5 file for detailed reporting"""
        report_data = {
            'device_info': {},
            'patient_info': {},
            'event_info': {},
            'ecg_data': {},
            'vitals': {},
            'waveforms': {},
            'analysis_info': {}
        }

        try:
            if not os.path.exists(source_file):
                return report_data

            with h5py.File(source_file, 'r') as f:
                # Extract metadata from the actual structure
                if 'metadata' in f:
                    metadata = f['metadata']

                    # Extract all metadata fields dynamically
                    for key in metadata.keys():
                        try:
                            item = metadata[key]
                            if isinstance(item, h5py.Dataset):
                                value = item[()]
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')

                                # Categorize the metadata
                                if key == 'device_info':
                                    report_data['device_info']['device_info'] = value
                                elif key == 'data_quality_score':
                                    report_data['data_quality_score'] = value
                                elif key == 'patient_id':
                                    report_data['patient_info']['patient_id'] = value
                                else:
                                    # Store other metadata in device_info or event_info
                                    if 'sampling_rate' in key or 'alarm' in key or 'seconds' in key:
                                        report_data['event_info'][key] = value
                                    else:
                                        report_data['device_info'][key] = value

                            elif isinstance(item, h5py.Group):
                                # Handle groups if any exist
                                group_data = {}
                                for subkey in item.keys():
                                    try:
                                        subvalue = item[subkey][()]
                                        if isinstance(subvalue, bytes):
                                            subvalue = subvalue.decode('utf-8')
                                        group_data[subkey] = subvalue
                                    except:
                                        pass

                                if group_data:
                                    if key == 'device_info':
                                        report_data['device_info'].update(group_data)
                                    else:
                                        report_data['device_info'][key] = group_data

                        except Exception as e:
                            print(f"Error extracting metadata {key}: {e}")

                    # Ensure patient_id is set
                    if not report_data['patient_info'].get('patient_id'):
                        report_data['patient_info']['patient_id'] = patient_id

                # Extract event-specific data
                if event_id in f:
                    event_group = f[event_id]

                    # Event information from attributes (Event Condition)
                    if hasattr(event_group, 'attrs'):
                        for attr_name in event_group.attrs.keys():
                            val = event_group.attrs[attr_name]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8')
                            report_data['event_info'][attr_name] = val

                    # Extract UUID and timestamp from datasets if available
                    if 'uuid' in event_group:
                        try:
                            uuid_val = event_group['uuid'][()]
                            if isinstance(uuid_val, bytes):
                                uuid_val = uuid_val.decode('utf-8')
                            report_data['event_info']['uuid'] = uuid_val
                        except Exception as e:
                            print(f"Error extracting UUID: {e}")

                    # Extract event timestamp from /event_xxx/timestamp dataset
                    if 'timestamp' in event_group:
                        try:
                            timestamp_val = event_group['timestamp'][()]
                            # Store as event_timestamp to prioritize it over any other timestamp
                            report_data['event_info']['event_timestamp'] = timestamp_val
                            print(f"Extracted event timestamp: {timestamp_val}")
                        except Exception as e:
                            print(f"Error extracting event timestamp: {e}")

                    # ECG Lead data with quality analysis (from /ecg/ group)
                    ecg_leads = {}
                    total_data_quality_score = 0
                    valid_leads = 0

                    if 'ecg' in event_group:
                        ecg_group = event_group['ecg']

                        for lead_name in ecg_group.keys():
                            if not lead_name.startswith('pacer'):  # Skip pacer info datasets
                                try:
                                    lead_dataset = ecg_group[lead_name]
                                    lead_data = lead_dataset[:]
                                    sampling_rate = lead_dataset.attrs.get('sampling_rate', 200) if hasattr(lead_dataset, 'attrs') else 200

                                    # Calculate data quality score for this lead
                                    quality_score = self._calculate_data_quality_score(lead_data, sampling_rate)

                                    ecg_leads[lead_name] = {
                                        'data': lead_data,
                                        'length': len(lead_data),
                                        'sampling_rate': sampling_rate,
                                        'quality_score': quality_score
                                    }

                                    total_data_quality_score += quality_score
                                    valid_leads += 1
                                except Exception as e:
                                    print(f"Error processing ECG lead {lead_name}: {e}")

                        # Extract pacer information if available
                        if 'pacer_info' in ecg_group:
                            try:
                                report_data['ecg_data']['pacer_info'] = ecg_group['pacer_info'][()]
                            except:
                                pass
                        if 'pacer_offset' in ecg_group:
                            try:
                                report_data['ecg_data']['pacer_offset'] = ecg_group['pacer_offset'][()]
                            except:
                                pass

                    report_data['ecg_data']['leads'] = ecg_leads

                    # Overall data quality score
                    if valid_leads > 0:
                        report_data['ecg_data']['overall_quality_score'] = total_data_quality_score / valid_leads
                    else:
                        report_data['ecg_data']['overall_quality_score'] = 0

                    # Vitals data (from /vitals/ group)
                    vitals_data = {}
                    if 'vitals' in event_group:
                        vitals_group = event_group['vitals']

                        for vital_name in vitals_group.keys():
                            try:
                                vital_group = vitals_group[vital_name]
                                vital_info = {}

                                # Extract all available data for this vital
                                for dataset_name in vital_group.keys():
                                    try:
                                        value = vital_group[dataset_name][()]
                                        if isinstance(value, bytes):
                                            value = value.decode('utf-8')
                                        vital_info[dataset_name] = value
                                    except:
                                        pass

                                if vital_info:  # Only add if we got some data
                                    vitals_data[vital_name] = vital_info

                            except Exception as e:
                                print(f"Error processing vital {vital_name}: {e}")

                    report_data['vitals'] = vitals_data

                    # Waveform data (PPG, Respiratory)
                    waveforms = {}

                    # PPG data from /ppg/ group
                    if 'ppg' in event_group:
                        ppg_group = event_group['ppg']
                        for key in ppg_group.keys():
                            try:
                                waveform_data = ppg_group[key][:]
                                sampling_rate = ppg_group[key].attrs.get('sampling_rate', 300) if hasattr(ppg_group[key], 'attrs') else 300
                                waveforms[f"PPG_{key}"] = {
                                    'data': waveform_data,
                                    'sampling_rate': sampling_rate,
                                    'length': len(waveform_data)
                                }
                            except Exception as e:
                                print(f"Error processing PPG data {key}: {e}")

                    # Respiratory data from /resp/ group
                    if 'resp' in event_group:
                        resp_group = event_group['resp']
                        for key in resp_group.keys():
                            try:
                                waveform_data = resp_group[key][:]
                                sampling_rate = resp_group[key].attrs.get('sampling_rate', 133) if hasattr(resp_group[key], 'attrs') else 133
                                waveforms[f"RESP_{key}"] = {
                                    'data': waveform_data,
                                    'sampling_rate': sampling_rate,
                                    'length': len(waveform_data)
                                }
                            except Exception as e:
                                print(f"Error processing Respiratory data {key}: {e}")

                    report_data['waveforms'] = waveforms

                # Ensure event UUID is available
                if 'uuid' not in report_data['event_info']:
                    report_data['event_info']['uuid'] = f"{patient_id}_{event_id}"

        except Exception as e:
            st.warning(f"Error extracting report data: {e}")

        return report_data

    def _render_event_report(self, patient_id: str, event_id: str, source_file: str, ai_data: Dict):
        """Render comprehensive event report"""
        st.header(f"📋 Event Report: {event_id}")

        # Extract comprehensive data from HDF5
        report_data = self._extract_event_report_data(patient_id, event_id, source_file)

        # Information Section
        st.subheader("📊 Information Section")

        # Create structured DataFrames for Information Section
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            # Device Information DataFrame
            st.markdown("**Device Information:**")
            device_info = report_data.get('device_info', {})
            if device_info:
                device_rows = []
                for key, value in device_info.items():
                    device_rows.append({
                        'Parameter': key.replace('_', ' ').title(),
                        'Value': str(value)
                    })
                device_df = pd.DataFrame(device_rows)
                st.dataframe(device_df, use_container_width=True, hide_index=True)
            else:
                st.info("No device information available")

            # Patient and Quality Information DataFrame
            st.markdown("**Patient & Quality Information:**")
            patient_quality_rows = []

            # Patient ID
            patient_quality_rows.append({
                'Parameter': 'Patient ID',
                'Value': patient_id
            })

            # Data Quality Score
            metadata_quality = report_data.get('data_quality_score', None)
            ecg_data = report_data.get('ecg_data', {})
            calculated_quality = ecg_data.get('overall_quality_score', 0)

            if metadata_quality is not None:
                overall_quality = metadata_quality
                quality_color = "🟢" if overall_quality >= 80 else "🟡" if overall_quality >= 60 else "🔴"
                quality_source = "HDF5 Metadata"
            else:
                overall_quality = calculated_quality
                quality_color = "🟢" if overall_quality >= 80 else "🟡" if overall_quality >= 60 else "🔴"
                quality_source = "Calculated"

            patient_quality_rows.append({
                'Parameter': 'Data Quality Score',
                'Value': f"{quality_color} {overall_quality:.1f}/100 ({quality_source})"
            })

            patient_quality_df = pd.DataFrame(patient_quality_rows)
            st.dataframe(patient_quality_df, use_container_width=True, hide_index=True)

        with info_col2:
            # Event Information DataFrame
            st.markdown("**Event Information:**")
            event_info = report_data.get('event_info', {})
            event_rows = []

            # Event name/condition
            condition = event_info.get('condition', ai_data.get('event_condition', 'Unknown'))
            event_rows.append({
                'Parameter': 'Event Condition',
                'Value': condition
            })

            # Event UUID
            event_uuid = event_info.get('uuid', f"{patient_id}_{event_id}")
            event_rows.append({
                'Parameter': 'Event UUID',
                'Value': event_uuid
            })

            # Event timestamp
            if 'event_timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['event_timestamp'], unit='s')
                event_rows.append({
                    'Parameter': 'Event Time',
                    'Value': event_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            elif 'timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['timestamp'], unit='s')
                event_rows.append({
                    'Parameter': 'Event Time',
                    'Value': event_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })

            # Heart rate
            if 'heart_rate' in event_info:
                event_rows.append({
                    'Parameter': 'Heart Rate',
                    'Value': f"{event_info['heart_rate']} BPM"
                })

            # Additional event details
            for key, value in event_info.items():
                if key not in ['condition', 'uuid', 'event_timestamp', 'timestamp', 'heart_rate']:
                    key_display = key.replace('_', ' ').title()

                    # Format specific fields nicely
                    if 'sampling_rate' in key:
                        formatted_value = f"{value} Hz"
                    elif 'alarm_time_epoch' in key:
                        try:
                            alarm_time = pd.to_datetime(value, unit='s')
                            formatted_value = alarm_time.strftime('%Y-%m-%d %H:%M:%S')
                            key_display = "Alarm Time"
                        except:
                            formatted_value = str(value)
                    elif 'seconds' in key:
                        formatted_value = f"{value}s"
                    else:
                        formatted_value = str(value)

                    event_rows.append({
                        'Parameter': key_display,
                        'Value': formatted_value
                    })

            # Calculate and show monitoring start time
            if 'alarm_time_epoch' in event_info and 'alarm_offset_seconds' in event_info:
                try:
                    alarm_time = float(event_info['alarm_time_epoch'])
                    offset = float(event_info['alarm_offset_seconds'])
                    monitoring_start = pd.to_datetime(alarm_time - offset, unit='s')
                    event_rows.append({
                        'Parameter': 'Monitoring Start Time',
                        'Value': monitoring_start.strftime('%Y-%m-%d %H:%M:%S')
                    })
                except:
                    pass

            if event_rows:
                event_df = pd.DataFrame(event_rows)
                st.dataframe(event_df, use_container_width=True, hide_index=True)
            else:
                st.info("No event information available")

        # Diagnosis Section
        st.subheader("🔬 Diagnosis Section")

        # ECG Lead Configuration Summary DataFrame
        st.markdown("**ECG Lead Configuration:**")
        ecg_data = report_data.get('ecg_data', {})
        leads = ecg_data.get('leads', {})

        if leads:
            # Get configured leads from API
            configured_leads = self._get_lead_configuration()

            # Create ECG Configuration Summary
            config_rows = []
            config_rows.append({
                'Configuration': 'ECG Leads Used for AI Analysis',
                'Value': ', '.join(configured_leads) if configured_leads else 'Not available'
            })
            config_rows.append({
                'Configuration': 'Available ECG Leads in Data',
                'Value': ', '.join(leads.keys())
            })

            if configured_leads:
                available_configured = [lead for lead in configured_leads if lead in leads.keys()]
                config_rows.append({
                    'Configuration': 'Configured Leads with Data',
                    'Value': ', '.join(available_configured) if available_configured else 'None'
                })

            config_df = pd.DataFrame(config_rows)
            st.dataframe(config_df, use_container_width=True, hide_index=True)

            # Get common data specifications from first lead
            first_lead_info = next(iter(leads.values()))
            common_data_length = first_lead_info.get('length', 0)
            common_sampling_rate = first_lead_info.get('sampling_rate', 200)

            # Display ECG data specifications once
            st.markdown(f"**ECG Data Specifications:** {common_data_length:,} samples per lead at {common_sampling_rate} Hz sampling rate")

            # ECG Lead Analysis DataFrame (without repetitive data length/sampling rate)
            st.markdown("**ECG Lead Quality & Anomaly Analysis:**")
            lead_analysis_rows = []

            for lead_name, lead_info in leads.items():
                # Mark if this lead is configured for AI analysis
                is_configured = lead_name in configured_leads if configured_leads else False
                config_status = "✅ AI Configured" if is_configured else "⭕ Not Configured"

                # Only show quality score and anomalous chunks for configured leads
                if is_configured:
                    quality_score = lead_info.get('quality_score', 0)
                    quality_icon = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
                    quality_text = f"{quality_icon} {quality_score:.1f}/100"

                    # Get actual anomaly percentage from database for configured leads
                    try:
                        conn = sqlite3.connect(self.db_path)
                        query = """
                            SELECT
                                COUNT(CASE WHEN anomaly_status = 'anomaly' THEN 1 END) * 100.0 / COUNT(*) as anomaly_pct
                            FROM chunks
                            WHERE event_id = ? AND source_file LIKE ? AND lead_name = ?
                        """
                        result = pd.read_sql_query(query, conn, params=[event_id, f"%{patient_id}%", lead_name])
                        conn.close()
                        anomaly_percentage = result['anomaly_pct'].iloc[0] if not result.empty else 0
                        anomaly_text = f"{anomaly_percentage:.1f}%"
                    except Exception:
                        anomaly_text = "N/A"
                else:
                    # For non-configured leads, show N/A for AI-specific metrics
                    quality_text = "N/A"
                    anomaly_text = "N/A"

                lead_analysis_rows.append({
                    'ECG Lead': lead_name,
                    'AI Configuration': config_status,
                    'Quality Score': quality_text,
                    'Anomalous Chunks': anomaly_text
                })

            if lead_analysis_rows:
                lead_analysis_df = pd.DataFrame(lead_analysis_rows)
                st.dataframe(lead_analysis_df, use_container_width=True, hide_index=True)

            # ECG Waveform plots placeholder
            st.info("📈 ECG Waveform plots would be displayed here (7-lead ECG with anomaly chunk highlighting)")

            # ECG Chunk Analysis Table - Only Anomalous Chunks
            st.markdown("**ECG Chunk Analysis (Anomalous Chunks Only):**")
            try:
                # Get chunk-level data for this specific event from database - only anomalous chunks
                conn = sqlite3.connect(self.db_path)
                query = """
                    SELECT lead_name, event_id, chunk_id, anomaly_status, anomaly_type, error_score
                    FROM chunks
                    WHERE event_id = ? AND source_file LIKE ? AND anomaly_status = 'anomaly'
                    ORDER BY lead_name, chunk_id
                """
                chunk_data = pd.read_sql_query(query, conn, params=[event_id, f"%{patient_id}%"])
                conn.close()

                if not chunk_data.empty:
                    chunk_analysis_rows = []

                    # Constants for offset calculation - corrected for 12-second ECG strips
                    sampling_rate = 200        # Hz - actual ECG sampling rate
                    total_strip_duration = 12  # seconds
                    total_samples = sampling_rate * total_strip_duration  # 2400 samples

                    # For a 12-second strip with chunk numbers like 0, 1120, 2100:
                    # If chunk 2100 should be near the end (around 10-11 seconds):
                    # Then each chunk number unit ≈ 12 seconds / 2400 = 0.005 seconds
                    # Or more likely: chunk_number represents sample number directly

                    # Try direct sample-to-time conversion
                    # chunk_number might be the sample number within the 12-second strip

                    for _, chunk_row in chunk_data.iterrows():
                        # Extract chunk number from chunk_id
                        chunk_id_raw = chunk_row['chunk_id']
                        offset_time = "0.0s"  # Default fallback

                        try:
                            # Handle different data types
                            if chunk_id_raw is None or pd.isna(chunk_id_raw):
                                chunk_id = 0
                            elif isinstance(chunk_id_raw, str):
                                cleaned = chunk_id_raw.strip()
                                if cleaned.isdigit():
                                    chunk_id = int(cleaned)
                                elif cleaned.startswith('chunk_'):
                                    # Handle chunk_eventid_chunknum format
                                    parts = cleaned.split('_')
                                    if len(parts) >= 3:  # chunk_eventid_chunknum
                                        chunk_num_str = parts[-1]  # Get last part (chunk number)
                                        if chunk_num_str.isdigit():
                                            chunk_id = int(chunk_num_str)
                                        else:
                                            raise ValueError(f"Last part '{chunk_num_str}' is not a digit")
                                    else:
                                        raise ValueError(f"Unexpected chunk format: '{cleaned}'")
                                else:
                                    # Try to extract all digits and use the last one
                                    import re
                                    digits = re.findall(r'\d+', cleaned)
                                    if digits:
                                        chunk_id = int(digits[-1])  # Use last number instead of first
                                    else:
                                        raise ValueError(f"No digits found in '{cleaned}'")
                            elif isinstance(chunk_id_raw, (int, float)):
                                chunk_id = int(chunk_id_raw)
                            else:
                                raise ValueError(f"Unexpected type: {type(chunk_id_raw)}")

                            # Calculate offset - assume chunk_id is the sample number within the strip
                            offset_seconds = chunk_id / sampling_rate

                            # Clamp to 12-second strip duration for sanity check
                            if offset_seconds > total_strip_duration:
                                offset_seconds = offset_seconds % total_strip_duration

                            # Format offset time
                            offset_time = f"{offset_seconds:.1f}s"

                        except Exception as e:
                            # Fallback - try to extract chunk number
                            try:
                                import re
                                chunk_str = str(chunk_id_raw)
                                if chunk_str.startswith('chunk_'):
                                    # Split by underscore and get the last part
                                    parts = chunk_str.split('_')
                                    if len(parts) >= 3:
                                        chunk_id = int(parts[-1])  # Get last part (chunk number)
                                    else:
                                        chunk_id = 0
                                else:
                                    # Extract last number
                                    numbers = re.findall(r'\d+', chunk_str)
                                    chunk_id = int(numbers[-1]) if numbers else 0

                                offset_seconds = chunk_id / sampling_rate
                                # Clamp to 12-second strip
                                if offset_seconds > total_strip_duration:
                                    offset_seconds = offset_seconds % total_strip_duration
                                offset_time = f"{offset_seconds:.1f}s"
                            except:
                                offset_time = f"Chunk {chunk_id_raw}"

                        # Format anomaly type
                        anomaly_type = chunk_row['anomaly_type'] if chunk_row['anomaly_type'] and chunk_row['anomaly_type'] != 'None' else 'Unknown Anomaly'

                        # Format error score safely
                        try:
                            error_score = f"{float(chunk_row['error_score']):.4f}"
                        except (ValueError, TypeError):
                            error_score = str(chunk_row['error_score'])

                        chunk_analysis_rows.append({
                            'ECG Lead': chunk_row['lead_name'],
                            'Chunk ID': chunk_row['chunk_id'],
                            'Offset of Anomaly in Strip': offset_time,
                            'Anomaly Type': anomaly_type,
                            'Error Score': error_score
                        })

                    if chunk_analysis_rows:
                        chunk_analysis_df = pd.DataFrame(chunk_analysis_rows)
                        st.dataframe(chunk_analysis_df, use_container_width=True, hide_index=True)
                        st.caption(f"📊 Showing {len(chunk_analysis_rows)} anomalous chunks (Sampling rate: {sampling_rate} Hz, Strip duration: {total_strip_duration}s)")
                    else:
                        st.info("No anomalous chunks detected for this event")
                else:
                    st.info("No anomalous chunks found for this event")
            except Exception as e:
                st.warning(f"Could not load chunk analysis data: {e}")

        else:
            st.warning("No ECG lead data available in HDF5 file")

        # Additional Waveforms DataFrame
        waveforms = report_data.get('waveforms', {})
        if waveforms:
            st.markdown("**Additional Waveforms Summary:**")
            waveform_rows = []

            for waveform_name, waveform_info in waveforms.items():
                # Calculate duration in seconds
                duration = waveform_info['length'] / waveform_info['sampling_rate']

                waveform_rows.append({
                    'Waveform': waveform_name.replace('_', ' '),
                    'Samples': f"{waveform_info['length']:,}",
                    'Sampling Rate': f"{waveform_info['sampling_rate']} Hz",
                    'Duration': f"{duration:.2f}s"
                })

            if waveform_rows:
                waveform_df = pd.DataFrame(waveform_rows)
                st.dataframe(waveform_df, use_container_width=True, hide_index=True)

            st.info("📊 PPG and Respiratory waveform plots would be displayed here (requires plotting implementation)")
        else:
            st.info("No additional waveform data available")

        # Vital Signs DataFrame
        vitals = report_data.get('vitals', {})
        if vitals:
            st.markdown("**Basic Vital Signs Data:**")
            vitals_rows = []

            # Get event timestamp for relative time calculation
            event_info = report_data.get('event_info', {})
            event_timestamp = None
            if 'event_timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['event_timestamp'], unit='s')
            elif 'timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['timestamp'], unit='s')

            # Define the order of vital signs as requested
            vital_order = ['HR', 'RespRate', 'SpO2', 'Pulse', 'Temp', 'Systolic', 'Diastolic', 'XL_Posture']

            # Process vitals in the specified order
            for vital_name in vital_order:
                if vital_name in vitals:
                    vital_info = vitals[vital_name]

                    # Format vital name for display
                    display_name = vital_name.replace('_', ' ').title()

                    # Handle special case for RespRate display name
                    if vital_name == 'RespRate':
                        display_name = 'RR (Respiratory Rate)'
                    elif vital_name == 'HR':
                        display_name = 'HR (Heart Rate)'
                    elif vital_name == 'SpO2':
                        display_name = 'SpO2 (Oxygen Saturation)'
                    elif vital_name == 'XL_Posture':
                        display_name = 'XL Posture'

                    # Extract common fields
                    value = vital_info.get('value', 'N/A')
                    units = vital_info.get('units', '')
                    timestamp = vital_info.get('timestamp', 'N/A')
                    lower_threshold = vital_info.get('lower_threshold', 'N/A')
                    upper_threshold = vital_info.get('upper_threshold', 'N/A')

                    # Convert timestamp and calculate relative time
                    formatted_timestamp = 'N/A'
                    relative_time = 'N/A'

                    if isinstance(timestamp, (int, float)):
                        try:
                            vital_timestamp = pd.to_datetime(timestamp, unit='s')
                            formatted_timestamp = vital_timestamp.strftime('%Y-%m-%d %H:%M:%S')

                            # Calculate relative time difference w.r.t event time
                            if event_timestamp is not None:
                                time_diff = vital_timestamp - event_timestamp
                                total_seconds = int(time_diff.total_seconds())

                                if total_seconds == 0:
                                    relative_time = "At Event"
                                elif total_seconds > 0:
                                    # After event
                                    if total_seconds < 60:
                                        relative_time = f"+{total_seconds}s"
                                    elif total_seconds < 3600:
                                        minutes = total_seconds // 60
                                        seconds = total_seconds % 60
                                        relative_time = f"+{minutes}m {seconds}s" if seconds > 0 else f"+{minutes}m"
                                    else:
                                        hours = total_seconds // 3600
                                        minutes = (total_seconds % 3600) // 60
                                        relative_time = f"+{hours}h {minutes}m" if minutes > 0 else f"+{hours}h"
                                else:
                                    # Before event
                                    abs_seconds = abs(total_seconds)
                                    if abs_seconds < 60:
                                        relative_time = f"-{abs_seconds}s"
                                    elif abs_seconds < 3600:
                                        minutes = abs_seconds // 60
                                        seconds = abs_seconds % 60
                                        relative_time = f"-{minutes}m {seconds}s" if seconds > 0 else f"-{minutes}m"
                                    else:
                                        hours = abs_seconds // 3600
                                        minutes = (abs_seconds % 3600) // 60
                                        relative_time = f"-{hours}h {minutes}m" if minutes > 0 else f"-{hours}h"
                        except:
                            formatted_timestamp = str(timestamp)

                    # Determine if value is within normal range
                    status = "Normal"
                    if (isinstance(value, (int, float)) and
                        isinstance(lower_threshold, (int, float)) and
                        isinstance(upper_threshold, (int, float))):
                        if value < lower_threshold:
                            status = "⬇️ Low"
                        elif value > upper_threshold:
                            status = "⬆️ High"
                        else:
                            status = "✅ Normal"

                    vitals_rows.append({
                        'Vital Sign': display_name,
                        'Value': f"{value} {units}".strip(),
                        'Normal Range': f"{lower_threshold}-{upper_threshold} {units}".strip() if lower_threshold != 'N/A' else 'N/A',
                        'Status': status,
                        'Relative Time': relative_time,
                        'Timestamp': formatted_timestamp
                    })

            if vitals_rows:
                vitals_df = pd.DataFrame(vitals_rows)
                st.dataframe(vitals_df, use_container_width=True, hide_index=True)
            else:
                st.info("No vital signs data could be parsed")
        else:
            st.info("No vital signs data available")

        # Analysis Information Section
        st.subheader("⚡ Analysis Information")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            # AI Analysis Results DataFrame
            st.markdown("**AI Analysis Results:**")
            ai_analysis_rows = []

            ai_analysis_rows.append({
                'Analysis Parameter': 'AI Verdict',
                'Result': ai_data.get('ai_verdict', 'Unknown')
            })

            ai_analysis_rows.append({
                'Analysis Parameter': 'Event Condition',
                'Result': ai_data.get('event_condition', 'Unknown')
            })

            error_score = ai_data.get('error_score', 'N/A')
            if error_score != 'N/A':
                ai_analysis_rows.append({
                    'Analysis Parameter': 'Average Error Score',
                    'Result': f"{error_score:.4f}" if isinstance(error_score, (int, float)) else str(error_score)
                })
            else:
                ai_analysis_rows.append({
                    'Analysis Parameter': 'Average Error Score',
                    'Result': 'N/A'
                })

            ai_analysis_df = pd.DataFrame(ai_analysis_rows)
            st.dataframe(ai_analysis_df, use_container_width=True, hide_index=True)

        with analysis_col2:
            # Processing Information DataFrame
            st.markdown("**Processing Information:**")
            processing_rows = []

            # Analysis timestamp
            analysis_timestamp = ai_data.get('timestamp', 'Unknown')
            if analysis_timestamp != 'Unknown':
                try:
                    if isinstance(analysis_timestamp, str):
                        formatted_timestamp = pd.to_datetime(analysis_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        formatted_timestamp = str(analysis_timestamp)
                except:
                    formatted_timestamp = str(analysis_timestamp)
            else:
                formatted_timestamp = 'Unknown'

            processing_rows.append({
                'Processing Parameter': 'Analysis Timestamp',
                'Value': formatted_timestamp
            })

            # Processing time placeholder
            processing_rows.append({
                'Processing Parameter': 'Time to Analyze',
                'Value': '[Would be calculated from processing logs]'
            })

            # Extract model version from device info
            device_info = report_data.get('device_info', {})
            model_version = device_info.get('model_version', device_info.get('software_version', device_info.get('ai_model_version', 'Unknown')))
            processing_rows.append({
                'Processing Parameter': 'Model Version',
                'Value': str(model_version)
            })

            processing_df = pd.DataFrame(processing_rows)
            st.dataframe(processing_df, use_container_width=True, hide_index=True)

        # === VITALS SIGNS ANALYSIS SECTION ===
        st.markdown("---")
        self.render_vitals_analysis_for_report(patient_id, event_id)

        # Export and Navigation Section (moved to end)
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            # PDF Export button
            pdf_data = self._generate_pdf_report(patient_id, event_id, report_data, ai_data)
            if pdf_data:
                st.download_button(
                    label="📄 Export as PDF",
                    data=pdf_data,
                    file_name=f"ECG_Report_{patient_id}_{event_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key=f"pdf_export_{event_id}"
                )

        with col2:
            # Close button
            if st.button("🔙 Back to Patient Analysis", key=f"back_from_report_{event_id}"):
                st.session_state.selected_event_for_report = None
                st.rerun()

        with col3:
            # Report metadata
            st.caption(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption("🔒 Confidential Medical Information")

    def _generate_pdf_report(self, patient_id: str, event_id: str, report_data: Dict, ai_data: Dict) -> bytes:
        """Generate comprehensive PDF report including all Patient Analysis content"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from io import BytesIO
            import datetime

            # Create PDF in memory
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue,
                spaceBefore=12,
                spaceAfter=6
            )

            subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.darkgreen,
                spaceBefore=8,
                spaceAfter=4
            )

            # Story elements
            story = []

            # Header
            story.append(Paragraph("RMS.AI Event Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # === PATIENT ANALYSIS SECTION ===
            story.append(Paragraph("Patient Analysis Summary", heading_style))

            # Get all patient data from database for comprehensive analysis
            try:
                conn = sqlite3.connect(self.db_path)
                query = "SELECT * FROM chunks WHERE source_file LIKE ?"
                patient_data = pd.read_sql_query(query, conn, params=[f"%{patient_id}%"])
                conn.close()

                if not patient_data.empty:
                    # Extract patient ID and get ground truth
                    patient_data['patient_id'] = patient_data['source_file'].str.extract(r'([A-Z0-9]+)_\d{4}-\d{2}\.h5')[0]
                    patient_data = patient_data[patient_data['patient_id'] == patient_id]

                    # Extract heart rate from metadata JSON (same logic as dashboard)
                    def extract_heart_rate(metadata_json):
                        if pd.isna(metadata_json) or not metadata_json:
                            return None
                        try:
                            metadata = json.loads(metadata_json)
                            return metadata.get('heart_rate', None)
                        except (json.JSONDecodeError, TypeError):
                            return None

                    patient_data['heart_rate'] = patient_data['metadata'].apply(extract_heart_rate)
                    ground_truth_conditions = self._get_ground_truth_conditions(patient_data)

                    # Calculate patient summary stats - handle different timestamp column names
                    agg_dict = {
                        'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
                        'anomaly_status': lambda x: (x == 'anomaly').sum(),
                        'error_score': 'mean',
                        'chunk_id': 'count',
                        'heart_rate': 'first'
                    }

                    # Add timestamp field if it exists
                    if 'timestamp' in patient_data.columns:
                        agg_dict['timestamp'] = 'first'
                    elif 'processing_timestamp' in patient_data.columns:
                        agg_dict['processing_timestamp'] = 'first'

                    patient_events = patient_data.groupby('event_id', as_index=False).agg(agg_dict).rename(columns={'chunk_id': 'total_chunks'})

                    # AI verdict calculation with severity order

                    # Use unified AI verdict function with HR-based logic
                    patient_events['ai_verdict'] = patient_events.apply(
                        lambda row: self.get_unified_ai_verdict(
                            row['anomaly_type'],
                            row['heart_rate'],
                            row['anomaly_status']
                        ), axis=1
                    )
                    patient_events['event_condition'] = patient_events['event_id'].map(ground_truth_conditions).fillna('Unknown')

                    # Patient Summary Statistics
                    total_events = len(patient_events)
                    anomaly_events = (patient_events['ai_verdict'] != 'Normal').sum()
                    avg_error = patient_events['error_score'].mean()

                    # Calculate accuracy using proper comparison logic
                    matches = 0
                    valid_events = 0
                    for _, row in patient_events.iterrows():
                        ai_verdict = row['ai_verdict']
                        event_condition = row['event_condition']

                        # Skip unknown ground truth from accuracy calculation
                        if event_condition == 'Unknown':
                            continue

                        valid_events += 1

                        # Use same logic as Event Summary comparison - only exact matches count
                        if ai_verdict == event_condition:
                            matches += 1

                    accuracy = (matches / valid_events) * 100 if valid_events > 0 else 0

                    # Patient Summary Table
                    summary_data = [
                        ['Metric', 'Value'],
                        ['Patient ID', patient_id],
                        ['Total Events', f"{total_events:,}"],
                        ['AI Detected Anomalies', f"{anomaly_events:,}"],
                        ['AI Accuracy vs Event Condition', f"{accuracy:.1f}%"],
                        ['Average Error Score', f"{avg_error:.4f}"],
                        ['Report Generated', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    ]

                    summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
                    summary_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))

                    story.append(summary_table)
                    story.append(Spacer(1, 12))

                    # Skip the Event Analysis table as requested
                    story.append(PageBreak())

            except Exception as e:
                story.append(Paragraph(f"Note: Patient analysis data not available: {e}", styles['Normal']))
                story.append(Spacer(1, 12))

            # === VITALS ANALYSIS SECTION ===
            try:
                story.append(PageBreak())
                story.append(Paragraph("Vital Signs Analysis", heading_style))
                story.append(Spacer(1, 12))

                # Try to perform vitals analysis
                from rmsai_vitals_analysis import RMSAIVitalsAnalyzer

                # Find HDF5 file containing this event
                data_dir = "data"
                if os.path.exists(data_dir):
                    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
                    target_hdf5 = None

                    for hdf5_file in hdf5_files:
                        hdf5_path = os.path.join(data_dir, hdf5_file)
                        try:
                            with h5py.File(hdf5_path, 'r') as f:
                                if event_id in f:
                                    target_hdf5 = hdf5_path
                                    break
                        except:
                            continue

                    if target_hdf5:
                        # Initialize analyzer and extract data
                        analyzer = RMSAIVitalsAnalyzer()
                        vitals_data = analyzer.extract_vitals_history_from_hdf5(target_hdf5, event_id)
                        analysis_results = analyzer.analyze_vital_trends(vitals_data)

                        # Current EWS Status
                        current_ews = analysis_results['current_ews']
                        story.append(Paragraph("Early Warning System (EWS) Assessment", subheading_style))

                        ews_data = [
                            ['Parameter', 'Value'],
                            ['EWS Score', str(current_ews['total_score'])],
                            ['Risk Category', current_ews['risk_category']],
                            ['Clinical Response', current_ews['clinical_response']]
                        ]

                        # Add trend if available
                        ews_trend = analysis_results.get('ews_trend_analysis')
                        if ews_trend:
                            ews_data.append(['Overall Trend', f"{ews_trend['overall_trend'].title()} (Confidence: {ews_trend['confidence']})"])
                            ews_data.append(['Clinical Interpretation', ews_trend['clinical_interpretation']])

                        ews_table = Table(ews_data, colWidths=[2*inch, 4*inch])
                        ews_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))

                        story.append(ews_table)
                        story.append(Spacer(1, 12))

                        # EWS Score Breakdown
                        story.append(Paragraph("EWS Score Breakdown", subheading_style))
                        breakdown_data = [['Vital Sign', 'Value', 'EWS Points', 'Notes']]

                        for vital_name, vital_info in current_ews['score_breakdown'].items():
                            breakdown_data.append([
                                vital_name,
                                str(vital_info.get('value', 'N/A')),
                                str(vital_info.get('score', 0)),
                                vital_info.get('note', '')
                            ])

                        breakdown_table = Table(breakdown_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
                        breakdown_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))

                        story.append(breakdown_table)
                        story.append(Spacer(1, 12))

                        # Clinical Summary
                        clinical_summary = analyzer.generate_clinical_summary(analysis_results)

                        story.append(Paragraph("Clinical Assessment Summary", subheading_style))
                        clinical_data = [
                            ['Assessment Area', 'Findings'],
                            ['Monitoring Frequency', clinical_summary['monitoring_frequency'].title()],
                        ]

                        if clinical_summary.get('recommendations'):
                            recommendations_text = '\n'.join([f"• {rec}" for rec in clinical_summary['recommendations']])
                            clinical_data.append(['Recommendations', recommendations_text])

                        if clinical_summary.get('individual_vital_alerts'):
                            alerts_text = '\n'.join([f"• {alert['vital']}: {alert['trend']} - {alert['interpretation']}"
                                                   for alert in clinical_summary['individual_vital_alerts']])
                            clinical_data.append(['Vital Sign Alerts', alerts_text])

                        clinical_table = Table(clinical_data, colWidths=[1.5*inch, 4.5*inch])
                        clinical_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.moccasin),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))

                        story.append(clinical_table)
                        story.append(Spacer(1, 12))

                        # Individual Vitals History Summary
                        story.append(Paragraph("Vital Signs History Summary", subheading_style))

                        vitals_history_data = [['Vital Sign', 'Current Value', 'Trend Analysis', 'History Points']]

                        individual_vitals = analysis_results.get('individual_vitals', {})
                        for vital_name, vital_info in vitals_data.items():
                            if vital_name in analyzer.ews_weights:
                                current_val = f"{vital_info['current_value']} {vital_info['units']}"
                                history_count = len(vital_info['history'])

                                trend_info = individual_vitals.get(vital_name, {})
                                if trend_info.get('status') == 'analyzed':
                                    trend_text = f"{trend_info['trend'].title()} (R²={trend_info['r_squared']:.3f})"
                                else:
                                    trend_text = "Insufficient data"

                                vitals_history_data.append([
                                    vital_name,
                                    current_val,
                                    trend_text,
                                    str(history_count + 1)  # +1 for current value
                                ])

                        vitals_history_table = Table(vitals_history_data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 1*inch])
                        vitals_history_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))

                        story.append(vitals_history_table)

                    else:
                        story.append(Paragraph("Note: HDF5 file containing vital signs data not found", styles['Normal']))
                else:
                    story.append(Paragraph("Note: Data directory not found for vitals analysis", styles['Normal']))

            except Exception as e:
                story.append(Paragraph(f"Note: Vitals analysis not available: {str(e)}", styles['Normal']))

            story.append(Spacer(1, 12))

            # === SPECIFIC EVENT REPORT SECTION ===
            story.append(Paragraph(f"Detailed Event Report: {event_id}", heading_style))

            # Information Section
            story.append(Paragraph("Information Section", subheading_style))

            # Device and Patient Info Table
            device_info = report_data.get('device_info', {})
            event_info = report_data.get('event_info', {})
            ecg_data = report_data.get('ecg_data', {})

            info_data = [
                ['Parameter', 'Value'],
                ['Patient ID', patient_id],
                ['Event ID', event_id],
                ['Event Condition', event_info.get('condition', ai_data.get('event_condition', 'Unknown'))],
                ['AI Verdict', ai_data.get('ai_verdict', 'Unknown')],
                ['Event UUID', event_info.get('uuid', f"{patient_id}_{event_id}")],
            ]

            # Add event timestamp with proper formatting
            if 'event_timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['event_timestamp'], unit='s')
                info_data.append(['Event Time', event_timestamp.strftime('%Y-%m-%d %H:%M:%S')])
            elif 'timestamp' in event_info:
                event_timestamp = pd.to_datetime(event_info['timestamp'], unit='s')
                info_data.append(['Event Time', event_timestamp.strftime('%Y-%m-%d %H:%M:%S')])

            # Data quality scores
            metadata_quality = report_data.get('data_quality_score', None)
            calculated_quality = ecg_data.get('overall_quality_score', 0)
            overall_quality = metadata_quality if metadata_quality is not None else calculated_quality
            quality_source = "HDF5 Metadata" if metadata_quality is not None else "Calculated"
            info_data.append(['Data Quality Score', f"{overall_quality:.1f}/100 ({quality_source})"])

            # Add device info if available
            for key, value in device_info.items():
                if key not in ['device_info']:  # Avoid duplicate entries
                    info_data.append([key.replace('_', ' ').title(), str(value)])

            info_table = Table(info_data, colWidths=[2.5*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(info_table)
            story.append(Spacer(1, 12))

            # Diagnosis Section
            story.append(Paragraph("Diagnosis Section", subheading_style))

            # ECG Lead Configuration and Analysis
            leads = ecg_data.get('leads', {})
            if leads:
                # Get configured leads from API
                try:
                    configured_leads = self._get_lead_configuration()
                except:
                    configured_leads = []

                # ECG Configuration Summary
                config_data = [
                    ['Configuration', 'Value'],
                    ['ECG Leads Used for AI Analysis', ', '.join(configured_leads) if configured_leads else 'Not available'],
                    ['Available ECG Leads in Data', ', '.join(leads.keys())],
                ]

                if configured_leads:
                    available_configured = [lead for lead in configured_leads if lead in leads.keys()]
                    config_data.append(['Configured Leads with Data', ', '.join(available_configured) if available_configured else 'None'])

                config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
                config_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(Paragraph("ECG Lead Configuration:", styles['Normal']))
                story.append(config_table)
                story.append(Spacer(1, 8))

                # Get common data length and sampling rate for display
                first_lead_info = next(iter(leads.values()))
                common_data_length = first_lead_info.get('length', 0)
                common_sampling_rate = first_lead_info.get('sampling_rate', 500)

                # Add data specifications before the table
                story.append(Paragraph(f"ECG Data Specifications: {common_data_length:,} samples per lead at {common_sampling_rate} Hz sampling rate", styles['Normal']))
                story.append(Spacer(1, 6))

                # ECG Lead Quality Analysis (simplified without repetitive data length/sampling rate)
                lead_data = [['ECG Lead', 'AI Configuration', 'Quality Score', 'Anomalous Chunks']]
                for lead_name, lead_info in leads.items():
                    is_configured = lead_name in configured_leads if configured_leads else False
                    config_status = "✓ AI Configured" if is_configured else "⭕ Not Configured"

                    # Only show quality score and anomalous chunks for configured leads
                    if is_configured:
                        quality_score = lead_info.get('quality_score', 0)
                        quality_score_text = f"{quality_score:.1f}/100"
                        # Get actual anomaly percentage from database for configured leads
                        try:
                            conn = sqlite3.connect(self.db_path)
                            query = """
                                SELECT
                                    COUNT(CASE WHEN anomaly_status = 'anomaly' THEN 1 END) * 100.0 / COUNT(*) as anomaly_pct
                                FROM chunks
                                WHERE event_id = ? AND source_file LIKE ? AND lead_name = ?
                            """
                            result = pd.read_sql_query(query, conn, params=[event_id, f"%{patient_id}%", lead_name])
                            conn.close()
                            anomaly_percentage = result['anomaly_pct'].iloc[0] if not result.empty else 0
                            anomaly_percentage_text = f"{anomaly_percentage:.1f}%"
                        except Exception:
                            anomaly_percentage_text = "N/A"
                    else:
                        # For non-configured leads, show N/A for AI-specific metrics
                        quality_score_text = "N/A"
                        anomaly_percentage_text = "N/A"

                    lead_data.append([
                        lead_name,
                        config_status,
                        quality_score_text,
                        anomaly_percentage_text
                    ])

                lead_table = Table(lead_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                lead_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(Paragraph("ECG Lead Quality & Anomaly Analysis:", styles['Normal']))
                story.append(lead_table)
                story.append(Spacer(1, 12))

                # ECG Waveform plots banner
                waveform_banner_style = ParagraphStyle(
                    'WaveformBanner',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.darkblue,
                    backColor=colors.lightgrey,
                    borderColor=colors.grey,
                    borderWidth=1,
                    borderPadding=6,
                    alignment=1  # Center alignment
                )
                story.append(Paragraph("📈 ECG Waveform plots would be displayed here (7-lead ECG with anomaly chunk highlighting)", waveform_banner_style))
                story.append(Spacer(1, 12))

                # ECG Chunk Analysis Table - Only Anomalous Chunks
                try:
                    # Get chunk-level data for this specific event from database - only anomalous chunks
                    conn = sqlite3.connect(self.db_path)
                    query = """
                        SELECT lead_name, event_id, chunk_id, anomaly_status, anomaly_type, error_score
                        FROM chunks
                        WHERE event_id = ? AND source_file LIKE ? AND anomaly_status = 'anomaly'
                        ORDER BY lead_name, chunk_id
                    """
                    chunk_data = pd.read_sql_query(query, conn, params=[event_id, f"%{patient_id}%"])
                    conn.close()

                    if not chunk_data.empty:
                        # Constants for offset calculation - corrected for 12-second ECG strips
                        sampling_rate = 200        # Hz - actual ECG sampling rate
                        total_strip_duration = 12  # seconds
                        total_samples = sampling_rate * total_strip_duration  # 2400 samples

                        chunk_data_table = [['ECG Lead', 'Chunk ID', 'Offset of Anomaly in Strip', 'Anomaly Type', 'Error Score']]

                        for _, chunk_row in chunk_data.iterrows():
                            try:
                                # Try multiple conversion approaches
                                chunk_id_raw = chunk_row['chunk_id']

                                # Handle different data types
                                if pd.isna(chunk_id_raw):
                                    chunk_id = 0
                                elif isinstance(chunk_id_raw, (int, float)):
                                    chunk_id = int(chunk_id_raw)
                                elif isinstance(chunk_id_raw, str):
                                    cleaned = chunk_id_raw.strip()
                                    if cleaned.isdigit():
                                        chunk_id = int(cleaned)
                                    elif cleaned.startswith('chunk_'):
                                        # Handle chunk_eventid_chunknum format
                                        parts = cleaned.split('_')
                                        if len(parts) >= 3:  # chunk_eventid_chunknum
                                            chunk_num_str = parts[-1]  # Get last part (chunk number)
                                            if chunk_num_str.isdigit():
                                                chunk_id = int(chunk_num_str)
                                            else:
                                                raise ValueError(f"Last part '{chunk_num_str}' is not a digit")
                                        else:
                                            raise ValueError(f"Unexpected chunk format: '{cleaned}'")
                                    else:
                                        # Try to extract all digits and use the last one
                                        import re
                                        digits = re.findall(r'\d+', cleaned)
                                        if digits:
                                            chunk_id = int(digits[-1])  # Use last number instead of first
                                        else:
                                            raise ValueError(f"No digits found in '{cleaned}'")
                                else:
                                    raise ValueError(f"Unexpected type: {type(chunk_id_raw)}")

                                # Calculate offset - assume chunk_id is the sample number within the strip
                                offset_seconds = chunk_id / sampling_rate

                                # Clamp to 12-second strip duration for sanity check
                                if offset_seconds > total_strip_duration:
                                    offset_seconds = offset_seconds % total_strip_duration

                                # Format offset time
                                if offset_seconds < 60:
                                    offset_time = f"{offset_seconds:.1f}s"
                                else:
                                    minutes = int(offset_seconds // 60)
                                    seconds = offset_seconds % 60
                                    offset_time = f"{minutes}m {seconds:.1f}s"

                            except (ValueError, TypeError) as e:
                                # Enhanced fallback - try to extract chunk number
                                try:
                                    import re
                                    chunk_str = str(chunk_row['chunk_id'])
                                    if chunk_str.startswith('chunk_'):
                                        # Split by underscore and get the last part
                                        parts = chunk_str.split('_')
                                        if len(parts) >= 3:
                                            chunk_id = int(parts[-1])  # Get last part (chunk number)
                                        else:
                                            chunk_id = 0
                                    else:
                                        # Extract last number
                                        numbers = re.findall(r'\d+', chunk_str)
                                        chunk_id = int(numbers[-1]) if numbers else 0

                                    offset_seconds = chunk_id / sampling_rate
                                    # Clamp to 12-second strip
                                    if offset_seconds > total_strip_duration:
                                        offset_seconds = offset_seconds % total_strip_duration
                                    if offset_seconds < 60:
                                        offset_time = f"{offset_seconds:.1f}s"
                                    else:
                                        minutes = int(offset_seconds // 60)
                                        seconds = offset_seconds % 60
                                        offset_time = f"{minutes}m {seconds:.1f}s"
                                except:
                                    offset_time = f"Chunk {chunk_row['chunk_id']}"

                            # Format anomaly type
                            anomaly_type = chunk_row['anomaly_type'] if chunk_row['anomaly_type'] and chunk_row['anomaly_type'] != 'None' else 'Unknown Anomaly'

                            # Format error score safely
                            try:
                                error_score = f"{float(chunk_row['error_score']):.4f}"
                            except (ValueError, TypeError):
                                error_score = str(chunk_row['error_score'])

                            chunk_data_table.append([
                                chunk_row['lead_name'],
                                str(chunk_row['chunk_id']),
                                offset_time,
                                anomaly_type,
                                error_score
                            ])

                        chunk_table = Table(chunk_data_table, colWidths=[1*inch, 0.8*inch, 1.4*inch, 1.5*inch, 1*inch])
                        chunk_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),
                            ('FONTSIZE', (0, 1), (-1, -1), 7),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                        ]))

                        story.append(Paragraph("ECG Chunk Analysis (Anomalous Chunks Only):", styles['Normal']))
                        story.append(chunk_table)
                        story.append(Paragraph(f"Note: Showing {len(chunk_data_table)-1} anomalous chunks. Sampling rate: {sampling_rate} Hz, Strip duration: {total_strip_duration}s",
                                             ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
                        story.append(Spacer(1, 12))
                    else:
                        story.append(Paragraph("No anomalous chunks detected for this event.", styles['Normal']))
                        story.append(Spacer(1, 8))
                except Exception as e:
                    story.append(Paragraph(f"Note: Could not load chunk analysis data: {e}", styles['Normal']))
                    story.append(Spacer(1, 8))

            # Vital Signs Table (ordered as requested)
            vitals = report_data.get('vitals', {})
            if vitals:
                # Get event timestamp for relative time calculation
                event_timestamp = None
                if 'event_timestamp' in event_info:
                    event_timestamp = pd.to_datetime(event_info['event_timestamp'], unit='s')
                elif 'timestamp' in event_info:
                    event_timestamp = pd.to_datetime(event_info['timestamp'], unit='s')

                vital_order = ['HR', 'RespRate', 'SpO2', 'Pulse', 'Temp', 'Systolic', 'Diastolic', 'XL_Posture']
                vitals_data = [['Vital Sign', 'Value', 'Normal Range', 'Status', 'Relative Time', 'Timestamp']]

                for vital_name in vital_order:
                    if vital_name in vitals:
                        vital_info = vitals[vital_name]
                        display_name = vital_name.replace('_', ' ').title()
                        if vital_name == 'RespRate':
                            display_name = 'RR (Respiratory Rate)'
                        elif vital_name == 'HR':
                            display_name = 'HR (Heart Rate)'
                        elif vital_name == 'SpO2':
                            display_name = 'SpO2 (Oxygen Saturation)'
                        elif vital_name == 'XL_Posture':
                            display_name = 'XL Posture'

                        value = vital_info.get('value', 'N/A')
                        units = vital_info.get('units', '')
                        lower_threshold = vital_info.get('lower_threshold', 'N/A')
                        upper_threshold = vital_info.get('upper_threshold', 'N/A')
                        timestamp = vital_info.get('timestamp', 'N/A')

                        # Convert timestamp and calculate relative time
                        formatted_timestamp = 'N/A'
                        relative_time = 'N/A'

                        if isinstance(timestamp, (int, float)):
                            try:
                                vital_timestamp = pd.to_datetime(timestamp, unit='s')
                                formatted_timestamp = vital_timestamp.strftime('%Y-%m-%d %H:%M:%S')

                                # Calculate relative time difference w.r.t event time
                                if event_timestamp is not None:
                                    time_diff = vital_timestamp - event_timestamp
                                    total_seconds = int(time_diff.total_seconds())

                                    if total_seconds == 0:
                                        relative_time = "At Event"
                                    elif total_seconds > 0:
                                        # After event
                                        if total_seconds < 60:
                                            relative_time = f"+{total_seconds}s"
                                        elif total_seconds < 3600:
                                            minutes = total_seconds // 60
                                            seconds = total_seconds % 60
                                            relative_time = f"+{minutes}m {seconds}s" if seconds > 0 else f"+{minutes}m"
                                        else:
                                            hours = total_seconds // 3600
                                            minutes = (total_seconds % 3600) // 60
                                            relative_time = f"+{hours}h {minutes}m" if minutes > 0 else f"+{hours}h"
                                    else:
                                        # Before event
                                        abs_seconds = abs(total_seconds)
                                        if abs_seconds < 60:
                                            relative_time = f"-{abs_seconds}s"
                                        elif abs_seconds < 3600:
                                            minutes = abs_seconds // 60
                                            seconds = abs_seconds % 60
                                            relative_time = f"-{minutes}m {seconds}s" if seconds > 0 else f"-{minutes}m"
                                        else:
                                            hours = abs_seconds // 3600
                                            minutes = (abs_seconds % 3600) // 60
                                            relative_time = f"-{hours}h {minutes}m" if minutes > 0 else f"-{hours}h"
                            except:
                                formatted_timestamp = str(timestamp)

                        # Determine status
                        status = "Normal"
                        if (isinstance(value, (int, float)) and
                            isinstance(lower_threshold, (int, float)) and
                            isinstance(upper_threshold, (int, float))):
                            if value < lower_threshold:
                                status = "Low"
                            elif value > upper_threshold:
                                status = "High"
                            else:
                                status = "Normal"

                        vitals_data.append([
                            display_name,
                            f"{value} {units}".strip(),
                            f"{lower_threshold}-{upper_threshold} {units}".strip() if lower_threshold != 'N/A' else 'N/A',
                            status,
                            relative_time,
                            formatted_timestamp
                        ])

                vitals_table = Table(vitals_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 0.6*inch, 0.8*inch, 1.2*inch])
                vitals_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(Paragraph("Vital Signs Analysis:", styles['Normal']))
                story.append(vitals_table)
                story.append(Spacer(1, 12))

                # PPG and Respiratory waveform plots banner
                story.append(Paragraph("📊 PPG and Respiratory waveform plots would be displayed here (requires plotting implementation)", waveform_banner_style))
                story.append(Spacer(1, 12))

            # Analysis Information
            story.append(Paragraph("Analysis Information", subheading_style))

            analysis_data = [
                ['Analysis Parameter', 'Result'],
                ['AI Verdict', ai_data.get('ai_verdict', 'Unknown')],
                ['Event Condition', ai_data.get('event_condition', 'Unknown')],
                ['Average Error Score', f"{ai_data.get('error_score', 0):.4f}" if isinstance(ai_data.get('error_score'), (int, float)) else str(ai_data.get('error_score', 'N/A'))],
                ['Analysis Timestamp', str(ai_data.get('timestamp', 'Unknown'))],
                ['Model Version', str(device_info.get('model_version', device_info.get('software_version', device_info.get('ai_model_version', 'Unknown'))))]
            ]

            analysis_table = Table(analysis_data, colWidths=[2.5*inch, 3*inch])
            analysis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(analysis_table)
            story.append(Spacer(1, 20))

            # Footer
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=1
            )
            story.append(Paragraph("🔒 Confidential Medical Information - RMSAI ECG Analysis System", footer_style))

            # Build PDF
            doc.build(story)

            # Get PDF data
            buffer.seek(0)
            pdf_data = buffer.getvalue()
            buffer.close()

            return pdf_data

        except ImportError:
            st.warning("PDF export requires reportlab library. Install with: pip install reportlab")
            return None
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            return None

    def render_patient_analysis(self, chunks_df: pd.DataFrame):
        """Render combined Patient Analysis view with both events and patient-specific data"""
        st.header("👥 Patient Analysis")
        st.markdown("*Comprehensive patient view with event-level analysis and AI vs event condition comparison*")

        # Option to filter by file existence
        filter_by_files = st.checkbox("🗂️ Only show patients with existing data files", value=True,
                                     help="Filter out patients whose source files no longer exist")

        if len(chunks_df) == 0:
            st.warning("No patient data available")
            return

        # Patient selection - sort by most recent processing timestamp
        if 'patient_id' in chunks_df.columns:
            # Extract patient IDs
            chunks_df['patient_id'] = chunks_df['source_file'].apply(self._extract_patient_id)

            # Filter by file existence if requested
            if filter_by_files:
                existing_files_mask = chunks_df['source_file'].apply(os.path.exists)
                chunks_df = chunks_df[existing_files_mask]
                if len(chunks_df) == 0:
                    st.warning("No patients found with existing data files")
                    return

            # Get most recent timestamp per patient
            patient_timestamps = chunks_df.groupby('patient_id')['processing_timestamp'].max()
            # Sort patients by most recent timestamp (descending)
            patients = patient_timestamps.sort_values(ascending=False).index.tolist()

            # Debug info for troubleshooting
            with st.expander("🔧 Debug Info", expanded=False):
                st.write(f"**Database path:** {os.path.abspath(self.db_path)}")
                st.write(f"**Database exists:** {os.path.exists(self.db_path)}")
                st.write(f"**Working directory:** {os.getcwd()}")
                st.write(f"**Total chunks loaded:** {len(chunks_df)}")
                st.write(f"**Patients found:** {len(patients)}")

                # Show unique source files
                st.write(f"**Source files in database:**")
                unique_sources = sorted(chunks_df['source_file'].unique())
                for source in unique_sources:
                    patient_id = self._extract_patient_id(source)
                    count = len(chunks_df[chunks_df['source_file'] == source])
                    exists = os.path.exists(source)
                    status = "✅" if exists else "❌"
                    st.write(f"  {source} → {patient_id} ({count} chunks) {status}")

                st.write(f"**Patient order (by latest activity):**")
                for i, patient in enumerate(patients):
                    latest = patient_timestamps[patient]
                    chunk_count = len(chunks_df[chunks_df['patient_id'] == patient])
                    st.write(f"  {i+1}. {patient} - {chunk_count} chunks, latest: {latest}")
        else:
            patients = []

        if not patients:
            st.warning("No patient IDs found in the data")
            return

        selected_patient = st.selectbox("Select Patient", patients, key="patient_analysis_select")

        if not selected_patient:
            return

        # Filter data for selected patient
        patient_data = chunks_df[chunks_df['patient_id'] == selected_patient]

        # Initialize session state for event reports
        if 'selected_event_for_report' not in st.session_state:
            st.session_state.selected_event_for_report = None

        # Check if we should show an event report instead of the main analysis
        if st.session_state.selected_event_for_report:
            event_id = st.session_state.selected_event_for_report

            # Find the source file for this event
            event_data = patient_data[patient_data['event_id'] == event_id]
            if not event_data.empty:
                source_file = event_data.iloc[0]['source_file']

                # Prepare AI data for the report
                ai_data = {
                    'event_condition': 'Unknown',  # Will be filled from HDF5
                    'ai_verdict': 'Unknown',
                    'error_score': event_data['error_score'].mean(),
                    'timestamp': event_data['processing_timestamp'].iloc[0]
                }

                # Get ground truth for this specific event
                ground_truth_conditions = self._get_ground_truth_conditions(event_data)
                ai_data['event_condition'] = ground_truth_conditions.get(event_id, 'Unknown')

                # Get AI verdict for this event using unified HR-enhanced logic
                event_anomaly_types = [t for t in event_data['anomaly_type'].dropna().tolist() if t and t != 'None']
                anomaly_status_count = (event_data['anomaly_status'] == 'anomaly').sum()
                heart_rate = event_data['heart_rate'].iloc[0] if 'heart_rate' in event_data.columns else None

                # Use unified AI verdict function
                ai_data['ai_verdict'] = self.get_unified_ai_verdict(
                    event_anomaly_types,
                    heart_rate,
                    anomaly_status_count
                )

                # Render the report
                self._render_event_report(selected_patient, event_id, source_file, ai_data)
                return
            else:
                st.error(f"No data found for event {event_id}")
                st.session_state.selected_event_for_report = None
                st.rerun()

        # Get ground truth conditions
        ground_truth_conditions = self._get_ground_truth_conditions(patient_data)

        # Patient summary stats
        st.subheader(f"Patient {selected_patient} - Summary")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # Calculate patient-level stats
        patient_events = patient_data.groupby('event_id', as_index=False).agg({
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'error_score': 'mean',
            'timestamp': 'first',
            'chunk_id': 'count',
            'heart_rate': 'first'
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Add AI verdict using unified function with HR-based logic
        def debug_patient_analysis_verdict(row):
            result = self.get_unified_ai_verdict(
                row['anomaly_type'],
                row['heart_rate'],
                row['anomaly_status']
            )
            # DEBUG: Print for event_1006 in Patient Analysis
            if str(row.get('heart_rate')) == "109.1":
                print(f"🔍 DEBUG Patient Analysis event_1006: HR={row['heart_rate']}, anomaly_count={row['anomaly_status']}, anomalies={row['anomaly_type']}")
                print(f"🔍 DEBUG Patient Analysis: AI Verdict Result = {result}")
            return result

        patient_events['ai_verdict'] = patient_events.apply(debug_patient_analysis_verdict, axis=1)
        patient_events['event_condition'] = patient_events['event_id'].map(ground_truth_conditions).fillna('Unknown')

        with col1:
            total_events = len(patient_events)
            st.metric("Total Events", f"{total_events:,}")

        with col2:
            anomaly_events = (patient_events['ai_verdict'] != 'Normal').sum()
            st.metric("AI Detected Anomalies", f"{anomaly_events:,}")

        # Calculate performance metrics for this patient
        total_events = len(patient_events)
        if total_events > 0:
            # Get valid events (excluding Unknown ground truth)
            valid_events = patient_events[patient_events['event_condition'] != 'Unknown']

            if len(valid_events) > 0:
                ai_predictions = valid_events['ai_verdict'].tolist()
                ground_truth = valid_events['event_condition'].tolist()

                # Calculate clinical grouped metrics (more clinically relevant than exact matching)
                overall_metrics = self.calculate_clinical_grouped_metrics(ai_predictions, ground_truth)

                with col3:
                    st.metric("AI Accuracy", f"{overall_metrics['accuracy']*100:.1f}%")

                with col4:
                    st.metric("Precision", f"{overall_metrics['precision']*100:.1f}%",
                             help="Clinical Grouped: Normal, Rhythm Abnormalities (AF/VT), Rate Abnormalities (Tachy/Brady)")

                with col5:
                    st.metric("Recall", f"{overall_metrics['recall']*100:.1f}%",
                             help="Clinical Grouped: Groups similar conditions for more meaningful evaluation")

                with col6:
                    st.metric("F1 Score", f"{overall_metrics['f1']*100:.1f}%",
                             help="Uses clinical grouping instead of exact condition matching")
            else:
                with col3:
                    st.metric("AI Accuracy", "N/A")
                with col4:
                    st.metric("Precision", "N/A")
                with col5:
                    st.metric("Recall", "N/A")
                with col6:
                    st.metric("F1 Score", "N/A")
        else:
            with col3:
                st.metric("AI Accuracy", "N/A")
            with col4:
                st.metric("Precision", "N/A")
            with col5:
                st.metric("Recall", "N/A")
            with col6:
                st.metric("F1 Score", "N/A")

        # Event Summary Table (with AI verdicts showing highest severity only)
        st.subheader("Event Summary - AI Verdicts")

        # Create event summary table
        event_summary = patient_events[['event_id', 'event_condition', 'ai_verdict', 'heart_rate', 'error_score', 'total_chunks']].copy()
        event_summary.columns = ['Event ID', 'Event Condition', 'AI Verdict', 'Heart Rate (BPM)', 'Avg Error', 'Total Chunks']
        event_summary['Avg Error'] = event_summary['Avg Error'].round(4)
        event_summary['Heart Rate (BPM)'] = event_summary['Heart Rate (BPM)'].round(1)

        # Add comparison column with detailed mismatch detection
        def compare_enhanced_verdict(row):
            ai_verdict = row['AI Verdict']
            ground_truth = row['Event Condition']

            if ground_truth == 'Unknown':
                return "🔍 Unknown GT"
            elif ai_verdict == 'Normal' and ground_truth == 'Normal':
                return "✅ Match"
            elif ai_verdict == 'Normal' and ground_truth != 'Normal':
                return "❌ Missed"
            elif ai_verdict != 'Normal' and ground_truth == 'Normal':
                return "❌ False Positive"
            elif ai_verdict != 'Normal' and ground_truth != 'Normal' and ground_truth != 'Unknown':
                # Both detected anomalies - check if types match
                if ai_verdict == ground_truth:
                    return "✅ Match"
                else:
                    return "🔄 Mismatch"  # Different anomaly types detected
            else:
                return "❓ Unclear"

        event_summary['Comparison'] = event_summary.apply(compare_enhanced_verdict, axis=1)

        # Sort by event ID
        event_summary = event_summary.sort_values('Event ID')

        # Display event summary table
        st.dataframe(event_summary, use_container_width=True, hide_index=True)

        # Event-level analysis table
        st.subheader("Event Analysis - AI vs Event Condition")

        # Aggregate data by event and lead for detailed view
        events_by_lead = patient_data.groupby(['event_id', 'lead_name'], as_index=False).agg({
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
            'error_score': 'mean',
            'timestamp': 'first',
            'chunk_id': 'count',
            'heart_rate': 'first'  # Heart rate is same for all chunks in an event
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Use unified AI verdict function with HR-enhanced logic for lead-level analysis
        # Note: HR logic applies at event level, so we apply it even for individual leads
        def process_lead_ai_verdict_enhanced(row):
            anomaly_types_list = row['anomaly_type']
            total_chunks = row['total_chunks']
            anomaly_count = row['anomaly_status']
            heart_rate = row['heart_rate']

            # First, try unified HR-enhanced logic
            base_verdict = self.get_unified_ai_verdict(anomaly_types_list, heart_rate, anomaly_count)

            # If we have anomalies and want to show percentage for low confidence
            if base_verdict != "Normal" and len(set(anomaly_types_list)) == 1 and anomaly_count > 0:
                percentage = (anomaly_count / total_chunks) * 100
                if percentage < 30:
                    return f"{base_verdict} ({percentage:.0f}%)"
                else:
                    return base_verdict
            else:
                return base_verdict

        events_by_lead['ai_verdict'] = events_by_lead.apply(process_lead_ai_verdict_enhanced, axis=1)

        # Add ground truth
        events_by_lead['event_condition'] = events_by_lead['event_id'].map(ground_truth_conditions).fillna('Unknown')

        # Format for display
        display_events = events_by_lead[['event_id', 'lead_name', 'event_condition', 'ai_verdict', 'error_score', 'timestamp']].copy()
        display_events.columns = ['Event ID', 'Lead', 'Event Condition', 'AI Verdict', 'Avg Error Score', 'Timestamp']
        display_events['Avg Error Score'] = display_events['Avg Error Score'].round(4)
        display_events['Timestamp'] = pd.to_datetime(display_events['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')


        # Sort by timestamp descending
        display_events = display_events.sort_values('Timestamp', ascending=False)

        # Add comparison column with detailed mismatch detection
        def compare_verdict(row):
            event_condition = row['Event Condition']
            ai_verdict = row['AI Verdict']

            # Handle percentage annotations (e.g., "Tachycardia (15%)") while preserving condition names like "(MIT-BIH)"
            import re
            # Remove only percentage annotations like (25%) at the end, but keep condition identifiers like (MIT-BIH)
            ai_verdict_clean = re.sub(r' \(\d+%\)$', '', ai_verdict)

            if event_condition == 'Unknown':
                return "🔍 Unknown GT"
            elif ai_verdict_clean == 'Normal' and event_condition == 'Normal':
                return "✅ Match"
            elif ai_verdict_clean == 'Normal' and event_condition != 'Normal':
                return "❌ Missed"
            elif ai_verdict_clean != 'Normal' and event_condition == 'Normal':
                return "❌ False Positive"
            elif ai_verdict_clean != 'Normal' and event_condition != 'Normal' and event_condition != 'Unknown':
                # Both detected anomalies - check if types match
                if ai_verdict_clean == event_condition:
                    return "✅ Match"
                else:
                    return "🔄 Mismatch"  # Different anomaly types detected
            else:
                return "❓ Unclear"

        display_events['Comparison'] = display_events.apply(compare_verdict, axis=1)

        # Filters for event table
        col1, col2, col3 = st.columns(3)

        with col1:
            event_ids = ['All'] + list(display_events['Event ID'].unique())
            selected_event = st.selectbox("Filter by Event", event_ids, key="event_filter")

        with col2:
            conditions = ['All'] + list(display_events['Event Condition'].unique())
            selected_condition = st.selectbox("Filter by Condition", conditions, key="condition_filter")

        with col3:
            comparisons = ['All'] + list(display_events['Comparison'].unique())
            selected_comparison = st.selectbox("Filter by Comparison", comparisons, key="comparison_filter")

        # Apply filters
        filtered_events = display_events.copy()
        if selected_event != 'All':
            filtered_events = filtered_events[filtered_events['Event ID'] == selected_event]
        if selected_condition != 'All':
            filtered_events = filtered_events[filtered_events['Event Condition'] == selected_condition]
        if selected_comparison != 'All':
            filtered_events = filtered_events[filtered_events['Comparison'] == selected_comparison]

        # Pagination
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="events_pagination")

        if len(filtered_events) > 0:
            total_pages = (len(filtered_events) - 1) // rows_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1), key="events_page")

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            # Display events with View Report buttons
            current_page_events = filtered_events.iloc[start_idx:end_idx]

            # Header row
            header_cols = st.columns([2, 1.5, 2, 2, 1.5, 2, 1.5, 1])
            header_cols[0].markdown("**Event ID**")
            header_cols[1].markdown("**Lead**")
            header_cols[2].markdown("**Event Condition**")
            header_cols[3].markdown("**AI Verdict**")
            header_cols[4].markdown("**Error Score**")
            header_cols[5].markdown("**Timestamp**")
            header_cols[6].markdown("**Comparison**")
            header_cols[7].markdown("**Action**")
            st.divider()

            for idx, (_, row) in enumerate(current_page_events.iterrows()):
                with st.container():
                    cols = st.columns([2, 1.5, 2, 2, 1.5, 2, 1.5, 1])

                    cols[0].write(row['Event ID'])
                    cols[1].write(row['Lead'])
                    cols[2].write(row['Event Condition'])
                    cols[3].write(row['AI Verdict'])
                    cols[4].write(f"{row['Avg Error Score']:.4f}")
                    cols[5].write(row['Timestamp'])
                    cols[6].write(row['Comparison'])

                    # View Event Report button (renamed)
                    if cols[7].button("📋 View Event Report", key=f"view_report_{row['Event ID']}_{idx}_{page}"):
                        st.session_state.selected_event_for_report = row['Event ID']
                        st.rerun()

                    st.divider()

            st.info(f"Showing {min(rows_per_page, len(filtered_events) - start_idx)} of {len(filtered_events)} events (Page {page} of {total_pages})")
        else:
            st.warning("No events match the selected filters")


        # Performance Stats Section
        st.subheader("Processing Performance")

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            # Time to process single event (all configured leads)
            avg_processing_time = events_by_lead.groupby('event_id')['total_chunks'].sum().mean() * 10  # 10ms per chunk estimate
            st.metric("Avg Event Processing Time", f"{avg_processing_time:.0f}ms")

        with perf_col2:
            # Throughput calculation
            if len(events_by_lead) > 1:
                time_span = (pd.to_datetime(events_by_lead['timestamp']).max() -
                           pd.to_datetime(events_by_lead['timestamp']).min()).total_seconds()
                if time_span > 0:
                    unique_events = events_by_lead['event_id'].nunique()
                    throughput = (unique_events * 60) / time_span  # events/min
                    st.metric("Throughput", f"{throughput:.1f} events/min")
                else:
                    st.metric("Throughput", "N/A")
            else:
                st.metric("Throughput", "N/A")

        with perf_col3:
            # Total chunks processed for this patient
            total_chunks = len(patient_data)
            st.metric("Total Chunks Processed", f"{total_chunks:,}")

        with perf_col4:
            # Average chunks per event
            avg_chunks_per_event = total_chunks / total_events if total_events > 0 else 0
            st.metric("Avg Chunks/Event", f"{avg_chunks_per_event:.0f}")

        # Detailed Performance Metrics
        if len(patient_events) > 0:
            valid_events = patient_events[patient_events['event_condition'] != 'Unknown']
            if len(valid_events) > 0:
                st.subheader("Detailed Performance Metrics")

                ai_predictions = valid_events['ai_verdict'].tolist()
                ground_truth = valid_events['event_condition'].tolist()

                # Calculate condition-specific metrics
                condition_metrics = self.calculate_condition_specific_metrics(ai_predictions, ground_truth)

                # Display overall metrics with confusion matrix details
                overall = condition_metrics['Overall_Anomaly_Detection']

                perf_detail_col1, perf_detail_col2 = st.columns(2)

                with perf_detail_col1:
                    st.write("**Confusion Matrix & Overall Performance**")

                    # Create confusion matrix visualization
                    confusion_data = [
                        ['', 'Predicted Normal', 'Predicted Anomaly'],
                        ['Actual Normal', f"TN: {overall['tn']}", f"FP: {overall['fp']}"],
                        ['Actual Anomaly', f"FN: {overall['fn']}", f"TP: {overall['tp']}"]
                    ]

                    st.table(pd.DataFrame(confusion_data[1:], columns=confusion_data[0]))

                    st.write("**Metrics Explanation:**")
                    st.write(f"• **True Positives (TP)**: {overall['tp']} - Anomalies correctly detected")
                    st.write(f"• **False Positives (FP)**: {overall['fp']} - Normal events incorrectly flagged")
                    st.write(f"• **False Negatives (FN)**: {overall['fn']} - Anomalies missed")
                    st.write(f"• **True Negatives (TN)**: {overall['tn']} - Normal events correctly identified")

                with perf_detail_col2:
                    st.write("**Condition-Specific Performance**")

                    # Show metrics for each specific condition
                    condition_data = []

                    for condition, metrics in condition_metrics.items():
                        if condition != 'Overall_Anomaly_Detection' and metrics['support'] > 0:
                            condition_data.append({
                                'Condition': condition,
                                'Precision': f"{metrics['precision']*100:.1f}%",
                                'Recall': f"{metrics['recall']*100:.1f}%",
                                'F1 Score': f"{metrics['f1']*100:.1f}%",
                                'Support': metrics['support'],
                                'TP': metrics['tp'],
                                'FP': metrics['fp'],
                                'FN': metrics['fn']
                            })

                    if condition_data:
                        st.dataframe(pd.DataFrame(condition_data), use_container_width=True)

                        st.write("**Clinical Interpretation:**")
                        st.write("• **Precision**: How reliable are positive predictions?")
                        st.write("• **Recall**: How many actual cases are caught?")
                        st.write("• **F1 Score**: Balanced measure of both")
                        st.write("• **Support**: Number of actual cases of this condition")
                    else:
                        st.info("No specific conditions with sufficient data for detailed analysis")

        # Comparison Summary Chart
        if len(patient_events) > 0:
            st.subheader("AI Performance Visualization")

            comparison_counts = display_events['Comparison'].value_counts()

            if len(comparison_counts) > 0:
                fig = px.pie(
                    values=comparison_counts.values,
                    names=comparison_counts.index,
                    title=f"AI vs Event Condition Comparison for Patient {selected_patient}",
                    color_discrete_map={
                        '✅ Match': 'green',
                        '❌ Missed': 'red',
                        '❌ False Positive': 'orange',
                        '🔍 Unknown GT': 'gray',
                        '❓ Unclear': 'yellow'
                    }
                )
                st.plotly_chart(fig, width='stretch')

    def render_vitals_analysis(self, selected_patient: str, patient_events: pd.DataFrame):
        """Render comprehensive vitals analysis section with EWS scoring and trends"""
        st.subheader("🫀 Vital Signs Analysis")
        st.markdown("*Early Warning System (EWS) scoring, trend analysis, and clinical assessment*")

        if len(patient_events) == 0:
            st.warning("No patient events available for vitals analysis")
            return

        # Event selection for vitals analysis
        event_options = sorted(patient_events['event_id'].unique())
        selected_event = st.selectbox(
            "Select Event for Detailed Vitals Analysis",
            event_options,
            key="vitals_event_select"
        )

        if not selected_event:
            return

        # Import the vitals analyzer
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from rmsai_vitals_analysis import RMSAIVitalsAnalyzer
        except ImportError as e:
            st.error(f"Could not import vitals analysis module: {e}")
            return

        # Find HDF5 file for the selected event
        try:
            # Look for HDF5 files in data directory
            data_dir = "data"
            if not os.path.exists(data_dir):
                st.error("Data directory not found. Please ensure HDF5 files are in the 'data' directory.")
                return

            # Find HDF5 file that contains this event
            hdf5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

            if not hdf5_files:
                st.error("No HDF5 files found in data directory.")
                return

            # Try to find the file containing the selected event
            target_hdf5 = None
            for hdf5_file in hdf5_files:
                hdf5_path = os.path.join(data_dir, hdf5_file)
                try:
                    with h5py.File(hdf5_path, 'r') as f:
                        if selected_event in f:
                            target_hdf5 = hdf5_path
                            break
                except:
                    continue

            if not target_hdf5:
                st.error(f"Could not find HDF5 file containing event {selected_event}")
                st.info(f"Available HDF5 files: {hdf5_files}")
                return

            # Initialize analyzer
            analyzer = RMSAIVitalsAnalyzer(window_size=6, recent_values_count=3)

            # Extract vitals data from HDF5
            with st.spinner("Analyzing vital signs data..."):
                vitals_data = analyzer.extract_vitals_history_from_hdf5(target_hdf5, selected_event)
                analysis_results = analyzer.analyze_vital_trends(vitals_data)

            # Display current EWS status
            st.subheader("📊 Current Early Warning System (EWS) Status")

            current_ews = analysis_results['current_ews']
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("EWS Score", current_ews['total_score'])

            with col2:
                st.metric("Risk Category", current_ews['risk_category'])

            with col3:
                risk_color = current_ews['risk_color']
                st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>●</span> {current_ews['risk_category']}",
                           unsafe_allow_html=True)

            with col4:
                # Show trend if available
                ews_trend = analysis_results.get('ews_trend_analysis')
                if ews_trend:
                    trend_emoji = {
                        'improving': '📈',
                        'deteriorating': '📉',
                        'stable': '➡️'
                    }.get(ews_trend['overall_trend'], '❓')
                    st.metric("Trend", f"{trend_emoji} {ews_trend['overall_trend'].title()}")

            # Clinical Response
            st.info(f"**Clinical Response:** {current_ews['clinical_response']}")

            # Generate and display plots
            plots = analyzer.create_vitals_dashboard_plots(vitals_data, analysis_results)

            # EWS Trend Plot
            if 'ews_trend' in plots:
                st.subheader("📈 EWS Score Trend Analysis")
                st.plotly_chart(plots['ews_trend'], use_container_width=True)

            # Current Risk Gauge
            if 'current_risk_gauge' in plots:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plots['current_risk_gauge'], use_container_width=True)

                with col2:
                    st.subheader("🎯 EWS Score Breakdown")
                    breakdown_data = []
                    for vital_name, vital_info in current_ews['score_breakdown'].items():
                        breakdown_data.append({
                            'Vital Sign': vital_name,
                            'Value': vital_info.get('value', 'N/A'),
                            'EWS Points': vital_info.get('score', 0),
                            'Notes': vital_info.get('note', '')
                        })

                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

            # Individual Vitals Trends
            if 'individual_vitals' in plots:
                st.subheader("📊 Individual Vitals Trends")
                st.plotly_chart(plots['individual_vitals'], use_container_width=True)

            # Clinical Summary
            st.subheader("🏥 Clinical Assessment Summary")

            clinical_summary = analyzer.generate_clinical_summary(analysis_results)

            # Current Status
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Patient Status:**")
                st.markdown(f"- EWS Score: **{clinical_summary['current_status']['ews_score']}**")
                st.markdown(f"- Risk Category: **{clinical_summary['current_status']['risk_category']}**")
                st.markdown(f"- Monitoring Frequency: **{clinical_summary['monitoring_frequency'].title()}**")

            with col2:
                st.markdown("**Trend Assessment:**")
                if clinical_summary.get('trend_assessment'):
                    trend_info = clinical_summary['trend_assessment']
                    st.markdown(f"- Overall Trend: **{trend_info['overall_trend'].title()}**")
                    st.markdown(f"- Confidence: **{trend_info['confidence'].title()}**")
                    st.markdown(f"- Clinical Interpretation: {trend_info['interpretation']}")

            # Recommendations
            if clinical_summary.get('recommendations'):
                st.markdown("**Clinical Recommendations:**")
                for rec in clinical_summary['recommendations']:
                    st.markdown(f"• {rec}")

            # Individual Vital Alerts
            if clinical_summary.get('individual_vital_alerts'):
                st.subheader("⚠️ Individual Vital Sign Alerts")
                alert_data = []
                for alert in clinical_summary['individual_vital_alerts']:
                    alert_data.append({
                        'Vital Sign': alert['vital'],
                        'Trend': alert['trend'].title(),
                        'Clinical Interpretation': alert['interpretation'],
                        'Trend Strength': f"{alert['confidence']:.2f}"
                    })

                st.dataframe(pd.DataFrame(alert_data), use_container_width=True, hide_index=True)

            # Detailed EWS History (expandable)
            with st.expander("🔍 Detailed EWS History Analysis", expanded=False):
                ews_detailed_history = analysis_results['ews_history'].get('detailed_history', [])

                if ews_detailed_history:
                    st.markdown("**Complete EWS calculation history with vital signs breakdown:**")

                    detailed_data = []
                    for entry in ews_detailed_history:
                        vitals_str = ', '.join([f"{k}: {v}" for k, v in entry['vitals_used'].items()])
                        detailed_data.append({
                            'Timestamp': entry['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                            'EWS Score': entry['ews_breakdown']['total_score'],
                            'Risk Category': entry['ews_breakdown']['risk_category'],
                            'Vitals Used': vitals_str
                        })

                    st.dataframe(pd.DataFrame(detailed_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No detailed EWS history available")

        except Exception as e:
            st.error(f"Error performing vitals analysis: {str(e)}")
            st.exception(e)

    def render_vitals_analysis_for_report(self, patient_id: str, event_id: str):
        """Render vitals analysis section specifically for the View Report page"""
        st.subheader("🫀 Vital Signs Analysis")
        st.markdown("*Early Warning System (EWS) scoring and clinical assessment for this specific event*")

        # Import the vitals analyzer
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from rmsai_vitals_analysis import RMSAIVitalsAnalyzer
        except ImportError as e:
            st.error(f"Could not import vitals analysis module: {e}")
            return

        # Find HDF5 file for the selected event
        try:
            # Look for HDF5 files in data directory
            data_dir = "data"
            if not os.path.exists(data_dir):
                st.error("Data directory not found. Please ensure HDF5 files are in the 'data' directory.")
                return

            # Find HDF5 file that contains this event
            hdf5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

            if not hdf5_files:
                st.error("No HDF5 files found in data directory.")
                return

            # Try to find the file containing the selected event
            target_hdf5 = None
            for hdf5_file in hdf5_files:
                hdf5_path = os.path.join(data_dir, hdf5_file)
                try:
                    with h5py.File(hdf5_path, 'r') as f:
                        if event_id in f:
                            target_hdf5 = hdf5_path
                            break
                except:
                    continue

            if not target_hdf5:
                st.error(f"Could not find HDF5 file containing event {event_id}")
                st.info(f"Available HDF5 files: {hdf5_files}")
                return

            # Initialize analyzer
            analyzer = RMSAIVitalsAnalyzer(window_size=6, recent_values_count=3)

            # Extract vitals data from HDF5
            with st.spinner("Analyzing vital signs data..."):
                vitals_data = analyzer.extract_vitals_history_from_hdf5(target_hdf5, event_id)
                analysis_results = analyzer.analyze_vital_trends(vitals_data)

            # Display current EWS status in a more compact format for the report
            current_ews = analysis_results['current_ews']

            # EWS Summary Card
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("EWS Score", current_ews['total_score'])
                risk_color = current_ews['risk_color']
                st.markdown(f"**Risk Level:** <span style='color: {risk_color}; font-weight: bold;'>{current_ews['risk_category']}</span>",
                           unsafe_allow_html=True)

            with col2:
                # Show trend if available
                ews_trend = analysis_results.get('ews_trend_analysis')
                if ews_trend:
                    trend_emoji = {
                        'improving': '📈 ↗️',
                        'deteriorating': '📉 ↘️',
                        'stable': '➡️ →'
                    }.get(ews_trend['overall_trend'], '❓')
                    st.metric("Trend", f"{ews_trend['overall_trend'].title()}")
                    st.markdown(f"**Direction:** {trend_emoji}")
                    st.caption(f"Confidence: {ews_trend['confidence']}")
                else:
                    st.metric("Trend", "Insufficient Data")

            with col3:
                clinical_summary = analyzer.generate_clinical_summary(analysis_results)
                st.markdown("**Monitoring Frequency:**")
                st.markdown(f"**{clinical_summary['monitoring_frequency'].title()}**")

                # Show alert status
                if clinical_summary.get('individual_vital_alerts'):
                    st.warning(f"⚠️ {len(clinical_summary['individual_vital_alerts'])} vital sign alerts")
                else:
                    st.success("✅ No vital sign alerts")

            # Clinical Response
            st.info(f"**Clinical Response:** {current_ews['clinical_response']}")

            # EWS Score Breakdown in a compact table
            st.subheader("📊 EWS Score Breakdown")

            breakdown_data = []
            for vital_name, vital_info in current_ews['score_breakdown'].items():
                breakdown_data.append({
                    'Vital Sign': vital_name,
                    'Value': vital_info.get('value', 'N/A'),
                    'EWS Points': vital_info.get('score', 0),
                    'Status': 'Missing' if vital_info.get('note') else 'Normal' if vital_info.get('score', 0) == 0 else 'Alert'
                })

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            # EWS Scoring Template Reference
            with st.expander("📋 EWS Scoring Template (NEWS2)", expanded=False):
                st.markdown("**Reference scoring template used for EWS calculations:**")

                # Import configuration
                from config import get_ews_scoring_template, get_ews_risk_categories

                ews_template = get_ews_scoring_template()
                risk_categories = get_ews_risk_categories()

                # Display scoring template for each vital
                for vital_name, vital_config in ews_template.items():
                    st.markdown(f"**{vital_config['display_name']} ({vital_config['units']})**")

                    template_data = []
                    for range_config in vital_config['ranges']:
                        template_data.append({
                            'Range': range_config['range'],
                            'EWS Points': range_config['score']
                        })

                    template_df = pd.DataFrame(template_data)
                    st.dataframe(template_df, use_container_width=True, hide_index=True)

                # Display risk categories
                st.markdown("**Risk Categories:**")
                risk_data = []
                for level, config in risk_categories.items():
                    score_min, score_max = config['score_range']
                    score_range = f"{score_min}-{score_max}" if score_max != float('inf') else f"{score_min}+"

                    risk_data.append({
                        'Total Score': score_range,
                        'Risk Level': config['category'],
                        'Monitoring Frequency': config['monitoring_frequency'],
                        'Clinical Response': config['clinical_response']
                    })

                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True, hide_index=True)

            # Generate and display plots (compact versions for report)
            plots = analyzer.create_vitals_dashboard_plots(vitals_data, analysis_results)

            # Show EWS trend plot if available
            if 'ews_trend' in plots and ews_trend:
                st.subheader("📈 EWS Trend Analysis")
                st.plotly_chart(plots['ews_trend'], use_container_width=True, height=400)

            # Individual vitals in a more compact view
            if 'individual_vitals' in plots:
                with st.expander("📊 Individual Vitals Trends", expanded=False):
                    st.plotly_chart(plots['individual_vitals'], use_container_width=True)

            # Clinical recommendations in a compact format
            if clinical_summary.get('recommendations'):
                st.subheader("🏥 Clinical Recommendations")
                for i, rec in enumerate(clinical_summary['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")

            # Individual vital alerts if any
            if clinical_summary.get('individual_vital_alerts'):
                st.subheader("⚠️ Vital Sign Alerts")
                for alert in clinical_summary['individual_vital_alerts']:
                    st.warning(f"**{alert['vital']}**: {alert['trend'].title()} trend - {alert['interpretation']}")

        except Exception as e:
            st.error(f"Error performing vitals analysis: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.exception(e)

    def render_api_status(self):
        """Render API connection status"""
        api_stats = self.get_api_stats()

        if api_stats:
            st.sidebar.success("🟢 API Connected")
            st.sidebar.json(api_stats)
        else:
            st.sidebar.error("🔴 API Disconnected")
            st.sidebar.info("Start API server: `python api_server.py`")

    def run(self):
        """Run the dashboard"""
        # Sidebar configuration
        st.sidebar.title("🔧 Dashboard Settings")

        # Lead selection configuration
        st.sidebar.subheader("Lead Configuration")
        st.sidebar.info(f"Currently processing {len(self.selected_leads)} leads: {', '.join(self.selected_leads)}")

        if st.sidebar.button("🔄 Refresh Lead Info"):
            self.selected_leads = self._get_selected_leads()
            st.rerun()

        # Auto-refresh setting
        refresh_setting = self.render_header()

        # API status
        self.render_api_status()

        # Cache settings
        st.sidebar.subheader("Cache Settings")
        use_cache = st.sidebar.checkbox("Enable Caching", value=True)
        cache_duration = st.sidebar.slider("Cache Duration (seconds)", 10, 300, 30)

        if st.sidebar.button("Clear Cache"):
            st.session_state.data_cache = {}
            st.session_state.cache_time = {}
            st.success("Cache cleared! Refresh page to see updated data.")

        # Show cache status
        if hasattr(st.session_state, 'cache_time') and st.session_state.cache_time:
            main_cache_time = st.session_state.cache_time.get("main_data")
            if main_cache_time:
                cache_age = time.time() - main_cache_time
                st.sidebar.info(f"Data cached {cache_age:.0f}s ago")
        else:
            st.sidebar.info("No cached data")

        # Load data
        chunks_df, files_df = self.load_data(use_cache, cache_duration)

        # Render main components
        self.render_overview_metrics(chunks_df, files_df)

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Timeline", "🔍 Conditions", "🔬 ECG Leads",
            "⚠️ Anomalies", "💊 System Health", "🔎 Similarity",
            "👥 Patient Analysis"
        ])

        with tab1:
            self.render_anomaly_timeline(chunks_df)

        with tab2:
            self.render_condition_analysis(chunks_df)

        with tab3:
            self.render_lead_analysis(chunks_df)

        with tab4:
            self.render_recent_anomalies(chunks_df)

        with tab5:
            self.render_system_health(files_df)

        with tab6:
            self.render_similarity_search(chunks_df)

        with tab7:
            self.render_patient_analysis(chunks_df)

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**RMSAI Dashboard v1.0**")
        st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Initialize and run dashboard
    dashboard = RMSAIDashboard()
    dashboard.run()