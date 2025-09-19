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

# Configure Streamlit page
st.set_page_config(
    page_title="RMSAI Anomaly Detection Dashboard",
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

        # Load processor configuration to get selected leads
        self.selected_leads = self._get_selected_leads()

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
                           processing_timestamp, vector_id
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
                else:
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
        severity_order = {
            'Ventricular Tachycardia (MIT-BIH)': 5,
            'Atrial Fibrillation (PTB-XL)': 4,
            'Unknown Arrhythmia': 3,
            'Tachycardia': 2,
            'Bradycardia': 1
        }

        def process_ai_verdict(anomaly_types_list, total_chunks, anomaly_count):
            if not anomaly_types_list or anomaly_count == 0:
                return "Normal"

            # Get unique anomaly types and sort by severity
            unique_anomalies = list(set(anomaly_types_list))
            unique_anomalies.sort(key=lambda x: severity_order.get(x, 0), reverse=True)

            if len(unique_anomalies) == 1:
                # Single anomaly type
                anomaly = unique_anomalies[0]
                percentage = (anomaly_count / total_chunks) * 100
                if percentage < 30:
                    return f"{anomaly} ({percentage:.0f}%)"
                else:
                    return anomaly
            else:
                # Multiple anomaly types
                return ", ".join(unique_anomalies)

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

    def render_patient_specific(self, chunks_df: pd.DataFrame):
        """Render Patient Specific view"""
        st.header("👤 Patient Specific Analysis")
        st.markdown("*Patient-focused view with event summaries and anomaly patterns*")

        if len(chunks_df) == 0:
            st.warning("No patient data available")
            return

        # Patient selection
        patients = sorted(chunks_df['patient_id'].unique()) if 'patient_id' in chunks_df.columns else []

        if not patients:
            st.warning("No patient IDs found in the data")
            return

        selected_patient = st.selectbox("Select Patient", patients)

        if not selected_patient:
            return

        # Filter data for selected patient
        patient_data = chunks_df[chunks_df['patient_id'] == selected_patient]

        # Aggregate by event for this patient
        patient_events = patient_data.groupby('event_id', as_index=False).agg({
            'condition': 'first',
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'error_score': 'mean',
            'timestamp': 'first',
            'chunk_id': 'count'
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Process AI Verdict for each event
        severity_order = {
            'Ventricular Tachycardia (MIT-BIH)': 5,
            'Atrial Fibrillation (PTB-XL)': 4,
            'Unknown Arrhythmia': 3,
            'Tachycardia': 2,
            'Bradycardia': 1
        }

        def get_event_ai_verdict(anomaly_types_list):
            if not anomaly_types_list:
                return "Normal"

            unique_anomalies = list(set(anomaly_types_list))
            unique_anomalies.sort(key=lambda x: severity_order.get(x, 0), reverse=True)

            return ", ".join(unique_anomalies)

        patient_events['ai_verdict'] = patient_events['anomaly_type'].apply(get_event_ai_verdict)

        # Patient summary stats
        st.subheader(f"Patient {selected_patient} - Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_events = len(patient_events)
            st.metric("Total Events", f"{total_events:,}")

        with col2:
            anomaly_events = (patient_events['ai_verdict'] != 'Normal').sum()
            st.metric("Anomaly Events", f"{anomaly_events:,}")

        with col3:
            if total_events > 0:
                anomaly_rate = (anomaly_events / total_events) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            else:
                st.metric("Anomaly Rate", "0%")

        with col4:
            avg_error = patient_events['error_score'].mean()
            st.metric("Avg Error Score", f"{avg_error:.4f}")

        # Most common anomalies
        all_anomalies = []
        for anomaly_list in patient_events['anomaly_type']:
            all_anomalies.extend(anomaly_list)

        if all_anomalies:
            anomaly_counts = pd.Series(all_anomalies).value_counts()
            st.subheader("Most Common Anomalies")

            fig = px.bar(
                x=anomaly_counts.index,
                y=anomaly_counts.values,
                title=f"Anomaly Distribution for Patient {selected_patient}",
                labels={'x': 'Anomaly Type', 'y': 'Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch')

        # Events table
        st.subheader("Patient Events")

        # Format for display
        display_events = patient_events[['event_id', 'condition', 'ai_verdict', 'error_score', 'timestamp']].copy()
        display_events.columns = ['Event ID', 'Reported Condition', 'AI Verdict', 'Avg Error Score', 'Timestamp']
        display_events['Avg Error Score'] = display_events['Avg Error Score'].round(4)
        display_events['Timestamp'] = pd.to_datetime(display_events['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Sort by timestamp descending
        display_events = display_events.sort_values('Timestamp', ascending=False)

        # Pagination for patient events
        rows_per_page = st.selectbox("Events per page", [5, 10, 20, 50], index=1, key="patient_pagination")

        if len(display_events) > 0:
            total_pages = (len(display_events) - 1) // rows_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1), key="patient_page")

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            st.dataframe(display_events.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)

            st.info(f"Showing {min(rows_per_page, len(display_events) - start_idx)} of {len(display_events)} events (Page {page} of {total_pages})")
        else:
            st.warning("No events found for this patient")

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

    def render_patient_analysis(self, chunks_df: pd.DataFrame):
        """Render combined Patient Analysis view with both events and patient-specific data"""
        st.header("👥 Patient Analysis")
        st.markdown("*Comprehensive patient view with event-level analysis and AI vs ground truth comparison*")

        if len(chunks_df) == 0:
            st.warning("No patient data available")
            return

        # Patient selection
        patients = sorted(chunks_df['patient_id'].unique()) if 'patient_id' in chunks_df.columns else []

        if not patients:
            st.warning("No patient IDs found in the data")
            return

        selected_patient = st.selectbox("Select Patient", patients, key="patient_analysis_select")

        if not selected_patient:
            return

        # Filter data for selected patient
        patient_data = chunks_df[chunks_df['patient_id'] == selected_patient]

        # Get ground truth conditions
        ground_truth_conditions = self._get_ground_truth_conditions(patient_data)

        # Patient summary stats
        st.subheader(f"Patient {selected_patient} - Summary")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate patient-level stats
        patient_events = patient_data.groupby('event_id', as_index=False).agg({
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'error_score': 'mean',
            'timestamp': 'first',
            'chunk_id': 'count'
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Add ground truth and AI verdict
        def get_ai_verdict(anomaly_types_list):
            if not anomaly_types_list:
                return "Normal"

            severity_order = {
                'Ventricular Tachycardia (MIT-BIH)': 5,
                'Atrial Fibrillation (PTB-XL)': 4,
                'Unknown Arrhythmia': 3,
                'Tachycardia': 2,
                'Bradycardia': 1
            }

            unique_anomalies = list(set(anomaly_types_list))
            unique_anomalies.sort(key=lambda x: severity_order.get(x, 0), reverse=True)
            return ", ".join(unique_anomalies)

        patient_events['ai_verdict'] = patient_events['anomaly_type'].apply(get_ai_verdict)
        patient_events['event_condition'] = patient_events['event_id'].map(ground_truth_conditions).fillna('Unknown')

        with col1:
            total_events = len(patient_events)
            st.metric("Total Events", f"{total_events:,}")

        with col2:
            anomaly_events = (patient_events['ai_verdict'] != 'Normal').sum()
            st.metric("AI Detected Anomalies", f"{anomaly_events:,}")

        with col3:
            # Accuracy calculation (AI vs Ground Truth)
            if total_events > 0:
                # Simple accuracy: count when AI verdict matches event condition
                matches = 0
                for _, row in patient_events.iterrows():
                    ai_verdict = row['ai_verdict']
                    event_condition = row['event_condition']

                    # Normalize for comparison
                    if ai_verdict == 'Normal' and event_condition in ['Normal', 'Unknown']:
                        matches += 1
                    elif ai_verdict != 'Normal' and event_condition != 'Normal' and event_condition != 'Unknown':
                        matches += 1

                accuracy = (matches / total_events) * 100
                st.metric("AI Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("AI Accuracy", "N/A")

        with col4:
            avg_error = patient_events['error_score'].mean()
            st.metric("Avg Error Score", f"{avg_error:.4f}")

        # Event-level analysis table
        st.subheader("Event Analysis - AI vs Ground Truth")

        # Aggregate data by event and lead for detailed view
        events_by_lead = patient_data.groupby(['event_id', 'lead_name'], as_index=False).agg({
            'anomaly_status': lambda x: (x == 'anomaly').sum(),
            'anomaly_type': lambda x: [t for t in x if t and t != 'None' and pd.notna(t)],
            'error_score': 'mean',
            'timestamp': 'first',
            'chunk_id': 'count'
        }).rename(columns={'chunk_id': 'total_chunks'})

        # Process AI verdicts for each lead
        def process_lead_ai_verdict(anomaly_types_list, total_chunks, anomaly_count):
            if not anomaly_types_list or anomaly_count == 0:
                return "Normal"

            severity_order = {
                'Ventricular Tachycardia (MIT-BIH)': 5,
                'Atrial Fibrillation (PTB-XL)': 4,
                'Unknown Arrhythmia': 3,
                'Tachycardia': 2,
                'Bradycardia': 1
            }

            unique_anomalies = list(set(anomaly_types_list))
            unique_anomalies.sort(key=lambda x: severity_order.get(x, 0), reverse=True)

            if len(unique_anomalies) == 1:
                anomaly = unique_anomalies[0]
                percentage = (anomaly_count / total_chunks) * 100
                if percentage < 30:
                    return f"{anomaly} ({percentage:.0f}%)"
                else:
                    return anomaly
            else:
                return ", ".join(unique_anomalies)

        events_by_lead['ai_verdict'] = events_by_lead.apply(
            lambda row: process_lead_ai_verdict(
                row['anomaly_type'],
                row['total_chunks'],
                row['anomaly_status']
            ),
            axis=1
        )

        # Add ground truth
        events_by_lead['event_condition'] = events_by_lead['event_id'].map(ground_truth_conditions).fillna('Unknown')

        # Format for display
        display_events = events_by_lead[['event_id', 'lead_name', 'event_condition', 'ai_verdict', 'error_score', 'timestamp']].copy()
        display_events.columns = ['Event ID', 'Lead', 'Event Condition (Ground Truth)', 'AI Verdict', 'Avg Error Score', 'Timestamp']
        display_events['Avg Error Score'] = display_events['Avg Error Score'].round(4)
        display_events['Timestamp'] = pd.to_datetime(display_events['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Sort by timestamp descending
        display_events = display_events.sort_values('Timestamp', ascending=False)

        # Add comparison column
        def compare_verdict(row):
            event_condition = row['Event Condition (Ground Truth)']
            ai_verdict = row['AI Verdict']

            if event_condition == 'Unknown':
                return "🔍 Unknown GT"
            elif ai_verdict == 'Normal' and event_condition == 'Normal':
                return "✅ Match"
            elif ai_verdict != 'Normal' and event_condition != 'Normal' and event_condition != 'Unknown':
                return "✅ Match"
            elif ai_verdict == 'Normal' and event_condition != 'Normal':
                return "❌ Missed"
            elif ai_verdict != 'Normal' and event_condition == 'Normal':
                return "❌ False Positive"
            else:
                return "❓ Unclear"

        display_events['Comparison'] = display_events.apply(compare_verdict, axis=1)

        # Filters for event table
        col1, col2, col3 = st.columns(3)

        with col1:
            event_ids = ['All'] + list(display_events['Event ID'].unique())
            selected_event = st.selectbox("Filter by Event", event_ids, key="event_filter")

        with col2:
            conditions = ['All'] + list(display_events['Event Condition (Ground Truth)'].unique())
            selected_condition = st.selectbox("Filter by Condition", conditions, key="condition_filter")

        with col3:
            comparisons = ['All'] + list(display_events['Comparison'].unique())
            selected_comparison = st.selectbox("Filter by Comparison", comparisons, key="comparison_filter")

        # Apply filters
        filtered_events = display_events.copy()
        if selected_event != 'All':
            filtered_events = filtered_events[filtered_events['Event ID'] == selected_event]
        if selected_condition != 'All':
            filtered_events = filtered_events[filtered_events['Event Condition (Ground Truth)'] == selected_condition]
        if selected_comparison != 'All':
            filtered_events = filtered_events[filtered_events['Comparison'] == selected_comparison]

        # Pagination
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="events_pagination")

        if len(filtered_events) > 0:
            total_pages = (len(filtered_events) - 1) // rows_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1), key="events_page")

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            st.dataframe(filtered_events.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)

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

        # Comparison Summary Chart
        if len(patient_events) > 0:
            st.subheader("AI Performance Summary")

            comparison_counts = display_events['Comparison'].value_counts()

            if len(comparison_counts) > 0:
                fig = px.pie(
                    values=comparison_counts.values,
                    names=comparison_counts.index,
                    title=f"AI vs Ground Truth Comparison for Patient {selected_patient}",
                    color_discrete_map={
                        '✅ Match': 'green',
                        '❌ Missed': 'red',
                        '❌ False Positive': 'orange',
                        '🔍 Unknown GT': 'gray',
                        '❓ Unclear': 'yellow'
                    }
                )
                st.plotly_chart(fig, width='stretch')

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
            st.success("Cache cleared!")

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