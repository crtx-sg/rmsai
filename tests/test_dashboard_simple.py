#!/usr/bin/env python3
"""
Simplified dashboard test to isolate the issue
"""

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="RMSAI Dashboard Test",
    page_icon="💓",
    layout="wide"
)

st.title("🔍 RMSAI Dashboard Data Test")

def load_data():
    """Load data from SQLite database"""
    try:
        with sqlite3.connect("rmsai_metadata.db") as conn:
            # Main chunks data
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

            return chunks_df, files_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()

# Load data
chunks_df, files_df = load_data()

st.write("## Data Loading Results")
st.write(f"Chunks loaded: **{len(chunks_df)}**")
st.write(f"Files loaded: **{len(files_df)}**")

if len(chunks_df) > 0:
    st.write("## Chunks Data Overview")

    # Calculate metrics
    total_chunks = len(chunks_df)
    total_anomalies = len(chunks_df[chunks_df['anomaly_status'] == 'anomaly'])
    anomaly_rate = (total_anomalies / total_chunks * 100) if total_chunks > 0 else 0
    avg_error = chunks_df['error_score'].mean()

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", total_chunks)

    with col2:
        st.metric("Anomalies", total_anomalies)

    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

    with col4:
        st.metric("Avg Error Score", f"{avg_error:.4f}")

    st.write("### Sample Data")
    st.dataframe(chunks_df.head(10))

    # Charts
    st.write("### Anomaly Status Distribution")
    anomaly_counts = chunks_df['anomaly_status'].value_counts()
    st.bar_chart(anomaly_counts)

    st.write("### Lead Name Distribution")
    lead_counts = chunks_df['lead_name'].value_counts()
    st.bar_chart(lead_counts)

    st.write("### Error Score Distribution")
    st.histogram_chart(chunks_df['error_score'])

else:
    st.warning("No chunks data found!")

if len(files_df) > 0:
    st.write("## Files Data Overview")
    st.dataframe(files_df)
else:
    st.warning("No files data found!")

st.write("## Raw Database Info")
try:
    with sqlite3.connect("rmsai_metadata.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM processed_files")
        file_count = cursor.fetchone()[0]

        st.write(f"Direct database query - Chunks: {chunk_count}, Files: {file_count}")
except Exception as e:
    st.error(f"Database query error: {e}")