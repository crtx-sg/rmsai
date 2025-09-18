#!/usr/bin/env python3
"""
Test script to debug dashboard data loading issues
"""

import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

def test_database_connection():
    """Test if database exists and is accessible"""
    db_path = "rmsai_metadata.db"

    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} does not exist")
        return False

    print(f"✅ Database file {db_path} exists")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ Database connection successful")
            print(f"Tables: {[table[0] for table in tables]}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_data_loading():
    """Test the same data loading logic as the dashboard"""
    db_path = "rmsai_metadata.db"

    try:
        with sqlite3.connect(db_path) as conn:
            # Test chunks query
            chunks_df = pd.read_sql_query("""
                SELECT chunk_id, event_id, source_file, lead_name,
                       anomaly_status, anomaly_type, error_score,
                       processing_timestamp, vector_id
                FROM chunks
                ORDER BY processing_timestamp DESC
                LIMIT 5000
            """, conn)

            # Test files query
            files_df = pd.read_sql_query("""
                SELECT filename, file_hash, status, processing_timestamp, error_message
                FROM processed_files
                ORDER BY processing_timestamp DESC
                LIMIT 1000
            """, conn)

            print(f"\n📊 Data Loading Results:")
            print(f"Chunks loaded: {len(chunks_df)}")
            print(f"Files loaded: {len(files_df)}")

            if len(chunks_df) > 0:
                print(f"\n📈 Chunks Data Analysis:")
                print(f"Columns: {list(chunks_df.columns)}")
                print(f"Anomalies: {len(chunks_df[chunks_df['anomaly_status'] == 'anomaly'])}")
                print(f"Average error score: {chunks_df['error_score'].mean():.4f}")
                print(f"Date range: {chunks_df['processing_timestamp'].min()} to {chunks_df['processing_timestamp'].max()}")

                print(f"\nSample data:")
                print(chunks_df.head(3).to_string())
            else:
                print("❌ No chunks data found")

            if len(files_df) > 0:
                print(f"\n📁 Files Data Analysis:")
                print(f"Columns: {list(files_df.columns)}")
                print(f"Completed files: {len(files_df[files_df['status'] == 'completed'])}")

                print(f"\nSample data:")
                print(files_df.head(3).to_string())
            else:
                print("❌ No files data found")

            return chunks_df, files_df

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def test_dashboard_metrics(chunks_df, files_df):
    """Test the metrics calculation logic"""
    print(f"\n🎯 Dashboard Metrics Test:")

    # Calculate metrics
    total_chunks = len(chunks_df)
    total_anomalies = len(chunks_df[chunks_df['anomaly_status'] == 'anomaly']) if len(chunks_df) > 0 else 0
    anomaly_rate = (total_anomalies / total_chunks * 100) if total_chunks > 0 else 0
    files_processed = len(files_df[files_df['status'] == 'completed']) if len(files_df) > 0 else 0
    avg_error = chunks_df['error_score'].mean() if len(chunks_df) > 0 else 0

    print(f"Total chunks: {total_chunks}")
    print(f"Total anomalies: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print(f"Files processed: {files_processed}")
    print(f"Average error score: {avg_error:.4f}")

    # System status
    if len(chunks_df) > 0:
        latest_processing = pd.to_datetime(chunks_df['processing_timestamp'].max())
        time_since_last = datetime.now() - latest_processing
        print(f"Latest processing: {latest_processing}")
        print(f"Time since last processing: {time_since_last}")

        if time_since_last.total_seconds() < 300:  # 5 minutes
            status = "🟢 Active"
        elif time_since_last.total_seconds() < 1800:  # 30 minutes
            status = "🟡 Recent"
        else:
            status = "🔴 Inactive"

        print(f"System status: {status}")

if __name__ == "__main__":
    print("🔍 RMSAI Dashboard Data Loading Test")
    print("=" * 50)

    # Test database connection
    if not test_database_connection():
        sys.exit(1)

    # Test data loading
    chunks_df, files_df = test_data_loading()

    # Test metrics calculation
    test_dashboard_metrics(chunks_df, files_df)

    print(f"\n✅ Test completed successfully!")