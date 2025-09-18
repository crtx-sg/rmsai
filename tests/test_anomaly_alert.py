#!/usr/bin/env python3
"""
Test script to verify anomaly alert functionality
"""

import sqlite3
import pandas as pd

def test_anomaly_calculation():
    """Test the same anomaly calculation logic as the dashboard"""
    db_path = "rmsai_metadata.db"

    try:
        with sqlite3.connect(db_path) as conn:
            chunks_df = pd.read_sql_query("""
                SELECT chunk_id, event_id, source_file, lead_name,
                       anomaly_status, anomaly_type, error_score,
                       processing_timestamp, vector_id
                FROM chunks
                ORDER BY processing_timestamp DESC
                LIMIT 5000
            """, conn)

            print(f"📊 Anomaly Alert Test Results:")
            print(f"Total chunks loaded: {len(chunks_df)}")

            if len(chunks_df) > 0:
                # Same calculation as dashboard
                total_chunks = len(chunks_df)
                total_anomalies = len(chunks_df[chunks_df['anomaly_status'] == 'anomaly'])
                anomaly_rate = (total_anomalies / total_chunks * 100) if total_chunks > 0 else 0

                print(f"Total chunks: {total_chunks:,}")
                print(f"Total anomalies: {total_anomalies:,}")
                print(f"Anomaly rate: {anomaly_rate:.1f}%")
                print(f"Alert threshold: 20%")
                print(f"Should show alert: {anomaly_rate > 20}")

                # Check anomaly_status values
                status_counts = chunks_df['anomaly_status'].value_counts()
                print(f"\nAnomaly status distribution:")
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")

                # Test the actual alert condition
                if anomaly_rate > 20:
                    alert_text = f"⚠️ High Anomaly Rate Alert: {anomaly_rate:.1f}% of recent chunks detected as anomalies. This may indicate a system issue or unusual patient conditions."
                    print(f"\n✅ Alert should display:")
                    print(f"'{alert_text}'")
                else:
                    print(f"\n❌ Alert should NOT display (rate <= 20%)")

            else:
                print("❌ No data found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_anomaly_calculation()