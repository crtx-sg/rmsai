#!/usr/bin/env python3
"""
ECG Chunking Strategy Analysis
=============================

Analyzes how ECG events are chunked for processing with the LSTM autoencoder model.
Shows the relationship between ECG sample count, chunking parameters, and resulting chunks.
"""

import sqlite3
import pandas as pd
import numpy as np

def analyze_chunking_strategy():
    """Analyze the ECG chunking strategy used in RMSAI"""

    print("=" * 70)
    print("ECG CHUNKING STRATEGY ANALYSIS")
    print("=" * 70)

    # Configuration from the processing pipeline
    ECG_SAMPLES_PER_EVENT = 2400  # 12 seconds at 200Hz
    CHUNK_SIZE = 140  # seq_len for LSTM model
    STEP_SIZE = CHUNK_SIZE // 2  # 50% overlap = 70

    print(f"\n📊 CHUNKING CONFIGURATION:")
    print(f"   ECG Samples per Event: {ECG_SAMPLES_PER_EVENT}")
    print(f"   Model Chunk Size: {CHUNK_SIZE} samples")
    print(f"   Step Size (Overlap): {STEP_SIZE} samples (50% overlap)")
    print(f"   Sampling Rate: 200 Hz")
    print(f"   Event Duration: {ECG_SAMPLES_PER_EVENT / 200} seconds")
    print(f"   Chunk Duration: {CHUNK_SIZE / 200} seconds")

    # Calculate theoretical number of chunks
    max_chunks_per_lead = (ECG_SAMPLES_PER_EVENT - CHUNK_SIZE) // STEP_SIZE + 1

    print(f"\n🔢 THEORETICAL CHUNKING:")
    print(f"   Max chunks per lead: {max_chunks_per_lead}")
    print(f"   Formula: (2400 - 140) // 70 + 1 = {max_chunks_per_lead}")
    print(f"   Last chunk starts at sample: {(max_chunks_per_lead - 1) * STEP_SIZE}")
    print(f"   Last chunk ends at sample: {(max_chunks_per_lead - 1) * STEP_SIZE + CHUNK_SIZE}")

    # Load actual data from database
    try:
        with sqlite3.connect("rmsai_metadata.db") as conn:
            # Get chunking statistics
            stats_query = """
            SELECT
                COUNT(DISTINCT event_id) as total_events,
                COUNT(DISTINCT lead_name) as total_leads,
                COUNT(*) as total_chunks,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT event_id), 1) as avg_chunks_per_event,
                ROUND(COUNT(*) * 1.0 / (COUNT(DISTINCT event_id) * COUNT(DISTINCT lead_name)), 1) as avg_chunks_per_lead
            FROM chunks
            """

            stats = pd.read_sql_query(stats_query, conn).iloc[0]

            print(f"\n📈 ACTUAL PROCESSING RESULTS:")
            print(f"   Total Events: {int(stats['total_events'])}")
            print(f"   Total Leads: {int(stats['total_leads'])}")
            print(f"   Total Chunks: {int(stats['total_chunks'])}")
            print(f"   Avg Chunks per Event: {stats['avg_chunks_per_event']}")
            print(f"   Avg Chunks per Lead: {stats['avg_chunks_per_lead']}")

            # Per-event analysis
            event_query = """
            SELECT
                event_id,
                COUNT(*) as total_chunks,
                COUNT(DISTINCT lead_name) as leads_processed,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT lead_name), 1) as chunks_per_lead
            FROM chunks
            GROUP BY event_id
            ORDER BY event_id
            """

            events_df = pd.read_sql_query(event_query, conn)

            print(f"\n📋 PER-EVENT BREAKDOWN:")
            print("   Event ID     | Total Chunks | Leads | Chunks/Lead")
            print("   " + "-" * 50)
            for _, row in events_df.iterrows():
                print(f"   {row['event_id']:<12} |     {int(row['total_chunks']):<8} |   {int(row['leads_processed']):<3} |    {row['chunks_per_lead']:<5}")

            # Lead analysis for one event
            lead_query = """
            SELECT
                lead_name,
                COUNT(*) as chunk_count
            FROM chunks
            WHERE event_id = 'event_1001'
            GROUP BY lead_name
            ORDER BY lead_name
            """

            leads_df = pd.read_sql_query(lead_query, conn)

            print(f"\n🎛️  CHUNKS PER LEAD (Sample Event: event_1001):")
            print("   Lead Name | Chunks")
            print("   " + "-" * 18)
            for _, row in leads_df.iterrows():
                print(f"   {row['lead_name']:<9} |   {int(row['chunk_count'])}")

            # Validate chunking logic
            actual_chunks_per_lead = int(leads_df['chunk_count'].iloc[0])

            print(f"\n✅ VALIDATION:")
            print(f"   Expected chunks per lead: {max_chunks_per_lead}")
            print(f"   Actual chunks per lead: {actual_chunks_per_lead}")

            if actual_chunks_per_lead == max_chunks_per_lead:
                print("   ✓ Chunking matches theoretical calculation!")
            else:
                print("   ⚠️  Chunking differs from theoretical - checking logic...")

                # Check if it's due to filtering or processing constraints
                print(f"\n   📝 POSSIBLE REASONS FOR DIFFERENCE:")
                print(f"      - Processing may limit chunks to avoid incomplete windows")
                print(f"      - Quality filtering may remove some chunks")
                print(f"      - Edge case handling in the sliding window")

    except Exception as e:
        print(f"Error accessing database: {e}")

    # Coverage analysis
    print(f"\n📊 COVERAGE ANALYSIS:")
    actual_chunks_per_lead = 10  # From our data
    coverage_samples = (actual_chunks_per_lead - 1) * STEP_SIZE + CHUNK_SIZE
    coverage_percentage = (coverage_samples / ECG_SAMPLES_PER_EVENT) * 100

    print(f"   Samples covered: {coverage_samples} out of {ECG_SAMPLES_PER_EVENT}")
    print(f"   Coverage: {coverage_percentage:.1f}%")
    print(f"   Uncovered samples: {ECG_SAMPLES_PER_EVENT - coverage_samples}")
    print(f"   Uncovered duration: {(ECG_SAMPLES_PER_EVENT - coverage_samples) / 200:.2f} seconds")

    print(f"\n🔄 OVERLAP ANALYSIS:")
    overlap_samples = CHUNK_SIZE - STEP_SIZE
    overlap_percentage = (overlap_samples / CHUNK_SIZE) * 100
    print(f"   Overlap per chunk: {overlap_samples} samples ({overlap_percentage:.0f}%)")
    print(f"   Non-overlapping portion: {STEP_SIZE} samples per chunk")
    print(f"   Total unique coverage: {(actual_chunks_per_lead * STEP_SIZE + overlap_samples)} samples")

def analyze_chunk_positions():
    """Show the exact positioning of chunks within ECG events"""

    CHUNK_SIZE = 140
    STEP_SIZE = 70
    ECG_SAMPLES = 2400

    print(f"\n" + "=" * 70)
    print("CHUNK POSITIONING ANALYSIS")
    print("=" * 70)

    print(f"\n📍 CHUNK POSITIONS (first 10 chunks):")
    print("   Chunk # | Start | End   | Samples | Duration")
    print("   " + "-" * 45)

    for i in range(10):
        start = i * STEP_SIZE
        end = start + CHUNK_SIZE

        if end <= ECG_SAMPLES:
            duration = CHUNK_SIZE / 200
            print(f"   {i+1:7} | {start:5} | {end:5} | {CHUNK_SIZE:7} | {duration:6.2f}s")
        else:
            print(f"   {i+1:7} | Would exceed ECG data length")
            break

if __name__ == "__main__":
    analyze_chunking_strategy()
    analyze_chunk_positions()