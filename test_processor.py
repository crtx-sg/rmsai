#!/usr/bin/env python3
"""
Test script for RMSAI LSTM Autoencoder Processor
"""

import time
import threading
from pathlib import Path
from rmsai_lstm_autoencoder_proc import RMSAIProcessor

def test_processor():
    """Test the RMSAI processor with existing files"""
    print("Testing RMSAI LSTM Autoencoder Processor")
    print("=" * 50)

    # Initialize processor
    processor = RMSAIProcessor()

    # Test with existing files
    print("\n1. Processing existing files...")
    processor.process_existing_files()

    # Get initial stats
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    print(f"Total chunks processed: {stats.get('total_chunks', 0)}")
    print(f"Anomalies detected: {stats.get('anomaly_counts', {}).get('anomaly', 0)}")
    print(f"Normal chunks: {stats.get('anomaly_counts', {}).get('normal', 0)}")
    print(f"Files processed: {stats.get('files_processed', 0)}")
    print(f"Average error score: {stats.get('avg_error_score', 0):.4f}")

    # Test file monitoring for a short time
    print(f"\n2. Testing file monitoring (10 seconds)...")
    print("Generate a new HDF5 file to test real-time processing")

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=processor.start)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for 10 seconds
    time.sleep(10)

    # Stop monitoring
    processor.stop()

    # Final stats
    final_stats = processor.get_processing_stats()
    print(f"\nFinal Statistics:")
    print(f"Total chunks processed: {final_stats.get('total_chunks', 0)}")
    print(f"Files processed: {final_stats.get('files_processed', 0)}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_processor()