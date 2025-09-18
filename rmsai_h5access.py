#!/usr/bin/env python3
"""
VIOS HDF5 Data Access Patterns
==============================

Comprehensive examples for accessing VIOS v3 HDF5 datasets.
Demonstrates various patterns for reading ECG, PPG, and vital signs data
from event-based physiological monitoring datasets.

Usage:
    python rmsai_h5access.py [hdf5_file_path]

If no file path is provided, the script will look for HDF5 files in the ./data/ directory.
"""

import h5py
import numpy as np
import argparse
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional

def list_available_files(data_dir: str = "data") -> List[str]:
    """List available HDF5 files in the data directory."""
    if not os.path.exists(data_dir):
        return []

    pattern = os.path.join(data_dir, "*.h5")
    return glob.glob(pattern)

def print_file_info(hdf5_file: str) -> None:
    """Print basic information about the HDF5 file."""
    print(f"\n{'='*60}")
    print(f"HDF5 File Information: {os.path.basename(hdf5_file)}")
    print(f"{'='*60}")

    with h5py.File(hdf5_file, 'r') as f:
        # File size
        file_size = os.path.getsize(hdf5_file)
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

        # Metadata
        if 'metadata' in f:
            metadata = f['metadata']
            print(f"\nMetadata:")
            if 'patient_id' in metadata:
                patient_id = metadata['patient_id'][()].decode()
                print(f"  Patient ID: {patient_id}")
            print(f"  ECG sampling rate: {metadata['sampling_rate_ecg'][()]} Hz")
            print(f"  PPG sampling rate: {metadata['sampling_rate_ppg'][()]} Hz")
            print(f"  Respiratory sampling rate: {metadata['sampling_rate_resp'][()]} Hz")
            print(f"  Pre-event duration: {metadata['seconds_before_event'][()]} seconds")
            print(f"  Post-event duration: {metadata['seconds_after_event'][()]} seconds")
            print(f"  Data quality score: {metadata['data_quality_score'][()]:.3f}")
            if 'device_info' in metadata:
                device_info = metadata['device_info'][()].decode()
                print(f"  Device info: {device_info}")

            # First alarm time
            alarm_epoch = metadata['alarm_time_epoch'][()]
            alarm_datetime = datetime.fromtimestamp(alarm_epoch)
            print(f"  First alarm time: {alarm_datetime}")

        # Count events
        event_keys = [k for k in f.keys() if k.startswith('event_')]
        print(f"\nEvents: {len(event_keys)} total")

        # Count conditions
        conditions = {}
        for event_key in event_keys:
            condition = f[event_key].attrs['condition']
            conditions[condition] = conditions.get(condition, 0) + 1

        print("  Condition distribution:")
        for condition, count in sorted(conditions.items()):
            percentage = (count / len(event_keys)) * 100
            print(f"    {condition}: {count} ({percentage:.1f}%)")

def demonstrate_basic_access(hdf5_file: str) -> None:
    """Demonstrate basic data access patterns."""
    print(f"\n{'='*60}")
    print("Basic Data Access Patterns")
    print(f"{'='*60}")

    with h5py.File(hdf5_file, 'r') as f:
        # Access metadata
        ecg_fs = f['metadata']['sampling_rate_ecg'][()]
        ppg_fs = f['metadata']['sampling_rate_ppg'][()]
        pre_event_duration = f['metadata']['seconds_before_event'][()]
        post_event_duration = f['metadata']['seconds_after_event'][()]

        print(f"Sampling rates - ECG: {ecg_fs} Hz, PPG: {ppg_fs} Hz")
        print(f"Event window: {pre_event_duration}s pre + {post_event_duration}s post")

        # Access first event
        event_keys = sorted([k for k in f.keys() if k.startswith('event_')])
        if not event_keys:
            print("No events found in file!")
            return

        first_event = f[event_keys[0]]

        # Basic event info
        condition = first_event.attrs['condition']
        heart_rate = first_event.attrs['heart_rate']
        event_uuid = first_event['uuid'][()].decode()
        event_timestamp = first_event['timestamp'][()]
        event_datetime = datetime.fromtimestamp(event_timestamp)

        print(f"\nFirst Event ({event_keys[0]}):")
        print(f"  Condition: {condition}")
        print(f"  Heart Rate: {heart_rate} bpm")
        print(f"  Timestamp: {event_datetime}")
        print(f"  UUID: {event_uuid}")

        # Access ECG signals
        ecg_leads = ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']
        print(f"\n  ECG Signals ({len(ecg_leads)} leads):")
        for lead in ecg_leads:
            if lead in first_event['ecg']:
                signal = first_event['ecg'][lead][:]
                print(f"    {lead}: {len(signal)} samples, "
                      f"range: {signal.min():.3f} to {signal.max():.3f} mV")

        # Access pacer information with comprehensive analysis
        pacer_data = analyze_pacer_data(first_event)
        if 'info' in pacer_data:
            info = pacer_data['info']
            print(f"\n  Pacer Information: 0x{info['raw_value']:08X}")
            print(f"    Type: {info['type']} ({info['type_name']})")
            if info['type'] > 0:
                print(f"    Rate: {info['rate']} bpm")
                print(f"    Amplitude: {info['amplitude']}")
                print(f"    Status flags: 0x{info['status_flags']:02X}")

        if 'timing' in pacer_data:
            timing = pacer_data['timing']
            print(f"\n  Pacer Timing:")
            print(f"    Offset: {timing['offset_samples']} samples ({timing['offset_seconds']:.3f} seconds)")
            print(f"    Position: {timing['window_percent']:.1f}% through ECG window ({timing['timing_category']} timing)")

        if 'signal_analysis' in pacer_data:
            signal = pacer_data['signal_analysis']
            print(f"    Signal at pacer spike: {signal['pacer_amplitude_at_spike']:.3f} mV")
            if signal['pre_pacer_avg'] is not None and signal['post_pacer_avg'] is not None:
                spike_magnitude = abs(signal['pacer_amplitude_at_spike'] - signal['pre_pacer_avg'])
                print(f"    Spike magnitude: {spike_magnitude:.3f} mV")

        # Access PPG signal
        ppg_signal = first_event['ppg']['PPG'][:]
        print(f"\n  PPG Signal: {len(ppg_signal)} samples, "
              f"range: {ppg_signal.min():.3f} to {ppg_signal.max():.3f} mV")

        # Access Respiratory signal
        resp_signal = first_event['resp']['RESP'][:]
        print(f"\n  Respiratory Signal: {len(resp_signal)} samples, "
              f"range: {resp_signal.min():.1f} to {resp_signal.max():.1f}")

        # Access vital signs
        print(f"\n  Vital Signs:")
        vital_names = ['HR', 'Pulse', 'SpO2', 'Systolic', 'Diastolic', 'RespRate', 'Temp', 'XL_Posture']
        for vital_name in vital_names:
            if vital_name in first_event['vitals']:
                vital_group = first_event['vitals'][vital_name]
                value = vital_group['value'][()]
                units = vital_group['units'][()].decode()
                timestamp = vital_group['timestamp'][()]
                vital_datetime = datetime.fromtimestamp(timestamp)

                print(f"    {vital_name}: {value} {units} at {vital_datetime}")

                # Show thresholds for all vitals except XL_Posture
                if vital_name != 'XL_Posture':
                    if 'upper_threshold' in vital_group and 'lower_threshold' in vital_group:
                        upper_threshold = vital_group['upper_threshold'][()]
                        lower_threshold = vital_group['lower_threshold'][()]
                        print(f"      Thresholds: {lower_threshold} - {upper_threshold} {units}")
                else:
                    # Show special attributes for XL_Posture
                    if 'step_count' in vital_group:
                        step_count = vital_group['step_count'][()]
                        print(f"      Step count: {step_count}")
                    if 'time_since_posture_change' in vital_group:
                        time_since_change = vital_group['time_since_posture_change'][()]
                        print(f"      Time since posture change: {time_since_change} seconds ({time_since_change//60} minutes)")

def analyze_pacer_data(event_group: h5py.Group) -> Dict[str, Any]:
    """Comprehensive analysis of pacer information and timing."""
    pacer_data = {}

    ecg_group = event_group['ecg']

    # Extract pacer information
    if 'pacer_info' in ecg_group:
        pacer_info = ecg_group['pacer_info'][()]

        # Decode bit-packed information
        pacer_type = pacer_info & 0xFF
        pacer_rate = (pacer_info >> 8) & 0xFF
        pacer_amplitude = (pacer_info >> 16) & 0xFF
        status_flags = (pacer_info >> 24) & 0xFF

        pacer_data['info'] = {
            'raw_value': pacer_info,
            'type': pacer_type,
            'type_name': ['None', 'Single', 'Dual', 'Biventricular'][min(pacer_type, 3)],
            'rate': pacer_rate if pacer_type > 0 else None,
            'amplitude': pacer_amplitude if pacer_type > 0 else None,
            'status_flags': status_flags if pacer_type > 0 else None
        }

    # Extract pacer timing
    if 'pacer_offset' in ecg_group:
        pacer_offset = ecg_group['pacer_offset'][()]
        time_offset = pacer_offset / 200.0  # Convert to seconds (200 Hz ECG)

        # Analyze timing characteristics
        window_percent = (pacer_offset / 2400.0) * 100  # Percentage through 12-second window

        if window_percent <= 25:
            timing_category = "Early"
        elif window_percent >= 75:
            timing_category = "Late"
        else:
            timing_category = "Mid"

        pacer_data['timing'] = {
            'offset_samples': pacer_offset,
            'offset_seconds': time_offset,
            'window_percent': window_percent,
            'timing_category': timing_category,
            'max_samples': 2400
        }

    # Extract ECG signal at pacer location for analysis
    if 'pacer_offset' in ecg_group and 'ECG1' in ecg_group:
        ecg_signal = ecg_group['ECG1'][:]
        if pacer_offset < len(ecg_signal):
            # Get signal values around pacer spike
            start_idx = max(0, pacer_offset - 10)
            end_idx = min(len(ecg_signal), pacer_offset + 10)

            pacer_data['signal_analysis'] = {
                'pacer_amplitude_at_spike': ecg_signal[pacer_offset],
                'surrounding_signal': ecg_signal[start_idx:end_idx],
                'pre_pacer_avg': ecg_signal[start_idx:pacer_offset].mean() if pacer_offset > start_idx else None,
                'post_pacer_avg': ecg_signal[pacer_offset+1:end_idx].mean() if pacer_offset+1 < end_idx else None
            }

    return pacer_data

def extract_vitals_from_event(event_group: h5py.Group) -> Dict[str, Any]:
    """Extract all vital signs from an event group."""
    vitals = {}
    vital_names = ['HR', 'Pulse', 'SpO2', 'Systolic', 'Diastolic', 'RespRate', 'Temp', 'XL_Posture']

    for vital_name in vital_names:
        if vital_name in event_group['vitals']:
            vital_group = event_group['vitals'][vital_name]
            vital_data = {
                'value': vital_group['value'][()],
                'units': vital_group['units'][()].decode(),
                'timestamp': vital_group['timestamp'][()],
                'datetime': datetime.fromtimestamp(vital_group['timestamp'][()])
            }

            # Add thresholds for all vitals except XL_Posture
            if vital_name != 'XL_Posture':
                if 'upper_threshold' in vital_group:
                    vital_data['upper_threshold'] = vital_group['upper_threshold'][()]
                if 'lower_threshold' in vital_group:
                    vital_data['lower_threshold'] = vital_group['lower_threshold'][()]
            else:
                # Add special attributes for XL_Posture
                if 'step_count' in vital_group:
                    vital_data['step_count'] = vital_group['step_count'][()]
                if 'time_since_posture_change' in vital_group:
                    vital_data['time_since_posture_change'] = vital_group['time_since_posture_change'][()]

            vitals[vital_name] = vital_data

    return vitals

def demonstrate_event_iteration(hdf5_file: str) -> None:
    """Demonstrate iterating through all events."""
    print(f"\n{'='*60}")
    print("Event Iteration Patterns")
    print(f"{'='*60}")

    with h5py.File(hdf5_file, 'r') as f:
        event_keys = sorted([k for k in f.keys() if k.startswith('event_')])
        print(f"Found {len(event_keys)} events\n")

        for i, event_key in enumerate(event_keys[:5]):  # Show first 5 events
            event = f[event_key]
            condition = event.attrs['condition']
            hr = event.attrs['heart_rate']
            uuid = event['uuid'][()].decode()
            timestamp = event['timestamp'][()]
            event_datetime = datetime.fromtimestamp(timestamp)

            print(f"{event_key}: {condition} (HR: {hr} bpm) - {uuid[:8]}... at {event_datetime}")

        if len(event_keys) > 5:
            print(f"... and {len(event_keys) - 5} more events")

def demonstrate_condition_filtering(hdf5_file: str) -> None:
    """Demonstrate filtering events by medical condition."""
    print(f"\n{'='*60}")
    print("Filtering Events by Condition")
    print(f"{'='*60}")

    def filter_events_by_condition(f: h5py.File, target_condition: str) -> List[Dict]:
        """Filter events by specific medical condition."""
        matching_events = []

        for event_key in [k for k in f.keys() if k.startswith('event_')]:
            event = f[event_key]
            condition = event.attrs['condition']

            if condition == target_condition:
                matching_events.append({
                    'event_id': event_key,
                    'heart_rate': event.attrs['heart_rate'],
                    'timestamp': event['timestamp'][()],
                    'uuid': event['uuid'][()].decode()
                })

        return matching_events

    with h5py.File(hdf5_file, 'r') as f:
        # Get all unique conditions
        conditions = set()
        for event_key in [k for k in f.keys() if k.startswith('event_')]:
            conditions.add(f[event_key].attrs['condition'])

        # Filter and display for each condition
        for condition in sorted(conditions):
            matching_events = filter_events_by_condition(f, condition)
            print(f"\n{condition}: {len(matching_events)} events")

            if matching_events:
                # Show statistics
                heart_rates = [event['heart_rate'] for event in matching_events]
                avg_hr = np.mean(heart_rates)
                min_hr = np.min(heart_rates)
                max_hr = np.max(heart_rates)

                print(f"  Heart rate - Avg: {avg_hr:.1f}, Range: {min_hr}-{max_hr} bpm")

                # Show first few events
                for event in matching_events[:3]:
                    event_time = datetime.fromtimestamp(event['timestamp'])
                    print(f"    {event['event_id']}: HR {event['heart_rate']} at {event_time}")

                if len(matching_events) > 3:
                    print(f"    ... and {len(matching_events) - 3} more")

def demonstrate_batch_processing(hdf5_file: str) -> None:
    """Demonstrate batch processing of multiple events."""
    print(f"\n{'='*60}")
    print("Batch Processing Example")
    print(f"{'='*60}")

    def process_all_events(f: h5py.File) -> List[Dict]:
        """Process all events and extract key information."""
        results = []

        ecg_fs = f['metadata']['sampling_rate_ecg'][()]

        for event_key in sorted([k for k in f.keys() if k.startswith('event_')]):
            event = f[event_key]

            # Extract basic event info
            event_data = {
                'event_id': event_key,
                'condition': event.attrs['condition'],
                'heart_rate': event.attrs['heart_rate'],
                'uuid': event['uuid'][()].decode(),
                'timestamp': event['timestamp'][()],
                'datetime': datetime.fromtimestamp(event['timestamp'][()]),
            }

            # Extract ECG statistics
            ecg_lead_i = event['ecg']['ECG1'][:]
            event_data['ecg_stats'] = {
                'samples': len(ecg_lead_i),
                'mean': np.mean(ecg_lead_i),
                'std': np.std(ecg_lead_i),
                'min': np.min(ecg_lead_i),
                'max': np.max(ecg_lead_i),
                'sampling_rate': ecg_fs
            }

            # Extract vitals summary
            vitals = extract_vitals_from_event(event)
            event_data['vitals_summary'] = {
                name: data['value'] for name, data in vitals.items()
            }

            results.append(event_data)

        return results

    with h5py.File(hdf5_file, 'r') as f:
        events = process_all_events(f)

        print(f"Processed {len(events)} events")

        # Show summary statistics
        conditions = [e['condition'] for e in events]
        heart_rates = [e['heart_rate'] for e in events]

        print(f"\nSummary Statistics:")
        print(f"  Total events: {len(events)}")
        print(f"  Unique conditions: {len(set(conditions))}")
        print(f"  Heart rate range: {min(heart_rates):.1f} - {max(heart_rates):.1f} bpm")
        print(f"  Average heart rate: {np.mean(heart_rates):.1f} bpm")

        # Show ECG signal statistics
        ecg_means = [e['ecg_stats']['mean'] for e in events]
        ecg_stds = [e['ecg_stats']['std'] for e in events]

        print(f"\nECG Signal Statistics (Lead I):")
        print(f"  Mean amplitude: {np.mean(ecg_means):.3f} ± {np.std(ecg_means):.3f} mV")
        print(f"  Signal noise (avg std): {np.mean(ecg_stds):.3f} mV")

        # Show sample processed data
        print(f"\nSample Processed Events:")
        for event in events[:3]:
            print(f"  {event['event_id']}: {event['condition']}")
            print(f"    HR: {event['vitals_summary'].get('HR', 'N/A')} bpm, "
                  f"SpO2: {event['vitals_summary'].get('SpO2', 'N/A')}%")
            print(f"    ECG: {event['ecg_stats']['samples']} samples, "
                  f"amplitude: {event['ecg_stats']['mean']:.3f} ± {event['ecg_stats']['std']:.3f} mV")

def demonstrate_time_analysis(hdf5_file: str) -> None:
    """Demonstrate time-based analysis of vital measurements."""
    print(f"\n{'='*60}")
    print("Time-Based Analysis")
    print(f"{'='*60}")

    with h5py.File(hdf5_file, 'r') as f:
        event_keys = sorted([k for k in f.keys() if k.startswith('event_')])

        if len(event_keys) < 2:
            print("Need at least 2 events for time analysis")
            return

        print("Analyzing vital measurement timing relative to alarm events...\n")

        # Analyze timing for first few events
        for event_key in event_keys[:3]:
            event = f[event_key]
            event_timestamp = event['timestamp'][()]
            event_datetime = datetime.fromtimestamp(event_timestamp)
            condition = event.attrs['condition']

            print(f"{event_key} ({condition}) - Alarm at {event_datetime}:")

            # Get vital timestamps and calculate offsets
            vital_names = ['HR', 'Pulse', 'SpO2', 'Systolic', 'Diastolic', 'RespRate', 'Temp']
            for vital_name in vital_names:
                if vital_name in event['vitals']:
                    vital_group = event['vitals'][vital_name]
                    vital_timestamp = vital_group['timestamp'][()]
                    vital_value = vital_group['value'][()]

                    # Calculate offset from alarm time
                    offset = vital_timestamp - event_timestamp

                    print(f"  {vital_name}: {vital_value} (offset: {offset:+.1f}s)")
            print()

def validate_file_structure(hdf5_file: str) -> bool:
    """Validate the structure of VIOS v3 HDF5 file."""
    print(f"\n{'='*60}")
    print("File Structure Validation")
    print(f"{'='*60}")

    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Check metadata
            if 'metadata' not in f:
                print("❌ Missing metadata group")
                return False

            metadata = f['metadata']
            required_metadata = [
                'patient_id', 'sampling_rate_ecg', 'sampling_rate_ppg', 'sampling_rate_resp',
                'alarm_time_epoch', 'alarm_offset_seconds', 'seconds_before_event',
                'seconds_after_event', 'data_quality_score', 'device_info'
            ]

            for field in required_metadata:
                if field not in metadata:
                    print(f"❌ Missing metadata field: {field}")
                    return False

            print("✓ Metadata structure valid")

            # Check events
            event_keys = [k for k in f.keys() if k.startswith('event_')]
            if len(event_keys) == 0:
                print("❌ No events found")
                return False

            print(f"✓ Found {len(event_keys)} events")

            # Validate event structure
            for i, event_key in enumerate(event_keys[:3]):  # Check first 3 events
                event = f[event_key]

                # Check required groups
                required_groups = ['ecg', 'ppg', 'resp', 'vitals']
                for group in required_groups:
                    if group not in event:
                        print(f"❌ Missing {group} group in {event_key}")
                        return False

                # Check required datasets
                if 'timestamp' not in event or 'uuid' not in event:
                    print(f"❌ Missing timestamp or uuid in {event_key}")
                    return False

                # Check ECG leads
                ecg_leads = ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']
                for lead in ecg_leads:
                    if lead not in event['ecg']:
                        print(f"❌ Missing ECG lead {lead} in {event_key}")
                        return False

                # Check pacer info (optional)
                if 'pacer_info' in event['ecg']:
                    print(f"✓ Pacer info found in {event_key}")

                if 'pacer_offset' in event['ecg']:
                    print(f"✓ Pacer offset found in {event_key}")

                # Check respiratory signal (required)
                if 'RESP' not in event['resp']:
                    print(f"❌ Missing RESP signal in {event_key}")
                    return False

                # Check PPG
                if 'PPG' not in event['ppg']:
                    print(f"❌ Missing PPG signal in {event_key}")
                    return False

                # Check vitals
                vital_names = ['HR', 'Pulse', 'SpO2', 'Systolic', 'Diastolic', 'RespRate', 'Temp', 'XL_Posture']
                for vital in vital_names:
                    if vital not in event['vitals']:
                        print(f"❌ Missing vital {vital} in {event_key}")
                        return False

                    vital_group = event['vitals'][vital]
                    if 'value' not in vital_group or 'units' not in vital_group or 'timestamp' not in vital_group:
                        print(f"❌ Missing value, units, or timestamp for {vital} in {event_key}")
                        return False

                if i == 0:  # Only print for first event
                    print(f"✓ Event structure valid (checked {event_key})")

            print("✓ HDF5 file structure validation passed")
            return True

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

def main():
    """Main function to demonstrate HDF5 access patterns."""
    parser = argparse.ArgumentParser(
        description="Demonstrate VIOS HDF5 data access patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python rmsai_h5access.py                          # Use first file found in ./data/
    python rmsai_h5access.py data/PT1234_2025-09.h5  # Use specific file
    python rmsai_h5access.py --list                   # List available files
        """
    )

    parser.add_argument('hdf5_file', nargs='?', help='Path to HDF5 file')
    parser.add_argument('--list', action='store_true', help='List available HDF5 files')

    args = parser.parse_args()

    # List files if requested
    if args.list:
        files = list_available_files()
        if files:
            print("Available HDF5 files:")
            for f in files:
                print(f"  {f}")
        else:
            print("No HDF5 files found in ./data/ directory")
        return

    # Determine which file to use
    hdf5_file = args.hdf5_file

    if not hdf5_file:
        # Look for files in data directory
        files = list_available_files()
        if not files:
            print("No HDF5 files found in ./data/ directory")
            print("Generate some data first with: python sim-hdf5-v3.py")
            return

        hdf5_file = files[0]
        print(f"Using first available file: {hdf5_file}")

    # Check if file exists
    if not os.path.exists(hdf5_file):
        print(f"Error: File not found: {hdf5_file}")
        return

    # Run demonstrations
    try:
        print(f"VIOS HDF5 Data Access Demonstration")
        print(f"File: {hdf5_file}")

        # Validate file structure first
        if not validate_file_structure(hdf5_file):
            print("File structure validation failed. Cannot proceed.")
            return

        # Run all demonstrations
        print_file_info(hdf5_file)
        demonstrate_basic_access(hdf5_file)
        demonstrate_event_iteration(hdf5_file)
        demonstrate_condition_filtering(hdf5_file)
        demonstrate_batch_processing(hdf5_file)
        demonstrate_time_analysis(hdf5_file)

        print(f"\n{'='*60}")
        print("Access Pattern Demonstration Complete")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error accessing HDF5 file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()