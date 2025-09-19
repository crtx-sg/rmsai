#!/usr/bin/env python3
"""
Script to show detailed vitals data from HDF5 file
"""

import h5py
import pandas as pd

def show_vitals_details(filename):
    """Show detailed vitals data from HDF5 file"""
    print(f"🩺 Detailed Vitals Information: {filename}")
    print("=" * 60)

    try:
        with h5py.File(filename, 'r') as f:
            # Look at first event for vitals structure
            event_key = 'event_1001'
            if event_key in f:
                event_group = f[event_key]

                # Event level information
                print(f"📋 Event: {event_key}")
                print(f"   Condition: {event_group.attrs.get('condition', 'Unknown')}")
                print(f"   Heart Rate: {event_group.attrs.get('heart_rate', 'Unknown')} BPM")
                print(f"   Timestamp: {event_group.attrs.get('event_timestamp', 'Unknown')}")
                print()

                if 'vitals' in event_group:
                    vitals_group = event_group['vitals']
                    print("🩺 Vitals Data Structure:")

                    vitals_data = []

                    for vital_name in vitals_group.keys():
                        vital_group = vitals_group[vital_name]
                        print(f"   📊 {vital_name}:")

                        vital_info = {}
                        vital_info['Vital'] = vital_name

                        for dataset_name in vital_group.keys():
                            try:
                                value = vital_group[dataset_name][()]
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                                vital_info[dataset_name] = value
                                print(f"      {dataset_name}: {value}")
                            except:
                                print(f"      {dataset_name}: [Unable to read]")

                        vitals_data.append(vital_info)
                        print()

                    # Create a summary table
                    if vitals_data:
                        print("📋 Vitals Summary Table:")
                        df = pd.DataFrame(vitals_data)
                        print(df.to_string(index=False))
                        print()

                # Show ECG data info
                if 'ecg' in event_group:
                    ecg_group = event_group['ecg']
                    print("📈 ECG Lead Information:")
                    for lead_name in ecg_group.keys():
                        if not lead_name.startswith('pacer'):
                            dataset = ecg_group[lead_name]
                            print(f"   {lead_name}: {dataset.shape} samples, dtype: {dataset.dtype}")
                    print()

                # Show waveform data
                if 'ppg' in event_group:
                    ppg_group = event_group['ppg']
                    print("🫀 PPG Data:")
                    for key in ppg_group.keys():
                        dataset = ppg_group[key]
                        print(f"   {key}: {dataset.shape} samples, dtype: {dataset.dtype}")
                    print()

                if 'resp' in event_group:
                    resp_group = event_group['resp']
                    print("🫁 Respiratory Data:")
                    for key in resp_group.keys():
                        dataset = resp_group[key]
                        print(f"   {key}: {dataset.shape} samples, dtype: {dataset.dtype}")
                    print()

    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")

if __name__ == "__main__":
    show_vitals_details("data/PT3079_2025-09.h5")