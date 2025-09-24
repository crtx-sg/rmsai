# This program converts the Vios vdx csv file to hdf5 file format
# Enhanced for RMSAI EWS (Early Warning System) compatibility
# ganesh
import pandas as pd
import h5py
import numpy as np
from datetime import datetime, timedelta
import uuid
import sys
import json
import os
import random

# --- JSON Serialization Helper ---
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def generate_vital_history(vital_name, current_value, current_timestamp, num_samples=20):
    """
    Generate realistic historical vital signs data for EWS trend analysis

    Args:
        vital_name: Name of the vital sign
        current_value: Current vital sign value
        current_timestamp: Current timestamp (epoch)
        num_samples: Number of historical samples to generate

    Returns:
        List of historical data points with timestamp, value
    """
    history = []

    # Define realistic variation ranges for different vitals
    variation_ranges = {
        'HR': {'min_change': -15, 'max_change': 20, 'volatility': 5},
        'Pulse': {'min_change': -15, 'max_change': 20, 'volatility': 5},
        'RespRate': {'min_change': -5, 'max_change': 8, 'volatility': 2},
        'SpO2': {'min_change': -8, 'max_change': 3, 'volatility': 2},
        'Systolic': {'min_change': -20, 'max_change': 25, 'volatility': 8},
        'Diastolic': {'min_change': -15, 'max_change': 20, 'volatility': 6},
        'Temp': {'min_change': -3.0, 'max_change': 4.0, 'volatility': 1.0},
        'XL_Posture': {'min_change': -30, 'max_change': 30, 'volatility': 10}
    }

    # Get variation parameters for this vital
    params = variation_ranges.get(vital_name, {'min_change': -10, 'max_change': 10, 'volatility': 3})

    # Generate time intervals (15-30 minutes apart, going backwards from current time)
    base_timestamp = current_timestamp

    for i in range(num_samples):
        # Time going backward (15-30 minutes intervals)
        time_offset = random.randint(900, 1800) * (i + 1)  # 15-30 minutes in seconds
        hist_timestamp = base_timestamp - time_offset

        # Generate realistic value variation
        if i == 0:
            # First historical point (closest to current) - small variation
            variation = random.uniform(-params['volatility'], params['volatility'])
        else:
            # More variation for older data, with some trend
            base_variation = random.uniform(params['min_change'], params['max_change'])
            noise = random.uniform(-params['volatility'], params['volatility'])
            variation = (base_variation * 0.1 * i) + noise

        hist_value = current_value + variation

        # Apply reasonable bounds for different vitals
        if vital_name in ['HR', 'Pulse']:
            hist_value = max(30, min(200, hist_value))
        elif vital_name == 'RespRate':
            hist_value = max(8, min(40, hist_value))
        elif vital_name == 'SpO2':
            hist_value = max(70, min(100, hist_value))
        elif vital_name == 'Systolic':
            hist_value = max(60, min(250, hist_value))
        elif vital_name == 'Diastolic':
            hist_value = max(40, min(150, hist_value))
        elif vital_name == 'Temp':
            hist_value = max(95.0, min(110.0, hist_value))

        history.append({
            'timestamp': int(hist_timestamp),
            'value': round(float(hist_value), 2) if vital_name == 'Temp' else int(hist_value)
        })

    # Sort by timestamp (oldest first)
    history.sort(key=lambda x: x['timestamp'])

    return history

# --- Main Conversion Logic ---
def convert_csv_to_hdf5(csv_path, enable_ews=True, history_samples=20):
    """
    Converts physiological data from a CSV file to a structured HDF5 file.

    Args:
        csv_path (str): Path to the input CSV file
        enable_ews (bool): Enable EWS enhancements (default: True)
        history_samples (int): Number of historical samples to generate (default: 20)

    Enhanced for RMSAI EWS (Early Warning System) compatibility:
    - Adds historical vital signs data for trend analysis
    - Includes extras JSON field with generated history
    - Compatible with NEWS2-based EWS scoring
    - Supports linear regression trend detection
    - Enables dashboard visualization and clinical decision support
    """
    print(f"Starting conversion of '{csv_path}'...")

    try:
        # Generate HDF5 filename from CSV path
        patient_id = os.path.basename(csv_path).split('_')[0]
        current_month = datetime.now().strftime('%Y-%m')
        output_dir = os.path.dirname(csv_path)
        hdf5_filename = f'{patient_id}_{current_month}.h5'
        hdf5_path = os.path.join(output_dir, hdf5_filename) if output_dir else hdf5_filename

        # Load the CSV file
        df = pd.read_csv(csv_path, skipinitialspace=True)
        print("CSV file loaded successfully.")

        # Clean column names by stripping leading/trailing spaces
        df.columns = [col.strip() for col in df.columns]

        with h5py.File(hdf5_path, 'w') as hf:
            print(f"HDF5 file '{hdf5_path}' created.")

            # --- Enhanced Metadata Group ---
            metadata = hf.create_group('metadata')
            metadata.create_dataset('patient_id', data=str(patient_id))
            metadata.create_dataset('sampling_rate_ecg', data=200.0)
            metadata.create_dataset('sampling_rate_ppg', data=75.0)
            metadata.create_dataset('sampling_rate_resp', data=33.33)
            metadata.create_dataset('alarm_time_epoch', data=convert_numpy_types(df['Epoch Time'].iloc[0]))
            metadata.create_dataset('alarm_offset_seconds', data=6.0)
            metadata.create_dataset('seconds_before_event', data=6.0)
            metadata.create_dataset('seconds_after_event', data=6.0)
            metadata.create_dataset('data_quality_score', data=0.95)
            metadata.create_dataset('device_info', data="RMSAI-VDX-Converter-v2.0")

            # EWS-specific metadata
            metadata.create_dataset('max_vital_history', data=history_samples)  # EWS trend analysis history length
            metadata.create_dataset('ews_enabled', data=1 if enable_ews else 0)  # Flag for EWS compatibility
            metadata.create_dataset('converter_version', data="2.0.0")
            metadata.create_dataset('conversion_timestamp', data=convert_numpy_types(int(datetime.now().timestamp())))

            print("Enhanced metadata group created with EWS compatibility.")

            # --- Event Group ---
            event_group = hf.create_group('event_001')
            event_group.create_dataset('timestamp', data=convert_numpy_types(df['Epoch Time'].iloc[0]))
            event_group.create_dataset('uuid', data=str(uuid.uuid4()))
            event_group.attrs['condition'] = "Auto-generated"

            hr_value = 0
            if 'HR' in df.columns:
                hr_series = df['HR'].replace(0, np.nan).dropna()
                if not hr_series.empty:
                    # Explicitly cast to standard Python int
                    hr_value = convert_numpy_types(hr_series.iloc[0])
            event_group.attrs['heart_rate'] = hr_value
            print("Event group 'event_001' created with all required components.")

            # --- ECG, PPG, and RESP Data ---
            ecg_group = event_group.create_group('ecg')
            ecg_leads = ['ECG1 #', 'ECG2 #', 'ECG3 #', 'aVL #', 'aVR #', 'aVF #', 'vVX #']
            for lead in ecg_leads:
                clean_lead = lead.replace(' #', '')
                ecg_group.create_dataset(clean_lead, data=df[lead].head(2400).values, compression='gzip')

            # Store pacer_info as a single standard integer
            ecg_group.create_dataset('pacer_info', data=0, dtype='i4')

            ppg_group = event_group.create_group('ppg')
            ppg_group.create_dataset('PPG', data=df['SpO2 #'].head(900).values, compression='gzip')

            resp_group = event_group.create_group('resp')
            resp_group.create_dataset('RESP', data=df['RESP #'].head(400).values, compression='gzip')
            print("ECG, PPG, and RESP data processed and stored.")

            # --- Vitals Data ---
            vitals_group = event_group.create_group('vitals')
            vitals_map = {
                'HR':       {'col': 'HR', 'units': 'bpm', 'low_col': 'HR threshold Low', 'high_col': 'HR threshold High'},
                'Pulse':    {'col': 'Pulse Rate', 'units': 'bpm', 'low_col': 'HR threshold Low', 'high_col': 'HR threshold High'},
                'RespRate': {'col': 'RR', 'units': 'bpm', 'low_col': 'RR threshold Low', 'high_col': 'RR threshold High'},
                'SpO2':     {'col': 'SpO2', 'units': '%', 'low_col': 'SpO2 threshold Low', 'high_col': 'SpO2 threshold High'},
                'Systolic': {'col': 'Systolic', 'units': 'mmHg', 'low_col': 'BP systolic threshold Low', 'high_col': 'BP Systolic threshold High'},
                'Diastolic':{'col': 'Diastolic', 'units': 'mmHg', 'low_col': 'BP Diastolic threshold Low', 'high_col': 'BP Diastolic threshold High'},
                'Temp':     {'col': 'Temp degree F', 'units': 'F', 'low_col': 'Temp threshold Low', 'high_col': 'Temp threshold High'},
            }

            for vital_name, info in vitals_map.items():
                value, timestamp = (0, convert_numpy_types(df['Epoch Time'].iloc[0]))

                if info['col'] in df.columns:
                    vital_series = df[info['col']].replace(0, np.nan).dropna()
                    if not vital_series.empty:
                        first_valid_index = vital_series.index[0]
                        value = vital_series.iloc[0]
                        timestamp = df['Epoch Time'].iloc[first_valid_index]

                vital_subgroup = vitals_group.create_group(vital_name)
                # Ensure all values are standard Python types
                vital_subgroup.create_dataset('value', data=convert_numpy_types(value))
                vital_subgroup.create_dataset('units', data=info['units'])
                vital_subgroup.create_dataset('timestamp', data=convert_numpy_types(timestamp))

                low_thresh, high_thresh = (0, 100)
                if info.get('low_col') and info['low_col'] in df.columns:
                    low_thresh_series = df[info['low_col']].replace(0, np.nan).dropna()
                    if not low_thresh_series.empty: low_thresh = low_thresh_series.iloc[0]
                if info.get('high_col') and info['high_col'] in df.columns:
                    high_thresh_series = df[info['high_col']].replace(0, np.nan).dropna()
                    if not high_thresh_series.empty: high_thresh = high_thresh_series.iloc[0]

                vital_subgroup.create_dataset('lower_threshold', data=convert_numpy_types(low_thresh))
                vital_subgroup.create_dataset('upper_threshold', data=convert_numpy_types(high_thresh))

                # --- EWS Enhancement: Add historical data in extras ---
                if enable_ews and value > 0:  # Only generate history if we have a valid current value
                    history = generate_vital_history(vital_name, value, timestamp, num_samples=history_samples)
                    extras_data = {
                        'history': history,
                        'generated_by': 'vdxcsv2hdf5_v2.0',
                        'generated_timestamp': int(datetime.now().timestamp()),
                        'trend_analysis_enabled': True
                    }
                    extras_json = json.dumps(extras_data, cls=NumpyEncoder)
                    vital_subgroup.create_dataset('extras', data=extras_json)
            print("Standard vitals data processed.")

            # --- XL_Posture Group ---
            posture_group = vitals_group.create_group('XL_Posture')
            posture_val, step_val, time_change_val, posture_time = (0, 0, 0, convert_numpy_types(df['Epoch Time'].iloc[0]))

            if 'XL Posture' in df.columns:
                posture_series = df['XL Posture'].replace(0, np.nan).dropna()
                if not posture_series.empty:
                    posture_idx = posture_series.index[0]
                    posture_val = posture_series.iloc[0]
                    posture_time = df['Epoch Time'].iloc[posture_idx]

            if 'XL step count' in df.columns:
                step_series = df['XL step count'].replace(0, np.nan).dropna()
                if not step_series.empty: step_val = step_series.iloc[0]

            if 'XL Time since change' in df.columns:
                time_change_series = df['XL Time since change'].replace(0, np.nan).dropna()
                if not time_change_series.empty: time_change_val = time_change_series.iloc[0]

            # Ensure all posture values are standard Python types
            posture_group.create_dataset('value', data=convert_numpy_types(posture_val))
            posture_group.create_dataset('units', data='degrees')
            posture_group.create_dataset('timestamp', data=convert_numpy_types(posture_time))
            posture_group.create_dataset('step_count', data=convert_numpy_types(step_val))
            posture_group.create_dataset('time_since_posture_change', data=convert_numpy_types(time_change_val))

            # --- EWS Enhancement: Add historical data for posture ---
            if enable_ews and posture_val > 0:  # Only generate history if we have a valid posture value
                posture_history = generate_vital_history('XL_Posture', posture_val, posture_time, num_samples=min(15, history_samples))
                posture_extras_data = {
                    'history': posture_history,
                    'generated_by': 'vdxcsv2hdf5_v2.0',
                    'generated_timestamp': int(datetime.now().timestamp()),
                    'trend_analysis_enabled': True,
                    'step_count': step_val,
                    'time_since_change': time_change_val
                }
                posture_extras_json = json.dumps(posture_extras_data, cls=NumpyEncoder)
                posture_group.create_dataset('extras', data=posture_extras_json)

            print("Enhanced XL_Posture vital processed with EWS compatibility.")

        print(f"\n🎉 EWS-Enhanced conversion successful!")
        print(f"📁 HDF5 file saved as: '{hdf5_path}'")
        print(f"🫀 EWS features: Historical vital signs data included for trend analysis")
        print(f"📊 Compatible with RMSAI EWS analysis and dashboard reporting")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execute the Conversion ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python vdxcsv2hdf5.py <path_to_csv_file> [--no-ews] [--history-samples N]")
        print("")
        print("Options:")
        print("  --no-ews           Disable EWS enhancements (no historical data)")
        print("  --history-samples  Number of historical samples to generate (default: 20)")
        print("")
        print("Examples:")
        print("  python vdxcsv2hdf5.py data.csv")
        print("  python vdxcsv2hdf5.py data.csv --history-samples 30")
        print("  python vdxcsv2hdf5.py data.csv --no-ews")
    else:
        csv_file_path = sys.argv[1]
        enable_ews = True
        history_samples = 20

        # Parse additional arguments
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--no-ews':
                enable_ews = False
            elif sys.argv[i] == '--history-samples' and i + 1 < len(sys.argv):
                try:
                    history_samples = int(sys.argv[i + 1])
                    if history_samples < 1 or history_samples > 100:
                        raise ValueError("History samples must be between 1 and 100")
                except ValueError as e:
                    print(f"Error: Invalid history samples value - {e}")
                    sys.exit(1)

        print(f"🔧 Configuration: EWS={'Enabled' if enable_ews else 'Disabled'}, History Samples={history_samples}")
        convert_csv_to_hdf5(csv_file_path, enable_ews=enable_ews, history_samples=history_samples)
