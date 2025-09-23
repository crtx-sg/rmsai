# This program converts the Vios vdx csv file to hdf5 file format
# ganesh
import pandas as pd
import h5py
import numpy as np
from datetime import datetime
import uuid
import sys
import json
import os

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

# --- Main Conversion Logic ---
def convert_csv_to_hdf5(csv_path):
    """
    Converts physiological data from a CSV file to a structured HDF5 file.
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

            # --- Metadata Group ---
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
            metadata.create_dataset('device_info', data="RMSAI-SimDevice-v1.0")
            print("Metadata group created with JSON-serializable types.")

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
            print("XL_Posture vital processed and stored.")

        print(f"\nConversion successful. HDF5 file saved as '{hdf5_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execute the Conversion ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python vdx2hdf5.py <path_to_csv_file>")
    else:
        csv_file_path = sys.argv[1]
        convert_csv_to_hdf5(csv_file_path)
