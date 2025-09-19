#!/usr/bin/env python3
"""
Script to check the metadata structure in HDF5 file
"""

import h5py

def check_metadata_structure(filename):
    """Check the metadata structure"""
    print(f"🔍 Checking metadata structure in: {filename}")
    print("=" * 60)

    try:
        with h5py.File(filename, 'r') as f:
            if 'metadata' in f:
                metadata = f['metadata']
                print("📁 /metadata/ group found")

                # Check what's in metadata
                print(f"   Keys: {list(metadata.keys())}")
                print(f"   Attributes: {list(metadata.attrs.keys())}")

                # Check each item in metadata
                for key in metadata.keys():
                    item = metadata[key]
                    print(f"\n   📊 {key}:")
                    print(f"      Type: {type(item)}")

                    if isinstance(item, h5py.Group):
                        print(f"      Group keys: {list(item.keys())}")
                        print(f"      Group attributes: {list(item.attrs.keys())}")
                    elif isinstance(item, h5py.Dataset):
                        print(f"      Dataset shape: {item.shape}")
                        print(f"      Dataset dtype: {item.dtype}")
                        try:
                            value = item[()]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            print(f"      Value: {value}")
                        except:
                            print("      Value: [Unable to read]")
            else:
                print("❌ No /metadata/ group found")
                print(f"   Available root groups: {list(f.keys())}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_metadata_structure("data/PT3079_2025-09.h5")