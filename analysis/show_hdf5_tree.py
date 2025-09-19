#!/usr/bin/env python3
"""
Script to show HDF5 file structure and tree
"""

import h5py
import os
import sys

def show_hdf5_tree(filename, max_depth=10):
    """Display the tree structure of an HDF5 file"""

    def print_structure(name, obj, level=0, max_level=10):
        if level > max_level:
            return

        indent = "  " * level

        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/ (Group)")

            # Show attributes if any
            if len(obj.attrs) > 0:
                print(f"{indent}  📋 Attributes:")
                for attr_name, attr_value in obj.attrs.items():
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8', errors='ignore')
                    print(f"{indent}    {attr_name}: {attr_value}")

            # Recurse into subgroups/datasets
            for key in obj.keys():
                print_structure(key, obj[key], level + 1, max_level)

        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}📄 {name} (Dataset)")
            print(f"{indent}  📊 Shape: {obj.shape}")
            print(f"{indent}  🔢 Dtype: {obj.dtype}")

            # Show attributes if any
            if len(obj.attrs) > 0:
                print(f"{indent}  📋 Attributes:")
                for attr_name, attr_value in obj.attrs.items():
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8', errors='ignore')
                    print(f"{indent}    {attr_name}: {attr_value}")

            # Show some sample data if small enough
            if obj.size < 50:
                try:
                    print(f"{indent}  📝 Sample data: {obj[:5]}")
                except:
                    print(f"{indent}  📝 Sample data: [Unable to read]")

    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return

    print(f"🗂️  HDF5 File Structure: {filename}")
    print("=" * 60)

    try:
        with h5py.File(filename, 'r') as f:
            print(f"📁 Root/")

            # Show root attributes
            if len(f.attrs) > 0:
                print("  📋 Root Attributes:")
                for attr_name, attr_value in f.attrs.items():
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8', errors='ignore')
                    print(f"    {attr_name}: {attr_value}")

            # Show structure
            for key in f.keys():
                print_structure(key, f[key], level=1, max_level=max_depth)

    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")

def main():
    # Get HDF5 files from data directory
    data_dir = "data"
    h5_files = []

    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.h5'):
                h5_files.append(os.path.join(data_dir, file))

    if not h5_files:
        print("❌ No HDF5 files found in data directory")
        return

    # Show structure for each file
    for i, h5_file in enumerate(h5_files):
        if i > 0:
            print("\n" + "=" * 80 + "\n")

        show_hdf5_tree(h5_file)

        # If multiple files, ask if user wants to see more
        if len(h5_files) > 1 and i < len(h5_files) - 1:
            print(f"\n📊 Found {len(h5_files)} HDF5 files total. Showing file {i+1}/{len(h5_files)}")

if __name__ == "__main__":
    main()