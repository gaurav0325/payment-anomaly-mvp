#!/usr/bin/env python3
"""
Debug script to check file paths and data availability
"""

import os
import sys

def check_paths():
    print("🔍 PATH DEBUGGING")
    print("=" * 50)
    
    # Current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    expected_files = ['requirements.txt', '.env', 'src']
    print(f"\nChecking for expected project files:")
    for file in expected_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓ EXISTS' if exists else '❌ MISSING'}")
    
    # Check data directory structure
    print(f"\nChecking data directory structure:")
    data_paths = [
        'data',
        'data/raw',
        'data/processed',
        'data/raw/sample_transactions.csv'
    ]
    
    for path in data_paths:
        exists = os.path.exists(path)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  {path}: {'✓ EXISTS' if exists else '❌ MISSING'} ({size} bytes)")
        else:
            print(f"  {path}: {'✓ EXISTS' if exists else '❌ MISSING'}")
    
    # List contents of data directories
    if os.path.exists('data'):
        print(f"\nContents of 'data' directory:")
        try:
            for item in os.listdir('data'):
                print(f"  - {item}")
        except Exception as e:
            print(f"  Error reading data directory: {e}")
    
    if os.path.exists('data/raw'):
        print(f"\nContents of 'data/raw' directory:")
        try:
            for item in os.listdir('data/raw'):
                item_path = os.path.join('data/raw', item)
                size = os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
                print(f"  - {item} ({size} bytes)")
        except Exception as e:
            print(f"  Error reading data/raw directory: {e}")
    
    # Test reading the CSV file
    csv_path = 'data/raw/sample_transactions.csv'
    if os.path.exists(csv_path):
        print(f"\n✅ Testing CSV file read:")
        try:
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                line_count = sum(1 for _ in f) + 1  # +1 for header
            print(f"  First line (header): {first_line}")
            print(f"  Total lines: {line_count}")
        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")
    else:
        print(f"\n❌ CSV file not found at: {csv_path}")
    
    # Check Python path
    print(f"\nPython executable: {sys.executable}")
    print(f"Python path: {sys.path[0]}")
    
    return os.path.exists(csv_path)

if __name__ == "__main__":
    success = check_paths()
    
    if not success:
        print("\n" + "="*50)
        print("🚨 ISSUE FOUND")
        print("="*50)
        print("The sample data file is missing.")
        print("\nTry running:")
        print("1. python generate_sample_data.py")
        print("2. Or python generate_sample_data_minimal.py")
        print("\nMake sure you're in the project root directory!")
    else:
        print("\n" + "="*50)
        print("✅ ALL PATHS LOOK GOOD")
        print("="*50)
        print("Data file exists and is readable.")