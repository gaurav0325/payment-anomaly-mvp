#!/usr/bin/env python3
"""
Quick script to update column names from merchant_id to channel_id
"""

import pandas as pd
import os

def update_column_names():
    """Update existing data to use channel terminology"""
    
    data_file = 'data/raw/sample_transactions.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the data generator first.")
        return False
    
    try:
        print("📝 Updating column names...")
        
        # Read the existing data
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} transactions")
        
        # Check if we need to rename columns
        if 'merchant_id' in df.columns:
            # Rename the column
            df = df.rename(columns={'merchant_id': 'channel_id'})
            print("✓ Renamed 'merchant_id' to 'channel_id'")
            
            # Save the updated data
            df.to_csv(data_file, index=False)
            print(f"✓ Updated data saved to: {data_file}")
            
        elif 'channel_id' in df.columns:
            print("✓ Data already uses 'channel_id' - no changes needed")
            
        else:
            print("❌ Neither 'merchant_id' nor 'channel_id' found in data")
            print("Available columns:", list(df.columns))
            return False
        
        # Display sample
        print("\nSample data:")
        print(df[['transaction_id', 'channel_id', 'amount', 'card_scheme']].head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating data: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Column Name Updater")
    print("="*30)
    
    success = update_column_names()
    
    if success:
        print("\n✅ SUCCESS!")
        print("Now restart Streamlit to see the changes.")
    else:
        print("\n❌ FAILED!")
        print("Try running the full data generator instead.")