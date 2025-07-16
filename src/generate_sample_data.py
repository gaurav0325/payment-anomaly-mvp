#!/usr/bin/env python3
"""
Payment Anomaly Detection MVP - Sample Data Generator
Generates realistic DART 312/313 payment data with anomalies for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_sample_dart_data(num_transactions=10000):
    """
    Generate sample DART 312/313 payment data with realistic patterns and anomalies
    """
    
    print(f"Generating {num_transactions} sample transactions...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic parameters
    merchants = [
        'BA_ECOM_001', 'BA_ECOM_002', 'BA_MOBILE_001', 'BA_KIOSK_001',
        'BA_CALLCENTER_001', 'BA_PARTNER_001', 'BA_PARTNER_002'
    ]
    
    card_schemes = ['VISA', 'MASTERCARD', 'AMEX', 'DISCOVER']
    currencies = ['GBP', 'USD', 'EUR']
    
    # Generate base transaction data
    transactions = []
    
    for i in range(num_transactions):
        # Generate timestamp (last 30 days)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Select merchant and scheme
        merchant = random.choice(merchants)
        scheme = random.choice(card_schemes)
        currency = random.choice(currencies)
        
        # Generate realistic transaction amount based on merchant type
        if 'ECOM' in merchant:
            base_amount = np.random.lognormal(mean=5.0, sigma=1.0)  # Higher online amounts
        elif 'MOBILE' in merchant:
            base_amount = np.random.lognormal(mean=3.5, sigma=0.8)  # Mobile payments
        else:
            base_amount = np.random.lognormal(mean=4.0, sigma=1.2)  # Other channels
        
        amount = round(max(10.0, base_amount), 2)
        
        # Calculate fees based on scheme and amount
        if scheme == 'VISA':
            interchange_rate = 0.015
            scheme_fee = 0.004
        elif scheme == 'MASTERCARD':
            interchange_rate = 0.014
            scheme_fee = 0.0035
        elif scheme == 'AMEX':
            interchange_rate = 0.025
            scheme_fee = 0.006
        else:  # DISCOVER
            interchange_rate = 0.016
            scheme_fee = 0.004
        
        interchange_fee = round(amount * interchange_rate, 2)
        scheme_fee_amount = round(amount * scheme_fee, 2)
        acquirer_fee = round(amount * 0.005, 2)  # 0.5% acquirer fee
        
        # Calculate net settlement
        net_settlement = round(amount - interchange_fee - scheme_fee_amount - acquirer_fee, 2)
        
        # Generate batch info
        batch_date = timestamp.strftime('%Y%m%d')
        batch_hour = timestamp.hour // 4 * 4  # Batches every 4 hours
        batch_id = f"BATCH_{batch_date}_{batch_hour:02d}"
        
        transaction = {
            'transaction_id': f'TXN_{timestamp.strftime("%Y%m%d")}_{i:06d}',
            'message_type': 'DART_312',
            'timestamp': timestamp,
            'settlement_date': timestamp.date(),
            'merchant_id': merchant,
            'card_scheme': scheme,
            'currency': currency,
            'amount': amount,
            'interchange_fee': interchange_fee,
            'scheme_fee': scheme_fee_amount,
            'acquirer_fee': acquirer_fee,
            'net_settlement': net_settlement,
            'batch_id': batch_id,
            'settlement_status': 'COMPLETED',
            'processing_time_ms': random.randint(100, 2000),
            'auth_code': f'AUTH_{random.randint(100000, 999999)}',
            'reference_number': f'REF_{random.randint(1000000, 9999999)}'
        }
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Add some calculated fields for anomaly detection
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['fee_rate'] = (df['interchange_fee'] + df['scheme_fee'] + df['acquirer_fee']) / df['amount']
    df['settlement_accuracy'] = df['net_settlement'] / (df['amount'] - df['interchange_fee'] - df['scheme_fee'] - df['acquirer_fee'])
    
    # Inject some anomalies for testing
    df = inject_anomalies(df)
    
    print(f"✓ Generated {len(df)} transactions")
    
    return df

def inject_anomalies(df, anomaly_rate=0.02):
    """
    Inject realistic anomalies into the dataset
    """
    print("Injecting anomalies...")
    
    num_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
    
    # Create anomaly tracking
    df['is_anomaly'] = False
    df['anomaly_type'] = None
    
    anomaly_type_counts = {}
    
    for idx in anomaly_indices:
        anomaly_type = random.choice([
            'high_amount', 'low_amount', 'fee_error', 'settlement_error', 
            'timing_anomaly', 'unusual_merchant_pattern'
        ])
        
        anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + 1
        
        df.loc[idx, 'is_anomaly'] = True
        df.loc[idx, 'anomaly_type'] = anomaly_type
        
        if anomaly_type == 'high_amount':
            # Create unusually high transaction
            df.loc[idx, 'amount'] = df.loc[idx, 'amount'] * random.uniform(10, 50)
            
        elif anomaly_type == 'low_amount':
            # Create unusually low transaction
            df.loc[idx, 'amount'] = random.uniform(0.01, 5.0)
            
        elif anomaly_type == 'fee_error':
            # Create fee calculation error
            df.loc[idx, 'interchange_fee'] = df.loc[idx, 'interchange_fee'] * random.uniform(0.1, 0.5)
            
        elif anomaly_type == 'settlement_error':
            # Create settlement calculation error
            df.loc[idx, 'net_settlement'] = df.loc[idx, 'net_settlement'] * random.uniform(0.8, 1.2)
            
        elif anomaly_type == 'timing_anomaly':
            # Create unusual processing time
            df.loc[idx, 'processing_time_ms'] = random.randint(10000, 60000)
            
        # Recalculate derived fields
        df.loc[idx, 'fee_rate'] = (df.loc[idx, 'interchange_fee'] + 
                                  df.loc[idx, 'scheme_fee'] + 
                                  df.loc[idx, 'acquirer_fee']) / df.loc[idx, 'amount']
    
    print(f"✓ Injected {num_anomalies} anomalies:")
    for atype, count in anomaly_type_counts.items():
        print(f"  - {atype}: {count}")
    
    return df

def generate_settlement_summary(df):
    """
    Generate daily settlement summary data
    """
    print("Generating settlement summary...")
    
    summary = df.groupby(['settlement_date', 'merchant_id', 'card_scheme']).agg({
        'transaction_id': 'count',
        'amount': 'sum',
        'interchange_fee': 'sum',
        'scheme_fee': 'sum',
        'acquirer_fee': 'sum',
        'net_settlement': 'sum',
        'processing_time_ms': 'mean'
    }).reset_index()
    
    summary.columns = [
        'settlement_date', 'merchant_id', 'card_scheme', 'transaction_count',
        'total_amount', 'total_interchange_fee', 'total_scheme_fee',
        'total_acquirer_fee', 'total_net_settlement', 'avg_processing_time'
    ]
    
    print(f"✓ Generated {len(summary)} settlement records")
    
    return summary

def display_sample_data(df):
    """
    Display sample data for verification
    """
    print("\n" + "="*80)
    print("SAMPLE DATA PREVIEW")
    print("="*80)
    
    print("\nFirst 3 transactions:")
    print(df[['transaction_id', 'merchant_id', 'amount', 'card_scheme', 'is_anomaly']].head(3).to_string())
    
    print(f"\nData Statistics:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Date range: {df['settlement_date'].min()} to {df['settlement_date'].max()}")
    print(f"  Total volume: £{df['amount'].sum():,.2f}")
    print(f"  Average amount: £{df['amount'].mean():.2f}")
    print(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")
    
    print(f"\nMerchant distribution:")
    print(df['merchant_id'].value_counts().to_string())
    
    print(f"\nCard scheme distribution:")
    print(df['card_scheme'].value_counts().to_string())
    
    print(f"\nAnomaly types:")
    print(df[df['is_anomaly']]['anomaly_type'].value_counts().to_string())

def main():
    """
    Main function to generate sample data
    """
    print("🚀 Payment Anomaly Detection MVP - Data Generator")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate sample data
    transactions_df = generate_sample_dart_data(10000)
    
    # Save transaction data
    transactions_df.to_csv('data/raw/sample_transactions.csv', index=False)
    print(f"✓ Saved transactions to: data/raw/sample_transactions.csv")
    
    # Generate and save settlement summary
    settlement_summary = generate_settlement_summary(transactions_df)
    settlement_summary.to_csv('data/processed/settlement_summary.csv', index=False)
    print(f"✓ Saved settlement summary to: data/processed/settlement_summary.csv")
    
    # Display sample data
    display_sample_data(transactions_df)
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Launch the dashboard: streamlit run src/frontend/streamlit_app.py")
    print("2. Or start the MCP server: python src/mcp_server/server.py")
    print("\nSample transaction IDs to test:")
    sample_ids = transactions_df['transaction_id'].head(5).tolist()
    for tid in sample_ids:
        print(f"  - {tid}")

if __name__ == "__main__":
    main()