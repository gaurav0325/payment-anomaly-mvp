#!/usr/bin/env python3
"""
Payment Anomaly Detection MVP - Sample Data Generator
Updated with Channel terminology instead of Merchant
"""

import json
import csv
import random
import os
from datetime import datetime, timedelta

def generate_sample_dart_data(num_transactions=1000):
    """
    Generate sample DART 312/313 payment data with Channel terminology
    """
    
    print(f"Generating {num_transactions} sample transactions...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define realistic BA payment channels
    channels = [
        'BA_ECOM_001',        # E-commerce website
        'BA_ECOM_002',        # E-commerce mobile site
        'BA_MOBILE_001',      # Mobile app
        'BA_KIOSK_001',       # Airport kiosks
        'BA_CALLCENTER_001',  # Call center
        'BA_PARTNER_001',     # Travel agent partners
        'BA_PARTNER_002'      # Corporate travel partners
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
        
        # Select channel and scheme
        channel = random.choice(channels)
        scheme = random.choice(card_schemes)
        currency = random.choice(currencies)
        
        # Generate realistic transaction amount based on channel type
        if 'ECOM' in channel:
            # Higher online amounts - flight bookings
            amount = max(50.0, random.uniform(100, 1500) * random.uniform(0.5, 3.0))
        elif 'MOBILE' in channel:
            # Mobile app - mix of bookings and ancillaries
            amount = max(20.0, random.uniform(50, 800) * random.uniform(0.8, 2.0))
        elif 'KIOSK' in channel:
            # Airport kiosks - typically smaller transactions
            amount = max(10.0, random.uniform(25, 400) * random.uniform(0.7, 1.5))
        elif 'CALLCENTER' in channel:
            # Call center - typically higher value bookings
            amount = max(100.0, random.uniform(200, 2000) * random.uniform(0.8, 2.5))
        else:  # PARTNER channels
            # Partner bookings - varied amounts
            amount = max(75.0, random.uniform(150, 1200) * random.uniform(0.6, 2.0))
        
        amount = round(amount, 2)
        
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
        
        # Create transaction record
        transaction = {
            'transaction_id': f'TXN_{timestamp.strftime("%Y%m%d")}_{i:06d}',
            'message_type': 'DART_312',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'settlement_date': timestamp.strftime('%Y-%m-%d'),
            'channel_id': channel,
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
            'reference_number': f'REF_{random.randint(1000000, 9999999)}',
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'fee_rate': round((interchange_fee + scheme_fee_amount + acquirer_fee) / amount, 4),
            'is_anomaly': False,
            'anomaly_type': None
        }
        
        transactions.append(transaction)
    
    # Inject anomalies
    transactions = inject_anomalies(transactions)
    
    print(f"✓ Generated {len(transactions)} transactions")
    
    return transactions

def inject_anomalies(transactions, anomaly_rate=0.02):
    """
    Inject realistic anomalies into the dataset
    """
    print("Injecting anomalies...")
    
    num_anomalies = int(len(transactions) * anomaly_rate)
    anomaly_indices = random.sample(range(len(transactions)), num_anomalies)
    
    anomaly_type_counts = {}
    
    for idx in anomaly_indices:
        anomaly_type = random.choice([
            'high_amount', 'low_amount', 'fee_error', 'settlement_error', 
            'timing_anomaly', 'unusual_channel_pattern'
        ])
        
        anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + 1
        
        transactions[idx]['is_anomaly'] = True
        transactions[idx]['anomaly_type'] = anomaly_type
        
        if anomaly_type == 'high_amount':
            # Create unusually high transaction
            transactions[idx]['amount'] = transactions[idx]['amount'] * random.uniform(10, 50)
            
        elif anomaly_type == 'low_amount':
            # Create unusually low transaction
            transactions[idx]['amount'] = random.uniform(0.01, 5.0)
            
        elif anomaly_type == 'fee_error':
            # Create fee calculation error
            transactions[idx]['interchange_fee'] = transactions[idx]['interchange_fee'] * random.uniform(0.1, 0.5)
            
        elif anomaly_type == 'settlement_error':
            # Create settlement calculation error
            transactions[idx]['net_settlement'] = transactions[idx]['net_settlement'] * random.uniform(0.8, 1.2)
            
        elif anomaly_type == 'timing_anomaly':
            # Create unusual processing time
            transactions[idx]['processing_time_ms'] = random.randint(10000, 60000)
        
        # Recalculate fee rate
        transactions[idx]['fee_rate'] = round((
            transactions[idx]['interchange_fee'] + 
            transactions[idx]['scheme_fee'] + 
            transactions[idx]['acquirer_fee']
        ) / transactions[idx]['amount'], 4)
    
    print(f"✓ Injected {num_anomalies} anomalies:")
    for atype, count in anomaly_type_counts.items():
        print(f"  - {atype}: {count}")
    
    return transactions

def save_to_csv(transactions, filename):
    """Save transactions to CSV file"""
    if not transactions:
        print("No transactions to save")
        return
    
    # Get field names from first transaction
    fieldnames = transactions[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

def display_sample_data(transactions):
    """Display sample data for verification"""
    if not transactions:
        print("No transactions to display")
        return
    
    print("\n" + "="*80)
    print("SAMPLE DATA PREVIEW")
    print("="*80)
    
    print("\nFirst 3 transactions:")
    for i, txn in enumerate(transactions[:3]):
        print(f"\n{i+1}. {txn['transaction_id']}")
        print(f"   Channel: {txn['channel_id']}")
        print(f"   Amount: £{txn['amount']:.2f}")
        print(f"   Card Scheme: {txn['card_scheme']}")
        print(f"   Anomaly: {txn['is_anomaly']}")
    
    # Calculate statistics
    total_amount = sum(txn['amount'] for txn in transactions)
    avg_amount = total_amount / len(transactions)
    anomaly_count = sum(1 for txn in transactions if txn['is_anomaly'])
    
    print(f"\nData Statistics:")
    print(f"  Total transactions: {len(transactions):,}")
    print(f"  Total volume: £{total_amount:,.2f}")
    print(f"  Average amount: £{avg_amount:.2f}")
    print(f"  Anomalies: {anomaly_count} ({anomaly_count/len(transactions)*100:.1f}%)")
    
    # Count by channel
    channel_counts = {}
    for txn in transactions:
        channel = txn['channel_id']
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
    
    print(f"\nChannel distribution:")
    for channel, count in sorted(channel_counts.items()):
        print(f"  {channel}: {count}")
    
    # Count by card scheme
    scheme_counts = {}
    for txn in transactions:
        scheme = txn['card_scheme']
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
    
    print(f"\nCard scheme distribution:")
    for scheme, count in sorted(scheme_counts.items()):
        print(f"  {scheme}: {count}")
    
    # Count anomaly types
    anomaly_types = {}
    for txn in transactions:
        if txn['is_anomaly'] and txn['anomaly_type']:
            atype = txn['anomaly_type']
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
    
    if anomaly_types:
        print(f"\nAnomaly types:")
        for atype, count in sorted(anomaly_types.items()):
            print(f"  {atype}: {count}")

def main():
    """Main function to generate sample data"""
    print("🚀 Payment Anomaly Detection MVP - Channel Data Generator")
    print("="*65)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate sample data
    transactions = generate_sample_dart_data(1000)
    
    # Save transaction data
    save_to_csv(transactions, 'data/raw/sample_transactions.csv')
    print(f"✓ Saved transactions to: data/raw/sample_transactions.csv")
    
    # Display sample data
    display_sample_data(transactions)
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Restart Streamlit to see updated terminology")
    print("2. Test the anomaly detection with channel data")
    print("3. Launch dashboard: streamlit run src/frontend/streamlit_app.py --server.port 8502")
    print("\nSample transaction IDs to test:")
    for i in range(min(5, len(transactions))):
        print(f"  - {transactions[i]['transaction_id']}")

if __name__ == "__main__":
    main()