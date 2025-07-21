import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import csv
import os

# --- DART 312/313 Sample Data Generator ---
def generate_dart_312_sample(num_records=500):
    """Generate DART 312 sample data with anomalies and detailed reasons."""
    records = []
    today = datetime.now()
    party_id = 'AL001'
    merchant_ids = [f'AL{str(i).zfill(3)}' for i in range(10)]
    mccs = ['4511', '4789', '4111']
    currencies = ['GBP', 'USD', 'EUR']
    rejection_codes = ['1300827893', '1610110033', '1610111253']
    # Header (00)
    header = {
        'record_type': '00',
        'data_file_date': today.strftime('%Y%m%d'),
        'party_id': party_id,
        'total_submitted': num_records,
        'total_accepted': int(num_records*0.95),
        'total_pending': int(num_records*0.03),
        'total_rejected': int(num_records*0.02),
        'total_transaction_value_accepted': 0,
        'total_settlement_value_accepted': 0,
        'total_transaction_value_pending': 0,
        'total_transaction_value_rejected': 0
    }
    records.append(header)
    # Accepted (01), Pending (02), Rejected (03)
    for i in range(num_records):
        base_date = today - timedelta(days=random.randint(0, 90))
        merchant_id = random.choice(merchant_ids)
        mcc = random.choice(mccs)
        currency = random.choice(currencies)
        amount = round(np.random.normal(150, 50), 2)
        is_anomaly = False
        anomaly_type = None
        anomaly_reason = None
        record_type = '01'
        # Inject anomalies
        if random.random() < 0.02:
            record_type = '03'
            is_anomaly = True
            anomaly_type = 'rejected_transaction'
            rejection_code = random.choice(rejection_codes)
            anomaly_reason = f"Rejected transaction: {rejection_code} - see DART spec for reason."
        elif random.random() < 0.03:
            record_type = '02'
            is_anomaly = True
            anomaly_type = 'pending_transaction'
            anomaly_reason = "Transaction is pending settlement, which is unusual for this age."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'duplicate_transaction'
            anomaly_reason = "Duplicate transaction detected (same PAN, amount, timestamp)."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'outlier_amount'
            amount = round(amount * 8, 2)
            anomaly_reason = "Transaction amount is much higher than channel average."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'timing_mismatch'
            anomaly_reason = "Settlement date is much later than transaction date."
        # Compose record
        rec = {
            'record_type': record_type,
            'data_file_date': today.strftime('%Y%m%d'),
            'party_id': party_id,
            'merchant_id': merchant_id,
            'mcc': mcc,
            'transaction_amount': amount,
            'transaction_currency': currency,
            'transaction_datetime': base_date.strftime('%Y%m%d %H%M%S'),
            'settlement_date': (base_date + timedelta(days=random.randint(1, 10))).strftime('%Y%m%d'),
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'anomaly_reason': anomaly_reason
        }
        if record_type == '03':
            rec['rejection_reason_code'] = rejection_code
        records.append(rec)
    # Trailer (99)
    trailer = {'record_type': '99', 'record_count': len(records)}
    records.append(trailer)
    return records

def generate_dart_313_sample(num_records=200):
    """Generate DART 313 sample data with anomalies and detailed reasons."""
    records = []
    today = datetime.now()
    party_id = 'AL001'
    currencies = ['GBP', 'USD', 'EUR']
    # Header (00)
    header = {
        'record_type': '00',
        'party_name': 'Airlines PLC',
        'settlement_currency': 'GBP',
        'party_id': party_id,
        'processing_date': today.strftime('%Y%m%d')
    }
    records.append(header)
    for i in range(num_records):
        base_date = today - timedelta(days=random.randint(0, 90))
        currency = random.choice(currencies)
        amount = round(np.random.normal(150, 50), 2)
        is_anomaly = False
        anomaly_type = None
        anomaly_reason = None
        record_type = '01'
        # Inject anomalies
        if random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'settlement_mismatch'
            anomaly_reason = "Settlement amount does not match corresponding 312 record."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'missing_confirmation'
            anomaly_reason = "Expected settlement confirmation missing for a 312 record."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'chargeback_anomaly'
            anomaly_reason = "Unusual chargeback value or frequency detected."
        elif random.random() < 0.03:
            is_anomaly = True
            anomaly_type = 'funding_anomaly'
            anomaly_reason = "Unexpected negative or positive swing in funding detected."
        rec = {
            'record_type': record_type,
            'party_id': party_id,
            'processing_date': base_date.strftime('%Y%m%d'),
            'currency': currency,
            'amount': amount,
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'anomaly_reason': anomaly_reason
        }
        records.append(rec)
    # Trailer (99)
    trailer = {'record_type': '99', 'record_count': len(records)}
    records.append(trailer)
    return records

def save_records_to_csv(records, filename):
    if not records:
        return
    # Collect all possible fieldnames
    fieldnames = set()
    for rec in records:
        fieldnames.update(rec.keys())
    fieldnames = list(fieldnames)
    # Ensure all records have all fields
    for rec in records:
        for field in fieldnames:
            if field not in rec:
                rec[field] = ""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    dart312 = generate_dart_312_sample(500)
    dart313 = generate_dart_313_sample(200)
    save_records_to_csv(dart312, 'data/processed/dart_312_sample.csv')
    save_records_to_csv(dart313, 'data/processed/dart_313_sample.csv')
    print("DART 312 and 313 sample files generated in data/processed/ with detailed anomaly reasons.")