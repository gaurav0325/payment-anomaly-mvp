#!/usr/bin/env python3
"""
Payment Anomaly Detection MVP - MCP Server
Simplified version for initial testing
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaymentAnomalyDetector:
    """
    Core anomaly detection engine for payment transactions
    """
    
    def __init__(self):
        self.data_path = 'data/raw/sample_transactions.csv'
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'amount', 'interchange_fee', 'scheme_fee', 'acquirer_fee',
            'processing_time_ms', 'hour_of_day', 'day_of_week', 'fee_rate'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """Load transaction data from CSV"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['settlement_date'] = pd.to_datetime(df['settlement_date'])
            
            logger.info(f"Loaded {len(df)} transactions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        # Handle missing values
        df_clean = df[self.feature_columns].fillna(0)
        
        # Scale features
        if not self.is_trained:
            features_scaled = self.scaler.fit_transform(df_clean)
        else:
            features_scaled = self.scaler.transform(df_clean)
            
        return features_scaled
    
    def train_model(self, df: pd.DataFrame):
        """Train the anomaly detection model"""
        try:
            logger.info("Training anomaly detection model...")
            
            # Prepare features
            features = self.prepare_features(df)
            
            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=0.05,  # Expect 5% anomalies
                random_state=42,
                n_estimators=100
            )
            
            self.model.fit(features)
            self.is_trained = True
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in transaction data"""
        try:
            if not self.is_trained:
                self.train_model(df)
            
            # Prepare features
            features = self.prepare_features(df)
            
            # Predict anomalies
            anomaly_predictions = self.model.predict(features)
            anomaly_scores = self.model.decision_function(features)
            
            # Add results to dataframe
            results = df.copy()
            results['ml_anomaly'] = anomaly_predictions == -1
            results['anomaly_score'] = anomaly_scores
            
            # Statistical anomaly detection
            results = self._add_statistical_anomalies(results)
            
            # Combine all anomaly flags
            results['final_anomaly'] = (
                results['ml_anomaly'] | 
                results['amount_anomaly'] | 
                results['fee_anomaly']
            )
            
            logger.info(f"Detected {results['final_anomaly'].sum()} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def _add_statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical anomaly detection"""
        # Amount anomalies (3-sigma rule)
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        df['amount_anomaly'] = (
            (df['amount'] > mean_amount + 3 * std_amount) |
            (df['amount'] < mean_amount - 3 * std_amount)
        )
        
        # Fee rate anomalies
        mean_fee_rate = df['fee_rate'].mean()
        std_fee_rate = df['fee_rate'].std()
        df['fee_anomaly'] = (
            (df['fee_rate'] > mean_fee_rate + 2 * std_fee_rate) |
            (df['fee_rate'] < mean_fee_rate - 2 * std_fee_rate)
        )
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """Generate anomaly summary statistics"""
        anomalies = df[df['final_anomaly'] == True]
        
        summary = {
            'total_transactions': len(df),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) * 100,
            'anomaly_types': {
                'ml_detected': int(df['ml_anomaly'].sum()),
                'amount_anomalies': int(df['amount_anomaly'].sum()),
                'fee_anomalies': int(df['fee_anomaly'].sum())
            },
            'severity_distribution': {
                'high': int((anomalies['anomaly_score'] < -0.5).sum()),
                'medium': int(((anomalies['anomaly_score'] >= -0.5) & 
                              (anomalies['anomaly_score'] < -0.2)).sum()),
                'low': int((anomalies['anomaly_score'] >= -0.2).sum())
            }
        }
        
        return summary

def test_anomaly_detection():
    """Test the anomaly detection system"""
    print("🔍 Testing Payment Anomaly Detection System")
    print("="*60)
    
    # Initialize detector
    detector = PaymentAnomalyDetector()
    
    # Load data
    print("Loading data...")
    df = detector.load_data()
    
    # Detect anomalies
    print("Detecting anomalies...")
    results = detector.detect_anomalies(df)
    
    # Get summary
    summary = detector.get_anomaly_summary(results)
    
    print("\n📊 ANOMALY DETECTION RESULTS:")
    print(f"Total transactions: {summary['total_transactions']:,}")
    print(f"Total anomalies: {summary['total_anomalies']:,}")
    print(f"Anomaly rate: {summary['anomaly_rate']:.2f}%")
    
    print(f"\nAnomaly types:")
    for atype, count in summary['anomaly_types'].items():
        print(f"  - {atype}: {count}")
    
    print(f"\nSeverity distribution:")
    for severity, count in summary['severity_distribution'].items():
        print(f"  - {severity}: {count}")
    
    # Show some example anomalies
    anomalies = results[results['final_anomaly'] == True]
    print(f"\n🚨 Top 5 Anomalies:")
    
    for i, (_, row) in enumerate(anomalies.head(5).iterrows()):
        print(f"\n{i+1}. {row['transaction_id']}")
        print(f"   Merchant: {row['merchant_id']}")
        print(f"   Amount: £{row['amount']:.2f}")
        print(f"   Anomaly Score: {row['anomaly_score']:.3f}")
        print(f"   Types: ML={row['ml_anomaly']}, Amount={row['amount_anomaly']}, Fee={row['fee_anomaly']}")
    
    print("\n✅ Testing completed successfully!")
    return detector, results

def main():
    """Main function for testing"""
    try:
        detector, results = test_anomaly_detection()
        
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("1. Launch dashboard: streamlit run src/frontend/streamlit_app.py")
        print("2. Test with sample transaction IDs from the data generator")
        print("3. Experiment with different date ranges and merchants")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you've run generate_sample_data.py first")
        print("2. Check that data/raw/sample_transactions.csv exists")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()