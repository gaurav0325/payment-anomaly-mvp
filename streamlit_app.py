"""
British Airways Payment Anomaly Detection Dashboard
Enhanced DART 312/313 Processing with Real-time Anomaly Detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from datetime import datetime, timedelta
import sys
import os
import csv

# Add src to path for imports
sys.path.append('src')

try:
    from core.dart_parser import DARTParser
    from core.anomaly_detector import EnhancedAnomalyDetector
except ImportError:
    st.error("Required modules not found. Please ensure you've followed the implementation guide.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="BA Payment Anomaly Detection",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for BA branding
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .severity-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .severity-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .severity-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample DART data for demonstration"""
    # Sample DART 312 data
    dart_312_data = {
        'record_type': ['00', '01', '01', '01', '02', '03', '99'],
        'party_id': ['BA001'] * 7,
        'merchant_id': ['', 'BA884', 'BA480', 'BA287', 'BA345', 'BA789', ''],
        'transaction_amount': [0, 834.48, 75000.00, 425.75, 89.75, 45.67, 0],
        'transaction_currency': ['', 'USD', 'GBP', 'GBP', 'GBP', 'GBP', ''],
        'settlement_amount': [0, 675.93, 75000.00, 425.75, 0, 0, 0],
        'mcc': ['', '4511', '4511', '4511', '4511', '0000', ''],
        'airline_ticket_number': ['', 'BA10386731', 'BA8095509', 'BA2341567', 'BA4567890', 'BA6789012', '']
    }
    
    # Sample DART 313 data
    dart_313_data = {
        'record_type': ['00', '01', '05', '08', '15', '16', '16', '99'],
        'party_id': ['BA001'] * 8,
        'amount': [0, 154275.95, 0, 0, 0, 0, 0, 0],
        'value_of_transactions': [0, 0, 156750.25, -2248.80, 0, 0, 0, 0],
        'chargeback_value': [0, 0, 0, 0, 0, 1250.00, 850.00, 0],
        'currency': ['GBP'] * 6 + ['USD', 'GBP'],
        'number_of_transactions': [0, 0, 1235, 1235, 0, 0, 0, 0],
        'dispute_id': [''] * 5 + ['DP789456', 'DP234567', '']
    }
    
    return pd.DataFrame(dart_312_data), pd.DataFrame(dart_313_data)

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

def main():
    # Header
    st.markdown('<h1 class="main-header">🛫 British Airways Payment Anomaly Detection</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Developer:** Gaurav (AI Solution Architect)")
    with col2:
        st.info("**Company:** BranchSpace")
    with col3:
        st.info("**Client:** British Airways")
    
    # Demo mode with sample data
    st.markdown("## 🎯 Demo Mode - Enhanced Anomaly Detection")
    
    dart_312_df, dart_313_df = load_sample_data()
    
    try:
        detector = EnhancedAnomalyDetector()
        
        # Process anomalies
        dart_312_anomalies = detector.detect_dart_312_anomalies(dart_312_df)
        dart_313_anomalies = detector.detect_dart_313_anomalies(dart_313_df)
        
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("DART 312 Anomalies", len(dart_312_anomalies))
        with col2:
            st.metric("DART 313 Anomalies", len(dart_313_anomalies))
        with col3:
            total_anomalies = len(dart_312_anomalies) + len(dart_313_anomalies)
            st.metric("Total Anomalies", total_anomalies)
        
        # Show anomalies
        if dart_312_anomalies or dart_313_anomalies:
            st.markdown("### 🚨 Detected Anomalies")
            
            all_anomalies = dart_312_anomalies + dart_313_anomalies
            for i, anomaly in enumerate(all_anomalies, 1):
                severity = anomaly.get('severity', 'UNKNOWN')
                anomaly_type = anomaly.get('type', 'Unknown')
                description = anomaly.get('description', 'No description')
                
                if severity == 'HIGH':
                    st.error(f"**{i}. {anomaly_type}** - {description}")
                elif severity == 'MEDIUM':
                    st.warning(f"**{i}. {anomaly_type}** - {description}")
                else:
                    st.info(f"**{i}. {anomaly_type}** - {description}")
        else:
            st.success("No anomalies detected in sample data!")
        
        # Show sample data
        with st.expander("View Sample DART 312 Data"):
            st.dataframe(dart_312_df)
        
        with st.expander("View Sample DART 313 Data"):
            st.dataframe(dart_313_df)
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Please ensure you've completed the setup from the implementation guide.")

if __name__ == "__main__":
    main()
