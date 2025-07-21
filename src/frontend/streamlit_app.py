import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import numpy as np
import csv
import io

# Page configuration
st.set_page_config(
    page_title="Airlines Payment Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .anomaly-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .anomaly-low {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class MockMCPClient:
    """
    Mock MCP client for demonstration purposes
    """
    
    def __init__(self):
        # Get the project root directory (2 levels up from this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_path = os.path.join(project_root, 'data', 'raw', 'sample_transactions.csv')
        
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Simulate MCP tool calls"""
        try:
            if not os.path.exists(self.data_path):
                return {"error": "Sample data not found. Please run the data generator first."}
            
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['settlement_date'] = pd.to_datetime(df['settlement_date'])
            
            if tool_name == "detect_payment_anomalies":
                return self._detect_anomalies(df, arguments)
            elif tool_name == "get_anomaly_summary":
                return self._get_summary(df, arguments)
            elif tool_name == "query_specific_transaction":
                return self._query_transaction(df, arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            return {"error": f"Error calling tool {tool_name}: {str(e)}"}
    
    def _detect_anomalies(self, df: pd.DataFrame, args: dict) -> dict:
        """Mock anomaly detection"""
        # Apply filters
        filtered_df = df.copy()
        
        if 'date_from' in args and args['date_from']:
            filtered_df = filtered_df[filtered_df['settlement_date'] >= pd.to_datetime(args['date_from'])]
        
        if 'date_to' in args and args['date_to']:
            filtered_df = filtered_df[filtered_df['settlement_date'] <= pd.to_datetime(args['date_to'])]
        
        if 'channel_id' in args and args['channel_id']:
            filtered_df = filtered_df[filtered_df['channel_id'] == args['channel_id']]
        
        # Simple anomaly detection (using known anomalies from sample data)
        anomalies = filtered_df[filtered_df['is_anomaly'] == True]
        
        response = {
            'analysis_period': {
                'from': filtered_df['settlement_date'].min().strftime('%Y-%m-%d'),
                'to': filtered_df['settlement_date'].max().strftime('%Y-%m-%d'),
                'total_transactions': len(filtered_df)
            },
            'anomalies_detected': len(anomalies),
            'anomaly_details': []
        }
        
        # Add top 10 anomalies
        for _, row in anomalies.head(10).iterrows():
            response['anomaly_details'].append({
                'transaction_id': row['transaction_id'],
                'channel_id': row['channel_id'],
                'amount': float(row['amount']),
                'anomaly_score': np.random.uniform(-1, -0.1),  # Mock score
                'anomaly_types': {
                    'ml_detected': True,
                    'amount_anomaly': row['anomaly_type'] in ['high_amount', 'low_amount'],
                    'fee_anomaly': row['anomaly_type'] == 'fee_error'
                },
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return response
    
    def _get_summary(self, df: pd.DataFrame, args: dict) -> dict:
        """Mock summary generation"""
        period = args.get('period', 'all')
        
        # Apply period filter
        if period == 'today':
            today = datetime.now().date()
            df = df[df['settlement_date'].dt.date == today]
        elif period == 'week':
            week_ago = datetime.now() - timedelta(days=7)
            df = df[df['settlement_date'] >= week_ago]
        elif period == 'month':
            month_ago = datetime.now() - timedelta(days=30)
            df = df[df['settlement_date'] >= month_ago]
        
        anomalies = df[df['is_anomaly'] == True]
        
        summary = {
            'total_transactions': len(df),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) * 100 if len(df) > 0 else 0,
            'anomaly_types': {
                'ml_detected': len(anomalies),
                'amount_anomalies': len(anomalies[anomalies['anomaly_type'].isin(['high_amount', 'low_amount'])]),
                'fee_anomalies': len(anomalies[anomalies['anomaly_type'] == 'fee_error'])
            },
            'severity_distribution': {
                'high': len(anomalies) // 3,
                'medium': len(anomalies) // 3,
                'low': len(anomalies) - (2 * (len(anomalies) // 3))
            }
        }
        
        return summary
    
    def _query_transaction(self, df: pd.DataFrame, args: dict) -> dict:
        """Mock transaction query"""
        transaction_id = args['transaction_id']
        transaction = df[df['transaction_id'] == transaction_id]
        
        if transaction.empty:
            return {"error": f"Transaction {transaction_id} not found"}
        
        row = transaction.iloc[0]
        
        response = {
            'transaction_id': transaction_id,
            'details': {
                'channel_id': row['channel_id'],
                'amount': float(row['amount']),
                'card_scheme': row['card_scheme'],
                'currency': row['currency'],
                'settlement_date': row['settlement_date'].strftime('%Y-%m-%d'),
                'fees': {
                    'interchange_fee': float(row['interchange_fee']),
                    'scheme_fee': float(row['scheme_fee']),
                    'acquirer_fee': float(row['acquirer_fee'])
                },
                'net_settlement': float(row['net_settlement'])
            },
            'anomaly_analysis': {
                'is_anomaly': bool(row['is_anomaly']),
                'anomaly_score': np.random.uniform(-1, -0.1) if row['is_anomaly'] else np.random.uniform(0.1, 0.5),
                'anomaly_types': {
                    'ml_detected': bool(row['is_anomaly']),
                    'amount_anomaly': row['anomaly_type'] in ['high_amount', 'low_amount'] if row['is_anomaly'] else False,
                    'fee_anomaly': row['anomaly_type'] == 'fee_error' if row['is_anomaly'] else False
                }
            }
        }
        
        return response

# Initialize MCP client
@st.cache_resource
def get_mcp_client():
    return MockMCPClient()

mcp_client = get_mcp_client()

# Helper functions

def load_dart_file(dart_type):
    """Load DART 312 or 313 sample data from processed directory."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if dart_type == 'DART 312':
            data_path = os.path.join(project_root, 'data', 'processed', 'dart_312_sample.csv')
        else:
            data_path = os.path.join(project_root, 'data', 'processed', 'dart_313_sample.csv')
        df = pd.read_csv(data_path)
        # Try to parse dates if present
        for col in ['transaction_datetime', 'settlement_date', 'processing_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='ignore')
        # Ensure is_anomaly is boolean
        if 'is_anomaly' in df.columns:
            df['is_anomaly'] = df['is_anomaly'].astype(str).str.lower().isin(['true', '1'])
        return df
    except Exception as e:
        return None

# --- File Upload Helper ---
def load_real_dart_file(dart_type, uploaded_file=None):
    """Load and parse real DART 312 or 313 data from upload or projdocs/backup/"""
    if uploaded_file is not None:
        f = io.StringIO(uploaded_file.getvalue().decode('utf-8'))
        reader = csv.reader(f)
        records = [row for row in reader]
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if dart_type == 'DART 312':
            data_path = os.path.join(project_root, 'projdocs', 'backup', 'WP_BADC_312TRC_V03_20250715_001.CSV')
        else:
            data_path = os.path.join(project_root, 'projdocs', 'backup', 'WP_BADC_313PRC_V03_20250715_001.CSV')
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            records = [row for row in reader]
    df = pd.DataFrame(records)
    df = df.rename(columns={0: 'record_type'})
    return df

# --- Anomaly Detection for Real DART Data ---
def detect_anomalies_312(df):
    # Add columns for anomaly flags and details
    df['is_anomaly'] = False
    df['anomaly_type'] = ''
    df['anomaly_reason'] = ''
    df['algorithm'] = ''
    # Example: Outlier detection for transaction amounts in record type 01
    if 'record_type' in df.columns:
        mask_01 = df['record_type'] == '01'
        if '23' in df.columns:  # Transaction Amount (field 23 in 312 type 01)
            try:
                amounts = pd.to_numeric(df.loc[mask_01, 23], errors='coerce')
                mean = amounts.mean()
                std = amounts.std()
                outlier_mask = (amounts > mean + 3*std) | (amounts < mean - 3*std)
                df.loc[mask_01 & outlier_mask, 'is_anomaly'] = True
                df.loc[mask_01 & outlier_mask, 'anomaly_type'] = 'outlier_amount'
                df.loc[mask_01 & outlier_mask, 'anomaly_reason'] = 'Transaction amount is a statistical outlier (3-sigma rule).'
                df.loc[mask_01 & outlier_mask, 'algorithm'] = 'Rule-based: Outlier Detection'
            except Exception:
                pass
        # Example: Duplicate detection (same PAN, amount, date)
        if all(x in df.columns for x in [11, 23, 21]):
            dups = df[mask_01].duplicated(subset=[11, 23, 21], keep=False)
            df.loc[mask_01 & dups, 'is_anomaly'] = True
            df.loc[mask_01 & dups, 'anomaly_type'] = 'duplicate_transaction'
            df.loc[mask_01 & dups, 'anomaly_reason'] = 'Duplicate transaction detected (same PAN, amount, date).'
            df.loc[mask_01 & dups, 'algorithm'] = 'Rule-based: Duplicate Check'
        # Example: Rejected transactions (record type 03)
        mask_03 = df['record_type'] == '03'
        df.loc[mask_03, 'is_anomaly'] = True
        df.loc[mask_03, 'anomaly_type'] = 'rejected_transaction'
        df.loc[mask_03, 'anomaly_reason'] = 'Transaction was rejected. See rejection reason code.'
        df.loc[mask_03, 'algorithm'] = 'Rule-based: Rejection Flag'
    return df

def detect_anomalies_313(df):
    # Add columns for anomaly flags and details
    df['is_anomaly'] = False
    df['anomaly_type'] = ''
    df['anomaly_reason'] = ''
    df['algorithm'] = ''
    # Example: Outlier detection for amount fields (field 7 or 8 depending on record type)
    if 'record_type' in df.columns:
        mask_16 = df['record_type'] == '16'
        if 7 in df.columns:
            try:
                amounts = pd.to_numeric(df.loc[mask_16, 7], errors='coerce')
                mean = amounts.mean()
                std = amounts.std()
                outlier_mask = (amounts > mean + 3*std) | (amounts < mean - 3*std)
                df.loc[mask_16 & outlier_mask, 'is_anomaly'] = True
                df.loc[mask_16 & outlier_mask, 'anomaly_type'] = 'outlier_chargeback'
                df.loc[mask_16 & outlier_mask, 'anomaly_reason'] = 'Chargeback value is a statistical outlier (3-sigma rule).'
                df.loc[mask_16 & outlier_mask, 'algorithm'] = 'Rule-based: Outlier Detection'
            except Exception:
                pass
        # Example: Negative settlement amounts (field 7 or 8 in some record types)
        for col in [7, 8]:
            if col in df.columns:
                neg_mask = pd.to_numeric(df[col], errors='coerce') < 0
                df.loc[neg_mask, 'is_anomaly'] = True
                df.loc[neg_mask, 'anomaly_type'] = 'negative_amount'
                df.loc[neg_mask, 'anomaly_reason'] = 'Negative value detected in settlement/chargeback.'
                df.loc[neg_mask, 'algorithm'] = 'Rule-based: Negative Value Check'
    return df

# --- Rejection Reason Code Definitions ---
def get_rejection_reason_description(code):
    """Get detailed description of DART 312 rejection reason codes."""
    rejection_codes = {
        '1300827893': 'Cardholder requested stop payment - Transaction declined by cardholder',
        '1610110033': 'Insufficient funds - Account has insufficient balance for transaction',
        '1610111253': 'Invalid card number - Card number format or checksum is invalid',
        '1610111254': 'Expired card - Card has passed its expiration date',
        '1610111255': 'Invalid CVV - Card verification value is incorrect',
        '1610111256': 'Card not present fraud - Transaction flagged for potential fraud',
        '1610111257': 'Do not honor - Issuing bank declined the transaction',
        '1610111258': 'Pick up card - Card should be retained by merchant',
        '1610111259': 'Refer to issuer - Contact card issuer for authorization',
        '1610111260': 'Invalid transaction - Transaction type not allowed for this card',
        '1610111261': 'Invalid amount - Transaction amount exceeds limits',
        '1610111262': 'Invalid merchant - Merchant ID not found or inactive',
        '1610111263': 'Duplicate transaction - Same transaction submitted multiple times',
        '1610111264': 'System error - Technical issue with payment processing',
        '1610111265': 'Network error - Communication issue with payment network',
        '1610111266': 'Timeout - Transaction timed out during processing',
        '1610111267': 'Invalid currency - Currency code not supported',
        '1610111268': 'Invalid MCC - Merchant category code not valid',
        '1610111269': 'AVS mismatch - Address verification failed',
        '1610111270': '3D Secure failure - 3D Secure authentication failed'
    }
    return rejection_codes.get(str(code), f'Unknown rejection code: {code}')

# --- Algorithm Definitions ---
def get_algorithm_definition(algorithm_name):
    """Get detailed definition and explanation of anomaly detection algorithms."""
    algorithms = {
        'rejected_transaction': {
            'name': 'Rejected Transaction Detection',
            'description': 'Identifies transactions that have been rejected by the payment processor or issuing bank.',
            'method': 'Rule-based detection using record type 03 (rejected transactions)',
            'threshold': 'Any transaction with record_type = "03"',
            'business_impact': 'High - Rejected transactions represent lost revenue and potential customer dissatisfaction',
            'investigation_priority': 'High - Immediate attention required to understand rejection reasons',
            'common_causes': [
                'Insufficient funds in customer account',
                'Invalid card details (expired, wrong CVV)',
                'Fraud detection triggers',
                'Card restrictions or limits',
                'Technical processing errors'
            ]
        },
        'pending_transaction': {
            'name': 'Pending Transaction Detection',
            'description': 'Identifies transactions that are pending settlement, which may indicate processing delays.',
            'method': 'Rule-based detection using record type 02 (pending transactions)',
            'threshold': 'Any transaction with record_type = "02"',
            'business_impact': 'Medium - Pending transactions affect cash flow timing',
            'investigation_priority': 'Medium - Monitor for unusual patterns or extended delays',
            'common_causes': [
                'Settlement processing delays',
                'Manual review requirements',
                'Network processing issues',
                'Merchant account holds',
                'Regulatory compliance checks'
            ]
        },
        'duplicate_transaction': {
            'name': 'Duplicate Transaction Detection',
            'description': 'Identifies potential duplicate transactions that may indicate processing errors or fraud.',
            'method': 'Statistical analysis comparing transaction amounts, timestamps, and merchant IDs',
            'threshold': 'Transactions with identical amount, merchant, and timestamp within 5 minutes',
            'business_impact': 'High - Duplicates can lead to double-charging customers',
            'investigation_priority': 'High - Immediate investigation required',
            'common_causes': [
                'Network retry mechanisms',
                'System processing errors',
                'Merchant terminal issues',
                'Intentional fraud attempts',
                'User interface double-clicks'
            ]
        },
        'outlier_amount': {
            'name': 'Transaction Amount Outlier Detection',
            'description': 'Identifies transactions with unusually high or low amounts compared to normal patterns.',
            'method': 'Statistical outlier detection using Z-score analysis (amounts > 3 standard deviations from mean)',
            'threshold': 'Transaction amount > 3σ from channel average',
            'business_impact': 'Medium - May indicate fraud, errors, or legitimate high-value transactions',
            'investigation_priority': 'Medium - Verify transaction legitimacy',
            'common_causes': [
                'Legitimate high-value purchases',
                'Fraudulent transactions',
                'Processing errors (decimal place issues)',
                'Test transactions',
                'Refund or chargeback processing'
            ]
        },
        'timing_mismatch': {
            'name': 'Settlement Timing Anomaly Detection',
            'description': 'Identifies transactions with unusual delays between transaction and settlement dates.',
            'method': 'Time-based analysis comparing transaction date to settlement date',
            'threshold': 'Settlement delay > 7 days from transaction date',
            'business_impact': 'Medium - Affects cash flow predictability',
            'investigation_priority': 'Low - Monitor for patterns',
            'common_causes': [
                'Weekend/holiday processing delays',
                'Manual review requirements',
                'Network processing issues',
                'Merchant account holds',
                'Regulatory compliance delays'
            ]
        },
        'settlement_mismatch': {
            'name': 'Settlement Amount Mismatch Detection',
            'description': 'Identifies discrepancies between DART 312 transaction amounts and DART 313 settlement amounts.',
            'method': 'Cross-file reconciliation between transaction and settlement records',
            'threshold': 'Difference > £0.01 between transaction and settlement amounts',
            'business_impact': 'High - Discrepancies affect financial reconciliation',
            'investigation_priority': 'High - Immediate reconciliation required',
            'common_causes': [
                'Processing fees or charges',
                'Currency conversion differences',
                'Partial settlements',
                'Processing errors',
                'Timing differences in reporting'
            ]
        },
        'missing_confirmation': {
            'name': 'Missing Settlement Confirmation Detection',
            'description': 'Identifies DART 312 transactions without corresponding DART 313 settlement confirmations.',
            'method': 'Cross-file analysis to find unmatched transaction records',
            'threshold': 'DART 312 transaction without DART 313 settlement within 3 days',
            'business_impact': 'High - Missing settlements affect financial reporting',
            'investigation_priority': 'High - Immediate investigation required',
            'common_causes': [
                'Settlement processing delays',
                'System integration issues',
                'Data transmission errors',
                'Manual processing requirements',
                'Network connectivity issues'
            ]
        },
        'chargeback_anomaly': {
            'name': 'Chargeback Pattern Anomaly Detection',
            'description': 'Identifies unusual patterns in chargeback amounts or frequencies.',
            'method': 'Statistical analysis of chargeback patterns and amounts',
            'threshold': 'Chargeback amount > 2σ from average or frequency > 5% of transactions',
            'business_impact': 'High - Chargebacks represent revenue loss and processing costs',
            'investigation_priority': 'High - Immediate fraud investigation required',
            'common_causes': [
                'Fraudulent transactions',
                'Customer disputes',
                'Processing errors',
                'Merchant service issues',
                'Card network disputes'
            ]
        },
        'funding_anomaly': {
            'name': 'Funding Swing Anomaly Detection',
            'description': 'Identifies unexpected positive or negative swings in funding amounts.',
            'method': 'Time-series analysis of funding patterns and amounts',
            'threshold': 'Funding swing > 50% from previous period average',
            'business_impact': 'High - Funding anomalies affect cash flow and financial planning',
            'investigation_priority': 'High - Immediate financial investigation required',
            'common_causes': [
                'Large transaction volumes',
                'Processing delays or accelerations',
                'System errors',
                'Regulatory changes',
                'Market events affecting payment volumes'
            ]
        }
    }
    return algorithms.get(algorithm_name, {
        'name': 'Unknown Algorithm',
        'description': 'Algorithm not defined',
        'method': 'Unknown',
        'threshold': 'Unknown',
        'business_impact': 'Unknown',
        'investigation_priority': 'Unknown',
        'common_causes': []
    })

# Main application
def main():
    st.markdown('<h1 class="main-header">🔍 Airlines Payment Anomaly Detection System (Real DART Data)</h1>', unsafe_allow_html=True)
    st.sidebar.header("Analysis Controls")
    dart_file = st.sidebar.selectbox("Select DART File", ["DART 312", "DART 313"])
    # File uploaders
    uploaded_312 = st.sidebar.file_uploader("Upload DART 312 File", type=["csv"], key="dart312") if dart_file == 'DART 312' else None
    uploaded_313 = st.sidebar.file_uploader("Upload DART 313 File", type=["csv"], key="dart313") if dart_file == 'DART 313' else None
    uploaded_file = uploaded_312 if dart_file == 'DART 312' else uploaded_313
    # Load data
    df = load_real_dart_file(dart_file, uploaded_file)
    if dart_file == 'DART 312':
        df = detect_anomalies_312(df)
    else:
        df = detect_anomalies_313(df)
    if df is None or df.empty:
        st.error(f"{dart_file} data not found or empty.")
        return
    # Date range filter (if date column present)
    date_col = 21 if dart_file == 'DART 312' else 13
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                df = df[(df[date_col].dt.date >= date_range[0]) & (df[date_col].dt.date <= date_range[1])]
        except Exception:
            pass
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Dashboard Overview", "Anomaly Detection", "Data Table"]
    )
    # Main content
    if analysis_type == "Dashboard Overview":
        show_dashboard_overview(df, dart_file)
    elif analysis_type == "Anomaly Detection":
        show_anomaly_detection(df, dart_file)
    elif analysis_type == "Data Table":
        show_data_table(df)

def show_dashboard_overview(df, dart_file):
    st.header(f"Dashboard Overview - {dart_file}")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
        st.metric("Total Anomalies", f"{anomaly_count:,}", delta=f"{(anomaly_count/len(df)*100):.1f}% rate" if len(df) else "")
    with col3:
        if 23 in df.columns:
            total_volume = pd.to_numeric(df[23], errors='coerce').sum()
            st.metric("Total Volume", f"£{total_volume:,.2f}")
        elif 7 in df.columns:
            total_volume = pd.to_numeric(df[7], errors='coerce').sum()
            st.metric("Total Volume", f"£{total_volume:,.2f}")
        else:
            st.metric("Total Volume", "N/A")
    
    st.header("📈 Data Visualizations")
    
    if 23 in df.columns or 7 in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            if 23 in df.columns:
                fig = px.histogram(
                    df, 
                    x=23, 
                    color='is_anomaly', 
                    nbins=50, 
                    title='Transaction Amount Distribution by Anomaly Status',
                    color_discrete_map={True: '#ff6b6b', False: '#4ecdc4'},
                    labels={'is_anomaly': 'Anomaly Status', 'count': 'Number of Transactions'}
                )
            else:
                fig = px.histogram(
                    df, 
                    x=7, 
                    color='is_anomaly', 
                    nbins=50, 
                    title='Amount Distribution by Anomaly Status',
                    color_discrete_map={True: '#ff6b6b', False: '#4ecdc4'},
                    labels={'is_anomaly': 'Anomaly Status', 'count': 'Number of Transactions'}
                )
            
            fig.update_layout(
                xaxis_title="Transaction Amount (£)",
                yaxis_title="Number of Transactions",
                legend_title="Anomaly Status"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Help text for amount distribution
            st.info("💡 **Chart Explanation:** This histogram shows the distribution of transaction amounts. Red bars represent anomalous transactions, while blue bars show normal transactions. Look for unusual spikes or patterns in the red bars that might indicate systematic issues.")
        
        with col2:
            st.subheader("Anomaly Type Breakdown")
            if 'anomaly_type' in df.columns and df['is_anomaly'].sum() > 0:
                anomaly_counts = df['anomaly_type'].value_counts()
                fig2 = px.bar(
                    x=anomaly_counts.index.astype(str), 
                    y=anomaly_counts.values, 
                    labels={'x': 'Anomaly Type', 'y': 'Count'},
                    title='Anomaly Type Distribution',
                    color=anomaly_counts.values,
                    color_continuous_scale='Reds'
                )
                fig2.update_layout(
                    xaxis_title="Anomaly Type",
                    yaxis_title="Number of Occurrences",
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Help text for anomaly type chart
                st.info("💡 **Chart Explanation:** This bar chart shows the frequency of each type of anomaly detected. Higher bars indicate more common anomaly types that may require priority attention. The color intensity indicates the relative frequency.")
            else:
                st.info("📊 No anomalies detected to display in the chart.")
        
        # Additional insights
        st.subheader("🔍 Key Insights")
        if 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
            anomalies = df[df['is_anomaly']]
            
            # Most common anomaly type
            if 'anomaly_type' in anomalies.columns:
                most_common = anomalies['anomaly_type'].mode().iloc[0] if not anomalies['anomaly_type'].mode().empty else 'None'
                st.markdown(f"**Most Common Anomaly:** {most_common}")
            
            # Anomaly rate trend
            if len(df) > 100:  # Only show if we have enough data
                st.markdown(f"**Overall Anomaly Rate:** {(len(anomalies)/len(df)*100):.2f}%")
                
                if (len(anomalies)/len(df)*100) > 5:
                    st.warning("⚠️ **High Anomaly Rate Detected:** Your anomaly rate is above 5%, which may indicate systematic issues requiring investigation.")
                elif (len(anomalies)/len(df)*100) > 2:
                    st.info("ℹ️ **Moderate Anomaly Rate:** Your anomaly rate is between 2-5%, which is within normal ranges but should be monitored.")
                else:
                    st.success("✅ **Low Anomaly Rate:** Your anomaly rate is below 2%, indicating good transaction processing health.")
        else:
            st.success("✅ **Clean Data:** No anomalies detected in the current dataset.")

def show_anomaly_detection(df, dart_file):
    st.header(f"Anomaly Detection - {dart_file}")
    
    # Summary statistics
    if 'is_anomaly' in df.columns:
        total_anomalies = df['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", f"{total_anomalies:,}")
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        with col3:
            if 'anomaly_type' in df.columns:
                unique_anomaly_types = df['anomaly_type'].nunique()
                st.metric("Anomaly Types", f"{unique_anomaly_types}")
    
    if 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
        anomalies = df[df['is_anomaly']]
        
        # Anomaly type distribution chart
        if 'anomaly_type' in anomalies.columns:
            st.subheader("📊 Anomaly Type Distribution")
            anomaly_counts = anomalies['anomaly_type'].value_counts()
            fig = px.pie(
                values=anomaly_counts.values, 
                names=anomaly_counts.index, 
                title="Distribution of Anomaly Types",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Help text for the chart
            st.info("💡 **Chart Explanation:** This pie chart shows the distribution of different types of anomalies detected in your data. The size of each slice represents the proportion of that anomaly type relative to all anomalies found.")
        
        st.subheader("🔍 Detailed Anomaly Analysis")
        
        for idx, row in anomalies.iterrows():
            anomaly_type = row.get('anomaly_type', 'Unknown')
            algorithm_info = get_algorithm_definition(anomaly_type)
            
            # Create expandable section for each anomaly
            with st.expander(f"Anomaly #{idx+1}: {algorithm_info['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Record Details:**
                    - **Record Type:** {row.get('record_type', 'N/A')}
                    - **Merchant/Party ID:** {row.get(2, row.get(1, 'N/A'))}
                    - **Amount:** £{row.get(23, row.get(7, 'N/A'))}
                    - **Date:** {row.get(21, row.get(13, 'N/A'))}
                    """)
                    
                    # Show rejection reason if available
                    if dart_file == 'DART 312' and 'rejection_reason_code' in row:
                        rejection_code = row.get('rejection_reason_code', '')
                        if rejection_code:
                            rejection_desc = get_rejection_reason_description(rejection_code)
                            st.markdown(f"""
                            **Rejection Details:**
                            - **Rejection Code:** {rejection_code}
                            - **Description:** {rejection_desc}
                            """)
                    
                    st.markdown(f"""
                    **Anomaly Reason:** {row.get('anomaly_reason', 'No reason provided')}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Risk Assessment:**
                    - **Business Impact:** {algorithm_info['business_impact']}
                    - **Investigation Priority:** {algorithm_info['investigation_priority']}
                    """)
                
                # Algorithm details
                st.markdown("### 🔬 Algorithm Details")
                st.markdown(f"""
                **Method:** {algorithm_info['method']}
                
                **Threshold:** {algorithm_info['threshold']}
                
                **Description:** {algorithm_info['description']}
                """)
                
                # Common causes
                if algorithm_info['common_causes']:
                    st.markdown("### 🎯 Common Causes")
                    for cause in algorithm_info['common_causes']:
                        st.markdown(f"- {cause}")
                
                st.markdown("---")
    else:
        st.success("✅ No anomalies detected in this data!")
        st.info("💡 **What this means:** Your payment data appears to be within normal parameters. This could indicate good transaction processing practices or that the data is from a period with minimal issues.")

def show_data_table(df):
    st.header("Data Table (showing up to 1000 rows)")
    max_rows = 1000
    if len(df) > max_rows:
        st.warning(f"Showing only the first {max_rows} rows out of {len(df)} total records due to browser and Pandas Styler limits.")
        df = df.head(max_rows)
    def highlight_anomalies(row):
        if 'is_anomaly' in row and row['is_anomaly']:
            return ['background-color: #ffebee'] * len(row)
        else:
            return [''] * len(row)
    st.dataframe(df.style.apply(highlight_anomalies, axis=1))

if __name__ == "__main__":
    main()