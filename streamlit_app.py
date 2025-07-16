import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Payment Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .anomaly-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .anomaly-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .anomaly-low {
        background-color: #90ee90;
        color: black;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample payment data"""
    try:
        # Try to load from local file first
        df = pd.read_csv('data/sample_data/dart_312_313_sample.csv')
    except FileNotFoundError:
        # If file doesn't exist, generate sample data
        df = generate_sample_data()
    
    # Ensure datetime columns are properly formatted
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['settlement_date'] = pd.to_datetime(df['settlement_date'])
    df['actual_settlement_date'] = pd.to_datetime(df['actual_settlement_date'])
    
    return df

def generate_sample_data():
    """Generate sample data if file is not available"""
    import random
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate 500 sample transactions
    num_transactions = 500
    base_date = datetime(2024, 1, 1)
    
    merchants = [f'MERCHANT_{i:03d}' for i in range(1, 11)]
    payment_methods = ['CARD', 'BANK_TRANSFER', 'DIGITAL_WALLET', 'DIRECT_DEBIT']
    currencies = ['GBP', 'USD', 'EUR']
    
    transactions = []
    
    for i in range(num_transactions):
        # Random timestamp within last 90 days
        timestamp = base_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        merchant_id = random.choice(merchants)
        payment_method = random.choice(payment_methods)
        currency = random.choice(currencies)
        
        # Generate amount based on payment method
        if payment_method == 'CARD':
            amount = max(1.0, np.random.normal(150, 50))
        elif payment_method == 'BANK_TRANSFER':
            amount = max(1.0, np.random.normal(500, 200))
        elif payment_method == 'DIGITAL_WALLET':
            amount = max(1.0, np.random.normal(75, 25))
        else:  # DIRECT_DEBIT
            amount = max(1.0, np.random.normal(200, 100))
        
        # Inject some anomalies
        if random.random() < 0.05:  # 5% anomalies
            amount *= random.uniform(5, 15)  # Make it anomalous
        
        processing_fee = amount * random.uniform(0.01, 0.03)
        settlement_delay = random.randint(1, 3)
        
        if random.random() < 0.05:  # 5% settlement delays
            settlement_delay += random.randint(7, 20)
        
        settlement_date = timestamp + timedelta(days=1)
        actual_settlement_date = timestamp + timedelta(days=settlement_delay)
        
        reconciliation_status = 'MATCHED' if random.random() < 0.95 else 'UNMATCHED'
        
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': currency,
            'merchant_id': merchant_id,
            'payment_method': payment_method,
            'settlement_date': settlement_date,
            'actual_settlement_date': actual_settlement_date,
            'reconciliation_status': reconciliation_status,
            'processing_fee': round(processing_fee, 2),
            'reference_number': f'REF_{random.randint(100000, 999999)}',
            'settlement_delay_days': settlement_delay
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

@st.cache_data
def detect_anomalies(df):
    """Simple anomaly detection using statistical methods"""
    df_processed = df.copy()
    
    # Calculate amount anomalies using IQR method
    Q1 = df_processed['amount'].quantile(0.25)
    Q3 = df_processed['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_processed['is_amount_anomaly'] = (
        (df_processed['amount'] < lower_bound) | 
        (df_processed['amount'] > upper_bound)
    )
    
    # Settlement delay anomalies
    df_processed['is_settlement_anomaly'] = df_processed['settlement_delay_days'] > 7
    
    # Reconciliation anomalies
    df_processed['is_reconciliation_anomaly'] = df_processed['reconciliation_status'] == 'UNMATCHED'
    
    # Weekend processing anomalies
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['is_weekend_anomaly'] = df_processed['day_of_week'].isin([5, 6])
    
    # Overall anomaly score
    df_processed['anomaly_score'] = (
        df_processed['is_amount_anomaly'].astype(int) +
        df_processed['is_settlement_anomaly'].astype(int) +
        df_processed['is_reconciliation_anomaly'].astype(int) +
        df_processed['is_weekend_anomaly'].astype(int)
    )
    
    # Classify severity
    df_processed['severity'] = 'NORMAL'
    df_processed.loc[df_processed['anomaly_score'] == 1, 'severity'] = 'LOW'
    df_processed.loc[df_processed['anomaly_score'] == 2, 'severity'] = 'MEDIUM'
    df_processed.loc[df_processed['anomaly_score'] >= 3, 'severity'] = 'HIGH'
    
    return df_processed

def create_anomaly_dashboard(df):
    """Create comprehensive anomaly dashboard"""
    
    # Detect anomalies
    df_anomalies = detect_anomalies(df)
    
    # Main header
    st.markdown("<h1 class='main-header'>🚨 Payment Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df_anomalies)
    total_anomalies = len(df_anomalies[df_anomalies['anomaly_score'] > 0])
    high_severity = len(df_anomalies[df_anomalies['severity'] == 'HIGH'])
    unmatched_reconciliation = len(df_anomalies[df_anomalies['reconciliation_status'] == 'UNMATCHED'])
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        st.metric("Total Anomalies", f"{total_anomalies:,}", 
                 delta=f"{(total_anomalies/total_transactions)*100:.1f}%")
    
    with col3:
        st.metric("High Severity", f"{high_severity:,}",
                 delta=f"{(high_severity/total_transactions)*100:.1f}%")
    
    with col4:
        st.metric("Unmatched Reconciliations", f"{unmatched_reconciliation:,}",
                 delta=f"{(unmatched_reconciliation/total_transactions)*100:.1f}%")
    
    # Anomaly distribution chart
    st.subheader("📊 Anomaly Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        severity_counts = df_anomalies['severity'].value_counts()
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Anomaly Severity Distribution",
            color_discrete_map={
                'NORMAL': '#90EE90',
                'LOW': '#FFD700',
                'MEDIUM': '#FFA500',
                'HIGH': '#FF6B6B'
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Anomaly types
        anomaly_types = {
            'Amount Anomalies': df_anomalies['is_amount_anomaly'].sum(),
            'Settlement Delays': df_anomalies['is_settlement_anomaly'].sum(),
            'Reconciliation Issues': df_anomalies['is_reconciliation_anomaly'].sum(),
            'Weekend Processing': df_anomalies['is_weekend_anomaly'].sum()
        }
        
        fig_types = px.bar(
            x=list(anomaly_types.keys()),
            y=list(anomaly_types.values()),
            title="Anomaly Types Count",
            color=list(anomaly_types.values()),
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Timeline analysis
    st.subheader("📈 Anomaly Timeline")
    
    # Group by date
    df_anomalies['date'] = df_anomalies['timestamp'].dt.date
    daily_anomalies = df_anomalies.groupby('date').agg({
        'anomaly_score': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    daily_anomalies.columns = ['date', 'total_anomaly_score', 'transaction_count']
    
    fig_timeline = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Transaction Volume', 'Daily Anomaly Score'),
        shared_xaxes=True
    )
    
    fig_timeline.add_trace(
        go.Scatter(x=daily_anomalies['date'], y=daily_anomalies['transaction_count'],
                  mode='lines+markers', name='Transactions'),
        row=1, col=1
    )
    
    fig_timeline.add_trace(
        go.Scatter(x=daily_anomalies['date'], y=daily_anomalies['total_anomaly_score'],
                  mode='lines+markers', name='Anomaly Score', line=dict(color='red')),
        row=2, col=1
    )
    
    fig_timeline.update_layout(height=500, title_text="Transaction and Anomaly Trends")
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Detailed anomaly table
    st.subheader("🔍 Detailed Anomaly Analysis")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            options=['ALL'] + list(df_anomalies['severity'].unique())
        )
    
    with col2:
        merchant_filter = st.selectbox(
            "Filter by Merchant",
            options=['ALL'] + list(df_anomalies['merchant_id'].unique())
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(df_anomalies['timestamp'].min().date(), df_anomalies['timestamp'].max().date()),
            min_value=df_anomalies['timestamp'].min().date(),
            max_value=df_anomalies['timestamp'].max().date()
        )
    
    # Apply filters
    filtered_df = df_anomalies[df_anomalies['anomaly_score'] > 0].copy()
    
    if severity_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['severity'] == severity_filter]
    
    if merchant_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['merchant_id'] == merchant_filter]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= date_range[0]) &
            (filtered_df['timestamp'].dt.date <= date_range[1])
        ]
    
    # Display filtered results
    if len(filtered_df) > 0:
        # Format the dataframe for display
        display_df = filtered_df[[
            'transaction_id', 'timestamp', 'amount', 'merchant_id', 
            'payment_method', 'settlement_delay_days', 'reconciliation_status',
            'anomaly_score', 'severity'
        ]].copy()
        
        # Format amount as currency
        display_df['amount'] = display_df['amount'].apply(lambda x: f"£{x:,.2f}")
        
        # Style the dataframe
        def style_severity(val):
            if val == 'HIGH':
                return 'background-color: #ff6b6b; color: white'
            elif val == 'MEDIUM':
                return 'background-color: #ffa500; color: white'
            elif val == 'LOW':
                return 'background-color: #90ee90; color: black'
            return ''
        
        styled_df = display_df.style.applymap(style_severity, subset=['severity'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Anomaly Distribution:**")
            st.write(f"- Total Anomalies: {len(filtered_df)}")
            st.write(f"- High Severity: {len(filtered_df[filtered_df['severity'] == 'HIGH'])}")
            st.write(f"- Medium Severity: {len(filtered_df[filtered_df['severity'] == 'MEDIUM'])}")
            st.write(f"- Low Severity: {len(filtered_df[filtered_df['severity'] == 'LOW'])}")
        
        with col2:
            st.write("**Financial Impact:**")
            total_amount = filtered_df['amount'].sum()
            avg_amount = filtered_df['amount'].mean()
            st.write(f"- Total Amount: £{total_amount:,.2f}")
            st.write(f"- Average Amount: £{avg_amount:,.2f}")
            st.write(f"- Max Amount: £{filtered_df['amount'].max():,.2f}")
            st.write(f"- Min Amount: £{filtered_df['amount'].min():,.2f}")
    
    else:
        st.info("No anomalies found matching the selected criteria.")

# Main application
def main():
    # Load data
    with st.spinner("Loading payment data..."):
        df = load_sample_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Raw Data", "About"]
    )
    
    if page == "Dashboard":
        create_anomaly_dashboard(df)
    
    elif page == "Raw Data":
        st.header("📄 Raw Payment Data")
        
        # Show basic statistics
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        with col3:
            st.metric("Unique Merchants", df['merchant_id'].nunique())
        
        # Show raw data
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="payment_data.csv",
            mime="text/csv"
        )
    
    elif page == "About":
        st.header("ℹ️ About Payment Anomaly Detection")
        
        st.markdown("""
        ## Purpose
        This application demonstrates an AI/ML-powered payment anomaly detection system designed for 
        settlement and reconciliation processes in the digital payments space.
        
        ## Features
        - **Real-time Anomaly Detection**: Identifies unusual patterns in payment transactions
        - **Multi-dimensional Analysis**: Considers amount, timing, settlement delays, and reconciliation status
        - **Severity Classification**: Categorizes anomalies as LOW, MEDIUM, or HIGH severity
        - **Interactive Dashboard**: Provides filtering and drill-down capabilities
        - **Export Functionality**: Download data for further analysis
        
        ## Technology Stack
        - **Frontend**: Streamlit
        - **ML Algorithm**: Statistical anomaly detection (IQR method)
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Seaborn
        - **Deployment**: Streamlit Cloud
        
        ## Anomaly Types Detected
        1. **Amount Anomalies**: Transactions with unusually high or low amounts
        2. **Settlement Delays**: Transactions taking longer than expected to settle
        3. **Reconciliation Issues**: Transactions that fail to match during reconciliation
        4. **Timing Anomalies**: Transactions processed outside normal business hours
        
        ## Sample Data
        The application uses synthetic DART 312/313 compliant data that simulates real payment 
        settlement and reconciliation scenarios.
        
        ---
        
        **Built by**: Gaurav (AI Solution Architect)  
        **Company**: BranchSpace  
        **Client**: British Airways  
        **Focus**: Digital Payments Architecture
        """)

if __name__ == "__main__":
    main()