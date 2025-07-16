import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="BA Payment Anomaly Detection",
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
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

class MockMCPClient:
    """
    Mock MCP client for demonstration purposes
    """
    
    def __init__(self):
        # Simplified path for cloud deployment
        self.data_path = 'data/sample_transactions.csv'
        
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Simulate MCP tool calls"""
        try:
            if not os.path.exists(self.data_path):
                return {"error": "Sample data not found. Using demo data."}
            
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
@st.cache_data
def load_sample_data():
    """Load sample data for visualizations"""
    try:
        df = pd.read_csv('data/sample_transactions.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Sample data file not found. Please ensure 'data/sample_transactions.csv' exists.")
        return None

def create_anomaly_chart(df):
    """Create anomaly detection chart"""
    if df is None:
        return None
    
    # Daily transaction volume
    daily_volume = df.groupby(df['settlement_date'].dt.date).agg({
        'transaction_id': 'count',
        'amount': 'sum',
        'is_anomaly': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Transaction volume
    fig.add_trace(go.Scatter(
        x=daily_volume['settlement_date'],
        y=daily_volume['transaction_id'],
        mode='lines+markers',
        name='Daily Transactions',
        line=dict(color='blue')
    ))
    
    # Anomaly count
    fig.add_trace(go.Scatter(
        x=daily_volume['settlement_date'],
        y=daily_volume['is_anomaly'],
        mode='markers',
        name='Daily Anomalies',
        marker=dict(color='red', size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Daily Transaction Volume and Anomalies',
        xaxis_title='Date',
        yaxis_title='Transaction Count',
        yaxis2=dict(
            title='Anomaly Count',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    return fig

def create_amount_distribution_chart(df):
    """Create transaction amount distribution chart"""
    if df is None:
        return None
    
    fig = px.histogram(
        df,
        x='amount',
        color='is_anomaly',
        nbins=50,
        title='Transaction Amount Distribution',
        labels={'amount': 'Transaction Amount (£)', 'count': 'Count'},
        color_discrete_map={True: 'red', False: 'blue'}
    )
    
    return fig

def create_channel_anomaly_chart(df):
    """Create channel anomaly chart"""
    if df is None:
        return None
    
    channel_stats = df.groupby('channel_id').agg({
        'transaction_id': 'count',
        'is_anomaly': 'sum',
        'amount': 'mean'
    }).reset_index()
    
    channel_stats['anomaly_rate'] = (channel_stats['is_anomaly'] / channel_stats['transaction_id']) * 100
    
    fig = px.bar(
        channel_stats,
        x='channel_id',
        y='anomaly_rate',
        title='Anomaly Rate by Channel',
        labels={'anomaly_rate': 'Anomaly Rate (%)', 'channel_id': 'Channel ID'}
    )
    
    return fig

# Main application
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 British Airways Payment Anomaly Detection</h1>', unsafe_allow_html=True)
    
    # Info box
    st.info("🚀 **MVP Demo** - Payment Settlement & Reconciliation Anomaly Detection System using AI/ML")
    
    # Load sample data
    df = load_sample_data()
    
    if df is None:
        st.error("❌ Sample data not available. This is a demo with limited functionality.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['settlement_date'].min().date(), df['settlement_date'].max().date()),
        min_value=df['settlement_date'].min().date(),
        max_value=df['settlement_date'].max().date()
    )
    
    # Channel filter
    channel_options = ['All'] + sorted(df['channel_id'].unique().tolist())
    selected_channel = st.sidebar.selectbox("Select Channel", channel_options)
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["🏠 Dashboard Overview", "🔍 Anomaly Detection", "🔎 Transaction Query"]
    )
    
    # Main content area
    if analysis_type == "🏠 Dashboard Overview":
        show_dashboard_overview(df)
    elif analysis_type == "🔍 Anomaly Detection":
        show_anomaly_detection(date_range, selected_channel)
    elif analysis_type == "🔎 Transaction Query":
        show_transaction_query()
    
    # Footer
    st.markdown("""
    <div class="footer">
        💡 Payment Anomaly Detection MVP | Built by Gaurav | AI/ML Solution Architecture Demo
    </div>
    """, unsafe_allow_html=True)

def show_dashboard_overview(df):
    """Show dashboard overview"""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{len(df):,}",
            delta=f"{len(df[df['settlement_date'] >= (datetime.now() - timedelta(days=1))])} today"
        )
    
    with col2:
        anomaly_count = df['is_anomaly'].sum()
        st.metric(
            "Total Anomalies",
            f"{anomaly_count:,}",
            delta=f"{(anomaly_count/len(df)*100):.1f}% rate"
        )
    
    with col3:
        total_volume = df['amount'].sum()
        st.metric(
            "Total Volume",
            f"£{total_volume:,.2f}",
            delta=f"£{df['amount'].mean():.2f} avg"
        )
    
    with col4:
        processing_time = df['processing_time_ms'].mean()
        st.metric(
            "Avg Processing Time",
            f"{processing_time:.0f}ms",
            delta=f"{df['processing_time_ms'].std():.0f}ms σ"
        )
    
    # Charts
    st.header("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        anomaly_chart = create_anomaly_chart(df)
        if anomaly_chart:
            st.plotly_chart(anomaly_chart, use_container_width=True)
    
    with col2:
        amount_chart = create_amount_distribution_chart(df)
        if amount_chart:
            st.plotly_chart(amount_chart, use_container_width=True)
    
    # Channel analysis
    channel_chart = create_channel_anomaly_chart(df)
    if channel_chart:
        st.plotly_chart(channel_chart, use_container_width=True)

def show_anomaly_detection(date_range, selected_channel):
    """Show anomaly detection interface"""
    st.header("🔍 Anomaly Detection")
    
    # Detection controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Analyzing transactions..."):
                # Prepare arguments
                args = {
                    'date_from': date_range[0].strftime('%Y-%m-%d'),
                    'date_to': date_range[1].strftime('%Y-%m-%d')
                }
                
                if selected_channel != 'All':
                    args['channel_id'] = selected_channel
                
                # Call MCP tool
                result = mcp_client.call_tool("detect_payment_anomalies", args)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.session_state.anomaly_results = result
    
    with col2:
        if st.button("📊 Get Summary"):
            with st.spinner("Generating summary..."):
                result = mcp_client.call_tool("get_anomaly_summary", {"period": "month"})
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.session_state.summary_results = result
    
    # Display results
    if hasattr(st.session_state, 'anomaly_results'):
        st.subheader("🚨 Anomaly Detection Results")
        
        result = st.session_state.anomaly_results
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", result['analysis_period']['total_transactions'])
        
        with col2:
            st.metric("Anomalies Detected", result['anomalies_detected'])
        
        with col3:
            if result['analysis_period']['total_transactions'] > 0:
                anomaly_rate = (result['anomalies_detected'] / result['analysis_period']['total_transactions']) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # Anomaly details
        if result['anomaly_details']:
            st.subheader("📋 Anomaly Details")
            
            for anomaly in result['anomaly_details']:
                severity = "high" if anomaly['anomaly_score'] < -0.5 else "medium" if anomaly['anomaly_score'] < -0.2 else "low"
                
                st.markdown(f"""
                <div class="metric-card anomaly-{severity}">
                    <strong>Transaction ID:</strong> {anomaly['transaction_id']}<br>
                    <strong>Channel:</strong> {anomaly['channel_id']}<br>
                    <strong>Amount:</strong> £{anomaly['amount']:.2f}<br>
                    <strong>Anomaly Score:</strong> {anomaly['anomaly_score']:.3f}<br>
                    <strong>Timestamp:</strong> {anomaly['timestamp']}<br>
                    <strong>Types:</strong> ML: {anomaly['anomaly_types']['ml_detected']}, Amount: {anomaly['anomaly_types']['amount_anomaly']}, Fee: {anomaly['anomaly_types']['fee_anomaly']}
                </div>
                """, unsafe_allow_html=True)
    
    # Display summary
    if hasattr(st.session_state, 'summary_results'):
        st.subheader("📊 Summary Statistics")
        
        summary = st.session_state.summary_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(summary['anomaly_types'])
        
        with col2:
            st.json(summary['severity_distribution'])

def show_transaction_query():
    """Show transaction query interface"""
    st.header("🔎 Transaction Query")
    
    # Query interface
    transaction_id = st.text_input("Enter Transaction ID", placeholder="e.g., TXN_20240714_000001")
    
    if st.button("🔍 Query Transaction") and transaction_id:
        with st.spinner("Querying transaction..."):
            result = mcp_client.call_tool("query_specific_transaction", {"transaction_id": transaction_id})
            
            if 'error' in result:
                st.error(result['error'])
            else:
                st.session_state.transaction_result = result
    
    # Display result
    if hasattr(st.session_state, 'transaction_result'):
        result = st.session_state.transaction_result
        
        # Transaction details
        st.subheader("💳 Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(result['details'])
        
        with col2:
            st.json(result['anomaly_analysis'])
        
        # Anomaly status
        if result['anomaly_analysis']['is_anomaly']:
            st.error("🚨 This transaction is flagged as an anomaly!")
        else:
            st.success("✅ This transaction appears normal.")

if __name__ == "__main__":
    main()