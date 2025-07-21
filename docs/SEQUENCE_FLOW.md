# Payment Anomaly Detection System - Sequence Flow Documentation

## 1. System Initialization Sequence

### 1.1 Application Startup
```
1. Streamlit Application Launch
   ├── Load configuration files
   ├── Initialize logging system
   ├── Set up error handling
   ├── Load CSS styles and custom components
   └── Display main dashboard header
```

### 1.2 User Interface Initialization
```
1. Sidebar Controls Setup
   ├── DART file type selector (312/313)
   ├── File upload widgets
   ├── Date range filters
   └── Analysis type selector

2. Main Dashboard Area
   ├── Header with system title
   ├── Metrics display area
   ├── Visualization containers
   └── Data table area
```

## 2. Data Loading Sequence

### 2.1 File Selection and Upload
```
1. User selects DART file type (312 or 313)
   ↓
2. File upload widget becomes active
   ↓
3. User either:
   ├── Uploads custom CSV file
   └── Uses default file from backup directory
   ↓
4. File validation begins
```

### 2.2 File Processing Sequence
```
1. File Validation
   ├── Check file format (CSV)
   ├── Validate file size (< 100MB)
   ├── Check file encoding (UTF-8)
   └── Verify file is not empty
   ↓

2. File Reading
   ├── Open file with UTF-8 encoding
   ├── Read CSV line by line
   ├── Parse each row into list
   └── Store in memory as DataFrame
   ↓

3. Data Structure Validation
   ├── Verify minimum required columns
   ├── Check record type column (column 0)
   ├── Validate record types (00, 01, 02, 03, 99)
   └── Ensure data integrity
```

## 3. Data Processing Sequence

### 3.1 DART 312 Processing Flow
```
1. DataFrame Creation
   ├── Convert CSV rows to DataFrame
   ├── Rename column 0 to 'record_type'
   ├── Identify record types in data
   └── Filter valid records (00, 01, 02, 03, 99)
   ↓

2. Record Type Processing
   ├── Header Records (00): Extract metadata
   ├── Transaction Records (01): Process accepted transactions
   ├── Pending Records (02): Process pending transactions
   ├── Rejected Records (03): Process rejected transactions
   └── Trailer Records (99): Validate record counts
   ↓

3. Data Cleaning
   ├── Remove header and trailer records
   ├── Handle missing values
   ├── Convert data types (amounts to numeric)
   ├── Standardize date formats
   └── Remove duplicate records
```

### 3.2 DART 313 Processing Flow
```
1. DataFrame Creation
   ├── Convert CSV rows to DataFrame
   ├── Rename column 0 to 'record_type'
   ├── Identify record types in data
   └── Filter valid records (00, 01, 99)
   ↓

2. Record Type Processing
   ├── Header Records (00): Extract party information
   ├── Settlement Records (01): Process settlement data
   └── Trailer Records (99): Validate record counts
   ↓

3. Data Cleaning
   ├── Remove header and trailer records
   ├── Handle missing values
   ├── Convert amounts to numeric
   ├── Standardize date formats
   └── Remove duplicate records
```

## 4. Anomaly Detection Sequence

### 4.1 DART 312 Anomaly Detection Flow
```
1. Initialize Anomaly Detection
   ├── Create 'is_anomaly' column (default False)
   ├── Create 'anomaly_type' column (default None)
   ├── Create 'anomaly_reason' column (default None)
   └── Create 'algorithm' column (default None)
   ↓

2. Rule-Based Anomaly Detection
   ├── Rejected Transaction Detection
   │   ├── Check record_type == '03'
   │   ├── Mark as anomaly
   │   ├── Set anomaly_type = 'rejected_transaction'
   │   ├── Extract rejection reason code
   │   └── Set algorithm = 'Rule-based detection'
   │
   ├── Pending Transaction Detection
   │   ├── Check record_type == '02'
   │   ├── Mark as anomaly
   │   ├── Set anomaly_type = 'pending_transaction'
   │   └── Set algorithm = 'Rule-based detection'
   │
   └── Duplicate Transaction Detection
       ├── Group by amount, merchant_id, timestamp
       ├── Find transactions within 5-minute window
       ├── Mark duplicates as anomalies
       ├── Set anomaly_type = 'duplicate_transaction'
       └── Set algorithm = 'Statistical analysis'
   ↓

3. Statistical Anomaly Detection
   ├── Amount Outlier Detection
   │   ├── Calculate mean and standard deviation
   │   ├── Identify amounts > 3σ from mean
   │   ├── Mark as anomalies
   │   ├── Set anomaly_type = 'outlier_amount'
   │   └── Set algorithm = 'Z-score analysis'
   │
   └── Timing Mismatch Detection
       ├── Calculate settlement delays
       ├── Identify delays > 7 days
       ├── Mark as anomalies
       ├── Set anomaly_type = 'timing_mismatch'
       └── Set algorithm = 'Time-based analysis'
   ↓

4. Anomaly Classification
   ├── Assign business impact levels
   ├── Set investigation priorities
   ├── Generate detailed reasons
   └── Create algorithm descriptions
```

### 4.2 DART 313 Anomaly Detection Flow
```
1. Initialize Anomaly Detection
   ├── Create anomaly columns
   └── Set default values
   ↓

2. Cross-File Anomaly Detection
   ├── Settlement Mismatch Detection
   │   ├── Compare DART 312 vs DART 313 amounts
   │   ├── Identify discrepancies > £0.01
   │   ├── Mark as anomalies
   │   └── Set algorithm = 'Cross-file reconciliation'
   │
   ├── Missing Confirmation Detection
   │   ├── Find DART 312 without DART 313
   │   ├── Mark as anomalies
   │   └── Set algorithm = 'Cross-file analysis'
   │
   └── Chargeback Anomaly Detection
       ├── Analyze chargeback patterns
       ├── Identify unusual frequencies
       ├── Mark as anomalies
       └── Set algorithm = 'Statistical analysis'
   ↓

3. Funding Anomaly Detection
   ├── Time-series analysis of funding
   ├── Identify swings > 50% from average
   ├── Mark as anomalies
   └── Set algorithm = 'Time-series analysis'
```

## 5. Dashboard Rendering Sequence

### 5.1 Analysis Type Selection
```
1. User selects analysis type:
   ├── Dashboard Overview
   ├── Anomaly Detection
   └── Data Table
   ↓

2. Route to appropriate function
   ├── show_dashboard_overview()
   ├── show_anomaly_detection()
   └── show_data_table()
```

### 5.2 Dashboard Overview Rendering
```
1. Calculate Key Metrics
   ├── Total records count
   ├── Total anomalies count
   ├── Anomaly rate percentage
   └── Total transaction volume
   ↓

2. Generate Visualizations
   ├── Transaction Amount Distribution
   │   ├── Create histogram with Plotly
   │   ├── Color-code by anomaly status
   │   ├── Add labels and titles
   │   └── Display with help text
   │
   └── Anomaly Type Breakdown
       ├── Create bar chart with Plotly
       ├── Show anomaly type frequencies
       ├── Add color intensity
       └── Display with help text
   ↓

3. Generate Key Insights
   ├── Most common anomaly type
   ├── Anomaly rate analysis
   ├── Risk level assessment
   └── Display recommendations
```

### 5.3 Anomaly Detection Rendering
```
1. Calculate Summary Statistics
   ├── Total anomalies
   ├── Anomaly rate
   └── Unique anomaly types
   ↓

2. Generate Anomaly Distribution Chart
   ├── Create pie chart with Plotly
   ├── Show anomaly type distribution
   ├── Add percentages and labels
   └── Display with help text
   ↓

3. Render Detailed Anomaly Cards
   ├── For each anomaly:
   │   ├── Extract record details
   │   ├── Get algorithm definition
   │   ├── Get rejection reason (if applicable)
   │   ├── Create expandable card
   │   ├── Display risk assessment
   │   ├── Show algorithm details
   │   └── List common causes
   └── Handle no anomalies case
```

### 5.4 Data Table Rendering
```
1. Data Preparation
   ├── Apply date filters (if selected)
   ├── Limit to 1000 rows (if large dataset)
   ├── Prepare highlighting function
   └── Generate warning (if truncated)
   ↓

2. Table Rendering
   ├── Create Streamlit dataframe
   ├── Apply anomaly highlighting
   ├── Display with pagination
   └── Show row count information
```

## 6. User Interaction Sequence

### 6.1 File Upload Interaction
```
1. User selects file
   ↓
2. File validation begins
   ↓
3. If validation fails:
   ├── Display error message
   └── Return to file selection
   ↓
4. If validation passes:
   ├── Load and process file
   ├── Run anomaly detection
   ├── Update dashboard
   └── Display success message
```

### 6.2 Filter Interaction
```
1. User adjusts date range
   ↓
2. Filter validation
   ↓
3. Apply filters to data
   ↓
4. Recalculate metrics
   ↓
5. Update visualizations
   ↓
6. Refresh dashboard display
```

### 6.3 Analysis Type Switching
```
1. User selects new analysis type
   ↓
2. Clear current display
   ↓
3. Load new analysis function
   ↓
4. Generate new visualizations
   ↓
5. Update dashboard content
```

## 7. Error Handling Sequence

### 7.1 File Processing Errors
```
1. File not found
   ├── Display error message
   ├── Provide file path information
   └── Suggest alternative actions
   ↓

2. File format errors
   ├── Validate file structure
   ├── Display specific error details
   └── Provide format requirements
   ↓

3. Data validation errors
   ├── Identify specific validation failures
   ├── Display error details
   └── Suggest data corrections
```

### 7.2 Processing Errors
```
1. Memory errors (large files)
   ├── Implement row limiting
   ├── Display warning message
   └── Continue with limited data
   ↓

2. Anomaly detection errors
   ├── Log error details
   ├── Continue with available data
   └── Display partial results
   ↓

3. Visualization errors
   ├── Handle empty datasets
   ├── Provide fallback displays
   └── Show informative messages
```

## 8. Performance Optimization Sequence

### 8.1 Data Loading Optimization
```
1. Lazy loading implementation
   ├── Load data only when needed
   ├── Cache processed results
   └── Implement pagination
   ↓

2. Memory management
   ├── Process data in chunks
   ├── Release unused memory
   └── Monitor memory usage
```

### 8.2 Rendering Optimization
```
1. Efficient visualization generation
   ├── Use Plotly for interactive charts
   ├── Implement chart caching
   └── Optimize chart configurations
   ↓

2. UI responsiveness
   ├── Async processing where possible
   ├── Progressive loading
   └── User feedback during processing
```

## 9. Logging and Monitoring Sequence

### 9.1 Application Logging
```
1. System events
   ├── Application startup/shutdown
   ├── File processing events
   ├── Anomaly detection results
   └── User interactions
   ↓

2. Error logging
   ├── File processing errors
   ├── Anomaly detection errors
   ├── UI rendering errors
   └── System performance issues
```

### 9.2 Performance Monitoring
```
1. Processing time tracking
   ├── File loading time
   ├── Anomaly detection time
   ├── Visualization generation time
   └── Overall response time
   ↓

2. Resource usage monitoring
   ├── Memory consumption
   ├── CPU usage
   ├── File size handling
   └── Concurrent user handling
``` 