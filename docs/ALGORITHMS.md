# Anomaly Detection Algorithms Documentation

## 1. Algorithms Used in This Project

### 1.1 Rule-Based Algorithms

#### a) Rejected Transaction Detection
- **Definition:** Flags transactions with record type '03' (rejected) in DART 312 files.
- **Description:** Any transaction marked as rejected by the payment processor or issuing bank is considered an anomaly. The rejection reason code is extracted and explained.
- **Business Use:** Identifies failed payments, helps in root cause analysis, and reduces revenue leakage.

#### b) Pending Transaction Detection
- **Definition:** Flags transactions with record type '02' (pending) in DART 312 files.
- **Description:** Transactions pending settlement for an unusual duration are flagged as anomalies, indicating possible processing delays or issues.
- **Business Use:** Helps monitor cash flow and operational bottlenecks.

#### c) Duplicate Transaction Detection
- **Definition:** Flags transactions with identical amount, merchant, and timestamp within a 5-minute window.
- **Description:** Detects possible double-charging or system errors by grouping and comparing transaction features.
- **Business Use:** Prevents customer disputes and financial errors.

### 1.2 Statistical Algorithms

#### d) Outlier Amount Detection
- **Definition:** Flags transactions with amounts greater than 3 standard deviations from the mean (Z-score analysis).
- **Description:** Identifies unusually high or low transaction amounts that may indicate fraud or errors.
- **Business Use:** Detects potential fraud, mistakes, or test transactions.

#### e) Timing Mismatch Detection
- **Definition:** Flags transactions where the settlement date is more than 7 days after the transaction date.
- **Description:** Identifies delays in settlement that may affect cash flow or indicate operational issues.
- **Business Use:** Ensures timely settlement and highlights process inefficiencies.

### 1.3 Cross-File and Advanced Rules (DART 313)

#### f) Settlement Mismatch Detection
- **Definition:** Compares DART 312 and DART 313 records for amount mismatches greater than £0.01.
- **Description:** Flags discrepancies between transaction and settlement records.
- **Business Use:** Ensures financial reconciliation and accuracy.

#### g) Missing Confirmation Detection
- **Definition:** Flags DART 312 transactions without a corresponding DART 313 settlement within 3 days.
- **Description:** Identifies missing settlements that may affect reporting.
- **Business Use:** Ensures completeness of settlement data.

#### h) Chargeback Anomaly Detection
- **Definition:** Flags chargeback amounts or frequencies that are statistical outliers.
- **Description:** Uses statistical analysis to detect unusual chargeback patterns.
- **Business Use:** Identifies fraud or service issues.

#### i) Funding Anomaly Detection
- **Definition:** Flags funding swings greater than 50% from the previous period average.
- **Description:** Uses time-series analysis to detect unexpected changes in funding.
- **Business Use:** Monitors cash flow and financial health.

---

## 2. What Else Can Be Used (Future Enhancements)

### 2.1 Machine Learning Algorithms
- **Isolation Forest:** Unsupervised anomaly detection for high-dimensional data.
- **One-Class SVM:** Identifies outliers in complex datasets.
- **Autoencoders (Neural Networks):** Learns normal patterns and flags deviations.
- **LOF (Local Outlier Factor):** Detects local density anomalies.
- **Random Cut Forest:** Scalable anomaly detection for streaming data.
- **Clustering (DBSCAN, K-Means):** Finds unusual clusters or points far from clusters.

### 2.2 Ensemble and Hybrid Approaches
- Combine rule-based and ML models for higher accuracy.
- Use supervised learning if labeled anomaly data is available.

### 2.3 Real-Time and Streaming Algorithms
- **Online anomaly detection:** For real-time payment streams (e.g., using AWS Kinesis, Apache Kafka).

---

## 3. Algorithm Definitions and Descriptions

- **Rule-Based:** Uses explicit business rules and thresholds. Easy to explain, fast, but may miss subtle patterns.
- **Statistical:** Uses statistical properties (mean, std, quantiles) to flag outliers. Good for numeric data.
- **Machine Learning:** Learns normal vs. anomalous patterns from data. Can detect complex, non-linear anomalies but may require more data and tuning.

---

## 4. References
- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [AWS Fraud Detection ML](https://aws.amazon.com/fraud-detector/)
- [PyOD: Python Outlier Detection](https://pyod.readthedocs.io/en/latest/) 