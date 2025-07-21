# Payment Anomaly Detection System - Quick README

## Overview
This project provides a dashboard for detecting and analyzing anomalies in Worldpay DART 312/313 payment files. It supports both rule-based and statistical anomaly detection, with detailed explanations and interactive visualizations.

## Features
- Upload and analyze DART 312/313 files (sample or real)
- Rule-based and statistical anomaly detection
- Detailed anomaly explanations and algorithm definitions
- Interactive dashboard with metrics, charts, and anomaly cards
- Data table with anomaly highlighting
- Export and reporting capabilities

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```bash
   streamlit run src/frontend/streamlit_app.py
   ```
3. Use the sidebar to upload/select DART files, filter data, and choose analysis type.

## Documentation
- **Architecture:** `docs/ARCHITECTURE.md`
- **Sequence Flow:** `docs/SEQUENCE_FLOW.md`
- **Algorithms:** `docs/ALGORITHMS.md`
- **Project Overview:** `docs/PROJECT_OVERVIEW.md`
- **Security & AWS:** `docs/SECURITY_AND_AWS.md`

## Tech Stack
- Python, Streamlit, Pandas, Plotly, NumPy
- (Optional) Scikit-learn for ML
- Docker, AWS (for production)

## Security
See `docs/SECURITY_AND_AWS.md` for best practices on cloud hosting and data protection.

## Contact
For questions or contributions, contact the project maintainer. 