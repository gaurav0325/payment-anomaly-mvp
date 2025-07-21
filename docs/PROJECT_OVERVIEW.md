# Payment Anomaly Detection System - Project Documentation

## Project Summary

This project is a payment anomaly detection and analytics dashboard for Worldpay DART 312/313 files. It enables users to upload, analyze, and visualize payment transaction data, highlighting anomalies using both rule-based and statistical methods. The dashboard provides detailed explanations, business impact, and investigation guidance for each anomaly.

## Key Features
- Upload and analyze DART 312/313 files (sample or real)
- Rule-based and statistical anomaly detection
- Detailed anomaly explanations and algorithm definitions
- Interactive dashboard with metrics, charts, and anomaly cards
- Data table with anomaly highlighting
- Export and reporting capabilities
- Modular, extensible codebase

## Technology Stack
- **Frontend:** Streamlit, Plotly, HTML/CSS
- **Backend:** Python 3.8+, Pandas, NumPy, Scikit-learn (optional for ML), CSV
- **Data Storage:** CSV files, Pandas DataFrames (in-memory)
- **Cloud/Infra:** Docker (optional), AWS (recommended for production)
- **Version Control:** Git

## File Structure
- `src/frontend/streamlit_app.py` — Main dashboard UI and logic
- `generate_sample_data.py` — Sample data generator
- `projdocs/backup/` — Real DART 312/313 files
- `docs/` — Documentation
- `requirements.txt` — Python dependencies

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run dashboard: `streamlit run src/frontend/streamlit_app.py`
3. Upload or select DART files, analyze, and export results

## Extensibility
- Add new anomaly detection rules or ML models in the backend
- Integrate with databases or cloud storage for large-scale data
- Enhance UI with more visualizations or export options

## Contact
For questions or contributions, contact the project maintainer. 