# PAYSAFE AI — Real-Time UPI Fraud Risk Telemetry & Decisioning Engine

A production-grade UPI transaction fraud detection and risk telemetry console built with **Python**, **scikit-learn**, and **Streamlit**.

![PAYSAFE AI Architecture](https://img.shields.io/badge/Security_Engine-v2.4-1d4ed8?style=flat-square)
![Model](https://img.shields.io/badge/Model-Random_Forest-0f172a?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9994-2563eb?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-98.98%25-0284c7?style=flat-square)

---

## Overview

PAYSAFE AI provides real-time fraud probability evaluation for Indian Unified Payments Interface (UPI) transactions. The machine learning pipeline is trained on **26,393 transaction records** with high-dimensional behavioral signals, featuring an institutional-grade security HUD and adaptive telemetry tracking.

### Key Capabilities

- **High-Performance Risk Estimator**: Random Forest baseline trained across 26.3K UPI transactions achieving **0.9994 ROC-AUC** and **98.98% Accuracy**.
- **Institutional Decisioning Protocol**:
  - **Tier 1 (Safe, 0–29%)**: Auto-Clearance & Instant Settlement Approved.
  - **Tier 2 (Elevated, 30–59%)**: Challenge Verification (Step-Up Biometric / SMS OTP).
  - **Tier 3 (Critical, 60–100%)**: Intercept & Freeze (Immediate hold and SIEM escalation).
- **Multi-Factor Behavioral Telemetry**:
  - Financial Metrics: Transaction Amount (INR), Velocity (1 Hr), Beneficiary Account Age.
  - Device & Network: New Payee, Unusual Device Fingerprint, Geolocation Anomaly.
  - Auth & Behavioral: Night Transaction (22:00–05:00), Rapid Failed Attempts, Time-Pressure / Social Engineering Signals.
- **Responsive Fintech Console**:
  - **Desktop (>1100px)**: Twin-panel side-by-side console fitted to single viewport height.
  - **Tablet (651px–1100px)**: 100% full-width stacked cards with 3-column input matrix.
  - **Mobile (≤650px)**: Touch-optimized single-column interface.
- **Enterprise Light Aesthetic**: Crisp white panels (`#ffffff`) on slate background (`#f8fafc`) with zero emoji dependencies (SVG vector iconography).

---

## Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/thewecas-dev/paysafe-ai.git
cd paysafe-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

The pre-trained model bundle is included in `models/fraud_model.joblib`. To retrain:

```bash
python train_model.py
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit theme & server configuration (Light mode)
├── data/
│   ├── fraud_dataset.csv     # 26,393 UPI transaction dataset
│   └── transactions.csv      # Fallback synthetic transaction seed
├── models/
│   └── fraud_model.joblib    # Serialized Random Forest bundle (estimator + metadata)
├── app.py                    # Interactive Streamlit risk telemetry console
├── train_model.py            # Model training & ROC-AUC evaluation pipeline
├── generate_data.py          # Synthetic data generator
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Machine Learning Pipeline

| Metric | Score |
| :--- | :--- |
| **Model** | Random Forest Classifier (100 Estimators) |
| **Training Records** | 26,393 Transactions |
| **Accuracy** | 98.98% |
| **ROC-AUC** | 0.9994 |
| **Inference Latency** | < 2.1 ms |

---

## Disclaimer

This system is an academic and engineering prototype designed for security risk analysis and research. It does not connect to NPCI, UPI live switch APIs, or real banking cores.
