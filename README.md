# 🔥 Network Log Analysis & Intrusion Detection

![GitHub](https://img.shields.io/github/license/your-username/your-repo)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange)

## 🌐 Overview
A high-performance **AI-powered Intrusion Detection System (IDS)** that analyzes network logs to detect anomalies and attacks using **machine learning**.

🚀 **Key Features**:
- Real-time log analysis with `tshark`
- Anomaly detection with `RandomForestClassifier`
- Interactive **Web Dashboard** for monitoring
- Secure model deployment using **FastAPI**
- CI/CD Integration with GitHub Actions ✅

## 🎯 Project Architecture
```
📂 security-log-analysis
 ├── 📂 data             # Raw network logs
 ├── 📂 models           # Trained ML models
 ├── 📂 dashboard        # Web-based monitoring
 ├── preprocess_data.py  # Data preprocessing script
 ├── train_model.py      # ML training script
 ├── infer.py            # Model inference
 ├── dashboard.html      # Live dashboard
 ├── requirements.txt    # Dependencies
 ├── .github/workflows   # CI/CD pipeline
```


## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd security-log-analysis

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage
```bash
# Run preprocessing
python preprocess_data.py

# Train model
python train_model.py

# Run inference
python infer.py --input data/sample_log.csv

# Launch the dashboard
python dashboard/app.py
```

## 🛡️ Security & Performance
✅ Uses **TShark** for efficient log collection.
✅ **Feature engineering** improves model accuracy.
✅ **Optimized RandomForest** for fast classification.

## 🛠️ Tech Stack
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Dash / Streamlit
- **CI/CD**: GitHub Actions + Docker

## 🤝 Contributing
Pull requests are welcome! **Guidelines:**
1. Fork the repo & create a new branch
2. Commit your changes
3. Open a PR 🚀



