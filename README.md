#Tea Garden Climate Risk Analyzer - Prototype
A lightweight, explainable prototype that uses historical climate data (temperature, rainfall, humidity + two additional features) to classify
climate risk and plant stress for tea gardens, and to generate actionable alerts and recommendations. Built as a Streamlit demo
with simple, interpretable ML models (Random Forest / Decision Tree) and a rule-based decision engine.

#Features
▪ Climate stress level classification (Healthy / Mild / Severe)
▪ Risk-level classification (Low / Moderate / High / Extreme)
▪ Simple rule-based alert engine for actionable recommendations
▪ Streamlit dashboard with dataset upload, EDA, manual input sliders, prediction panel

#Repository Structure
tea-climate-analyzer/
├─ data/
│  └─ Indian_Climate_Dataset_2024_2025.csv
├─ notebooks/
│  └─ eda.ipynb
├─ src/
│  ├─ features.py         # feature engineering and CSI label creation
│  ├─ modeling.py         # model training / evaluation / save/load
│  └─ app.py              # streamlit app
├─ models/
│  ├─ rf_risk.pkl
│  └─ dt_stress.pkl
├─ requirements.txt
├─ README.md              # this file
└─ LICENSE

#Dataset
data/Indian_Climate_Dataset_2024_2025.csv

#ML architecture
High-level pipeline:
1.Data ingestion and parsing (date handling)
2.Data cleaning (missing values, outlier handling)
3.Feature engineering (rolling aggregates, deviations, CSI)
4.Label creation (CSI → stress and risk labels)
5.Modeling:
  RandomForestClassifier for risk (baseline)
  DecisionTreeClassifier or LogisticRegression for stress (explainable)
6.Evaluation (accuracy, macro F1, confusion matrix, feature importances)
7.Model serialization (joblib.dump) and Streamlit-based delivery
8.Rule engine for alerts and recommendations

#Future Scope
▪ Integrate live weather APIs for near real-time inputs and automated model retraining on incoming data (live weather retraining).
▪ Add remote-sensing signals (NDVI / satellite rainfall validation).
▪ Include soil moisture, leaf wetness sensors, or farm-level sensors for better ground truth.
▪ Build a retraining pipeline and scheduled evaluation for model drift.
