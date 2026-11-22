# 📡 Telecom Customer Churn Prediction & Retention Analysis

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to discontinue their service ("churn") based on billing, contract, and service usage data, paired with an interactive **Streamlit** web dashboard for visual analysis and automated retention strategies.

---

## 🎯 Project Overview
Customer retention is critical in the telecommunications industry, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project:
1. Cleans and preprocesses the **IBM Telco Customer Churn** dataset.
2. Trains and compares **Logistic Regression** and **Decision Tree** classification models.
3. Evaluates model performance using **Accuracy** and **Confusion Matrices**.
4. Provides an interactive **Streamlit** UI to explore churn trends and predict churn risk for new customer profiles in real time.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `DecisionTreeClassifier`)
* **Model Serialization:** Joblib
* **Web Interface & Visualization:** Streamlit

---

## 📂 Project Structure
```text
churn-project/
├── train.py                # Data cleaning, encoding, model training & evaluation
├── app.py                  # Streamlit web dashboard & live churn predictor
├── telco_churn_clean.csv   # Preprocessed dataset used for dashboard charts
├── models.pkl              # Saved trained models and feature columns
└── requirements.txt        # Python dependencies