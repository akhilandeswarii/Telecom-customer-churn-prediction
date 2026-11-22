import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Collect customer dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Step 2: Clean and prepare the data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # Blank spaces ni NaN ga marchadam
df = df.dropna() # Missing rows remove cheyadam
df.to_csv("telco_churn_clean.csv", index=False) # UI graphs kosam clean CSV save cheyadam

features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Step 4: Compare with Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Step 5: Evaluate model performance
log_acc = accuracy_score(y_test, log_preds)
dt_acc = accuracy_score(y_test, dt_preds)

print(f"Logistic Regression Accuracy: {log_acc*100:.2f}%")
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, log_preds))
print("-" * 40)
print(f"Decision Tree Accuracy: {dt_acc*100:.2f}%")
print("Confusion Matrix (Decision Tree):\n", confusion_matrix(y_test, dt_preds))

# Save both models and accuracies for the UI
joblib.dump({
    "log_model": log_model,
    "dt_model": dt_model,
    "columns": X.columns.tolist(),
    "log_acc": log_acc,
    "dt_acc": dt_acc
}, "models.pkl")
print("\nDone! Models saved to models.pkl")