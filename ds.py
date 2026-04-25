import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. LOAD DATA
# ==========================================
# You can use 'Telco-Customer-Churn.csv' from Kaggle
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# ==========================================
# 2. PREPROCESSING (The "Force-Numeric" Version)
# ==========================================
# Drop ID
df.drop('customerID', axis=1, errors='ignore', inplace=True)

# 1. Convert TotalCharges and handle NaNs
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 2. Force the target variable 'Churn' to be 1 and 0
# This replaces the 'No'/'Yes' strings directly
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 3. Encode all other categorical columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 4. Final safety check: ensure everything is numeric
df = df.apply(pd.to_numeric)

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']
# ==========================================
# 3. HANDLE IMBALANCE USING SMOTE
# ==========================================
print(f"Original class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Resampled class distribution: {np.bincount(y_train_res)}")

# ==========================================
# 4. MODEL TRAINING (Random Forest)
# ==========================================
# Using Random Forest as it builds multiple Decision Trees for better accuracy
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Confusion Matrix ---")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ==========================================
# 6. FEATURE IMPORTANCE (The "Why")
# ==========================================
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Factors Driving Churn')
plt.show()