import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ROOT_DIR)

data_path = os.path.join(BASE_DIR, "data", "cleaned_full_data.csv")
df = pd.read_csv(data_path)

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = df.drop('y', axis=1)
target = df['y']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

model_dir = os.path.join(BASE_DIR, "models", "random_forest")
scaler_dir = os.path.join(BASE_DIR, "models", "logistic_regression")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(model_dir, "model_rf.pkl"))
pickle.dump(scaler, open(os.path.join(scaler_dir, "scaler_rf.pkl"), "wb"))
pickle.dump(label_encoders, open(os.path.join(model_dir, "label_encoder_rf.pkl"), "wb"))

# ✅ Save test data with column names
pd.DataFrame(X_test, columns=features.columns).to_csv(os.path.join(BASE_DIR, "data", "X_test.csv"), index=False)
y_test.to_frame(name='y').to_csv(os.path.join(BASE_DIR, "data", "y_test.csv"), index=False)

print("✅ Random Forest model, scaler, encoders, and test data saved successfully.")
