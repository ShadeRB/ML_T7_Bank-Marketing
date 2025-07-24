import pandas as pd
import joblib
from evaluation import evaluate_model, predict_model

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['y']

# Load Decision Tree model & scaler
model_dt = joblib.load('models/decision_tree/model_dt.pkl')
scaler_dt = joblib.load('models/logistic_regression/scaler_dt.pkl')

# Load Random Forest model & scaler
model_rf = joblib.load('models/random_forest/model_rf.pkl')
scaler_rf = joblib.load('models/logistic_regression/scaler_rf.pkl')

# Predict & Evaluate
y_pred_dt = predict_model(model_dt, scaler_dt.transform(X_test))
evaluate_model(y_test, y_pred_dt, model_name="Decision Tree")

y_pred_rf = predict_model(model_rf, scaler_rf.transform(X_test))
evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

# Optional: Show comparison bar chart
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Collect metrics
metrics = {
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
    ],
    "Precision": [
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf),
    ],
    "Recall": [
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf),
    ]
}

# Plot bar chart
import pandas as pd
df = pd.DataFrame(metrics)
df.set_index("Model").plot(kind="bar", figsize=(8, 5), colormap="coolwarm", legend=True)
plt.title("Model Evaluation Comparison")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.show()
