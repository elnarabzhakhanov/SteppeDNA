import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

print("Loading Data & Splitting identically to training...")
df = pd.read_csv("data/master_training_dataset.csv")

with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

X = df[feature_names].values
y = df["Label"].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)
X_test_s = scaler.transform(X_test)

print("Loading Models...")
nn_model = load_model("data/universal_nn.h5")
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")

print("Running Inference...")
nn_preds_test = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]
y_probs = (0.6 * xgb_preds_test) + (0.4 * nn_preds_test)

def print_metrics(threshold_val, name):
    print(f"\n==========================================")
    print(f" {name.upper()} THRESHOLD: {threshold_val}")
    print(f"==========================================")
    y_pred = (y_probs > threshold_val).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                  Predicted Benign   Predicted Pathogenic")
    print(f"Actual Benign        TN={cm[0][0]:<5}            FP={cm[0][1]}")
    print(f"Actual Pathogenic    FN={cm[1][0]:<5}            TP={cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))

with open("data/universal_threshold_ensemble.pkl", "rb") as f:
    opt_thresh = pickle.load(f)

print_metrics(opt_thresh, "Optimized")
print_metrics(0.5, "Absolute Zero Bias (0.50)")
