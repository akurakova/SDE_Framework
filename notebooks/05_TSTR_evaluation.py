import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -----------------------
# 1. Parse CLI argument
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python 05_TSTR_evaluation.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]
print(f"Running TSTR Evaluation for dataset: {dataset_name}\n")

# -----------------------
# 2. Paths and model names
# -----------------------
base_path = "../data"
model_names = ["Real", "tvae", "ctgan", "ctabgan", "great", "rtf"]
file_names = {
    "Real": f"processed/{dataset_name}_train.csv",
    "Test": f"processed/{dataset_name}_test.csv",
    "tvae": f"synthetic/tvae/{dataset_name}_tvae.csv",
    "ctgan": f"synthetic/ctgan/{dataset_name}_ctgan.csv",
    "ctabgan": f"synthetic/ctabgan/{dataset_name}_ctabgan.csv",
    "great": f"synthetic/great/{dataset_name}_great.csv",
    "rtf": f"synthetic/rtf/{dataset_name}_rtf.csv",
}

def load_datasets(base_path, model_names, file_names):
    datasets = {}
    for name in model_names:
        path = os.path.join(base_path, file_names[name])
        if not os.path.exists(path):
            print(f"Warning: Missing file for {name} at {path}")
            continue
        datasets[name] = pd.read_csv(path)
    datasets["Test"] = pd.read_csv(os.path.join(base_path, file_names["Test"]))
    return datasets

# Load data
datasets = load_datasets(base_path, model_names, file_names)

# -----------------------
# 3. Determine target
# -----------------------
target_col = dataset_name if dataset_name in datasets["Real"].columns else 'target'

real_train = datasets["Real"]
real_test = datasets["Test"]

X_real_test = real_test.drop(columns=[target_col])
y_real_test = real_test[target_col]

# -----------------------
# 4. Detect categorical columns
# -----------------------
categorical_cols = [
    col for col in X_real_test.select_dtypes(include=['object', 'category']).columns
    if X_real_test[col].nunique() <= 20
]

# One-hot encode real data
combined_real = pd.concat([real_train.drop(columns=[target_col]), X_real_test], axis=0)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(combined_real[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)

real_encoded = pd.DataFrame(encoded[:len(real_train)], columns=encoded_cols)
test_encoded = pd.DataFrame(encoded[len(real_train):], columns=encoded_cols)

X_real_train = real_train.drop(columns=[target_col] + categorical_cols).reset_index(drop=True)
X_real_test = X_real_test.drop(columns=categorical_cols).reset_index(drop=True)

X_real_train = pd.concat([X_real_train, real_encoded], axis=1)
X_real_test = pd.concat([X_real_test, test_encoded], axis=1)

# Scale
scaler = StandardScaler()
X_real_test_scaled = scaler.fit_transform(X_real_test)

# -----------------------
# 5. TSTR function
# -----------------------
def tstr_evaluation(synthetic_data):
    X_train = synthetic_data.drop(columns=[target_col])
    y_train = synthetic_data[target_col]

    X_cat = X_train[categorical_cols]
    X_num = X_train.drop(columns=categorical_cols)
    X_encoded = encoder.transform(X_cat)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols, index=X_train.index)

    X_train = pd.concat([X_num.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
    X_train_scaled = scaler.transform(X_train)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_real_test_scaled)
    y_prob = clf.predict_proba(X_real_test_scaled)[:, 1]

    acc = accuracy_score(y_real_test, y_pred)
    f1 = f1_score(y_real_test, y_pred)
    auc = roc_auc_score(y_real_test, y_prob)

    return acc, f1, auc

# -----------------------
# 6. Run evaluations
# -----------------------
tstr_results = {}

for model in model_names[1:]:
    if model not in datasets:
        continue
    acc, f1, auc = tstr_evaluation(datasets[model])
    tstr_results[model] = {"Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc}

# Baseline TRTR
acc, f1, auc = tstr_evaluation(real_train)
tstr_results["TRTR"] = {"Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc}

# -----------------------
# 7. Output results
# -----------------------
tstr_df = pd.DataFrame.from_dict(tstr_results, orient='index')
print("\nTSTR Evaluation Results:")
print(tstr_df.round(3))

# Optionally save
#tstr_df.to_csv(f"../results/{dataset_name}_tstr_results.csv")
