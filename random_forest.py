import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from libraries.core import *
from libraries.ml import *

df = pd.read_csv("Data/processed_data/matches_ml_xg_features.csv")

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Feature importance
importances = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

proba = model.predict_proba(X_test)

# Class order matters
classes = model.classes_
print("Class order:", classes)

proba_df = pd.DataFrame(
    proba,
    columns=[f"proba_{c}" for c in classes]
)

# Combine with predictions and true labels
results_df = X_test.reset_index(drop=True).copy()
results_df["true_result"] = y_test.reset_index(drop=True)
results_df["predicted_result"] = y_pred
results_df = pd.concat([results_df, proba_df], axis=1)

print(results_df.head())
results_df["max_proba"] = results_df[
    [col for col in results_df.columns if col.startswith("proba_")]
].max(axis=1)

high_confidence = results_df[results_df["max_proba"] >= 0.6]

print("High-confidence predictions:", high_confidence.shape[0])

results_df.to_csv(
    "Data/processed_data/match_predictions_with_probabilities.csv",
    index=False
)

print("✅ Probability predictions saved")

import joblib

joblib.dump(model, "train_model/random_forest_model.pkl")
print("✅ Model saved")
