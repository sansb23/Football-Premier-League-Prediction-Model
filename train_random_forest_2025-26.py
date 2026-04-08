import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# Load ML-ready dataset
# -------------------------------------------------
df = pd.read_csv(
    "Data/processed_data/matches_ml_xg_features.csv"
)

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------------------------
# Features & target
# -------------------------------------------------
X = df.drop("result", axis=1)
y = df["result"]

# -------------------------------------------------
# Train-test split (stratified)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Train Random Forest
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# Save model
# -------------------------------------------------
joblib.dump(
    model,
    "train_model/random_forest_model_2025_26.pkl"
)

print("\n✅ Model retrained & saved as random_forest_model_2025_26.pkl")
