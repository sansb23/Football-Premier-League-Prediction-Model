import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from libraries.core import *
from libraries.ml import *

import pandas as pd

df = pd.read_csv("Data/processed_data/matches_clean.csv")

def get_result_from_xg(row):
    if row["home_expected_goals_xg"] > row["away_expected_goals_xg"]:
        return "H"
    elif row["home_expected_goals_xg"] < row["away_expected_goals_xg"]:
        return "A"
    else:
        return "D"

df["result"] = df.apply(get_result_from_xg, axis=1)

print(df["result"].value_counts())
features = ["home_team", "away_team"]

X = df[features]
y = df["result"]



le = LabelEncoder()

X["home_team"] = le.fit_transform(X["home_team"])
X["away_team"] = le.fit_transform(X["away_team"])

ml_df = X.copy()
ml_df["result"] = y

ml_df.to_csv(
    "Data/processed_data/matches_ml_ready.csv",
    index=False
)

print("✅ ML-ready dataset created")
