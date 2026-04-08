import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from libraries.core import *


df = pd.read_csv("Data/processed_data/matches_clean.csv")

# Sort by date if available (VERY IMPORTANT)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# Create rolling xG features
df["home_avg_xg_last5"] = (
    df.groupby("home_team")["home_expected_goals_xg"]
      .shift(1)
      .rolling(5)
      .mean()
)

df["away_avg_xg_last5"] = (
    df.groupby("away_team")["away_expected_goals_xg"]
      .shift(1)
      .rolling(5)
      .mean()
)

# Difference feature
df["xg_diff"] = df["home_avg_xg_last5"] - df["away_avg_xg_last5"]

# Drop rows where rolling stats are not available
df = df.dropna(subset=["home_avg_xg_last5", "away_avg_xg_last5"])

# Target (already created earlier)
def get_result(row):
    if row["home_expected_goals_xg"] > row["away_expected_goals_xg"]:
        return "H"
    else:
        return "A"

df["result"] = df.apply(get_result, axis=1)

# Select ML features
ml_df = df[
    ["home_team", "away_team", "home_avg_xg_last5", "away_avg_xg_last5", "xg_diff", "result"]
]

ml_df.to_csv(
    "Data/processed_data/matches_ml_xg_features.csv",
    index=False
)

print("✅ Rolling xG features created")
print(ml_df.head())
def get_result(row, threshold=0.25):
    if abs(row["xg_diff"]) < threshold:
        return "D"
    elif row["xg_diff"] > 0:
        return "H"
    else:
        return "A"

df["result"] = df.apply(get_result, axis=1)

print(df["result"].value_counts())
ml_df = df[
    ["home_team", "away_team", "home_avg_xg_last5", "away_avg_xg_last5", "xg_diff", "result"]
]