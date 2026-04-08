import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from libraries.core import *


#  Load metadata
metadata_path = "Data/raw_data/premier_league_metadata.csv"
metadata_df = pd.read_csv(metadata_path)

print("Metadata loaded:", metadata_df.shape)

#  Filter ONLY matches data
matches_meta = metadata_df[metadata_df["dataset_type"] == "matches"]

print("Matches metadata:")
print(matches_meta[["season", "gameweek", "file_name", "rows"]])

#  Choose the BEST matches file (highest row count)
matches_meta = matches_meta.sort_values("rows", ascending=False)
matches_file = matches_meta.iloc[0]["source_path"]

print("Using matches file:", matches_file)

#  Load matches data
df_matches = pd.read_csv(matches_file, encoding="latin1")

print("Raw matches shape:", df_matches.shape)
print(df_matches.head())
print(df_matches.info())

#  BASIC CLEANING
df_clean = df_matches.copy()

df_clean.columns = df_clean.columns.str.strip().str.lower()
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(how="all")

print("Cleaned matches info:")
print(df_clean.info())

#  Save cleaned matches
df_clean.to_csv(
    "Data/processed_data/matches_clean.csv",
    index=False
)

print("✅ Cleaned matches data saved.")
