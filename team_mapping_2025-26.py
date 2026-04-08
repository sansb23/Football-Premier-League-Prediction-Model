import pandas as pd
import os

# -------------------------------------------------
# 1. LOAD MATCH DATA (SOURCE OF TRUTH FOR TEAM IDS)
# -------------------------------------------------
matches_path = "Data/processed_data/matches_ml_xg_features.csv"

if not os.path.exists(matches_path):
    raise FileNotFoundError(f"Matches file not found: {matches_path}")

df_matches = pd.read_csv(matches_path)

print("\n📂 Loaded matches file")
print("Columns found:")
print(df_matches.columns.tolist())

# -------------------------------------------------
# 2. AUTO-DETECT HOME / AWAY TEAM COLUMNS
# -------------------------------------------------
home_col = None
away_col = None

for col in df_matches.columns:
    col_l = col.lower()
    if "home" in col_l and "team" in col_l:
        home_col = col
    if "away" in col_l and "team" in col_l:
        away_col = col

if home_col is None or away_col is None:
    raise ValueError(
        "❌ Could not detect home/away team columns.\n"
        f"Columns available: {df_matches.columns.tolist()}"
    )

print(f"\n✅ Detected home team column: {home_col}")
print(f"✅ Detected away team column: {away_col}")

# -------------------------------------------------
# 3. COLLECT TEAM CODES FROM MATCHES
# -------------------------------------------------
match_team_codes = sorted(
    set(df_matches[home_col]).union(set(df_matches[away_col]))
)

print("\n⚽ Team codes found in matches:")
print(match_team_codes)
print("Total teams:", len(match_team_codes))

# -------------------------------------------------
# 4. FIND & LOAD 2025–26 teams.csv (AUTO-DETECT)
# -------------------------------------------------
base_dir = "C:\\Users\\jiyaa\\OneDrive\\Desktop\\Football_prediction_model\\Data\\raw_data\\football_metadata\\data\\Premier_League"
teams_file = None

for season in os.listdir(base_dir):
    season_path = os.path.join(base_dir, season)
    if os.path.isdir(season_path):
        candidate = os.path.join(season_path, "teams.csv")
        if os.path.exists(candidate):
            teams_file = candidate
            print("\n📄 Found teams.csv at:")
            print(teams_file)
            break

if teams_file is None:
    raise FileNotFoundError("❌ teams.csv not found in Premier_League folders")

teams_2526 = pd.read_csv(teams_file)

print("\n📂 Loaded teams.csv")
print("Teams file columns:")
print(teams_2526.columns.tolist())

# -------------------------------------------------
# 5. STANDARDIZE TEAMS FILE COLUMNS
# -------------------------------------------------
column_map = {}

if "id" in teams_2526.columns:
    column_map["id"] = "team_code"
if "name" in teams_2526.columns:
    column_map["name"] = "team_name"
if "short_name" in teams_2526.columns:
    column_map["short_name"] = "short_name"

teams_2526 = teams_2526.rename(columns=column_map)

required_cols = ["team_code", "team_name"]

for col in required_cols:
    if col not in teams_2526.columns:
        raise ValueError(f"❌ Required column missing in teams.csv: {col}")

# -------------------------------------------------
# 6. BUILD FINAL TEAM MAPPING (SAFE MERGE)
# -------------------------------------------------
team_map = pd.DataFrame({"team_code": match_team_codes})

team_map = team_map.merge(
    teams_2526[["team_code", "team_name"] + (
        ["short_name"] if "short_name" in teams_2526.columns else []
    )],
    on="team_code",
    how="left"
)

# -------------------------------------------------
# 7. HANDLE MISSING LABELS (NO DELETIONS)
# -------------------------------------------------
team_map["team_name"] = team_map["team_name"].fillna("Unknown / Legacy")

if "short_name" in team_map.columns:
    team_map["short_name"] = team_map["short_name"].fillna("UNK")

# -------------------------------------------------
# 8. SAVE OUTPUT
# -------------------------------------------------
output_path = "Data/processed_data/team_id_map_2025_26.csv"
team_map.to_csv(output_path, index=False)

print("\n✅ Team mapping created successfully")
print(team_map)

print(f"\n💾 Saved to: {output_path}")
