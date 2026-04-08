import pandas as pd
import os

# -------------------------------------------------
# 1. LOAD MATCH DATA (SOURCE OF TRUTH)
# -------------------------------------------------
matches_path = "Data/processed_data/matches_ml_xg_features.csv"
df_matches = pd.read_csv(matches_path)

# -------------------------------------------------
# 2. AUTO-DETECT HOME / AWAY TEAM COLUMNS
# -------------------------------------------------
home_col, away_col = None, None

for col in df_matches.columns:
    cl = col.lower()
    if "home" in cl and "team" in cl:
        home_col = col
    if "away" in cl and "team" in cl:
        away_col = col

if home_col is None or away_col is None:
    raise ValueError(
        f"Home/Away team columns not found. Columns: {df_matches.columns.tolist()}"
    )

print(f"✅ Home column: {home_col}")
print(f"✅ Away column: {away_col}")

# -------------------------------------------------
# 3. COLLECT TEAM CODES FROM MATCHES
# -------------------------------------------------
team_codes = sorted(
    set(df_matches[home_col]).union(set(df_matches[away_col]))
)

print("\n⚽ Team codes found in matches:")
print(team_codes)

# -------------------------------------------------
# 4. LOAD 2025–26 TEAMS.CSV
# -------------------------------------------------
teams_path = "C:\\Users\\jiyaa\\OneDrive\\Desktop\\Football_prediction_model\\Data\\raw_data\\football_metadata\\data\\Premier_League\\2025-2026\\teams.csv"
teams_df = pd.read_csv(teams_path)

teams_df = teams_df.rename(columns={
    "id": "team_code",
    "name": "team_name",
    "short_name": "short_name"
})

# -------------------------------------------------
# 5. BUILD BASE TEAM MAP
# -------------------------------------------------
team_map = pd.DataFrame({"team_code": team_codes})

team_map = team_map.merge(
    teams_df[["team_code", "team_name", "short_name"]],
    on="team_code",
    how="left"
)

# -------------------------------------------------
# 6. MANUAL FIX FOR LEGACY / MISMAPPED TEAMS
# -------------------------------------------------
legacy_fix = {
    21: "West Ham",
    31: "Crystal Palace",
    36: "Brighton",
    39: "Wolves",
    40: "Ipswich",
    43: "Man City",
    54: "Fulham",
    94: "Brentford"
}

team_map["team_name"] = team_map.apply(
    lambda r: legacy_fix.get(r["team_code"], r["team_name"]),
    axis=1
)

team_map["short_name"] = team_map["short_name"].fillna(
    team_map["team_name"].str[:3].str.upper()
)

team_map["team_name"] = team_map["team_name"].fillna("Unknown")

# -------------------------------------------------
# 7. SAVE FINAL TEAM MAP
# -------------------------------------------------
output_path = "Data/processed_data/team_id_map_2025_26.csv"
team_map.to_csv(output_path, index=False)

print("\n✅ FINAL TEAM MAPPING")
print(team_map)
print(f"\n💾 Saved to: {output_path}")
