import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import poisson

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("train_model/random_forest_model_2025_26.pkl")

# -------------------------------------------------
# Load historical ML feature data
# -------------------------------------------------
df = pd.read_csv("Data/processed_data/matches_ml_xg_features.csv")

# -------------------------------------------------
# Load team mapping
# -------------------------------------------------
team_map = pd.read_csv("Data/processed_data/team_id_map_2025_26.csv")

# -------------------------------------------------
# Helper: get team code from user input
# -------------------------------------------------
def get_team_code(team_input):
    team_input = team_input.strip().lower()
    cols = team_map.columns.str.lower()

    if "short_name" in cols:
        row = team_map[team_map["short_name"].str.lower() == team_input]
        if not row.empty:
            return int(row["team_code"].values[0])

    if "team_name" in cols:
        row = team_map[team_map["team_name"].str.lower() == team_input]
        if not row.empty:
            return int(row["team_code"].values[0])

    if "name" in cols:
        row = team_map[team_map["name"].str.lower() == team_input]
        if not row.empty:
            return int(row["team_code"].values[0])

    print("\n❌ Team not found. Available teams:")
    for _, r in team_map.iterrows():
        print(f"{r.get('short_name','')} | {r.get('team_name', r.get('name',''))}")
    raise SystemExit

# -------------------------------------------------
# Helper: rolling xG
# -------------------------------------------------
def get_team_avg_xg(team_code, side, df, n=5):
    if side == "home":
        values = df[df["home_team"] == team_code]["home_avg_xg_last5"]
    else:
        values = df[df["away_team"] == team_code]["away_avg_xg_last5"]

    if len(values) == 0:
        return df["home_avg_xg_last5"].mean()

    return values.tail(n).mean()

# -------------------------------------------------
# Build feature row
# -------------------------------------------------
def prepare_match_features(home_team, away_team):
    home_xg = get_team_avg_xg(home_team, "home", df)
    away_xg = get_team_avg_xg(away_team, "away", df)

    return pd.DataFrame([{
        "home_team": home_team,
        "away_team": away_team,
        "home_avg_xg_last5": home_xg,
        "away_avg_xg_last5": away_xg,
        "xg_diff": home_xg - away_xg
    }])

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
print("\n⚽ FOOTBALL MATCH PREDICTOR ⚽\n")

home_team_name = input("Enter HOME team name: ").strip()
away_team_name = input("Enter AWAY team name: ").strip()

home_code = get_team_code(home_team_name)
away_code = get_team_code(away_team_name)

# -------------------------------------------------
# ML Prediction
# -------------------------------------------------
match_df = prepare_match_features(home_code, away_code)

prediction = model.predict(match_df)[0]
proba = model.predict_proba(match_df)[0]
classes = model.classes_

prob_dict = dict(zip(classes, proba))
home_win_prob = prob_dict.get("H", 0)
away_win_prob = prob_dict.get("A", 0)

# -------------------------------------------------
# xG + Poisson Goal Model
# -------------------------------------------------
home_xg = match_df["home_avg_xg_last5"].values[0]
away_xg = match_df["away_avg_xg_last5"].values[0]
total_xg = home_xg + away_xg

# Draw probability from Poisson (score equality)
draw_prob_poisson = 0.0
max_goals = 6

for g in range(max_goals + 1):
    draw_prob_poisson += poisson.pmf(g, home_xg) * poisson.pmf(g, away_xg)

# -------------------------------------------------
# Blend ML + Poisson (calibrated)
# -------------------------------------------------
ml_total = home_win_prob + away_win_prob

home_win_final = home_win_prob * (1 - draw_prob_poisson) / ml_total
away_win_final = away_win_prob * (1 - draw_prob_poisson) / ml_total
draw_final = draw_prob_poisson

# -------------------------------------------------
# Winner name
# -------------------------------------------------
if home_win_final > away_win_final and home_win_final > draw_final:
    winning_team = home_team_name
elif away_win_final > home_win_final and away_win_final > draw_final:
    winning_team = away_team_name
else:
    winning_team = "Draw"

# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------
print("\n📊 MATCH PREDICTION (CALIBRATED)")
print("--------------------------------")
print(f"Match: {home_team_name} vs {away_team_name}")
print(f"Predicted Winner: {winning_team}")

print("\nWin Probabilities:")
print(f"{home_team_name} Win: {home_win_final * 100:.1f}%")
print(f"Draw: {draw_final * 100:.1f}%")
print(f"{away_team_name} Win: {away_win_final * 100:.1f}%")

# -------------------------------------------------
# Visualization
# -------------------------------------------------

# -------------------------------------------------
# BTTS + Over/Under 2.5
# -------------------------------------------------
p_home_0 = poisson.pmf(0, home_xg)
p_away_0 = poisson.pmf(0, away_xg)

btts_yes = 1 - p_home_0 - p_away_0 + (p_home_0 * p_away_0)
btts_no = 1 - btts_yes

under_25 = (
    poisson.pmf(0, total_xg) +
    poisson.pmf(1, total_xg) +
    poisson.pmf(2, total_xg)
)
over_25 = 1 - under_25

print("\n⚽ GOAL MARKETS (xG-based)")
print("-------------------------")
print(f"{home_team_name} xG: {home_xg:.2f}")
print(f"{away_team_name} xG: {away_xg:.2f}")
print(f"Total xG: {total_xg:.2f}")

print("\nBoth Teams To Score:")
print(f"YES: {btts_yes * 100:.1f}%")
print(f"NO : {btts_no * 100:.1f}%")

print("\nOver / Under 2.5 Goals:")
print(f"Over 2.5: {over_25 * 100:.1f}%")
print(f"Under 2.5: {under_25 * 100:.1f}%")

plt.figure()
plt.bar(
    ["Home Win", "Draw", "Away Win", "BTTS Yes", "BTTS No", "Over 2.5", "Under 2.5"],
    [home_win_final, draw_final, away_win_final, btts_yes, btts_no, over_25, under_25],
    color=['blue', 'gray', 'red', 'green', 'orange', 'purple', 'brown']
     
)
plt.ylabel("Probability")
plt.title(f"{home_team_name} vs {away_team_name} Outcome Probabilities")
plt.show()
