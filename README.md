# Football-Premier-League-Prediction-Model

A machine learning project that predicts Premier League match outcomes using Random Forest classification, xG (expected goals) features, and Poisson-based goal modelling.

---

## 📁 Project Structure

```
Football_prediction_model/
│
├── Data/
│   ├── raw_data/
│   │   └── clean_raw_data.py
│   │   └── metadata.py
│   │   └── Premier_League_metadata.csv
│   │   └── read_raw_data.py
│   │   └── run_metadata.py
│   └── processed_data/
│       ├── matches_ml_xg_features.csv
│       ├── matches_ml_ready.csv
│       ├── team_id_map_2025_26.csv
│       └── match_predictions_with_probabilities.csv
│
├── train_model/
│   ├── random_forest_model.pkl
│   └── random_forest_model_2025_26.pkl
│
├── libraries/
│   ├── core.py
│   └── ml.py
│
├── model1.py
├── random_forest.py
├── train_random_forest_2025-26.py
├── team_mapping.py
├── team_mapping_2025-26.py
└── predict_next_match.py
```

---

## 🧠 Models

### `model1.py` — Logistic Regression (Baseline)
A baseline classification model using Logistic Regression trained on `matches_ml_ready.csv`.

### `random_forest.py` — Random Forest v1
Trains a Random Forest classifier (200 estimators, max depth 6) on xG-based features. Outputs feature importances, win probabilities, and saves predictions to CSV.

### `train_random_forest_2025-26.py` — Random Forest v2 (2025–26 Season)
An improved Random Forest (300 estimators, max depth 7, min samples leaf 5) retrained specifically on 2025–26 season data with better regularisation.

---

## 🗂️ Key Scripts

| Script | Purpose |
|---|---|
| `model1.py` | Logistic Regression baseline model |
| `random_forest.py` | Train & evaluate RF v1, save model + probabilities |
| `train_random_forest_2025-26.py` | Retrain RF for 2025–26 season |
| `team_mapping.py` | Build team ID → name mapping from match data |
| `team_mapping_2025-26.py` | Auto-detect and map team codes for 2025–26 |
| `predict_next_match.py` | Interactive CLI to predict any upcoming match |

---

## 🔮 Prediction Pipeline (`predict_next_match.py`)

The prediction script combines two approaches for calibrated probability estimates:

1. **Random Forest (ML)** — Predicts Home Win / Draw / Away Win from historical xG features.
2. **Poisson Goal Model** — Uses rolling average xG to estimate score distributions and draw probability.
3. **Blended Output** — ML win probabilities are scaled using the Poisson-derived draw probability for better calibration.

### Additional Markets Output
- **BTTS (Both Teams To Score):** YES / NO probabilities
- **Over/Under 2.5 Goals:** Based on total xG via Poisson distribution
- **Bar chart visualisation** of all outcome probabilities

---

## ⚙️ Features Used

| Feature | Description |
|---|---|
| `home_team` | Encoded home team ID |
| `away_team` | Encoded away team ID |
| `home_avg_xg_last5` | Home team's rolling average xG over last 5 matches |
| `away_avg_xg_last5` | Away team's rolling average xG over last 5 matches |
| `xg_diff` | Difference between home and away xG |

**Target variable:** `result` — `H` (Home Win), `D` (Draw), `A` (Away Win)

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install pandas scikit-learn joblib matplotlib scipy
```

### 2. Prepare Data

Ensure `Data/processed_data/matches_ml_xg_features.csv` exists with the required feature columns.

### 3. Build Team Mapping

```bash
python team_mapping_2025-26.py
```

### 4. Train the Model

```bash
python train_random_forest_2025-26.py
```

### 5. Predict a Match

```bash
python predict_next_match.py
```

You will be prompted to enter the home and away team names:

```
⚽ FOOTBALL MATCH PREDICTOR ⚽

Enter HOME team name: Arsenal
Enter AWAY team name: Chelsea
```

---

## 📊 Example Output

```
📊 MATCH PREDICTION (CALIBRATED)
--------------------------------
Match: Arsenal vs Chelsea
Predicted Winner: Arsenal

Win Probabilities:
Arsenal Win:  54.3%
Draw:         22.1%
Chelsea Win:  23.6%

⚽ GOAL MARKETS (xG-based)
-------------------------
Arsenal xG:  1.72
Chelsea xG:  1.10
Total xG:    2.82

Both Teams To Score:
YES: 67.4%
NO : 32.6%

Over / Under 2.5 Goals:
Over 2.5:  58.9%
Under 2.5: 41.1%
```

---

## 🧪 Model Performance

The Random Forest model is evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1 per class)
- **Confusion Matrix**

Trained with an 80/20 stratified train-test split (`random_state=42`).

---

## 📌 Notes

- Team codes are integer IDs sourced from the football metadata `teams.csv`.
- The legacy team fix dictionary in `team_mapping.py` handles teams whose IDs differ between seasons (e.g., Man City → 43, Brentford → 94).
- Models are saved as `.pkl` files using `joblib` for fast reloading.
- The prediction script falls back to the dataset mean xG if a team has no historical records.

---

## 📦 Dependencies

| Library | Use |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Model training and evaluation |
| `joblib` | Model serialisation |
| `matplotlib` | Probability visualisation |
| `scipy` | Poisson distribution for goal modelling |
