# ⚽ Football Premier League Prediction Model

## 📌 Project Overview

This project is a **machine learning-based football prediction system** that predicts the outcome of Premier League matches.

In simple words:
👉 You enter two teams
👉 The model tells you:

* Who is likely to win 🏆
* Probability of Home Win / Away Win / Draw 📊
* Chances of goals (Over/Under, BTTS)

This is similar to how real sports analytics systems work using historical data and statistics.

---

## 🎯 Objective

The main goal of this project is:

* To predict match outcomes (Home Win / Away Win / Draw)
* To use **data instead of guessing**
* To understand how football analytics works in real life

Football prediction models usually rely on **team strength, past performance, and statistical patterns** ([Wikipedia][1])

---

## 🧠 How It Works (Simple Explanation)

### Step 1: Data Collection

* Match data (teams, results, stats)
* Player and team performance data
* Multiple seasons of Premier League data

---

### Step 2: Data Cleaning

* Removed missing values
* Standardized column names
* Combined multiple CSV files into one dataset

---

### Step 3: Feature Engineering

We created useful features like:

* Average expected goals (xG)
* Last 5 matches performance
* Difference between teams (xG diff)

These features help the model understand:
👉 Which team is stronger
👉 Current form of teams

---

### Step 4: Machine Learning Model

We used:

* Logistic Regression (baseline)
* Random Forest (final model ✅)

Why Random Forest?

* Works well with structured data
* Handles patterns better
* Gives better accuracy

---

### Step 5: Prediction System

The model predicts:

* Match result (H / A / D)
* Winning team name
* Probabilities of each outcome

We also added:

* 📊 Visualization (bar chart)
* ⚽ Goal predictions using Poisson distribution
  (commonly used in football analytics) ([arXiv][2])

---

## 📊 Features of the Project

✅ Predict match winner
✅ Probability of win/draw/loss
✅ Expected goals (xG-based)
✅ Both Teams To Score (BTTS)
✅ Over / Under 2.5 goals
✅ Visualization of predictions
✅ Handles new season data (2025–26)

---

## 🗂️ Project Structure

```
Football_prediction_model/
│
├── Data/
│   ├── raw_data/              # Raw datasets
│   ├── processed_data/        # Cleaned + ML-ready data
│
├── libraries/
│   ├── core.py               # Core imports
│   ├── ml.py                 # ML utilities
│
├── train_model/
│   ├── model1.py             # Logistic Regression
│   ├── random_forest.py      # Random Forest training
│   ├── predict_next_match.py # Final prediction script
│   ├── team_mapping_2025_26.py
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```
git clone https://github.com/sansb23/Football-Premier-League-Prediction-Model.git
cd Football-Premier-League-Prediction-Model
```

### Step 2: Install Libraries

```
python -m pip install pandas numpy scikit-learn matplotlib scipy joblib seaborn plotly
```

---

## ▶️ How to Run

### Train Model

```
python -m train_model.random_forest
```

### Predict Match

```
python -m train_model.predict_next_match
```

---

## 🧪 Example

```
Enter HOME team name: MUN
Enter AWAY team name: BOU
```

Output:

```
Match: Man Utd vs Bournemouth

Predicted Winner: Man Utd

Probabilities:
Home Win: 55%
Draw: 25%
Away Win: 20%
```

---

## 📈 Model Performance

* Accuracy: ~60–63%
* This is considered good in football prediction because:

  * Football is highly unpredictable
  * Even professional models struggle to exceed ~65%

---

## 🚧 Challenges Faced

* Handling missing data
* Different team IDs across seasons
* Updating model for new season (2025–26)
* Draw prediction imbalance

---

## 🔄 Future Improvements

* Add more features (shots, possession, player stats)
* Use deep learning models
* Build web app (Streamlit)
* Real-time match predictions
* Betting strategy simulation

---

## 🧑‍💻 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Joblib

---

## 💡 Key Learnings

* Data cleaning is the most important step
* Feature engineering improves accuracy
* ML models are only as good as the data
* Real-world projects require debugging and iteration

---

## 👤 Author

**Sans (Sanskriti Bhardwaj)**

* Data Science + Fashion + Sports Analytics

---

## ⭐ Final Note

This project is not just about predicting football matches —
it shows how **data + machine learning can be used to make decisions in real life**.
