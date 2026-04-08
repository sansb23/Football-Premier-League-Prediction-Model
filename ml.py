
# -----------------------------
# MACHINE LEARNING (Sklearn + Boosting)
# -----------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler
)
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
