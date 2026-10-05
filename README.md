# Electric Vehicle Purchase Prediction

Predicting how likely a person is to buy an electric vehicle (`Will_Buy_EV`) from demographic, lifestyle and charging-infrastructure features. The project is a binary classification task on a tabular dataset of roughly 670k rows, solved with a Random Forest baseline and a tuned XGBoost model.

**Result:** ROC AUC of **0.941** on the validation split and **0.941** on the Kaggle public leaderboard.

## Dataset

The data comes from the Kaggle competition [Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 
Unmodified copies of `train.csv` and `test.csv` are included in the [`data/`](data/) folder, so there is nothing to download.
The training set has **668,665 rows** and **15 columns**, with no missing values.

| Type | Features |
|---|---|
| Numeric | `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level` |
| Categorical | `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level` |
| Target | `Will_Buy_EV` |

## Approach

1. **Preprocessing**
   - One-hot encoding for `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible` and `Subsidy_Available`.
   - Label encoding for `Range_Anxiety_Level` and the target.
2. **Feature engineering**
   - `total_charging_stations`: stations near home + stations near work
   - `charging_stations_diff`: stations near work − stations near home
   - `subsidy_x_env`: subsidy availability × `Environmental_Concern_Level`
3. **Validation:** 5% of the training data is held out (`random_state=4`).
4. **Baseline:** default `RandomForestClassifier`.
5. **Main model:** `XGBClassifier` (100 trees, `binary:logistic`) tuned with 5-fold `GridSearchCV` scored by ROC AUC.
6. **Prediction:** probabilities for the test set from the best XGBoost model.

### Grid search space

| Parameter | Values |
|---|---|
| `max_depth` | 3, 6, 9 |
| `learning_rate` | 0.01, 0.1, 0.2 |
| `subsample` | 0.8, 1.0 |
| `colsample_bytree` | 0.8, 1.0 |

That is 36 combinations × 5 folds, so the search takes a while.

## Results

| Model | Train ROC AUC | Validation ROC AUC |
|---|---|---|
| Random Forest (baseline) | 1.000 | 0.931 |
| XGBoost (best grid-search model) | 0.945 | 0.941 |

Best XGBoost parameters: `max_depth=6`, `learning_rate=0.2`, `subsample=1.0`, `colsample_bytree=0.8`.

- The Random Forest memorises the training set (train AUC 1.000), while XGBoost shows almost no gap between train and validation (0.945 vs 0.941).
- The validation score matches the Kaggle public leaderboard score (0.941), so the hold-out split is a reliable estimate here.

## Getting started

```bash
git clone https://github.com/Ashkan-Rafiee/electric-vehicle-purchases-prediction.git
cd electric-vehicle-purchases-prediction

pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

mkdir -p outputs

jupyter notebook electric-vehicle-purchases.ipynb
```

Then use **Restart & Run All**. The final cell writes the predictions to `outputs/submission.csv`.

## Project structure

```
.
├── electric-vehicle-purchases.ipynb   # full pipeline: preprocessing, models, prediction
├── data/                              # train.csv, test.csv
├── outputs/                           # generated submission.csv
└── README.md
```

## Possible improvements

- The best `learning_rate` (0.2) sits at the edge of the search grid. More trees with early stopping would likely help.
- Add feature importance or SHAP plots to explain what drives the predictions.
- Compare with LightGBM or CatBoost, and replace the grid search with Optuna for faster tuning.
