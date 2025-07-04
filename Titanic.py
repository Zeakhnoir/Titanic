# Enhanced Titanic XGBoost pipeline with preprocessing & hyper‑parameter search

# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_CSV   = Path("train.csv")            # train features + Survived
TEST_CSV    = Path("test.csv")             # test features only
TEST_Y_CSV  = Path("gender_submission.csv")# sample labels or true labels if you have them
TARGET      = "Survived"
DROP_COLS   = ["Cabin", "Name", "Ticket"]
SEED        = 42

# ── LOAD ──────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV).drop(columns=DROP_COLS)
test_df  = pd.read_csv(TEST_CSV).drop(columns=DROP_COLS)

y_train  = train_df[TARGET]
X_train  = train_df.drop(columns=[TARGET])
X_test   = test_df.copy()

test_y_df = pd.read_csv(TEST_Y_CSV)
# gender_submission.csv has PassengerId + Survived; keep numeric Survived
if TARGET in test_y_df.columns:
    y_test = test_y_df[TARGET]
else:                       # fallback to the last numeric column
    y_test = test_y_df.select_dtypes(include="number").iloc[:, -1]

y_test = y_test.squeeze()

# ── PREPROCESSING PIPELINE ────────────────────────────────────────────────────
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols     = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_tf, numeric_cols),
    ("cat", categorical_tf, categorical_cols),
])

# ── MODEL & SEARCH SPACE ──────────────────────────────────────────────────────
base_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    booster="gbtree",
    random_state=SEED,
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", base_xgb),
])

param_dist = {
    "model__n_estimators":      [200, 300, 400, 500],
    "model__learning_rate":     [0.01, 0.05, 0.1],
    "model__max_depth":         [3, 4, 5, 6],
    "model__subsample":         [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "model__gamma":             [0, 0.1, 0.2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=SEED,
)

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("⏳  Running hyper‑parameter search …")
search.fit(X_train, y_train)
print("✅  Best CV accuracy:", search.best_score_)
print("🏷  Best parameters:")
for k, v in search.best_params_.items():
    print(f"   {k}: {v}")

best_clf = search.best_estimator_

# ── EVALUATE ON TEST LABELS (if available) ────────────────────────────────────
print("\n⏳  Evaluating on held‑out test set …")
y_pred = best_clf.predict(X_test)
print("Test accuracy:", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ── FEATURE IMPORTANCE PLOT (top 20) ──────────────────────────────────────────
print("\n🔎  Plotting top‑20 feature importances …")
ohe = best_clf.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_features = ohe.get_feature_names_out(categorical_cols)
feature_names = numeric_cols + cat_features.tolist()

importances = best_clf.named_steps["model"].feature_importances_
idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(9,6))
plt.title("Top‑20 XGBoost Feature Importances")
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=90)
plt.tight_layout()
plt.show()

# ── SAVE PREDICTIONS (optional) ─────────────────────────────────────────────
# submission = pd.DataFrame({
#     "PassengerId": test_df["PassengerId"],
#     "Survived":    y_pred
# })
# submission.to_csv("submission.csv", index=False)
# print("💾  submission.csv written.")

# -----------------------------------------------------------------------------
#  End of script – happy modelling!  ✨
