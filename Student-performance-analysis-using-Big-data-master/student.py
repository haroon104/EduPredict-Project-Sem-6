# made by Dharmendra Choudhary......VIT university, vellore, Tamil Nadu
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import mean_squared_error

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv("datasets.csv")

# drop unwanted column
df = df.drop("PlaceofBirth", axis=1)

print(df.describe())

# ------------------------------
# Visualization (EDA)
# ------------------------------
ls = [
    "gender",
    "Relation",
    "Topic",
    "SectionID",
    "GradeID",
    "NationalITy",
    "Class",
    "StageID",
    "Semester",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]

for i in ls:
    g = sns.countplot(x=i, data=df)
    plt.show()

print("Dataset shape:", df.shape)

# ------------------------------
# Preprocessing
# ------------------------------
target = df.pop("Class")
X = pd.get_dummies(df)

le = LabelEncoder()
y = le.fit_transform(target)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# ------------------------------
# Feature Importance (RandomForest)
# ------------------------------
feat_labels = X.columns
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print(
        "%2d) %-*s %f"
        % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    )

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feat_labels[indices])
plt.title("Feature Importances")
plt.show()

# ------------------------------
# Dimension Reduction (manual selection)
# ------------------------------
X_train_new = X_train.copy()
X_test_new = X_test.copy()

ls_keep = [
    "VisITedResources",
    "raisedhands",
    "AnnouncementsView",
    "StudentAbsenceDays_Above-7",
    "StudentAbsenceDays_Under-7",
    "Discussion",
]

X_train_new = X_train_new[ls_keep]
X_test_new = X_test_new[ls_keep]

# ------------------------------
# Spot checking algorithms
# ------------------------------
models = [
    ("LR", LinearRegression()),
    ("LASSO", Lasso()),
    ("EN", ElasticNet()),
    ("KNN", KNeighborsRegressor()),
    ("CART", DecisionTreeRegressor()),
    ("SVR", SVR()),
]

results, names = [], []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(
        model, X_train_new, y_train, cv=kfold, scoring="neg_mean_squared_error"
    )
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# ------------------------------
# Standardized Pipelines
# ------------------------------
pipelines = [
    ("ScaledLR", Pipeline([("Scaler", StandardScaler()), ("LR", LinearRegression())])),
    ("ScaledLASSO", Pipeline([("Scaler", StandardScaler()), ("LASSO", Lasso())])),
    ("ScaledEN", Pipeline([("Scaler", StandardScaler()), ("EN", ElasticNet())])),
    ("ScaledKNN", Pipeline([("Scaler", StandardScaler()), ("KNN", KNeighborsRegressor())])),
    ("ScaledCART", Pipeline([("Scaler", StandardScaler()), ("CART", DecisionTreeRegressor())])),
    ("ScaledSVR", Pipeline([("Scaler", StandardScaler()), ("SVR", SVR())])),
]

results, names = [], []
for name, model in pipelines:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(
        model, X_train_new, y_train, cv=kfold, scoring="neg_mean_squared_error"
    )
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

plt.figure()
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.title("Scaled Algorithm Comparison")
plt.show()

# ------------------------------
# Lasso tuning
# ------------------------------
scaler = StandardScaler().fit(X_train_new)
rescaledX = scaler.transform(X_train_new)
k_values = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
param_grid = dict(alpha=k_values)
model = Lasso()
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
grid = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error", cv=kfold
)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# ------------------------------
# Ensembles
# ------------------------------
ensembles = [
    ("ScaledAB", Pipeline([("Scaler", StandardScaler()), ("AB", AdaBoostRegressor())])),
    ("ScaledGBM", Pipeline([("Scaler", StandardScaler()), ("GBM", GradientBoostingRegressor())])),
    ("ScaledRF", Pipeline([("Scaler", StandardScaler()), ("RF", RandomForestRegressor())])),
    ("ScaledET", Pipeline([("Scaler", StandardScaler()), ("ET", ExtraTreesRegressor())])),
]

results, names = [], []
for name, model in ensembles:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(
        model, X_train_new, y_train, cv=kfold, scoring="neg_mean_squared_error"
    )
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

plt.figure()
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.title("Scaled Ensemble Algorithm Comparison")
plt.show()

# ------------------------------
# Tune AdaBoost
# ------------------------------
scaler = StandardScaler().fit(X_train_new)
rescaledX = scaler.transform(X_train_new)
param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
model = AdaBoostRegressor(random_state=7)
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
grid = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error", cv=kfold
)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# ------------------------------
# Final Model - GradientBoosting
# ------------------------------
scaler = StandardScaler().fit(X_train_new)
rescaledX = scaler.transform(X_train_new)
model = GradientBoostingRegressor(random_state=7, n_estimators=400)
model.fit(rescaledX, y_train)

# Test set predictions
rescaledValidationX = scaler.transform(X_test_new)
predictions = model.predict(rescaledValidationX)
print("Final Test MSE:", mean_squared_error(y_test, predictions))
