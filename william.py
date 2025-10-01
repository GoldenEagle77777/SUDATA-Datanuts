import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Load 
df = pd.read_csv("data.csv")

#  Features/targets 
feat_cols = [
    "weather_condition_severity",
    "port_congestion_level",
    "shipping_costs",
    "supplier_reliability_score",
    "lead_time_days",
    "historical_demand",
    "iot_temperature",
    "cargo_condition_status",
    "route_risk_level",
    "customs_clearance_time",
    "driver_behavior_score",
    "fatigue_monitoring_score",
]
X = df[feat_cols].copy()

y_cls = df["risk_classification"]           #  Low/Moderate/High
y_reg = df["delivery_time_deviation"]       # hours

# Train/test split
Xtr, Xte, ytr_c, yte_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
_,   _,  ytr_r, yte_r  = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Preprocess
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), feat_cols)], remainder="drop")

    # Models
clf = HistGradientBoostingClassifier(max_depth=6, class_weight="balanced", random_state=42)
reg = HistGradientBoostingRegressor(max_depth=6, random_state=42)

pipe_c = Pipeline([("pre", pre), ("model", clf)])
pipe_r = Pipeline([("pre", pre), ("model", reg)])

  #  Fit 
pipe_c.fit(Xtr, ytr_c)
pipe_r.fit(Xtr, ytr_r)

 #  Classification metrics + Confusion Matrix
pred_c = pipe_c.predict(Xte)
print("Classification accuracy:", accuracy_score(yte_c, pred_c))
print(classification_report(yte_c, pred_c))

labels = sorted(y_cls.unique())
cm = confusion_matrix(yte_c, pred_c, labels=labels)

plt.figure(figsize=(5.5,4.5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Risk Classification")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.tight_layout()
plt.show()

    #  Regression metrics 
pred_r = pipe_r.predict(Xte)
mae = mean_absolute_error(yte_r, pred_r)
mse = mean_squared_error(yte_r, pred_r)
rmse = np.sqrt(mse)
r2   = r2_score(yte_r, pred_r)
print(f"Regression  MAE={mae:.2f}h  RMSE={rmse:.2f}h  R^2={r2:.3f}")

    #  Permutation importances 
def plot_perm_importance(pipe, Xval, yval, title, n=10):
    result = permutation_importance(pipe, Xval, yval, n_repeats=8, random_state=7, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=feat_cols).sort_values(ascending=False).head(n)
    imp = imp[::-1]
    plt.figure(figsize=(6,4.5))
    plt.barh(imp.index, imp.values)
    plt.title(title)
    plt.xlabel("Mean importance (permutation)")
    plt.tight_layout()
    plt.show()

plot_perm_importance(pipe_c, Xte, yte_c, "Top drivers of Risk Classification")
plot_perm_importance(pipe_r, Xte, yte_r, "Top drivers of Delivery Time Deviation")

#  Partial Dependence Plots 
act_feats = ["port_congestion_level", "customs_clearance_time",
             "supplier_reliability_score", "lead_time_days"]

# For classifier: one pdp per class
for cls in pipe_c.classes_:
    fig, ax = plt.subplots(figsize=(10,6))
    PartialDependenceDisplay.from_estimator(
        pipe_c, Xte, act_feats, kind="average", ax=ax, target=cls
    )
    plt.suptitle(f"PDP — Effect on Risk Classification (class={cls})", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# For regressor: single pdp
fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
PartialDependenceDisplay.from_estimator(pipe_r, Xte, act_feats, kind="average", ax=ax)
plt.suptitle("PDP — Effect on Delivery Time Deviation", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

