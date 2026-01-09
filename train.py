import pandas as pd
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Create output directories
# -------------------------
Path("output/model").mkdir(parents=True, exist_ok=True)
Path("output/results").mkdir(parents=True, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("dataset/winequality-red.csv", sep=';')

X = df.drop("quality", axis=1)
y = df["quality"]

# -------------------------
# Feature selection
# -------------------------
corr = df.corr()["quality"].abs().sort_values(ascending=False)


# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Model (ONE experiment)
# -------------------------
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# REQUIRED PRINTS
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# -------------------------
# Save model
# -------------------------
joblib.dump(model, "output/model/model.pkl")

# -------------------------
# Save results
# -------------------------
results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results/results.json", "w") as f:
    json.dump(results, f, indent=4)
