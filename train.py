import pandas as pd

# Load dataset

df = pd.read_csv("dataset\winequality-red.csv", sep=';')

df.head()

num_samples = df.shape[0]
num_features = df.shape[1] - 1 
target_variable = "quality"
print(f"Number of Samples:{num_samples}, Number of Features:{num_features}, Target Variables:{target_variable}")
df.isnull().sum()

df.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
def run_experiment(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_lr = LinearRegression()
mse_1, r2_1 = run_experiment(model_lr, X_train, X_test, y_train, y_test)

mse_1, r2_1

