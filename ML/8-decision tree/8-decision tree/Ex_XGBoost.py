# Step 1: Install XGBoost if not already
# pip install xgboost

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 2: Load dataset (Boston Housing for regression)
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert to DMatrix (optimized format for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Step 5: Define parameters
params = {
    "objective": "reg:squarederror",  # regression task
    "max_depth": 4,
    "eta": 0.1,                       # learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# Step 6: Train model
num_rounds = 200
model = xgb.train(params, dtrain, num_rounds)

# Step 7: Make predictions
y_pred = model.predict(dtest)

# Step 8: Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Step 9: Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()