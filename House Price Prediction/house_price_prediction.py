import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("house_price_dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# 2. Separate features and target
X = data.drop("Price_PKR", axis=1)
y = data["Price_PKR"]

# 3. Define categorical and numerical columns
categorical_cols = ["Location", "Property_Type"]
numerical_cols = [
    "Area_Marla",
    "Bedrooms",
    "Bathrooms",
    "Age_Years",
    "Parking_Spaces"
]

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Linear Regression model
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

# 7. Linear Regression predictions
lr_pred = linear_model.predict(X_test)

# 8. Evaluate Linear Regression
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n--- Linear Regression ---")
print("MAE:", round(lr_mae, 2))
print("RMSE:", round(lr_rmse, 2))
print("R2 Score:", round(lr_r2, 4))

# 9. Random Forest model
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

rf_model.fit(X_train, y_train)

# 10. Random Forest predictions
rf_pred = rf_model.predict(X_test)

# 11. Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest ---")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 4))

# 12. Compare actual and predicted prices
results = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": rf_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

# 13. Custom house prediction
new_house = pd.DataFrame({
    "Area_Marla": [10],
    "Bedrooms": [4],
    "Bathrooms": [3],
    "Age_Years": [5],
    "Parking_Spaces": [2],
    "Location": ["Lahore"],
    "Property_Type": ["House"]
})

predicted_price = rf_model.predict(new_house)[0]

print("\n--- Custom Prediction ---")
print("House Details:")
print(new_house)

print("\nEstimated House Price:")
print("PKR", round(predicted_price))

# 14. Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price (PKR)")
plt.ylabel("Predicted Price (PKR)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 15. Simple model comparison
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2_Score": [lr_r2, rf_r2]
})

print("\n--- Model Comparison ---")
print(comparison)

# Project completed
print("\nHouse Price Prediction Project Completed!")
