import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset (Area vs Price)
data = {
    'Area': [800, 900, 1000, 1100, 1200, 1500],
    'Price': [40, 45, 50, 55, 60, 75]   # Prices in lakhs
}

df = pd.DataFrame(data)

# Step 1: Split data into input (X) and output (y)
X = df[['Area']]
y = df['Price']

# Step 2: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Step 3: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict house price for test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model using MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Area values:\n", X_test)
print("\nActual Prices:\n", y_test)
print("\nPredicted Prices:\n", y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
