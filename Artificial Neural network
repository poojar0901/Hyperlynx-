# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dataset: study hours vs pass/fail
# hours studied
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create ANN model
# hidden_layer_sizes=(4,) means one hidden layer with 4 neurons
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu',
                      max_iter=1000, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for a new student
new_hours = np.array([[3.5]])
new_hours_scaled = scaler.transform(new_hours)
prediction = model.predict(new_hours_scaled)

print("Predicted (1=Pass, 0=Fail):", prediction[0])
