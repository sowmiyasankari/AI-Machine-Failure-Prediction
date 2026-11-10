
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate sample machine sensor dataset
np.random.seed(42)

n = 1000

temperature = np.random.uniform(40, 120, n)
vibration = np.random.uniform(0.1, 5.0, n)
rpm = np.random.uniform(500, 3000, n)
pressure = np.random.uniform(1, 10, n)

failure = (
    (temperature > 90) &
    (vibration > 3.0) |
    (temperature > 105) |
    (vibration > 4.2)
).astype(int)

data = pd.DataFrame({
    "Temperature": temperature,
    "Vibration": vibration,
    "RPM": rpm,
    "Pressure": pressure,
    "Failure": failure
})

data.to_csv("machine_data.csv", index=False)

# Train machine learning model
X = data[["Temperature", "Vibration", "RPM", "Pressure"]]
y = data["Failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Training Completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
with open("machine_failure_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Dataset and trained model saved successfully!")
