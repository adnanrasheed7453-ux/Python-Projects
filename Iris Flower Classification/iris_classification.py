import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load Dataset
data = pd.read_csv("iris_dataset.csv")

print("----- Iris Flower Classification -----")
print(data.head())

# Separate Features and Target
X = data[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

y = data["species"]


# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))


# -------------------------------
# Logistic Regression
# -------------------------------

lr_model = LogisticRegression(max_iter=200)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n----- Logistic Regression -----")
print("Accuracy:", lr_accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# -------------------------------
# KNN
# -------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\n----- KNN -----")
print("Accuracy:", knn_accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))


# -------------------------------
# Decision Tree
# -------------------------------

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n----- Decision Tree -----")
print("Accuracy:", dt_accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))


# -------------------------------
# Custom Flower Prediction
# -------------------------------

print("\n----- Custom Flower Prediction -----")

sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

custom_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = lr_model.predict(custom_flower)

print("Flower Measurements:")
print("Sepal Length:", sepal_length)
print("Sepal Width:", sepal_width)
print("Petal Length:", petal_length)
print("Petal Width:", petal_width)

print("Predicted Flower:", prediction[0])