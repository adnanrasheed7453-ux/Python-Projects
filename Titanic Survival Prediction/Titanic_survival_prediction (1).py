import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Select Features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features].copy()
y = df["Survived"]

# Fill Missing Numerical Values
age_imputer = SimpleImputer(strategy="median")
X["Age"] = age_imputer.fit_transform(X[["Age"]]).ravel()

# Fill Missing Categorical Values
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# Encode Categorical Features
sex_encoder = LabelEncoder()
X["Sex"] = sex_encoder.fit_transform(X["Sex"])

embarked_encoder = LabelEncoder()
X["Embarked"] = embarked_encoder.fit_transform(X["Embarked"])

print("\nProcessed Data:")
print(X.head())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_prediction)

print("\nLogistic Regression Accuracy:")
print(logistic_accuracy)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logistic_prediction))

# Random Forest Model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_prediction = random_forest_model.predict(X_test)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)

print("\nRandom Forest Accuracy:")
print(random_forest_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, random_forest_prediction))

# Compare Models
print("\nModel Comparison:")
print("Logistic Regression:", logistic_accuracy)
print("Random Forest:", random_forest_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, random_forest_prediction)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.yticks([0, 1], ["Did Not Survive", "Survived"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# Sample Passenger Prediction
sample_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex": [sex_encoder.transform(["male"])[0]],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": [embarked_encoder.transform(["S"])[0]]
})

prediction = random_forest_model.predict(sample_passenger)

print("\nSample Passenger Prediction:")

if prediction[0] == 1:
    print("Passenger is predicted to SURVIVE.")
else:
    print("Passenger is predicted NOT TO SURVIVE.")