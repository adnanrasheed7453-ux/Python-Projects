# Fake News Detection - Simple Project

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Step 1: Dataset Load Garnu
# -----------------------------

data = pd.read_csv(r"C:\Users\GMS\Desktop\Python Project\fake_news.csv")

print("Dataset successfully load bhayo!")
print(data.head())

# -----------------------------
# Step 2: Text Cleaning
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["text"] = data["text"].apply(clean_text)

# Empty rows remove garnu
data = data[data["text"] != ""]

# -----------------------------
# Step 3: Labels Convert Garnu
# -----------------------------

if data["label"].dtype == "object":
    data["label"] = data["label"].str.lower()

    data["label"] = data["label"].map({
        "fake": 0,
        "real": 1
    })

data = data.dropna(subset=["label"])
data["label"] = data["label"].map({"fake": 0, "real": 1})

# -----------------------------
# Step 4: Train/Test Split Garnu
# -----------------------------

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Step 5: TF-IDF
# -----------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF complete bhayo!")

# -----------------------------
# Step 6: Logistic Regression
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# -----------------------------
# Step 7: Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n----- Model Results -----")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"],
    zero_division=0
))

# -----------------------------
# Step 8: Nai News Test Garnu
# -----------------------------

news = input("\nNews ko text enter garnu: ")

clean_news = clean_text(news)

news_tfidf = tfidf.transform([clean_news])

prediction = model.predict(news_tfidf)[0]

probability = model.predict_proba(news_tfidf)[0].max()

if prediction == 0:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")

print("Confidence:", round(probability * 100, 2), "%")

# -----------------------------
# Step 9: Top Words Ko Contribution
# -----------------------------

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# Real ko lagi top words
real_words = sorted(
    zip(feature_names, coefficients),
    key=lambda x: x[1],
    reverse=True
)[:15]

# Fake ko lagi top words
fake_words = sorted(
    zip(feature_names, coefficients),
    key=lambda x: x[1]
)[:15]

print("\n----- REAL News Ko Lagi Top Words -----")

for word, value in real_words:
    print(word, ":", round(value, 4))

print("\n----- FAKE News Ko Lagi Top Words -----")

for word, value in fake_words:
    print(word, ":", round(value, 4))

# -----------------------------
# Step 10: Second Model
# -----------------------------

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_pred)

print("\n----- Naive Bayes -----")
print("Accuracy:", nb_accuracy)