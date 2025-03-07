
import os
import pandas as pd
import fasttext
import fasttext.util
import json
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# ------------------- CONFIGURATION -------------------
SAVE_PATH = "./tfidf_product_classifier"
DATA_PATH = "/home/ubuntu/product_classifier/di-interview-product-classifier/dataset_splits/"  # Load pre-split datasets

# ------------------- LOAD DATA -------------------
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

print(f" Loaded train, val, and test datasets from {DATA_PATH}")

# **Use only `title` as text**
train_df["text"] = train_df["title_clean"].astype(str)
val_df["text"] = val_df["title_clean"].astype(str)
test_df["text"] = test_df["title_clean"].astype(str)

# ------------------- LABEL ENCODING -------------------
labels = train_df["productType"].unique() 
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_df["label"] = train_df["productType"].map(label2id).astype(int)
val_df["label"] = val_df["productType"].map(label2id).astype(int)
test_df["label"] = test_df["productType"].map(label2id).astype(int)

os.makedirs(SAVE_PATH, exist_ok=True)

# Save `id2label.json`
with open(os.path.join(SAVE_PATH, "id2label.json"), "w") as f:
    json.dump(id2label, f)

print(f" id2label.json saved in {SAVE_PATH}")


# ------------------- TF-IDF + RANDOM FOREST -------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
random_forest = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)  # Reduce trees and depth


pipeline_rf = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', random_forest)
])

pipeline_rf.fit(train_df["text"], train_df["label"])

# Save Random Forest Model
joblib.dump(pipeline_rf, os.path.join(SAVE_PATH, "random_forest_model.pkl"))
print(" Random Forest model training complete!")

# ------------------- EVALUATE MODELS -------------------
def evaluate_model(model, df, model_name):
    y_true = df["label"].tolist()
    y_pred = model.predict(df["text"])
    print(f"\n {model_name} Validation Performance:")
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))

# Evaluate models
evaluate_model(pipeline_rf, val_df, "Random Forest")

# ------------------- INFERENCE EXAMPLE -------------------
def classify_text(model, text):
    prediction = model.predict([text])[0]
    return {"product_type": id2label[prediction]}

sample_text = "Luxury leather handbag for women"
rf_prediction = classify_text(pipeline_rf, sample_text)

print(f"Sample Classification (Random Forest): {sample_text} -> {rf_prediction}")

