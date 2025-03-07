
import os
import pandas as pd
import fasttext
import fasttext.util
import json
from sklearn.metrics import classification_report

# ------------------- CONFIGURATION -------------------
SAVE_PATH = "./fasttext_product_classifier"
DATA_PATH = "/home/ubuntu/product_classifier/di-interview-product-classifier/dataset_splits/"  # Load pre-split datasets

# ------------------- LOAD DATA -------------------
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

print(f" Loaded train, val, and test datasets from {DATA_PATH}")

# **Use only title as text**
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

# Save id2label.json
with open(os.path.join(SAVE_PATH, "id2label.json"), "w") as f:
    json.dump(id2label, f)

print(f" id2label.json saved in {SAVE_PATH}")

# ------------------- FASTTEXT DATA PREPARATION -------------------
def save_fasttext_format(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            label = f"__label__{row['productType']}"
            text = row["text"].replace("\n", " ") 
            f.write(f"{label} {text}\n")

# Save train and validation data in FastText format
fasttext_train_path = os.path.join(SAVE_PATH, "train.txt")
fasttext_val_path = os.path.join(SAVE_PATH, "val.txt")
save_fasttext_format(train_df, fasttext_train_path)
save_fasttext_format(val_df, fasttext_val_path)

print(" FastText training and validation files saved")

# ------------------- TRAIN FASTTEXT MODEL -------------------
model = fasttext.train_supervised(
    input=fasttext_train_path,
    lr=0.3,         
    epoch=50,      
    wordNgrams=3,   
    bucket=500000,  
    dim=300,        
    loss="softmax"  
)

# Save trained model
model.save_model(os.path.join(SAVE_PATH, "fasttext_model.bin"))
print(" FastText model training complete!")

# ------------------- EVALUATE MODEL -------------------
def evaluate_fasttext(model, df):
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        label = row["productType"]
        text = row["text"]
        pred_label = model.predict(text)[0][0].replace("__label__", "")
        y_true.append(label)
        y_pred.append(pred_label)
    
    print(classification_report(y_true, y_pred))

print("\n FastText Validation Performance:")
evaluate_fasttext(model, val_df)

