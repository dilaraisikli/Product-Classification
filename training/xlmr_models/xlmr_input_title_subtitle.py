import os
import pandas as pd
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig, get_scheduler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

# ------------------- CONFIGURATION -------------------
MODEL_NAME = "xlm-roberta-base"  # Using XLM-R
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 7  
LEARNING_RATE = 1e-5
SAVE_PATH = "./xlmr_product_classifier_xlm_title_and_subtitle"
DATA_PATH = "/home/ubuntu/product_classifier/di-interview-product-classifier/dataset_splits/"  # Load pre-split datasets

# ------------------- LOAD PRE-SPLIT DATASETS -------------------
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

print(f" Loaded train, val, and test datasets from {DATA_PATH}")

# **Use only `title` as `text`**
# Fill NaN values
train_df["title_clean"] = train_df["title_clean"].fillna("").astype(str)
train_df["subtitle_clean"] = train_df["subtitle_clean"].fillna("").astype(str)
test_df["title_clean"] = test_df["title_clean"].fillna("").astype(str)
test_df["subtitle_clean"] = test_df["subtitle_clean"].fillna("").astype(str)
val_df["title_clean"] = val_df["title_clean"].fillna("").astype(str)
val_df["subtitle_clean"] = val_df["subtitle_clean"].fillna("").astype(str)
# Merge title and subtitle
train_df["text"] = train_df.apply(lambda row: row["title_clean"] if row["subtitle_clean"] == "" 
                       else row["title_clean"] + " " + row["subtitle_clean"], axis=1)
val_df["text"] = val_df.apply(lambda row: row["title_clean"] if row["subtitle_clean"] == "" 
                       else row["title_clean"] + " " + row["subtitle_clean"], axis=1)
test_df["text"] = test_df.apply(lambda row: row["title_clean"] if row["subtitle_clean"] == "" 
                       else row["title_clean"] + " " + row["subtitle_clean"], axis=1)

# Mapping product type labels to integers
labels = train_df["productType"].unique()  # Use train set for mapping
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_df["label"] = train_df["productType"].map(label2id).astype(int)
val_df["label"] = val_df["productType"].map(label2id).astype(int)
test_df["label"] = test_df["productType"].map(label2id).astype(int)

# Ensure SAVE_PATH exists before saving id2label.json
os.makedirs(SAVE_PATH, exist_ok=True)

# Save `id2label.json`
with open(os.path.join(SAVE_PATH, "id2label.json"), "w") as f:
    json.dump(id2label, f)

print(f" id2label.json saved in {SAVE_PATH}")

# ------------------- TOKENIZATION -------------------
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

class ProductDataset(Dataset):
    def __init__(self, df):
        self.texts = list(df["text"])
        self.labels = list(df["label"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create PyTorch datasets
train_dataset = ProductDataset(train_df)
val_dataset = ProductDataset(val_df)
test_dataset = ProductDataset(test_df)

# ------------------- BALANCED BATCH SAMPLING -------------------
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)

# Assign weights to each sample in the training dataset
sample_weights = np.array([class_weights[label] for label in train_df["label"]], dtype=np.float32)
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------- MODEL SETUP -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=len(label2id))
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
model.to(device)

# Ensure model saves correct label mapping
model.config.id2label = id2label
model.config.label2id = label2id

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS),
    num_training_steps=len(train_loader) * EPOCHS
)

# ------------------- TRAINING LOOP -------------------
def train():
    model.train()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        epoch_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")
        validate()

# ------------------- VALIDATION -------------------
def validate():
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {correct / total:.4f}")

# ------------------- RUN TRAINING -------------------
train()

# ------------------- SAVE MODEL -------------------
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n Model training complete!")
print(f" Model saved in: {SAVE_PATH}")

# ------------------- EVALUATE MODEL -------------------
def evaluate():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure only present labels are used
    unique_labels = sorted(set(all_labels))

    # Fix target_names mismatch
    target_names = [id2label[i] for i in unique_labels if i in id2label]

    print(classification_report(all_labels, all_preds, target_names=target_names))

evaluate()
