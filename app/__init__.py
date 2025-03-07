from transformers import BertForSequenceClassification, BertTokenizer
import torch

# ------------------- LOAD MODEL & TOKENIZER -------------------
#DEL_PATH = "/home/ubuntu/product_classifier/di-interview-product-classifier/training/mbert_product_classifier"
MODEL_PATH = "/home/ubuntu/product_classifier/di-interview-product-classifier/training/mbert_models/mbert_product_classifier_input_subtitle_and_title"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
id2label = model.config.id2label

# ------------------- PREDICTION FUNCTION -------------------
def predict_product_type(title: str, top_k: int = 3):
    """Predicts the product type for a given title."""
    encoding = tokenizer(
        title,
        padding="max_length",
        truncation=True,
        max_length=128, 
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

   
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
    top_k_labels = [id2label[idx.item()] for idx in top_k_indices[0]]

    # response
    result = {
        "title": title,
        "top_3_results": [
            {"product_type": top_k_labels[i], "score": round(top_k_probs[0][i].item(), 4)}
            for i in range(top_k)
        ],
        "predicted_product_type": top_k_labels[0],  # Best match
    }
    return result
