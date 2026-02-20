import torch
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from datasets import DatasetDict, Dataset


def analyze_subdomain_with_bert(query_text: str, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification) -> float:
    # 1. Isolate the subdomain
    # Example: "secretdata.example.com" -> parts: ["secretdata", "example", "com"]
    parts = query_text.split('.')
    
    if len(parts) > 2:
        # Join all parts before the domain and TLD (handles multi-level subdomains)
        subdomain = ".".join(parts[:-2]) 
    else:
        # If there's no subdomain (e.g., "google.com"), use the full name or return benign
        subdomain = query_text

    # 2. Tokenize ONLY the subdomain
    inputs = tokenizer(subdomain, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # 3. Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        
    malicious_prob = probs[0, 1].item()
    return malicious_prob

