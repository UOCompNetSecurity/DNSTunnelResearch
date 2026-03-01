import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datasets import Dataset

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

#======================================================================
# Initialize Models
#======================================================================
#Default Model
DEFAULT_MODEL_NAME = "amahdaouy/DomURLs_BERT"
DEFAULT_MODEL = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME, num_labels=2)
#Pretrained Model (Already trained on DNS queries)
PRETRAINED_MODEL_NAME = "models/default"
PRETRAINED_MODEL = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
#Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

#======================================================================
# Dataset Object
#======================================================================
class DNSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
#======================================================================
# Training Arguments
#======================================================================
# Balanced default - good starting point
TRAINING_ARGS_DEFAULT = TrainingArguments(
    output_dir="./domurls_bert_dns",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
)

# Fast iteration - fewer epochs, larger batch, useful for quick experiments
TRAINING_ARGS_FAST = TrainingArguments(
    output_dir="./domurls_bert_dns",
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=3e-5,
    warmup_steps=50,
    weight_decay=0.01,
)

# High accuracy - more epochs, smaller batch, lower learning rate for careful fine-tuning
TRAINING_ARGS_HIGH_ACCURACY = TrainingArguments(
    output_dir="./domurls_bert_dns",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=1e-5,
    warmup_steps=200,
    weight_decay=0.01,
)

# Low memory - small batches with gradient accumulation to compensate, useful if VRAM is limited
TRAINING_ARGS_LOW_MEMORY = TrainingArguments(
    output_dir="./domurls_bert_dns",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # effective batch size = 8 * 4 = 32
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
)

# Anti-overfitting - stronger regularization, useful if your dataset is small or imbalanced
TRAINING_ARGS_REGULARIZED = TrainingArguments(
    output_dir="./domurls_bert_dns",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.1,       # stronger weight decay
)

#======================================================================
# Data Processing
#======================================================================
def tokenize(texts, max_length=128) -> AutoTokenizer:
    return TOKENIZER(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def create_dataset(training_file:str) -> tuple[Dataset, Dataset]:
    df = pd.read_csv(training_file)
    # Clean the queries
    df['Query'] = df['Query'].str.lower().str.strip().str.rstrip('.')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Value'])
    # Tokenize data
    train_encodings = tokenize(train_df['Query'])
    val_encodings   = tokenize(val_df['Query'])
    # Create the dataset
    train_dataset = DNSDataset(train_encodings, train_df['Value'].tolist())
    val_dataset   = DNSDataset(val_encodings,   val_df['Value'].tolist())
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def train_and_save(train_dataset: Dataset, val_dataset: Dataset, output_name: str, model_name: str = "Default", arg_type: str = "Default") -> None:
    training_dict = {
        "Default":      TRAINING_ARGS_DEFAULT,
        "Fast":         TRAINING_ARGS_FAST,
        "HighAccuracy": TRAINING_ARGS_HIGH_ACCURACY,
        "LowMemory":    TRAINING_ARGS_LOW_MEMORY,
        "Regularized":  TRAINING_ARGS_REGULARIZED,
    }

    if arg_type not in training_dict:
        print(f"Unknown arg_type '{arg_type}'. Valid options: {list(training_dict.keys())}")
        print("Falling back to Default training args.")
        arg_type = "Default"

    training_args = training_dict[arg_type]
    training_args.output_dir = output_name  # keep output dir consistent with save location

    if model_name == "Default":
        training_model = DEFAULT_MODEL
    else:
        training_model = PRETRAINED_MODEL

    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(output_name)
    TOKENIZER.save_pretrained(output_name)

#======================================================================
# Prediction
#======================================================================
def determine_device() -> str:
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.mps.is_available():
        device = "mps" 
    elif torch.cpu.is_available():
        device = "cpu"
    else:
        raise ValueError("No devices available")
    return device

def predict_float(query:str, model:AutoModelForSequenceClassification, device: str) -> float:
    query = query.lower().strip().rstrip(".")
    inputs = TOKENIZER(query, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    return probabilities[0][1].item()  

def predict_binary(query:str, model:AutoModelForSequenceClassification, device: str) -> int:
    query = query.lower().strip().rstrip('.')
    inputs = TOKENIZER(query, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

    return pred
    