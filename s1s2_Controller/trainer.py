import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from collections import Counter

# Set the global random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The following settings improve reproducibility at the cost of performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration parameters
SEED = 42
set_seed(SEED)
MODEL_NAME = 'sentence-transformers/LaBSE'
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 5
LR_GRID = [2e-5]  # Learning rate options for hyperparameter search
WEIGHT_DECAY = 5e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './saved_models_labse'
LOG_FILE = './training_log_labse.txt'
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize and encode the text to input IDs and attention mask
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items() if k in ['input_ids', 'attention_mask']}
        return encoding, torch.tensor(label, dtype=torch.long)

# Define the classifier model based on a pre-trained transformer
class ModernBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load pre-trained transformer encoder
        self.bert = AutoModel.from_pretrained(model_name)
        # Classification head for binary classification
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        # Forward pass through transformer
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooler output for classification
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Training loop for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch, labels in tqdm(dataloader, desc="Training", leave=False):
        # Move inputs and labels to the device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

# Evaluation loop (validation or test)
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)
            logits = model(**batch)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, prec, f1

# Main training and evaluation logic
def main():
    # Reset seed at start of main
    set_seed(SEED)
    # Load dataset from CSV file
    df = pd.read_csv('data.csv')
    texts = df['task'].tolist()
    labels = df['s1_or_s2'].tolist()

    # Split data into train and test sets (80/20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Initialize tokenizer and cross-validator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Compute class weights for handling imbalance\m    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [total / label_counts[i] for i in range(len(label_counts))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    logf = open(LOG_FILE, 'w')
    best_models = []

    # 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        print(f"\n=== Fold {fold+1} Training ===")

        # Prepare fold-specific train and validation splits
        ft_texts = [train_texts[i] for i in train_idx]
        ft_labels = [train_labels[i] for i in train_idx]
        fv_texts = [train_texts[i] for i in val_idx]
        fv_labels = [train_labels[i] for i in val_idx]

        train_ds = TextDataset(ft_texts, ft_labels, tokenizer)
        val_ds = TextDataset(fv_texts, fv_labels, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        best_val_f1, best_lr, best_state = 0, None, None

        for lr in LR_GRID:
            # Reset seed before each hyperparameter run
            set_seed(SEED)
            model = ModernBERTClassifier(MODEL_NAME).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
            patience_cnt = 0

            for epoch in range(1, EPOCHS+1):
                print(f"Fold {fold+1} | LR {lr:.1e} | Epoch {epoch}/{EPOCHS}")
                tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion)
                val_loss, val_acc, val_prec, val_f1 = eval_epoch(model, val_loader, criterion)
                print(f"Train loss: {tr_loss:.4f}, F1: {tr_f1:.4f} | Val loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, F1: {val_f1:.4f}")
                logf.write(f"Fold {fold+1}, LR {lr}, Epoch {epoch}, TrL: {tr_loss:.4f}, TrF1: {tr_f1:.4f}, ValL: {val_loss:.4f}, ValF1: {val_f1:.4f}\n")

                if val_f1 > best_val_f1:
                    best_val_f1, best_lr, best_state = val_f1, lr, model.state_dict()
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Clean up after each hyperparameter iteration
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Save the best model state for this fold
        pt_path = os.path.join(OUTPUT_DIR, f"model_fold{fold+1}.pt")
        torch.save(best_state, pt_path)
        best_models.append({'path': pt_path, 'f1': best_val_f1, 'lr': best_lr})

    logf.close()

    # Evaluate the best model on the held-out test set
    best_models.sort(key=lambda x: x['f1'], reverse=True)
    best = best_models[0]
    print(f"\nBest model: {best['path']} with F1 {best['f1']:.4f} (LR {best['lr']})")

    final_model = ModernBERTClassifier(MODEL_NAME).to(DEVICE)
    final_model.load_state_dict(torch.load(best['path']))
    final_model.eval()

    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("\n=== Testing on held-out set ===")
    t_loss, t_acc, t_prec, t_f1 = eval_epoch(final_model, test_loader, nn.CrossEntropyLoss(weight=class_weights))
    print(f"Test Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, Prec: {t_prec:.4f}, F1: {t_f1:.4f}")

if __name__ == '__main__':
    main()
