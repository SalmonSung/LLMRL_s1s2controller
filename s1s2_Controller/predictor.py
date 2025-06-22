import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# 1. Set the global random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. Configuration parameters
SEED = 42
set_seed(SEED)
MODEL_NAME = 'sentence-transformers/LaBSE'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './saved_models_labse'  # Directory containing trained model .pt files
DATA_FILE = 'data.csv'  # CSV file for data loading

# 3. Dataset class matching the training script
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
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items() if k in ['input_ids', 'attention_mask']}
        return encoding, label

# 4. Model class matching the training script
class ModernBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load pre-trained encoder
        self.bert = AutoModel.from_pretrained(model_name)
        # Simple linear classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        # Forward pass through transformer encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # Obtain logits for two classes
        logits = self.classifier(pooled_output)
        return logits

# 5. Load dataset and split out the test set (80/20 split)
df = pd.read_csv(DATA_FILE)
texts = df['task'].tolist()
labels = df['s1_or_s2'].tolist()
_, test_texts, _, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# 6. Prepare the test DataLoader
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_ds = TextDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 7. Function to evaluate a single model on the test set
def evaluate_model(path):
    # Instantiate and load the model
    model = ModernBERTClassifier(MODEL_NAME).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            # Move inputs to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # Forward pass
            logits = model(**batch)
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # Argmax to get predicted labels
            pred_labels = np.argmax(probs, axis=1)
            preds.extend(pred_labels)
            truths.extend(labels)

    # Compute F1 score
    f1 = f1_score(truths, preds)
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    return f1, preds

# 8. Evaluate all saved models and select the top 3 by F1 score
model_files = sorted(
    [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]
)
model_scores = []
for path in model_files:
    f1, _ = evaluate_model(path)
    model_scores.append((path, f1))
    print(f'Model: {path}, Test F1: {f1:.4f}')

# Sort by descending F1 and pick top 3\model_scores.sort(key=lambda x: x[1], reverse=True)
top3 = [m[0] for m in model_scores[:3]]
print('\nTop 3 models selected for ensemble:', top3)

# 9. Sequentially load the top 3 models and collect their predictions
all_preds = []
for path in top3:
    _, preds = evaluate_model(path)
    all_preds.append(preds)

# Convert to NumPy array of shape (n_models, n_samples)
preds_array = np.stack(all_preds)

# 10. Hard voting ensemble: majority vote for each sample
ensemble_preds = []
for votes in preds_array.T:
    # Sum votes for class '1'
    ensemble_preds.append(1 if votes.sum() >= 2 else 0)

# 11. Compute ensemble metrics
acc = accuracy_score(test_labels, ensemble_preds)
f1 = f1_score(test_labels, ensemble_preds)
prec = precision_score(test_labels, ensemble_preds)
print(f"\nEnsemble Test Results -> Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")
