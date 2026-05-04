import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score
import os
from tqdm import tqdm

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def run_training(data_dir='data', models_dir='models'):
    # Parameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'bert-base-uncased'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    
    if not os.path.exists(train_path):
        data_dir = os.path.join('..', data_dir)
        models_dir = os.path.join('..', models_dir)
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')

    train_df = pd.read_csv(train_path).sample(n=2000, random_state=42) 
    val_df = pd.read_csv(val_path).sample(n=500, random_state=42)
    
    tokenizer = BertTokenizer.from_checkpoint(MODEL_NAME) if hasattr(BertTokenizer, 'from_checkpoint') else BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_ds = HateSpeechDataset(train_df.text.to_numpy(), train_df.label.to_numpy(), tokenizer, MAX_LEN)
    val_ds = HateSpeechDataset(val_df.text.to_numpy(), val_df.label.to_numpy(), tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, len(train_df))
        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(model, val_loader, device, len(val_df))
        print(f"Val loss {val_loss} accuracy {val_acc}")

        if val_acc > best_accuracy:
            save_path = os.path.join(models_dir, 'bert_hate_speech')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            best_accuracy = val_acc

    print("Training complete. Best Val Accuracy:", best_accuracy)

if __name__ == "__main__":
    run_training()
