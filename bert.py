import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU not available, using CPU")

# load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["review"].tolist(), df["label"].tolist()

train_texts, train_labels = load_data("train_dataset.csv")
dev_texts, dev_labels = load_data("dev_dataset.csv")
test_texts, test_labels = load_data("test_dataset.csv")

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts, labels, max_length=128):
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))

# evaluation
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1, all_labels, all_preds

# hyperparameter tuning: grid search over lr, epochs, batch_size
learning_rates = [5e-5]
num_epochs_list = [2]
batch_sizes = [16]
max_length = 128

best_acc = 0
best_model = None
best_config = {}

# tokenize datasets once
train_dataset = tokenize(train_texts, train_labels, max_length)
dev_dataset = tokenize(dev_texts, dev_labels, max_length)
test_dataset = tokenize(test_texts, test_labels, max_length)

# grid search loop
for lr, ep, bs in itertools.product(learning_rates, num_epochs_list, batch_sizes):
    print(f"\nlr={lr}, epochs={ep}, batch_size={bs}, max_len={max_length}")

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, pin_memory=True)

    # load BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * ep)

    # training loop
    model.train()
    for epoch in range(ep):
        print(f"Epoch {epoch + 1}/{ep}")
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # evaluate on dev
    acc, prec, rec, f1, _, _ = evaluate(model, dev_loader)
    print(f"Dev Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_config = {
            "lr": lr,
            "epochs": ep,
            "batch_size": bs,
            "max_length": max_length
        }

# final evaluation on test set
print("\nBest Config:", best_config)
test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(best_model, test_loader)

print("\n=== Final Test Metrics ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1 Score : {test_f1:.4f}")

# confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake (0)", "Genuine (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (BERT))")
plt.savefig("confusion_matrix_bert.png", bbox_inches="tight")
plt.show()

# === Print 10 False Positives and 10 False Negatives For Discussion ===
test_texts_full = test_texts  # already loaded earlier from CSV

false_positives = []
false_negatives = []

for idx, (text, true_label, pred_label) in enumerate(zip(test_texts_full, y_true, y_pred)):
    if len(false_positives) < 10 and true_label == 0 and pred_label == 1:
        false_positives.append((text, true_label, pred_label))
    elif len(false_negatives) < 10 and true_label == 1 and pred_label == 0:
        false_negatives.append((text, true_label, pred_label))
    if len(false_positives) >= 10 and len(false_negatives) >= 10:
        break

print("\n=== False Positives (Predicted Genuine, Actually Fake) ===")
for i, (text, true_label, pred_label) in enumerate(false_positives, 1):
    print(f"{i}. Text: {text}\n   True Label: {true_label}, Predicted: {pred_label}\n")

print("\n=== False Negatives (Predicted Fake, Actually Genuine) ===")
for i, (text, true_label, pred_label) in enumerate(false_negatives, 1):
    print(f"{i}. Text: {text}\n   True Label: {true_label}, Predicted: {pred_label}\n")
