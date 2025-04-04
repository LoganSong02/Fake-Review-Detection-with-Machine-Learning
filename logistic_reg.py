import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def predict(w, X):
    prob = sigmoid(X @ w)
    return np.where(prob >= 0.5, 1, 0)

def train(X, y, lr=0.1, num_iters=1000, l2=0.0):
    N, D = X.shape
    w = np.zeros(D)
    for _ in tqdm(range(num_iters)):
        prob = sigmoid(X @ w)
        grad = (1 / N) * (X.T @ (prob - y)) + l2 * w
        w -= lr * grad
    return w

def evaluate(w, X, y):
    y_pred = predict(w, X)
    acc = accuracy_score(y, y_pred)
    return acc, y_pred

# load data
train_df = pd.read_csv("train_dataset.csv")
dev_df = pd.read_csv("dev_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

# drop non-numeric columns
drop_cols = ['review', 'category']
X_train = train_df.drop(columns=drop_cols + ['label']).values
y_train = train_df['label'].values

X_dev = dev_df.drop(columns=drop_cols + ['label']).values
y_dev = dev_df['label'].values

X_test = test_df.drop(columns=drop_cols + ['label']).values
y_test = test_df['label'].values

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# hyperparameter grids
learning_rates = [1, 0.1, 0.01]
l2_regs = [0, 0.1, 0.01]
num_iters = 1000

results = []
best_acc = 0
best_model = None
best_params = {}

for lr in learning_rates:
    for l2 in l2_regs:
        print(f"\nTraining with learning rate = {lr}, L2 = {l2}")
        w = train(X_train, y_train, lr=lr, num_iters=num_iters, l2=l2)
        acc, _ = evaluate(w, X_dev, y_dev)
        results.append((lr, l2, acc))
        if acc > best_acc:
            best_acc = acc
            best_model = w
            best_params = {'lr': lr, 'l2': l2}

# print result table
print("\n=== Hyperparameter Results ===")
print("Learning Rate\tL2\t\tDev Accuracy")
for lr, l2, acc in results:
    print(f"{lr:<14}{l2:<8}\t{acc:.4f}")

# final evaluation on test set
print("\n=== Best Model Evaluation on Test Set ===")
test_acc, y_test_pred = evaluate(best_model, X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# confusion Matrix
cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake (0)", "Genuine (1)"])

fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")  
ax.set_title(f"Confusion Matrix (lr={best_params['lr']}, l2={best_params['l2']})")

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\n=== Test Metrics ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# save figure
plt.savefig("confusion_matrix_logistic_reg.png", bbox_inches="tight")
plt.show()


# === Print 10 False Positives and 10 False Negatives For Discussion===
test_reviews = test_df["review"].tolist()

false_positives = []
false_negatives = []

for idx, (text, true_label, pred_label) in enumerate(zip(test_reviews, y_test, y_test_pred)):
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
