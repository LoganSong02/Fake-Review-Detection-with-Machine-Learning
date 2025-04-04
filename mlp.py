import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# hyperparameter grid for MLP
hidden_layer_sizes = [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)]
learning_rates = [0.005, 0.01, 0.02]
num_iters = 200

# results
results = []
best_acc = 0
best_model = None
best_params = {}

# train and evaluate MLP for different hyperparameters
for hidden_layer_size in hidden_layer_sizes:
    for lr in learning_rates:
        print(f"\nTraining with hidden_layer_size = {hidden_layer_size}, learning rate = {lr}")
        
        # initialize and train the MLP model
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=num_iters, learning_rate_init=lr, random_state=42)
        mlp.fit(X_train, y_train)

        # evaluate on development set
        y_dev_pred = mlp.predict(X_dev)
        acc = accuracy_score(y_dev, y_dev_pred)
        results.append((hidden_layer_size, lr, acc))

        if acc > best_acc:
            best_acc = acc
            best_model = mlp
            best_params = {'hidden_layer_size': hidden_layer_size, 'lr': lr}

# print results
print("\n=== Hyperparameter Results ===")
print("Hidden Layer Size\tLearning Rate\tDev Accuracy")
for hidden_layer_size, lr, acc in results:
    print(f"{hidden_layer_size}\t\t{lr}\t\t{acc:.4f}")

# final evaluation on test set
print("\n=== Best Model Evaluation on Test Set ===")
y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

# confusion Matrix
cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake (0)", "Genuine (1)"])

fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_title(f"Confusion Matrix (hidden_layer_size={best_params['hidden_layer_size']}, lr={best_params['lr']})")

# print metrics
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\n=== Test Metrics ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# save figure
plt.savefig("confusion_matrix_mlp.png", bbox_inches="tight")
plt.show()

# === Print 10 False Positives and 10 False Negatives For Discussion ===
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
