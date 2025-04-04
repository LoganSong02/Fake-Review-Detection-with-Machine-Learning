from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# tokenize the review text
def tokenize(text):
    return text.split()

# load data and prepare dataset
train_df = pd.read_csv("train_dataset.csv")
dev_df = pd.read_csv("dev_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

train_data = [(tokenize(row['review']), row['label']) for _, row in train_df.iterrows()]
dev_data = [(tokenize(row['review']), row['label']) for _, row in dev_df.iterrows()]
test_data = [(tokenize(row['review']), row['label']) for _, row in test_df.iterrows()]

# vocabulary: all unique words in the training set
vocabulary = list(set(word for review, _ in train_data for word in review))

# the set of words present in the training data
label_counts = Counter(label for _, label in train_data)

# count word occurrences per label
word_counts = defaultdict(Counter)
for review, label in train_data:
    for word in review:
        word_counts[label][word] += 1

# Laplace-smoothed Naive Bayes prediction
def predict(words, label_counts, word_counts, vocabulary, laplace_smoothing=1):
    labels = list(label_counts.keys())
    label_samples = sum(label_counts.values())
    total_words = len(vocabulary)
    computed_probs = {}

    for label in labels:
        label_prob = np.log(label_counts[label] / label_samples)
        curr_prob = label_prob
        num_words_label = sum(word_counts[label].values())

        for word in words:
            num_word = word_counts[label][word]
            smoothed_prob = (num_word + laplace_smoothing) / (num_words_label + laplace_smoothing * total_words)
            curr_prob += np.log(smoothed_prob)
        
        computed_probs[label] = curr_prob
    
    return max(computed_probs, key=computed_probs.get)

# computes accuracy for dev set and full metrics for test set
def evaluate(label_counts, word_counts, vocabulary, dataset, laplace_smoothing=1, is_test=False):
    y_true = []
    y_pred = []
    for words, label in dataset:
        pred_label = predict(words, label_counts, word_counts, vocabulary, laplace_smoothing)
        y_true.append(label)
        y_pred.append(pred_label)
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    if is_test:  # Only calculate precision, recall, F1 score on test set
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake (0)", "Genuine (1)"])
        return accuracy, precision, recall, f1, disp
    else:
        return accuracy  # For dev set, we only return accuracy

# hyperparameters: testing different Laplace smoothing values
laplace_smoothing_values = [0.001, 0.01, 0.1, 0.5, 1, 2]

# storing results
results = []

# test each value of Laplace smoothing for dev set
for laplace_smoothing in laplace_smoothing_values:
    print(f"\nEvaluating with Laplace smoothing = {laplace_smoothing}")

    acc = evaluate(label_counts, word_counts, vocabulary, dev_data, laplace_smoothing)
    results.append((laplace_smoothing, acc))
    
    print(f"Dev Accuracy: {acc:.4f}")

# display all results in a table for dev accuracy
print("\n=== Dev Accuracy Results ===")
print(f"{'Laplace Smoothing':<20}{'Accuracy':<10}")
for laplace_smoothing, acc in results:
    print(f"{laplace_smoothing:<20}{acc:<10.4f}")

# final evaluation on test set for best Laplace smoothing
best_laplace_smoothing = max(results, key=lambda x: x[1])[0]
test_acc, test_precision, test_recall, test_f1, disp = evaluate(label_counts, word_counts, vocabulary, test_data, best_laplace_smoothing, is_test=True)

# print test metrics
print(f"\nTest Set Evaluation (Laplace Smoothing = {best_laplace_smoothing}):")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# save the confusion matrix plot for the test set
confusion_matrix_filename = "confusion_matrix_naive_bayes.png"
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Laplace Smoothing = {best_laplace_smoothing})")
plt.savefig(confusion_matrix_filename, bbox_inches="tight") 
plt.close()

# === Print 10 False Positives and 10 False Negatives For Discussion ===
test_reviews = test_df["review"].tolist()

false_positives = []
false_negatives = []

for idx, (words, label) in enumerate(test_data):
    pred = predict(words, label_counts, word_counts, vocabulary, best_laplace_smoothing)
    if len(false_positives) < 10 and label == 0 and pred == 1:
        false_positives.append((test_reviews[idx], label, pred))
    elif len(false_negatives) < 10 and label == 1 and pred == 0:
        false_negatives.append((test_reviews[idx], label, pred))
    if len(false_positives) >= 10 and len(false_negatives) >= 10:
        break

print("\n=== False Positives (Predicted Genuine, Actually Fake) ===")
for i, (text, true_label, pred_label) in enumerate(false_positives, 1):
    print(f"{i}. Text: {text}\n   True Label: {true_label}, Predicted: {pred_label}\n")

print("\n=== False Negatives (Predicted Fake, Actually Genuine) ===")
for i, (text, true_label, pred_label) in enumerate(false_negatives, 1):
    print(f"{i}. Text: {text}\n   True Label: {true_label}, Predicted: {pred_label}\n")



