import pandas as pd
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv('reviews_dataset.csv')

# rename the review text column
df = df.rename(columns={'text_': 'review'})

# clean the review column
df['review'] = df['review'].fillna("").astype(str)

# encode labels: OR -> 1 (Original), CG -> 0 (Generated)
df['label'] = df['label'].map({'OR': 1, 'CG': 0})

# feature extraction functions
def count_words(text):
    return len(text.split())

def count_sentences(text):
    rough_split = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in rough_split if s.strip()]
    return len(sentences)

def count_exclamations(text):
    return text.count('!')

def count_first_person(text):
    return len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', text, flags=re.IGNORECASE))

# feature engineering
df['review_length'] = df['review'].apply(count_words)
df['sentence_count'] = df['review'].apply(count_sentences)
df['exclamation_count'] = df['review'].apply(count_exclamations)
df['first_person_count'] = df['review'].apply(count_first_person)

# save the fully processed dataset
df.to_csv('processed_reviews_dataset.csv', index=False)

# split dataset into 70% train, 10% dev, and 20% test while preserving label distribution
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=2/3, stratify=temp_df['label'], random_state=42)

# print label distribution in each split for verification
print("Train label ratio:\n", train_df['label'].value_counts(normalize=True))
print("Dev label ratio:\n", dev_df['label'].value_counts(normalize=True))
print("Test label ratio:\n", test_df['label'].value_counts(normalize=True))

# save all three datasets
train_df.to_csv('train_dataset.csv', index=False)
dev_df.to_csv('dev_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("\nDataset processing and splitting completed. Files saved as: ")
print("   - processed_reviews_dataset.csv")
print("   - train_dataset.csv")
print("   - dev_dataset.csv")
print("   - test_dataset.csv")
