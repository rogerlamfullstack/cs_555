from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# 1. Load DailyDialog utterances
dataset = load_dataset(
    "daily_dialog",
    cache_dir="./hf_cache_clean"        # use a new local cache folder
)
print(dataset)

test_dialogues = dataset['test']['dialog']
true_labels_test = dataset['test']['emotion']  # Ground truth labels (emotion)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)
# 1 preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
dialogues = dataset['train']['dialog']
# flattened_dialogues = [turn for dialog in train_dialogues for turn in dialog]
# flattened_test_dialogues = [turn for dialog in test_dialogues for turn in dialog]
# flattened_train_labels = [label for sublist in dataset['train']['emotion'] for label in sublist]
# flattened_test_labels = [label for sublist in true_labels_test for label in sublist]

# # Print the lengths of flattened data to ensure consistency
# print(f"Number of flattened train dialogues: {len(flattened_dialogues)}")
# print(f"Number of flattened train labels: {len(flattened_train_labels)}")
all_original_utterances = []
for dialogue_list in dialogues:
    all_original_utterances.extend(dialogue_list)

proces_data = [preprocess_text(d) for d in all_original_utterances]
print("--- Processed Dialogues (Sample) ---")
for i in range(3):
    print(f"Original: {all_original_utterances[i]}")
    print(f"Processed: {proces_data[i]}\n")

# Apply TF-IDF to the dialogues
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(proces_data)
true_labels = dataset['train']['emotion']
true_labels = [label for sublist in true_labels for label in sublist]
num_unique_emotions = len(set(true_labels))

print("No of topic:", num_unique_emotions)