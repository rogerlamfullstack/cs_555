from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Load DailyDialog utterances
dataset = load_dataset(
    "daily_dialog",
    cache_dir="./hf_cache_clean"        # use a new local cache folder
)
print(dataset)

test_dialogues = dataset['test']['dialog']
true_labels_test = dataset['test']['emotion']  # Ground truth labels (emotion)

# 1 preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
train_dialogues = dataset['train']['dialog']
flattened_dialogues = [turn for dialog in train_dialogues for turn in dialog]
flattened_test_dialogues = [turn for dialog in test_dialogues for turn in dialog]
flattened_train_labels = [label for sublist in dataset['train']['emotion'] for label in sublist]
flattened_test_labels = [label for sublist in true_labels_test for label in sublist]

# Print the lengths of flattened data to ensure consistency
print(f"Number of flattened train dialogues: {len(flattened_dialogues)}")
print(f"Number of flattened test dialogues: {len(flattened_test_dialogues)}")
print(f"Number of flattened train labels: {len(flattened_train_labels)}")
print(f"Number of flattened test labels: {len(flattened_test_labels)}")

# Apply TF-IDF to the dialogues
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(flattened_dialogues).toarray()
