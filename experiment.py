from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import torch
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# === Load and preprocess ===
dataset = load_dataset("daily_dialog")
dialogs = dataset['test']['dialog']
emotions = dataset['test']['emotion']
utterances = [utt for dialog in dialogs for utt in dialog]
emotion_labels = [emo for dialog_emos in emotions for emo in dialog_emos]
num_unique_emotions = len(set(emotion_labels))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

utterances_cleaned = [preprocess(u) for u in utterances]

# ===  BERT embeddings ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

print('size of data', len(utterances_cleaned))
X = [get_bert_embedding(u) for u in utterances_cleaned]
X_np = np.vstack(X)
y_true = np.array(emotion_labels[:len(X_np)])
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=50).fit_transform(X_np)
# === KMeans clustering ===
kmeans = KMeans(n_clusters=num_unique_emotions, random_state=42).fit(X_np)
labels_kmeans = kmeans.labels_

silhouette_kmeans = silhouette_score(X_np, labels_kmeans)
calinski_kmeans = calinski_harabasz_score(X_np, labels_kmeans)
davies_kmeans = davies_bouldin_score(X_np, labels_kmeans)
ari_kmeans = adjusted_rand_score(y_true, labels_kmeans)

print(f"K-Means Silhouette Score: {silhouette_kmeans:.4f}")
print(f"K-Means Calinski-Harabasz Index: {calinski_kmeans:.2f}")
print(f"K-Means Davies-Bouldin Index: {davies_kmeans:.2f}")
if ari_kmeans is not None:
    print(f"K-Means Adjusted Rand Index: {ari_kmeans:.4f}")

# ===  DBSCAN clustering ===
dbscan = DBSCAN(eps=3.0, min_samples=30).fit(X_np)
labels_dbscan = dbscan.labels_
# Number of clusters (-1 means noise)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

# Output results
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print("Cluster labels:", labels_dbscan)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_np)

plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_dbscan, cmap='viridis', marker='o')
plt.colorbar(label='Cluster Label')
plt.title("DBSCAN Clustering (t-SNE visualization)")
plt.show()
# Remove noise points (-1) for evaluation
mask = labels_dbscan != -1
if np.sum(mask) > 0 and len(set(labels_dbscan[mask])) > 1:
    silhouette_dbscan = silhouette_score(X_np[mask], labels_dbscan[mask])
    calinski_dbscan = calinski_harabasz_score(X_np[mask], labels_dbscan[mask])
    davies_dbscan = davies_bouldin_score(X_np[mask], labels_dbscan[mask])
    ari_dbscan = adjusted_rand_score(y_true[mask], labels_dbscan[mask])
else:
    silhouette_dbscan = calinski_dbscan = davies_dbscan = ari_dbscan = None

print()
print("=== DBSCAN Evaluation ===")
if silhouette_dbscan is not None:
    print(f"DBSCAN Silhouette Score: {silhouette_dbscan:.4f}")
    print(f"DBSCAN Calinski-Harabasz Index: {calinski_dbscan:.2f}")
    print(f"DBSCAN Davies-Bouldin Index: {davies_dbscan:.2f}")
    print(f"DBSCAN Adjusted Rand Index: {ari_dbscan:.4f}")
else:
    print("DBSCAN clustering failed (only one cluster or all noise).")
