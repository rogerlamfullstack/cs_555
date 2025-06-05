# Apply K-Means clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from preprocessingv2 import * # Assuming this imports X and num_unique_emotions

# Ensure X is loaded and num_unique_emotions is defined from preprocessing.py
# For demonstration purposes, let's assume X is already a sparse matrix
# and num_unique_emotions is an integer.

# K-Means clustering
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Convert the sparse matrix X to a dense array for evaluation metrics
X_dense = X.toarray()

# Evaluate K-Means using different metrics
silhouette_kmeans = silhouette_score(X_dense, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(X_dense, kmeans_labels)
davies_kmeans = davies_bouldin_score(X_dense, kmeans_labels)

# Assuming you have true labels (if available)
# ARI requires true labels
# Adjust the following code if your dataset has a ground truth
# For this example, I'm assuming 'true_labels' is defined elsewhere.
# If you don't have true_labels, you cannot use adjusted_rand_score.
try:
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
except NameError:
    print("Warning: 'true_labels' not defined. Skipping Adjusted Rand Index calculation.")
    ari_kmeans = None # Or handle as appropriate

print(f"K-Means Silhouette Score: {silhouette_kmeans}")
print(f"K-Means Calinski-Harabasz Index: {calinski_kmeans}")
print(f"K-Means Davies-Bouldin Index: {davies_kmeans}")
if ari_kmeans is not None:
    print(f"K-Means Adjusted Rand Index: {ari_kmeans}")