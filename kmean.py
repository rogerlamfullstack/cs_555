# Apply K-Means clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from preprocessing import *
# K-Means clustering
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate K-Means using different metrics
silhouette_kmeans = silhouette_score(X, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(X, kmeans_labels)
davies_kmeans = davies_bouldin_score(X, kmeans_labels)
# Assuming you have true labels (if available)
true_labels = dataset['train']['emotion']
true_labels = [label for sublist in true_labels for label in sublist]

# ARI requires true labels
# Adjust the following code if your dataset has a ground truth
ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)

print(f"K-Means Silhouette Score: {silhouette_kmeans}")
print(f"K-Means Calinski-Harabasz Index: {calinski_kmeans}")
print(f"K-Means Davies-Bouldin Index: {davies_kmeans}")
print(f"K-Means Adjusted Rand Index: {ari_kmeans}")