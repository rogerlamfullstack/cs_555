# Apply DBSCAN clustering
from preprocessing import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score


dbscan = DBSCAN(eps=0.15, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Evaluate DBSCAN using different metrics
silhouette_dbscan = silhouette_score(X, dbscan_labels)
calinski_dbscan = calinski_harabasz_score(X, dbscan_labels)
davies_dbscan = davies_bouldin_score(X, dbscan_labels)

# ARI requires true labels
ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)

print(f"DBSCAN Silhouette Score: {silhouette_dbscan}")
print(f"DBSCAN Calinski-Harabasz Index: {calinski_dbscan}")
print(f"DBSCAN Davies-Bouldin Index: {davies_dbscan}")
print(f"DBSCAN Adjusted Rand Index: {ari_dbscan}")