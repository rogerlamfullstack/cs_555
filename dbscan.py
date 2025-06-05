# Apply DBSCAN clustering
from preprocessing import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import time 

X = X.toarray()
start_time = time.time()
num_of_eps = 1
dbscan = DBSCAN(eps=num_of_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"DBSCAN clustering took {elapsed_time:.2f} seconds")
print("no of eps:", num_of_eps)
# Evaluate DBSCAN using different metrics
silhouette_dbscan = silhouette_score(X, dbscan_labels)
calinski_dbscan = calinski_harabasz_score(X, dbscan_labels)
davies_dbscan = davies_bouldin_score(X, dbscan_labels)

# ARI requires true labels
ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)

print(f"DBSCAN Silhouette Score: {silhouette_dbscan}")
try:
    print(f"DBSCAN Calinski-Harabasz Index: {calinski_dbscan}")
except NameError:
    print("Warning: 'true_labels' not defined. Skipping DBSCAN Calinski-Harabasz Index calculation.")
    ari_kmeans = None # Or handle as appropriate

print(f"DBSCAN Davies-Bouldin Index: {davies_dbscan}")
print(f"DBSCAN Adjusted Rand Index: {ari_dbscan}")