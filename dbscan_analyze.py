from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from preprocessing import * # Assuming X is imported from here

# To help choose eps for DBSCAN with cosine metric
# Note: distances will be 1 - cosine_similarity
nn = NearestNeighbors(n_neighbors=5, metric='cosine') # n_neighbors should be min_samples
nn.fit(X)
distances, indices = nn.kneighbors(X)
# distances to the 4th nearest neighbor (min_samples - 1)
sorted_distances = np.sort(distances[:, 4], axis=0) 

plt.figure(figsize=(10, 6)) # Optional: make the figure larger for better readability
plt.plot(sorted_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("Distance to 4th nearest neighbor (eps)")
plt.title("k-distance graph for DBSCAN (cosine metric)")

# Save the plot to a file instead of showing it
plt.savefig("dbscan_k_distance_plot.png") 
plt.close() # Close the plot to free up memory

print("K-distance plot saved as dbscan_k_distance_plot.png")
# Look for the 'knee' in the plot to estimate eps