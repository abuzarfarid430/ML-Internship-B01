import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


X, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1])
plt.title("Original Moon Dataset")
plt.show()


eps_values = [0.1, 0.3, 0.5, 1.0]
min_samples_values = [3, 5, 10]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"eps={eps}, min_samples={min_samples}")
        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise}")
        print("-"*40)


# Good parameters for moons dataset
dbscan = DBSCAN(eps=0.3, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

# Identify noise
noise_points = db_labels == -1


plt.figure(figsize=(7,6))

# Plot clusters
unique_labels = set(db_labels)

for label in unique_labels:
    if label == -1:
        # Noise points
        plt.scatter(
            X_scaled[noise_points, 0],
            X_scaled[noise_points, 1],
            color='black',
            marker='x',
            label='Noise'
        )
    else:
        plt.scatter(
            X_scaled[db_labels == label, 0],
            X_scaled[db_labels == label, 1],
            label=f'Cluster {label}'
        )

plt.title("DBSCAN Clustering")
plt.legend()
plt.show()


n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print("Final DBSCAN clusters:", n_clusters_db)


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(12,5))

# DBSCAN
plt.subplot(1,2,1)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=db_labels)
plt.title("DBSCAN Result")

# K-Means
plt.subplot(1,2,2)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels)
plt.title("K-Means Result")

plt.show()


# Remove noise points for silhouette (DBSCAN)
if n_clusters_db > 1:
    mask = db_labels != -1
    db_silhouette = silhouette_score(X_scaled[mask], db_labels[mask])
    print("DBSCAN Silhouette Score:", round(db_silhouette,4))

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print("K-Means Silhouette Score:", round(kmeans_silhouette,4))