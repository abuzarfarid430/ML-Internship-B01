import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)

print("Dataset Preview:")
print(df.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia_values = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()




silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8,5))
plt.plot(K_range, silhouette_scores, marker='o', color='green')
plt.title("Silhouette Scores for Different K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()



optimal_k = 3   # Based on elbow & silhouette (Iris usually best at 3)
print(f"Optimal K selected: {optimal_k}")



# Random initialization
kmeans_random = KMeans(n_clusters=optimal_k, init='random', random_state=42)
labels_random = kmeans_random.fit_predict(X_scaled)

# k-means++ initialization
kmeans_plus = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
labels_plus = kmeans_plus.fit_predict(X_scaled)

print("Inertia (Random):", kmeans_random.inertia_)
print("Inertia (K-Means++):", kmeans_plus.inertia_)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters (k-means++)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels_plus,
    palette='Set1'
)
plt.title("K-Means Clusters (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()


df['Cluster'] = labels_plus


cluster_profile = df.groupby('Cluster').mean()

print("\nCluster Profile Statistics:")
print(cluster_profile)


df.to_csv("iris_clustered_output.csv", index=False)
print("\nClustered data saved as 'iris_clustered_output.csv'")