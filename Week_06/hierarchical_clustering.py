import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from scipy.cluster.hierarchy import dendrogram, linkage


iris = load_iris()
X = iris.data
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)

print("Dataset Preview:")
print(df.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


plt.figure(figsize=(10, 6))
Z = linkage(X_scaled, method='ward')
dendrogram(Z)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


methods = ['single', 'complete', 'average']

for method in methods:
    plt.figure(figsize=(10, 6))
    Z = linkage(X_scaled, method=method)
    dendrogram(Z)
    plt.title(f"Dendrogram ({method.capitalize()} Linkage)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

# Based on dendrogram (Iris usually best at 3)
optimal_clusters = 3
print(f"Optimal clusters selected: {optimal_clusters}")


agg_model = AgglomerativeClustering(
    n_clusters=optimal_clusters,
    linkage='ward'
)

agg_labels = agg_model.fit_predict(X_scaled)

df['Agg_Cluster'] = agg_labels


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=agg_labels,
    palette='Set2'
)
plt.title("Agglomerative Clustering (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Cluster'] = kmeans_labels


agg_silhouette = silhouette_score(X_scaled, agg_labels)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print("\nSilhouette Scores Comparison:")
print("Agglomerative Clustering:", round(agg_silhouette, 4))
print("K-Means Clustering:", round(kmeans_silhouette, 4))


df.to_csv("hierarchical_clustering_output.csv", index=False)
print("\nClustered data saved as 'hierarchical_clustering_output.csv'")