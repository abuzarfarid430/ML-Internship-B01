import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset shape:", X.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=50)
X_pca_50 = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_



plt.figure(figsize=(8,5))
plt.bar(range(1,51), explained_variance)
plt.title("Explained Variance Ratio (First 50 Components)")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.show()




cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(8,5))
plt.plot(range(1,51), cumulative_variance, marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.show()



pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)


plt.figure(figsize=(8,6))
plt.scatter(X_pca_2d[:,0], X_pca_2d[:,1], c=y, cmap='tab10', s=5)
plt.title("PCA 2D Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()


print("Running t-SNE (this may take some time)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled[:5000])   # Use subset for speed
y_subset = y[:5000]

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_subset, cmap='tab10', s=5)
plt.title("t-SNE 2D Visualization")
plt.colorbar()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled[:5000])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:,0], X_pca_3d[:,1], X_pca_3d[:,2],
           c=y[:5000], cmap='tab10', s=5)
ax.set_title("PCA 3D Visualization")
plt.show()


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---- Original Features ----
start = time.time()
clf_original = LogisticRegression(max_iter=1000)
clf_original.fit(X_train, y_train)
train_time_original = time.time() - start

y_pred_original = clf_original.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)


# ---- PCA Reduced (50 components) ----
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca_50, y, test_size=0.2, random_state=42
)

start = time.time()
clf_pca = LogisticRegression(max_iter=1000)
clf_pca.fit(X_train_pca, y_train)
train_time_pca = time.time() - start

y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print("\nClassification Results Comparison:")
print("Original Features Accuracy:", round(acc_original,4))
print("Original Training Time:", round(train_time_original,2), "seconds")

print("\nPCA (50 Components) Accuracy:", round(acc_pca,4))
print("PCA Training Time:", round(train_time_pca,2), "seconds")