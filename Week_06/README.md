# 📘 Week 6: Clustering & Dimensionality Reduction

This repository contains implementations of unsupervised learning
algorithms and dimensionality reduction techniques using Python and
Scikit-Learn.

The goal of this week is to understand how clustering works, how to
visualize high-dimensional data, and how different algorithms behave on
different types of datasets.

------------------------------------------------------------------------

# 📂 Tasks Overview

## ✅ Task 6.1 -- K-Means Clustering

File: kmeans_clustering.py

-   Implemented K-Means clustering
-   Used Iris dataset
-   Determined optimal number of clusters using:
    -   Elbow Method
    -   Silhouette Score
-   Compared initialization methods (random vs k-means++)
-   Applied PCA for 2D visualization
-   Created cluster profiles
-   Saved cluster assignments

------------------------------------------------------------------------

## ✅ Task 6.2 -- Hierarchical Clustering & Dendrograms

File: hierarchical_clustering.py

-   Implemented Agglomerative Clustering
-   Created dendrograms using ward, single, complete, and average
    linkage
-   Determined optimal clusters by cutting dendrogram
-   Compared results with K-Means
-   Calculated silhouette scores

------------------------------------------------------------------------

## ✅ Task 6.3 -- DBSCAN & Density-Based Clustering

File: dbscan_clustering.py

-   Implemented DBSCAN clustering
-   Used non-spherical dataset (make_moons)
-   Tuned eps and min_samples parameters
-   Identified noise points (label = -1)
-   Compared DBSCAN with K-Means
-   Created side-by-side comparison plots

------------------------------------------------------------------------

## ✅ Task 6.4 -- Dimensionality Reduction (PCA & t-SNE)

File: dimensionality_reduction.py

-   Applied PCA and t-SNE
-   Used MNIST dataset
-   Plotted explained variance ratio and cumulative variance
-   Created 2D and 3D visualizations
-   Compared PCA vs t-SNE
-   Trained classifier on original vs reduced features
-   Compared accuracy and training time

------------------------------------------------------------------------

# 🛠 Required Libraries

Install dependencies using:

pip install numpy pandas matplotlib seaborn scikit-learn scipy

------------------------------------------------------------------------

# 📊 Algorithms Implemented

Clustering: - K-Means - Agglomerative Clustering - DBSCAN

Dimensionality Reduction: - PCA - t-SNE

Evaluation: - Silhouette Score

Visualization: - 2D & 3D Scatter Plots - Dendrograms

------------------------------------------------------------------------

# 🎯 Key Learnings

-   Difference between centroid-based, hierarchical, and density-based
    clustering
-   How to determine optimal number of clusters
-   How DBSCAN detects noise and arbitrary shapes
-   How PCA reduces dimensions while preserving variance
-   Why t-SNE is better for visualization but not ideal for training
-   Impact of dimensionality reduction on training speed and accuracy

------------------------------------------------------------------------

# 🚀 Conclusion

Week 6 provided practical experience with clustering, density-based
models, dimensionality reduction, and high-dimensional data
visualization. These techniques are widely used in customer
segmentation, image processing, anomaly detection, and exploratory data
analysis.
