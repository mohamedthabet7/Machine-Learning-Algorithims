import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Euclidean Distance Function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# New Centroids Function
def new_centroids(old_clusters):
    new_centers = []
    for cluster in old_clusters:
        center = np.mean(cluster, axis=0)
        new_centers.append(center)
    return new_centers

# Define K-means Function
def kmeans(x, iterrations, k=0, set_centroids=None):
    # If k not provided, derive from set_centroids
    if k == 0 and set_centroids is not None:
        k = len(set_centroids)
    
    # If centroids Provided, Use them
    if set_centroids is not None and len(set_centroids) > 0:
        centroids = list(set_centroids)
    else:
        centroids = []
        for _ in range(k):
            random_x = np.random.uniform(x['X'].min(), x['X'].max())
            random_y = np.random.uniform(x['Y'].min(), x['Y'].max())
            centroids.append([random_x, random_y])
         
    clusters = [[] for _ in range(k)]
    
    for _ in range(iterrations):
        clusters = [[] for _ in range(k)]
        for i, row in x.iterrows():
            point = row[['X', 'Y']].values
            distances = []

            for center in centroids:
                distance = euclidean_distance(point, center)
                distances.append(distance)

            # Calculate Min distance from each Clusters Center, and store index
            min_distance = min(distances)
            cluster_choice = distances.index(min_distance)

            # Store the point in the assigned cluster index
            clusters[cluster_choice].append(point)
        
        # Update Centroids List with the mean of each cluster
        centroids = new_centroids(clusters)
        
    # Set Final Cluster to the last updated cluster
    final_cluster = clusters
    return final_cluster, centroids

if __name__ == "__main__":
    # Create DataFrame (Dataset 1: Known 8 Points)
    df = pd.DataFrame({
        'Point': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
        'X': [2, 2, 8, 5, 7, 6, 1, 4],
        'Y': [10, 5, 4, 8, 5, 4, 2, 9]
    })

    # Create Synthetic data (Dataset 2: 40 Random Points)
    np.random.seed(42)
    n_data = 40
    df1 = pd.DataFrame({
        'Point': [f'P{i+1}' for i in range(n_data)],
        'X': np.random.uniform(0, 10, n_data),
        'Y': np.random.uniform(0, 10, n_data)
    })

    # Create 3 initial centroids from Dataset 1
    c1 = df.loc[df["Point"] == "A1", ["X", "Y"]].values[0]
    c4 = df.loc[df["Point"] == "A4", ["X", "Y"]].values[0]
    c7 = df.loc[df["Point"] == "A7", ["X", "Y"]].values[0]
    centroids = [c1, c4, c7]

    # Run K-Means
    clusters, final_centroids = kmeans(x=df, iterrations=2, set_centroids=centroids)
    clusters1, final_centroids1 = kmeans(x=df1, iterrations=3, k=3)

    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dataset 1 Plot
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            axes[0].scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i+1}")

    axes[0].set_title("Dataset 1 (Predefined Centroids)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()

    # Dataset 2 Plot
    for i, cluster in enumerate(clusters1):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            axes[1].scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i+1}")

    axes[1].set_title("Dataset 2 (Random Centroids, 40 Points)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
