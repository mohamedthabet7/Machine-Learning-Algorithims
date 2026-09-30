import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Euclidean Distance Function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# New medoids Function: picks the point within the cluster that minimizes total distance to all other points
def new_medoids(old_clusters):
    new_medoids = []
    
    for cluster in old_clusters:
        distances = []
        for medoid in cluster:
            total_distance = 0
            for point in cluster:
                total_distance += euclidean_distance(point, medoid)
            distances.append(total_distance)
            
        min_distance = min(distances)
        medoid_choice = distances.index(min_distance)
        new_medoids.append(cluster[medoid_choice])
    
    return new_medoids

# Define K-Medoids Function
def kmedoids(x, iterrations, set_medoids):
    medoids = list(set_medoids)
    clusters = [[] for _ in range(len(medoids))]
    
    for _ in range(iterrations):
        clusters = [[] for _ in range(len(medoids))]
        for i, row in x.iterrows():
            point = row[['X', 'Y']].values
            distances = []

            for medoid in medoids:
                distance = euclidean_distance(point, medoid)
                distances.append(distance)

            # Calculate Min distance from each Cluster's Medoid, and store index
            min_distance = min(distances)
            cluster_choice = distances.index(min_distance)

            # Store the point in the assigned cluster index
            clusters[cluster_choice].append(point)
        
        # Update Medoids List with the point that minimizes intra-cluster distance
        medoids = new_medoids(clusters)
        
    # Set Final Cluster to the last updated cluster
    final_cluster = clusters
    return final_cluster, medoids

if __name__ == "__main__":
    # Create DataFrame
    df = pd.DataFrame({
        'Point': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
        'X': [2, 2, 8, 5, 7, 6, 1, 4],
        'Y': [10, 5, 4, 8, 5, 4, 2, 9]
    })

    # Create 3 initial Medoids
    c1 = df.loc[df["Point"] == "A1", ["X", "Y"]].values[0]
    c4 = df.loc[df["Point"] == "A4", ["X", "Y"]].values[0]
    c7 = df.loc[df["Point"] == "A7", ["X", "Y"]].values[0]
    medoids = [c1, c4, c7]

    # Call the kmedoids function
    clusters, final_medoids = kmedoids(x=df, iterrations=2, set_medoids=medoids)

    # Create Figure
    plt.figure(figsize=(10, 6))

    # Display Clusters
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            plt.scatter(
                cluster[:, 0],
                cluster[:, 1],
                label=f"Cluster {i+1}",
                s=80
            )

    # Plot final medoids
    medoids_arr = np.array(final_medoids)
    plt.scatter(
        medoids_arr[:, 0],
        medoids_arr[:, 1],
        color='black',
        marker='X',
        s=200,
        label='Medoids'
    )
      
    plt.title("K-Medoids Clustering (Dataset 1)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
