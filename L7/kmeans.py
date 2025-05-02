import numpy as np
from tqdm.auto import tqdm

class Kmeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None
    
    def fit(self, X, max_iterations=100):
        m, _ = X.shape

        # Randomly Select k centroids from sample
        self.centroids = X[np.random.randint(low=0, high = m, size = self.k)]

        for _ in tqdm(range(max_iterations)):

            # Label each data point centroid index that are near to them
            clusters = self.group_clusters(X)

            # Store previous iterations centroids for difference
            prev_centroid = self.centroids

            # Update centroids by taking mean of each labeled cluster
            self.update_centroid(clusters, X)

            # Check for difference between previous centroids and new updated centroids
            diff = prev_centroid - self.centroids
            if not diff.any():
                # Just to stop any further iterations if any one of them are same wrt previous iteration
                return

    # Assign Label to each point
    def group_clusters(self, X):
        cluster_idx = np.array(
            [self.closest_centroid(x) for x in X]
        )
        return cluster_idx
    
    # Label data point with closest centroid
    def closest_centroid(self, X):
        distances = np.linalg.norm(X - self.centroids)
        return np.argmin(distances)
    
    def update_centroid(self, clusters, X):
        for i in range(self.k):
            self.centroids[i] = np.mean(X[clusters==i], axis=0)
    
    def transform(self, X):
        cluster_idx = self.group_clusters(X)
        return cluster_idx
        

if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt
    X, y = datasets.make_blobs()
    kmean = Kmeans(3)
    kmean.fit(X)
    y_preds = kmean.transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


    plt.scatter(X[:, 0], X[:, 1], c=y_preds, cmap='viridis')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()



