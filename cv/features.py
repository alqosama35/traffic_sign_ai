import numpy as np
from sklearn.cluster import KMeans

class BoVW:
    def __init__(self, k=20):
        self.kmeans = KMeans(n_clusters=k)

    def fit(self, descriptors_list):
        all_desc = np.vstack([d for d in descriptors_list if d is not None])
        self.kmeans.fit(all_desc)

    def transform(self, desc):
        if desc is None:
            return np.zeros(self.kmeans.n_clusters)

        hist = np.zeros(self.kmeans.n_clusters)
        preds = self.kmeans.predict(desc)

        for p in preds:
            hist[p] += 1

        return hist / (np.linalg.norm(hist) + 1e-6)