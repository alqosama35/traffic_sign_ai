import numpy as np
from sklearn.cluster import KMeans

class BoVW:
    def __init__(self, k=20):
        self.k = k
        self.kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        self.fitted = False

    def fit(self, descriptors_list):
        valid_desc = [
            d for d in descriptors_list
            if d is not None and len(d) > 0
        ]

        if len(valid_desc) == 0:
            raise ValueError("No valid descriptors to train BoVW")

        all_desc = np.vstack(valid_desc)

        self.kmeans.fit(all_desc)
        self.fitted = True

    def transform(self, desc):
        if not self.fitted:
            raise RuntimeError("BoVW not fitted yet")

        if desc is None or len(desc) == 0:
            return np.zeros(self.k)

        hist = np.zeros(self.k)

        preds = self.kmeans.predict(desc)

        for p in preds:
            hist[p] += 1

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm

        return hist