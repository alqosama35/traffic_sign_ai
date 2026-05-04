import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class NaiveBayesSiftTrainer:
    """
    Classifier that works on BoVW histograms only.
    Input shape:
        - one sample:  (k,)
        - batch:       (n_samples, k)
    """

    def __init__(self):
        self.clf = GaussianNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, features, labels):
        """
        features: list/np.ndarray of BoVW histograms
                  shape = (n_samples, k)
        labels:   list/np.ndarray of class ids
        """
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels)

        if X.size == 0:
            raise ValueError("No training features were provided to NaiveBayesSiftTrainer.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D training features, got shape {X.shape}")

        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        self.is_fitted = True

    def predict(self, feature):
        """
        feature: one BoVW histogram, shape = (k,)
        returns: np.ndarray with one predicted label
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")

        x = np.asarray(feature, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D feature, got shape {x.shape}")

        x = self.scaler.transform(x)
        return self.clf.predict(x)

    def predict_batch(self, features):
        """
        features: batch of BoVW histograms, shape = (n_samples, k)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")

        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {X.shape}")

        X = self.scaler.transform(X)
        return self.clf.predict(X)