import cv2
import numpy as np

def segment(img):
    Z = img.reshape((-1, 3)).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        Z, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(img.shape)