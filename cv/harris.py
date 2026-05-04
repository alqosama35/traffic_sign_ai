import cv2
import numpy as np

def extract_harris(img, threshold=0.01):
    """
    Detect Harris corners and return keypoints
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    thresh = threshold * dst.max()

    keypoints = []
    ys, xs = np.where(dst > thresh)
    for (x, y) in zip(xs, ys):
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=3)
        keypoints.append(kp)

    return keypoints, dst