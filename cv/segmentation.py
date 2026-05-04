import cv2
import numpy as np

def segment_and_isolate_sign(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    Z = img_lab.reshape((-1, 3)).astype(np.float32) 

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape(img.shape[:2])
    
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()].reshape(img.shape)
    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_Lab2BGR)
    mask = np.uint8(labels == 1) * 255 
    
    return segmented_img, mask
