import cv2
import numpy as np
from cv.sift import extract_sift

def match_and_score(img1, img2):
    kp1, des1 = extract_sift(img1)
    kp2, des2 = extract_sift(img2)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    if len(good_matches) <= 4:
        return {
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return {
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0
        }

    inliers = int(np.sum(mask))
    total = len(good_matches)
    ratio = inliers / total

    return {
        "good_matches": total,
        "inliers": inliers,
        "inlier_ratio": ratio
    }