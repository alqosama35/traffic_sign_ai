import cv2

def pyramid(img):
    levels = []
    for _ in range(3):
        img = cv2.pyrDown(img)
        levels.append(img)
    return levels