import cv2

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    return img