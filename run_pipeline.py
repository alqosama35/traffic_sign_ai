import cv2
import glob
import os
from cv.preprocessing import preprocess
from cv.sift import extract_sift
from cv.features import BoVW
from cv.classifier import NB
from cv.evaluation import evaluate

# -------------------
# LOAD DATA (simple folder structure)
# -------------------
images = []
labels = []

for path in glob.glob("data/*/*.png"):
    label = os.path.basename(os.path.dirname(path))

    img = cv2.imread(path)
    img = preprocess(img)

    images.append(img)
    labels.append(label)

# -------------------
# SIFT FEATURES
# -------------------
descriptors = []
for img in images:
    _, des = extract_sift(img)
    descriptors.append(des)

# -------------------
# BoVW
# -------------------
bovw = BoVW(k=20)
bovw.fit(descriptors)

X = [bovw.transform(d) for d in descriptors]

# -------------------
# CLASSIFIER
# -------------------
clf = NB()
clf.train(X, labels)

pred = clf.predict(X)

# -------------------
# EVALUATION
# -------------------
acc, report = evaluate(labels, pred)

print("Accuracy:", acc)
print(report)