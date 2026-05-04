import cv2
import pandas as pd
import os

from cv.sift import extract_sift
from cv.features import BoVW
from cv.naive_bayes import NaiveBayesSiftTrainer
from cv.segmentation import segment_and_isolate_sign
from cv.matching import match_and_score
from cv.evaluation import run_evaluation


BASE_PATH = "data"
TRAIN_PATH = os.path.join(BASE_PATH, "Train")
TEST_PATH = os.path.join(BASE_PATH, "Test.csv")


# =========================
# LOAD TRAIN FROM FOLDERS
# =========================
def load_train(max_classes=None, max_per_class=None):
    images = []
    labels = []
    class_ids = sorted(os.listdir(TRAIN_PATH))

    if max_classes is not None:
        class_ids = class_ids[:max_classes]

    for class_id in class_ids:
        class_folder = os.path.join(TRAIN_PATH, class_id)

        if not os.path.isdir(class_folder):
            continue

        img_names = os.listdir(class_folder)

        if max_per_class is not None:
            img_names = img_names[:max_per_class]

        for img_name in img_names:
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            images.append(img)
            labels.append(int(class_id))

    return images, labels


# =========================
# LOAD TEST FROM CSV
# =========================
def load_test(max_samples=None):
    df = pd.read_csv(TEST_PATH)

    if max_samples is not None:
        df = df.iloc[:max_samples]

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(BASE_PATH, row["Path"])
        img = cv2.imread(img_path)

        if img is None:
            continue

        images.append(img)
        labels.append(int(row["ClassId"]))

    return images, labels


# =========================
# BUILD TRAIN HISTOGRAMS
# =========================
def build_bovw_features(images, labels, bovw):
    X = []
    y = []

    for img, label in zip(images, labels):
        _, des = extract_sift(img)

        if des is None or len(des) == 0:
            continue

        hist = bovw.transform(des)
        X.append(hist)
        y.append(label)

    return X, y


# =========================
# EVALUATION
# =========================
def run_full_evaluation(clf, bovw, test_imgs, test_labels, match_func):
    print("\nPreparing evaluation...")

    # 1) Classification
    y_pred = []
    for img in test_imgs:
        _, des = extract_sift(img)

        if des is None or len(des) == 0:
            pred = -1
        else:
            hist = bovw.transform(des)
            pred = clf.predict(hist)[0]

        y_pred.append(pred)

    # 2) Segmentation
    pred_masks = []
    gt_masks = []

    for img in test_imgs:
        seg, mask = segment_and_isolate_sign(img)
        pred_masks.append(mask)

        # Placeholder only.
        # Replace with real GT masks if available.
        gt_masks.append(mask)

    # 3) Matching inputs
    images = test_imgs
    labels = test_labels

    # 4) Run evaluation
    results = run_evaluation(
        y_true=test_labels,
        y_pred=y_pred,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        images=images,
        labels=labels,
        sift_match_func=match_func
    )

    print("\n===== FINAL RESULTS =====")
    print(results)

    return results


# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")

    train_imgs, train_labels = load_train(max_classes=10, max_per_class=100)
    test_imgs, test_labels = load_test(max_samples=100)

    print("Train:", len(train_imgs))
    print("Test:", len(test_imgs))

    # =========================
    # FEATURE EXTRACTION + BoVW TRAIN SET
    # =========================
    print("Building vocabulary...")

    train_descs = []
    for img in train_imgs:
        _, des = extract_sift(img)
        if des is not None and len(des) > 0:
            train_descs.append(des)

    bovw = BoVW(k=50)
    bovw.fit(train_descs)

    # Build histograms for classifier training
    print("Preparing training histograms...")
    X_train, y_train = build_bovw_features(train_imgs, train_labels, bovw)

    # =========================
    # TRAIN
    # =========================
    print("Training classifier...")
    clf = NaiveBayesSiftTrainer()
    clf.train(X_train, y_train)
    print("Training done!")

    # =========================
    # TEST SAMPLE
    # =========================
    print("\nTesting...")

    if len(test_imgs) > 0:
        _, des = extract_sift(test_imgs[0])
        if des is not None and len(des) > 0:
            hist = bovw.transform(des)
            pred = clf.predict(hist)[0]
            print("Predicted class:", pred)

    # =========================
    # SEGMENTATION
    # =========================
    if len(test_imgs) > 0:
        seg, mask = segment_and_isolate_sign(test_imgs[0])

        cv2.namedWindow("seg", cv2.WINDOW_NORMAL)
        cv2.imshow("seg", seg)
        cv2.resizeWindow("seg", 600, 400)

        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", mask)
        cv2.resizeWindow("mask", 600, 400)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # =========================
    # MATCHING
    # =========================
    if len(test_imgs) > 5:
        scores = match_and_score(test_imgs[2], test_imgs[5])
        print(scores)

    print("\nRunning full evaluation...")

    run_full_evaluation(
        clf=clf,
        bovw=bovw,
        test_imgs=test_imgs,
        test_labels=test_labels,
        match_func=match_and_score
    )

    print("Done!")


if __name__ == "__main__":
    main()