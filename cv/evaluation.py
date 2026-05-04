import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)

# =========================
# 1. CLASSIFICATION EVAL
# =========================

def evaluate_classification(y_true, y_pred, name="CV Baseline"):
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    print(f"\n===== {name} Classification =====")
    print("Accuracy:", acc)
    print("Macro Precision:", macro_p)
    print("Macro Recall:", macro_r)
    print("Macro F1:", macro_f1)

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred))

    return acc, macro_p, macro_r, macro_f1


# =========================
# 2. SEGMENTATION IoU
# =========================

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0
    return intersection / union


def evaluate_segmentation(pred_masks, gt_masks):
    ious = []

    for p, g in zip(pred_masks, gt_masks):
        ious.append(compute_iou(p, g))

    miou = np.mean(ious)

    print("\n===== Segmentation =====")
    print("Mean IoU:", miou)

    return miou


# =========================
# 3. SIFT MATCHING ACCURACY
# =========================

def evaluate_sift_matching(images, labels, match_func, threshold=10):
    correct = 0
    total = 0

    for i in range(len(images)):
        for j in range(i + 1, len(images)):

            matches, _, _ = match_func(images[i], images[j])

            same_class = (labels[i] == labels[j])
            predicted_same = len(matches) > threshold

            if predicted_same == same_class:
                correct += 1

            total += 1

    acc = correct / total

    print("\n===== SIFT Matching =====")
    print("Matching Accuracy:", acc)

    return acc


# =========================
# 4. FULL PIPELINE WRAPPER
# =========================

def run_evaluation(
    y_true, y_pred,
    pred_masks, gt_masks,
    images, labels,
    sift_match_func
):

    results = {}

    # Classification
    acc, p, r, f1 = evaluate_classification(y_true, y_pred)
    results["acc"] = acc
    results["prec"] = p
    results["rec"] = r
    results["f1"] = f1

    # Segmentation
    results["miou"] = evaluate_segmentation(pred_masks, gt_masks)

    # SIFT Matching
    results["match_acc"] = evaluate_sift_matching(
        images, labels, sift_match_func
    )

    return results


# =========================
# 5. COMPARISON TABLE (FR-CV-08)
# =========================

def build_comparison_table(cv_res, mob_res, ga_res, eff_res=None):

    data = {
        "Model": ["CV Baseline", "MobileNetV2", "GA-Optimized"],
        "Accuracy": [cv_res["acc"], mob_res["acc"], ga_res["acc"]],
        "Precision": [cv_res["prec"], mob_res["prec"], ga_res["prec"]],
        "Recall": [cv_res["rec"], mob_res["rec"], ga_res["rec"]],
        "F1": [cv_res["f1"], mob_res["f1"], ga_res["f1"]],
        "mIoU (CV only)": [cv_res["miou"], None, None]
    }

    if eff_res:
        data["Model"].append("EfficientNet-B0")
        data["Accuracy"].append(eff_res["acc"])
        data["Precision"].append(eff_res["prec"])
        data["Recall"].append(eff_res["rec"])
        data["F1"].append(eff_res["f1"])
        data["mIoU (CV only)"].append(None)

    df = pd.DataFrame(data)

    print("\n===== FINAL COMPARISON TABLE =====")
    print(df)

    return df