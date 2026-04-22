import os
import shutil
import random
import pandas as pd
from PIL import Image
import kagglehub

# ======================
# 0) Class Names
# ======================
CLASS_NAMES = {
0:'Speed limit 20', 1:'Speed limit 30', 2:'Speed limit 50', 3:'Speed limit 60',
4:'Speed limit 70', 5:'Speed limit 80', 6:'End speed 80', 7:'Speed limit 100',
8:'Speed limit 120', 9:'No passing', 10:'No passing >3.5t',
11:'Right of way jn.', 12:'Priority road', 13:'Yield', 14:'Stop',
15:'No vehicles', 16:'No veh >3.5t', 17:'No entry', 18:'General caution',
19:'Danger curve L', 20:'Danger curve R', 21:'Double curve',
22:'Bumpy road', 23:'Slippery road', 24:'Narrow road R',
25:'Road works', 26:'Traffic signals', 27:'Pedestrians',
28:'Children crossing', 29:'Bicycles crossing',
30:'Beware of ice', 31:'Wild animals', 32:'End restrictions',
33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
36:'Ahead or right', 37:'Ahead or left', 38:'Keep right',
39:'Keep left', 40:'Roundabout', 41:'End no passing',
42:'End no pass >3.5t'
}

# ======================
# Helpers
# ======================
def clean_name(name):
    return name.replace(">", "").replace(" ", "_")

def resize_image(src, dst, size=(224, 224)):
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)  # 🔥 أفضل جودة
        img.save(dst)
    except Exception as e:
        print(f"Error with {src}: {e}")

# ======================
# 1) Download Dataset (FIXED)
# ======================

def download_dataset():
    print("⬇ Downloading dataset using kagglehub...")

    path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    )

    print(" Downloaded at:", path)

    # نقل الداتا لملف المشروع
    if os.path.exists("data"):
        shutil.rmtree("data")

    shutil.copytree(path, "data")

    print(" Dataset moved to ./data")

# ======================
# 2) Split Train -> Train + Val
# ======================
def split_train_val(train_path, output_path="dataset", val_ratio=0.15):
    random.seed(42)

    train_out = os.path.join(output_path, "train")
    val_out = os.path.join(output_path, "val")

    for class_id in os.listdir(train_path):
        class_folder = os.path.join(train_path, class_id)

        if not os.path.isdir(class_folder):
            continue

        class_name = clean_name(CLASS_NAMES[int(class_id)])

        images = [img for img in os.listdir(class_folder)
                  if img.endswith(('.png', '.jpg', '.ppm'))]

        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_ratio))

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            for img in img_list:
                src = os.path.join(class_folder, img)

                dst_folder = os.path.join(output_path, split, class_name)
                os.makedirs(dst_folder, exist_ok=True)

                dst = os.path.join(dst_folder, img)
                resize_image(src, dst)

    print(" Train/Val split + resize done.")

# ======================
# 3) Organize Test
# ======================

def organize_test(test_path, csv_path, output_path="dataset"):
    test_out = os.path.join(output_path, "test")
    df = pd.read_csv(csv_path)

    df["ClassName"] = df["ClassId"].map(CLASS_NAMES)
    df["ClassName"] = df["ClassName"].apply(clean_name)

    for _, row in df.iterrows():
        img_name = row["Path"]
        img_name = os.path.basename(img_name)

        class_name = row["ClassName"]

        src = os.path.join(test_path, img_name)
        dst_folder = os.path.join(test_out, class_name)

        os.makedirs(dst_folder, exist_ok=True)

        dst = os.path.join(dst_folder, img_name)

        resize_image(src, dst)

    df.to_csv(os.path.join(output_path, "test_labeled.csv"), index=False)

    print(" Test organized correctly")


# ======================
# 4) Distribution Check
# ======================
def check_distribution(path):
    print(f"\n {path}")
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            print(f"{cls}: {len(os.listdir(cls_path))}")

# ======================
# 5) build annotation
# ======================

def build_annotation(dataset_path="dataset"):
    data = []

    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            # reverse mapping (class name → id)
            class_id = None
            for k, v in CLASS_NAMES.items():
                if clean_name(v) == class_name:
                    class_id = k
                    break

            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                data.append({
                    "image_name": img,
                    "image_path": img_path,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name
                })

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(dataset_path, "annotations.csv"), index=False)

    print(" Annotation file created ✔")
# ======================
# 5) Run Everything
# ======================
def prepare_data():
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")

    download_dataset()

    base_path = "data"

    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")
    csv_path = os.path.join(base_path, "Test.csv")

    split_train_val(train_path)
    organize_test(test_path, csv_path)
    build_annotation("dataset")

    check_distribution("dataset/train")
    check_distribution("dataset/val")
    check_distribution("dataset/test")

    print("\n Dataset Ready!")
# ======================
# RUN
# ======================
if __name__ == "__main__":
    prepare_data()