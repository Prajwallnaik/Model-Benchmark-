import os
import random
import shutil

# Change this to where your current folders exist
SOURCE_DIR = r"C:\Users\prajw\Downloads\tomoto"   # contains Tomato_Late_blight, Tomato_Early_blight
TARGET_DIR = "data"

TRAIN_RATIO = 0.8
random.seed(42)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_dir = os.path.join(TARGET_DIR, "train", cls)
    val_dir = os.path.join(TARGET_DIR, "val", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, img))

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(val_dir, img))

print("✅ Dataset split completed!")
