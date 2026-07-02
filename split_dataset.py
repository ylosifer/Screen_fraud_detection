import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = "dataset"
CLASSES = ["real", "screen"]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


def split_class(class_name):
    source_dir = os.path.join(DATASET_DIR, class_name)

    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Train split
    train_imgs, temp_imgs = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Validation/Test split
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    for split_name, file_list in splits.items():
        dest_dir = os.path.join(DATASET_DIR, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for file in file_list:
            shutil.copy2(
                os.path.join(source_dir, file),
                os.path.join(dest_dir, file)
            )

    print(
        f"{class_name}: "
        f"Train={len(train_imgs)}, "
        f"Val={len(val_imgs)}, "
        f"Test={len(test_imgs)}"
    )


def main():
    for class_name in CLASSES:
        split_class(class_name)

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()