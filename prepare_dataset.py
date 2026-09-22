# import os
# import shutil
# import random
# from datasets import load_dataset

# # ============================================================
# # SETTINGS
# # ============================================================

# DATASET_NAME = "okhater/lolv2-real"

# TRAIN_RATIO = 0.8
# SEED = 42

# # ============================================================
# # DESTINATION FOLDERS
# # ============================================================

# TRAIN_LOW = "dataset/train/low"
# TRAIN_HIGH = "dataset/train/high"

# VAL_LOW = "dataset/val/low"
# VAL_HIGH = "dataset/val/high"

# TEST_LOW = "dataset/test/low"
# TEST_HIGH = "dataset/test/high"

# folders = [
#     TRAIN_LOW,
#     TRAIN_HIGH,
#     VAL_LOW,
#     VAL_HIGH,
#     TEST_LOW,
#     TEST_HIGH
# ]

# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

# # ============================================================
# # DOWNLOAD LOL-v2-Real FROM HUGGING FACE
# # ============================================================

# print("Downloading/loading LOL-v2-Real...")
# print("This may take some time the first time.")

# ds = load_dataset(DATASET_NAME)

# print("\nDataset loaded successfully!")
# print(ds)

# # ============================================================
# # CHECK DATASET
# # ============================================================

# print("\nDataset information:")

# for split in ds:
#     print(split, ":", len(ds[split]))

# print("\nColumns:")
# print(ds["train"].column_names)

# # ============================================================
# # GET IMAGES
# # ============================================================

# dataset = ds["train"]

# print("\nProcessing images...")

# input_images = []
# gt_images = []

# for i, item in enumerate(dataset):

#     label = item["label"]

#     # label 0 = GT
#     # label 1 = Input
#     #
#     # We identify the image type using the label.

#     if label == 0:
#         gt_images.append((i, item["image"]))

#     elif label == 1:
#         input_images.append((i, item["image"]))

# print("\nGT images    :", len(gt_images))
# print("Input images :", len(input_images))

# # ============================================================
# # IMPORTANT
# # ============================================================
# #
# # The Hugging Face dataset contains images as a single dataset.
# # We need to pair Input and GT images using their original
# # filename/path information where possible.
# #
# # Instead of relying on dataset order, we create image files
# # from the dataset and use the dataset index.
# # ============================================================

# # Save all images temporarily

# TEMP_INPUT = "data/hf_input"
# TEMP_GT = "data/hf_gt"

# os.makedirs(TEMP_INPUT, exist_ok=True)
# os.makedirs(TEMP_GT, exist_ok=True)

# print("\nSaving temporary images...")

# for i, image in input_images:

#     filename = f"{i:05d}.png"

#     image.save(
#         os.path.join(TEMP_INPUT, filename)
#     )

# for i, image in gt_images:

#     filename = f"{i:05d}.png"

#     image.save(
#         os.path.join(TEMP_GT, filename)
#     )

# print("Temporary images saved.")

# # ============================================================
# # CREATE PAIRS
# # ============================================================

# # NOTE:
# # The dataset contains Input and GT images.
# # We pair them according to their original ordering.

# input_count = len(input_images)
# gt_count = len(gt_images)

# pair_count = min(input_count, gt_count)

# print("\nPossible pairs:", pair_count)

# pairs = []

# for i in range(pair_count):

#     input_index, input_image = input_images[i]
#     gt_index, gt_image = gt_images[i]

#     pairs.append(
#         (
#             input_index,
#             gt_index
#         )
#     )

# # ============================================================
# # SHUFFLE PAIRS
# # ============================================================

# random.seed(SEED)
# random.shuffle(pairs)

# # ============================================================
# # TRAIN / VALIDATION SPLIT
# # ============================================================

# train_count = int(len(pairs) * TRAIN_RATIO)

# train_pairs = pairs[:train_count]
# val_pairs = pairs[train_count:]

# print("\n========================================")
# print("DATASET SPLIT")
# print("========================================")

# print("Total pairs :", len(pairs))
# print("Train pairs :", len(train_pairs))
# print("Val pairs   :", len(val_pairs))
# print("Test pairs  :", pair_count)

# # ============================================================
# # COPY TRAIN
# # ============================================================

# print("\nCreating training dataset...")

# for pair_number, (input_index, gt_index) in enumerate(train_pairs):

#     filename = f"{pair_number:05d}.png"

#     shutil.copy2(
#         os.path.join(TEMP_INPUT, f"{input_index:05d}.png"),
#         os.path.join(TRAIN_LOW, filename)
#     )

#     shutil.copy2(
#         os.path.join(TEMP_GT, f"{gt_index:05d}.png"),
#         os.path.join(TRAIN_HIGH, filename)
#     )

# # ============================================================
# # COPY VALIDATION
# # ============================================================

# print("Creating validation dataset...")

# for pair_number, (input_index, gt_index) in enumerate(val_pairs):

#     filename = f"{pair_number:05d}.png"

#     shutil.copy2(
#         os.path.join(TEMP_INPUT, f"{input_index:05d}.png"),
#         os.path.join(VAL_LOW, filename)
#     )

#     shutil.copy2(
#         os.path.join(TEMP_GT, f"{gt_index:05d}.png"),
#         os.path.join(VAL_HIGH, filename)
#     )

# # ============================================================
# # TEST DATA
# # ============================================================

# print("Creating test dataset...")

# # The dataset structure contains Test/Input and Test/GT,
# # but load_dataset() may expose them as part of the dataset.
# #
# # For now, test pairs are created from the remaining paired
# # images.

# for pair_number, (input_index, gt_index) in enumerate(pairs):

#     filename = f"{pair_number:05d}.png"

#     shutil.copy2(
#         os.path.join(TEMP_INPUT, f"{input_index:05d}.png"),
#         os.path.join(TEST_LOW, filename)
#     )

#     shutil.copy2(
#         os.path.join(TEMP_GT, f"{gt_index:05d}.png"),
#         os.path.join(TEST_HIGH, filename)
#     )

# # ============================================================
# # FINAL CHECK
# # ============================================================

# print("\n========================================")
# print("DATASET PREPARATION COMPLETE")
# print("========================================")

# print("Train low  :", len(os.listdir(TRAIN_LOW)))
# print("Train high :", len(os.listdir(TRAIN_HIGH)))

# print("Val low    :", len(os.listdir(VAL_LOW)))
# print("Val high   :", len(os.listdir(VAL_HIGH)))

# print("Test low   :", len(os.listdir(TEST_LOW)))
# print("Test high  :", len(os.listdir(TEST_HIGH)))

# print("\nDataset is ready!")

# from datasets import load_dataset

# print("Loading LOL-v2-Real...")

# ds = load_dataset("okhater/lolv2-real")

# print("\n================================")
# print("DATASET INFORMATION")
# print("================================")

# print(ds)

# for split in ds:
#     print(f"\nSplit: {split}")
#     print("Number of samples:", len(ds[split]))
#     print("Columns:", ds[split].column_names)

# print("\n================================")
# print("FIRST SAMPLE")
# print("================================")

# for split in ds:
#     print(f"\n--- {split} ---")
#     print(ds[split][0])



# from datasets import load_dataset

# ds = load_dataset("okhater/lolv2-real")

# dataset = ds["train"]

# print("Total images:", len(dataset))

# print("\nLabel counts:")
# print(dataset["label"])

# print("\nFirst 20 labels:")

# for i in range(20):
#     print(i, dataset[i]["label"], dataset[i]["image"].size)


# from datasets import load_dataset
# from collections import Counter

# ds = load_dataset("okhater/lolv2-real")
# dataset = ds["train"]

# labels = dataset["label"]

# print("Total images:", len(dataset))
# print("Label counts:", Counter(labels))

# print("\nWhere does label change?")

# previous = labels[0]

# for i, label in enumerate(labels):
#     if label != previous:
#         print("Label changes at index:", i)
#         print("Previous label:", previous)
#         print("New label:", label)
#         break


# from datasets import load_dataset

# ds = load_dataset("okhater/lolv2-real")
# dataset = ds["train"]

# print("Total:", len(dataset))

# # Print label ranges
# start = 0
# previous = dataset[0]["label"]

# for i in range(1, len(dataset)):
#     current = dataset[i]["label"]

#     if current != previous:
#         print(f"Index {start} to {i-1}: label {previous}")
#         start = i
#         previous = current

# print(f"Index {start} to {len(dataset)-1}: label {previous}")

import os
import shutil
import random
from pathlib import Path
from huggingface_hub import snapshot_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "okhater/lolv2-real"

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Where the downloaded Hugging Face repository will be stored
HF_DATASET_DIR = BASE_DIR / "hf_lolv2_real"

# Final RetinexNet dataset
OUTPUT_DIR = BASE_DIR / "dataset"

# Train/Validation split
VAL_RATIO = 0.20

# Reproducibility
RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    folders = [
        OUTPUT_DIR / "train" / "low",
        OUTPUT_DIR / "train" / "high",

        OUTPUT_DIR / "val" / "low",
        OUTPUT_DIR / "val" / "high",

        OUTPUT_DIR / "test" / "low",
        OUTPUT_DIR / "test" / "high",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("Output folders created.")


# ============================================================
# DOWNLOAD DATASET FROM HUGGING FACE
# ============================================================

def download_dataset():

    print("\n" + "=" * 60)
    print("DOWNLOADING LOL-v2 REAL DATASET")
    print("=" * 60)

    print(f"Repository: {REPO_ID}")
    print(f"Download location: {HF_DATASET_DIR}")

    if HF_DATASET_DIR.exists():

        print("\nDataset already downloaded.")
        print("Using existing downloaded files.")

    else:

        print("\nDownloading dataset...")

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(HF_DATASET_DIR),
        )

        print("\nDataset download completed.")


# ============================================================
# FIND IMAGE PAIRS
# ============================================================

def find_pairs(input_dir, gt_dir):

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found:\n{gt_dir}"
        )

    # Supported image extensions
    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    input_images = {
        file.name: file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    }

    gt_images = {
        file.name: file
        for file in gt_dir.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    }

    # Only use filenames that exist in BOTH folders
    common_names = sorted(
        set(input_images.keys()) & set(gt_images.keys())
    )

    missing_gt = sorted(
        set(input_images.keys()) - set(gt_images.keys())
    )

    missing_input = sorted(
        set(gt_images.keys()) - set(input_images.keys())
    )

    print(f"\nInput images : {len(input_images)}")
    print(f"GT images    : {len(gt_images)}")
    print(f"Valid pairs  : {len(common_names)}")

    if missing_gt:
        print(
            f"\nWARNING: {len(missing_gt)} input images "
            "do not have matching GT images."
        )

    if missing_input:
        print(
            f"WARNING: {len(missing_input)} GT images "
            "do not have matching input images."
        )

    pairs = []

    for name in common_names:
        pairs.append(
            (
                input_images[name],
                gt_images[name],
                name,
            )
        )

    return pairs


# ============================================================
# COPY IMAGE PAIRS
# ============================================================

def copy_pair(low_path, high_path, filename, split):

    low_destination = OUTPUT_DIR / split / "low" / filename
    high_destination = OUTPUT_DIR / split / "high" / filename

    shutil.copy2(low_path, low_destination)
    shutil.copy2(high_path, high_destination)


# ============================================================
# PROCESS TRAIN DATA
# ============================================================

def process_train_data():

    print("\n" + "=" * 60)
    print("PROCESSING TRAIN DATA")
    print("=" * 60)

    train_input = HF_DATASET_DIR / "Train" / "Input"
    train_gt = HF_DATASET_DIR / "Train" / "GT"

    pairs = find_pairs(train_input, train_gt)

    if len(pairs) == 0:
        raise RuntimeError("No training image pairs found.")

    # Shuffle pairs while keeping low/high together
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    val_count = int(len(pairs) * VAL_RATIO)
    train_count = len(pairs) - val_count

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    print(f"\nTotal training pairs : {len(pairs)}")
    print(f"Train pairs          : {len(train_pairs)}")
    print(f"Validation pairs     : {len(val_pairs)}")

    # Copy training pairs
    print("\nCopying training images...")

    for index, (low, high, filename) in enumerate(train_pairs, start=1):

        copy_pair(
            low,
            high,
            filename,
            "train"
        )

        if index % 50 == 0 or index == len(train_pairs):
            print(
                f"Train: {index}/{len(train_pairs)}"
            )

    # Copy validation pairs
    print("\nCopying validation images...")

    for index, (low, high, filename) in enumerate(val_pairs, start=1):

        copy_pair(
            low,
            high,
            filename,
            "val"
        )

        if index % 50 == 0 or index == len(val_pairs):
            print(
                f"Val:   {index}/{len(val_pairs)}"
            )

    return len(train_pairs), len(val_pairs)


# ============================================================
# PROCESS TEST DATA
# ============================================================

def process_test_data():

    print("\n" + "=" * 60)
    print("PROCESSING TEST DATA")
    print("=" * 60)

    test_input = HF_DATASET_DIR / "Test" / "Input"
    test_gt = HF_DATASET_DIR / "Test" / "GT"

    pairs = find_pairs(test_input, test_gt)

    if len(pairs) == 0:
        raise RuntimeError("No test image pairs found.")

    print(f"\nTotal test pairs: {len(pairs)}")

    print("\nCopying test images...")

    for index, (low, high, filename) in enumerate(pairs, start=1):

        copy_pair(
            low,
            high,
            filename,
            "test"
        )

        if index % 50 == 0 or index == len(pairs):
            print(
                f"Test: {index}/{len(pairs)}"
            )

    return len(pairs)


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset():

    print("\n" + "=" * 60)
    print("VERIFYING FINAL DATASET")
    print("=" * 60)

    all_correct = True

    for split in ["train", "val", "test"]:

        low_dir = OUTPUT_DIR / split / "low"
        high_dir = OUTPUT_DIR / split / "high"

        low_files = {
            file.name
            for file in low_dir.iterdir()
            if file.is_file()
        }

        high_files = {
            file.name
            for file in high_dir.iterdir()
            if file.is_file()
        }

        missing_high = low_files - high_files
        missing_low = high_files - low_files

        print(f"\n{split.upper()}")
        print("-" * 30)
        print(f"Low images  : {len(low_files)}")
        print(f"High images : {len(high_files)}")

        if missing_high:
            print(
                f"ERROR: {len(missing_high)} low images "
                "have no high counterpart."
            )
            all_correct = False

        if missing_low:
            print(
                f"ERROR: {len(missing_low)} high images "
                "have no low counterpart."
            )
            all_correct = False

        if not missing_high and not missing_low:
            print("Pairing     : OK")

    return all_correct


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       LOL-v2 REAL DATASET PREPARATION")
    print("       FOR RETINEXNET")
    print("=" * 60)

    try:

        # Step 1
        create_output_folders()

        # Step 2
        download_dataset()

        # Step 3
        train_count, val_count = process_train_data()

        # Step 4
        test_count = process_test_data()

        # Step 5
        verified = verify_dataset()

        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)

        print(f"\nTrain pairs : {train_count}")
        print(f"Val pairs   : {val_count}")
        print(f"Test pairs  : {test_count}")

        print(f"\nDataset location:")
        print(OUTPUT_DIR)

        print("\nFinal structure:")

        print("""
dataset/
├── train/
│   ├── low/
│   └── high/
├── val/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
""")

        if verified:
            print("✓ All low/high image pairs are correctly matched.")
        else:
            print("⚠ Dataset verification found problems.")

        print("\nYou can now train RetinexNet.")

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(f"\n{type(e).__name__}: {e}")

        print(
            "\nPlease check the error above and run the script again."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()