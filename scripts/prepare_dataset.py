import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ISIC_CSV = os.path.join(PROJECT_ROOT,
                        "dataset",
                        "ISIC2019",
                        "ISIC_2019_Training_GroundTruth.csv")

HAM_CSV: str = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "HAM10000",
    "skin-cancer-mnist-ham10000",
    "HAM10000_metadata.csv"
)

PAD_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "PAD_UFES_20",
    "imgs_part_1",
    "metadata.csv"
)

IMAGE_DIR = os.path.join(PROJECT_ROOT,
                         "dataset",
                         "data",
                         "images")

print("="*60)
print("Loading datasets...")
print("="*60)

isic = pd.read_csv(ISIC_CSV)
ham = pd.read_csv(HAM_CSV)
pad = pd.read_csv(PAD_CSV)

print("\nISIC2019")
print(isic.head())
print()

print("HAM10000")
print(ham.head())
print()

print("PAD-UFES-20")
print(pad.head())
print()

print("="*60)
print("Dataset Shapes")
print("="*60)

print("ISIC :", isic.shape)
print("HAM  :", ham.shape)
print("PAD  :", pad.shape)
# ==========================================
# Prepare ISIC2019
# ==========================================

isic_rows = []

disease_columns = [
    "MEL","NV","BCC","AK","BKL",
    "DF","VASC","SCC","UNK"
]

for _, row in isic.iterrows():

    image_name = row["image"] + ".jpg"

    diagnosis = "non-BCC"

    for d in disease_columns:
        if row[d] == 1:
            if d == "BCC":
                diagnosis = "BCC"
            break

    isic_rows.append({
        "image_path": image_name,
        "label": diagnosis,
        "source": "ISIC2019"
    })

isic_df = pd.DataFrame(isic_rows)

print("\nISIC Converted")
print(isic_df.head())
# ==========================================
# Prepare HAM10000
# ==========================================

ham_rows = []

for _, row in ham.iterrows():

    image_name = row["image_id"] + ".jpg"

    diagnosis = "BCC" if row["dx"] == "bcc" else "non-BCC"

    ham_rows.append({
        "image_path": image_name,
        "label": diagnosis,
        "source": "HAM10000"
    })

ham_df = pd.DataFrame(ham_rows)

print("\nHAM Converted")
print(ham_df.head())
# ==========================================
# Prepare PAD-UFES-20
# ==========================================

pad_rows = []

for _, row in pad.iterrows():

    image_name = row["img_id"]

    diagnosis = "BCC" if row["diagnostic"] == "BCC" else "non-BCC"

    pad_rows.append({
        "image_path": image_name,
        "label": diagnosis,
        "source": "PAD_UFES_20"
    })

pad_df = pd.DataFrame(pad_rows)

print("\nPAD Converted")
print(pad_df.head())
# ==========================================
# Merge all datasets
# ==========================================

merged_df = pd.concat(
    [isic_df, ham_df, pad_df],
    ignore_index=True
)

print("\nMerged Dataset Shape:", merged_df.shape)

print("\nClass Distribution:")
print(merged_df["label"].value_counts())

print("\nSource Distribution:")
print(merged_df["source"].value_counts())
# ==========================================
# Verify image files exist
# ==========================================

# ==========================================
# Verify image files exist
# ==========================================

import glob

image_files = {}

# Search every image recursively inside dataset/
for ext in ("*.jpg", "*.jpeg", "*.png"):
    pattern = os.path.join(PROJECT_ROOT, "dataset", "**", ext)

    for img in glob.glob(pattern, recursive=True):
        image_files[os.path.basename(img)] = img

print(f"\nTotal images found on disk: {len(image_files)}")

merged_df["full_path"] = merged_df["image_path"].map(image_files)

missing = merged_df["full_path"].isna().sum()

print(f"Missing images: {missing}")

merged_df = merged_df.dropna(subset=["full_path"])

print(f"Remaining images: {len(merged_df)}")
# --------------------------------------------
# Train / Validation / Test Split
# --------------------------------------------

train_df, temp_df = train_test_split(
    merged_df,
    test_size=0.30,
    stratify=merged_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Save CSV files
output_dir = os.path.join(PROJECT_ROOT, "dataset", "data")
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("\n======================================")
print("Dataset Split Completed")
print("======================================")
print("Train :", len(train_df))
print("Validation :", len(val_df))
print("Test :", len(test_df))

print("\nCSV files saved to:")
print(output_dir)