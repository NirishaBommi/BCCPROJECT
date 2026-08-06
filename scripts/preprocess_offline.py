import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSVS = [
    os.path.join(PROJECT_ROOT, "dataset", "data", "train.csv"),
    os.path.join(PROJECT_ROOT, "dataset", "data", "val.csv"),
    os.path.join(PROJECT_ROOT, "dataset", "data", "test.csv")
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "data", "images_preprocessed")

def remove_hair(image: np.ndarray, radius: int = 15) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, int(0.15 * 255), 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

def reinhard_normalise(image: np.ndarray,
                       target_mean: np.ndarray = np.array([65.3, 12.1, 14.8], dtype=np.float32),
                       target_std: np.ndarray = np.array([22.1, 9.6, 8.4], dtype=np.float32)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    def norm_channel(ch, t_mean, t_std):
        src_mean, src_std = ch.mean(), ch.std() + 1e-6
        return (ch - src_mean) / src_std * t_std + t_mean

    lab_norm = cv2.merge([
        np.clip(norm_channel(l, target_mean[0], target_std[0]), 0, 255),
        np.clip(norm_channel(a, target_mean[1], target_std[1]), 0, 255),
        np.clip(norm_channel(b, target_mean[2], target_std[2]), 0, 255),
    ]).astype(np.uint8)

    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

def process_single_image(args):
    src_path, dest_path = args
    if os.path.exists(dest_path):
        return True
    try:
        image = cv2.imread(src_path)
        if image is None:
            return False
        image = remove_hair(image)
        image = reinhard_normalise(image)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        cv2.imwrite(dest_path, image)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    print("Collecting images for preprocessing...")
    all_tasks = []
    
    # Read unique source paths
    for csv_path in INPUT_CSVS:
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            src_path = row['full_path']
            # Make output path relative to preprocessed folder
            rel_path = row['image_path']
            dest_path = os.path.join(OUTPUT_DIR, rel_path)
            all_tasks.append((src_path, dest_path))

    # Remove duplicates from tasks
    unique_tasks = list(set(all_tasks))
    print(f"Total unique images to process: {len(unique_tasks)}")

    # Run multiprocessing pool
    num_workers = max(1, cpu_count() - 1)
    print(f"Starting preprocessing with {num_workers} processes...")
    
    success_count = 0
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, unique_tasks), total=len(unique_tasks)):
            if result:
                success_count += 1

    print(f"Preprocessing completed. Successfully processed {success_count}/{len(unique_tasks)} images.")

if __name__ == '__main__':
    main()
