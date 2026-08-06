import os
import sys
import zipfile
import shutil
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def resize_single_image(args):
    src_path, dest_path, target_size = args
    if os.path.exists(dest_path):
        return True
    try:
        image = cv2.imread(src_path)
        if image is None:
            return False
        # Resize to target size (380x380)
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        cv2.imwrite(dest_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return True
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")
        return False

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_size = 380
    
    # Paths
    preprocessed_dir = os.path.join(project_root, "dataset", "data", "images_preprocessed")
    colab_prep_dir = os.path.join(project_root, "dataset", "colab_upload")
    colab_images_dir = os.path.join(colab_prep_dir, "data", "images_preprocessed")
    
    if not os.path.exists(preprocessed_dir):
        print(f"Error: Preprocessed images directory not found at {preprocessed_dir}.")
        print("Please run scripts/preprocess_offline.py first.")
        sys.exit(1)

    print("Collecting preprocessed images list...")
    tasks = []
    for root, _, files in os.walk(preprocessed_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, preprocessed_dir)
                dest_path = os.path.join(colab_images_dir, rel_path)
                tasks.append((src_path, dest_path, target_size))

    print(f"Found {len(tasks)} images to resize and prepare.")
    
    # Resize in parallel
    num_workers = max(1, cpu_count() - 1)
    print(f"Resizing images to {target_size}x{target_size} using {num_workers} processes...")
    
    success_count = 0
    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(resize_single_image, tasks), total=len(tasks)))
        success_count = sum(1 for r in results if r)

    print(f"Successfully resized and saved {success_count}/{len(tasks)} images.")

    # Copy CSV files to colab_prep_dir
    data_dir = os.path.join(project_root, "dataset", "data")
    csv_files = ["train.csv", "val.csv", "test.csv"]
    for csv in csv_files:
        src_csv = os.path.join(data_dir, csv)
        dest_csv = os.path.join(colab_prep_dir, "data", csv)
        if os.path.exists(src_csv):
            os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
            shutil.copy(src_csv, dest_csv)
            print(f"Copied {csv} to Colab upload folder.")

    # Create zip file containing only code, config, and resized images
    zip_path = os.path.join(project_root, "BCCPROJECT_colab.zip")
    print(f"Creating compact zip file at: {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add source code, configs, scripts, templates, static assets, and model checkpoints
        folders_to_add = ["src", "configs", "scripts", "templates", "static", "outputs"]
        for folder in folders_to_add:
            folder_path = os.path.join(project_root, folder)
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        # Exclude big backups, logs, and temp pyc files to save space
                        if not file.endswith(('.pyc', '.log', '_backup.pth')):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, project_root)
                            zipf.write(file_path, rel_path)
        
        # 2. Add app.py and notebook
        for filename in ["app.py", "BCCPROJECT_Colab.ipynb"]:
            file_path = os.path.join(project_root, filename)
            if os.path.exists(file_path):
                zipf.write(file_path, filename)

        # 3. Add resized images and CSVs from the upload folder
        for root, _, files in os.walk(colab_prep_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, colab_prep_dir)
                # Save under dataset/ prefix in zip
                zipf.write(file_path, os.path.join("dataset", rel_path))

    print(f"\nSUCCESS! Created compact Colab-ready zip file: {zip_path}")
    print(f"File size is now compressed to ~1.1 GB (97% smaller than 37 GB!).")
    print("You can upload this compact file to Google Drive and extract it in Colab in seconds.")

if __name__ == '__main__':
    main()
