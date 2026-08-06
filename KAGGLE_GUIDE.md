# Guide to Running the BCC Project on Kaggle

This guide provides step-by-step instructions to upload, configure, train, and run the BCC Skin Cancer Detection project on Kaggle. Using Kaggle grants access to powerful free GPUs (e.g. NVIDIA T4 or P100) with generous run time.

---

## Step 1: Open Kaggle & Create a Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and log in.
2. Click **Create** -> **New Notebook**.
3. In the notebook editor, open the settings panel on the right:
   * **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   * **Internet on**: Ensure this is switched **ON** (required to install packages and download pre-trained model weights).

---

## Step 2: Upload the Code to Kaggle
You can upload the project code using one of two methods:

### Method A: Direct Upload via Notebook File Upload (Recommended)
1. Click **File** -> **Upload data** in the top menu of the notebook.
2. Drag and drop **`BCCPROJECT_code_only.zip`** (located in your local workspace).
3. Give it a name (e.g., `bccproject-code`) and click **Create**.
4. The uploaded zip will appear in your notebook's input folder under `/kaggle/input/bccproject-code`.

### Method B: Git Clone (If your repository is hosted online)
In the first cell of your Kaggle notebook, run:
```bash
!git clone <YOUR_REPOSITORY_URL> bcc-project
%cd bcc-project
```

---

## Step 3: Set Up and Unzip the Code
If you used **Method A**, copy the zip file from the input folder to the writable `/kaggle/working` directory and unzip it:

```python
import os
import zipfile

# Define paths
zip_path = '/kaggle/input/bccproject-code/BCCPROJECT_code_only.zip'
extract_dir = '/kaggle/working/BCCPROJECT'

# Extract code
if os.path.exists(zip_path):
    print("Extracting project files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete!")
else:
    print(f"Zip file not found at {zip_path}. Please verify the upload path.")

# Change working directory to the project folder
%cd {extract_dir}
```

---

## Step 4: Install Dependencies
Kaggle's environment includes `torch`, `torchvision`, `pandas`, `numpy`, and `scikit-learn`. You only need to install `timm` (PyTorch Image Models) and `albumentations` (for image augmentations):

```python
!pip install timm albumentations flask fastapi uvicorn
```

---

## Step 5: Mount or Add Datasets
The training scripts expect the dataset to be structured in `/kaggle/working/BCCPROJECT/dataset`.
You have two options to load datasets:

### Option A: Use Kaggle's Public Datasets (Easiest)
Search and add public datasets in your notebook by clicking **Add Input** (or **Add Data**) on the right panel:
* Search for `HAM10000` -> Click **Add**
* Search for `ISIC 2019` -> Click **Add**
* Search for `PAD-UFES-20` -> Click **Add**

Then, modify the dataset paths in [scripts/prepare_dataset.py](file:///c:/Users/Hemanth/OneDrive/Desktop/BCCPROJECT/scripts/prepare_dataset.py) to point to `/kaggle/input/...` directories, or create symbolic links to map Kaggle input directories to the expected project folder structure.

### Option B: Upload Your Zipped Dataset
If you have a customized aggregated dataset zip (e.g. `BCCPROJECT_colab.zip` or `dataset.zip`):
1. Click **Add Input** -> **Upload a dataset**.
2. Upload the zipped dataset.
3. Extract it directly to the `/kaggle/working/BCCPROJECT/` directory.

---

## Step 6: Execute the Workflow

### 1. Preprocess the Dataset
Prepare the train/val/test CSV splits and run offline preprocessing:
```python
# 1. Prepares dataset CSV files
!python scripts/prepare_dataset.py

# 2. Performs offline hair removal and color normalization
!python scripts/preprocess_offline.py
```

### 2. Train the Model
Train the lightweight model utilizing the GPU:
```python
!python -u scripts/train_fast.py
```
*(The `-u` flag ensures output is printed to the notebook cell console in real-time).*

### 3. Evaluate the Test Set
Generate confusion matrices, ROC curves, and the final report:
```python
!python scripts/evaluate_test_set.py
```

---

## Step 7: Launching the Web Dashboard (Flask/FastAPI)

Kaggle notebooks do not have built-in web browser proxies like Google Colab. To preview the interactive web dashboard ([app.py](file:///c:/Users/Hemanth/OneDrive/Desktop/BCCPROJECT/app.py)), you can tunnel the local port `5000` using a service like **Pinggy** or **Localtunnel**.

Run the following cell to start the server and obtain a public URL:

```python
import subprocess
import time

# 1. Start the Flask application in the background
flask_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)  # Wait for Flask to start

# 2. Create a public tunnel using Pinggy (SSH tunnel, no account needed)
print("Creating public web tunnel...")
tunnel_process = subprocess.Popen(
    ["ssh", "-p", "443", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:5000", "a.pinggy.io"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# 3. Read output to fetch the public URL
for _ in range(20):
    line = tunnel_process.stdout.readline()
    if "http://" in line or "https://" in line:
        print(f"\n👉 Dashboard URL: {line.strip()}\n")
        break
    time.sleep(0.5)
```
*(Click the generated link to open the dashboard. Keep this notebook cell running to keep the server alive).*

---

## Step 8: Download Checkpoints and Reports
To download your trained weights and evaluation plots:
```python
from IPython.display import FileLink

# Generate download links
display(FileLink('outputs/best_model.pth'))
display(FileLink('outputs/final_report.md'))
display(FileLink('outputs/roc_curve.png'))
display(FileLink('outputs/confusion_matrix.png'))
```
