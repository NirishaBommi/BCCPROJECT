# Guide to Running the BCC Project on Google Colab

Follow these steps to upload and run your project on Google Colab using a free or paid cloud GPU (e.g. T4, L4, or A100).

---

## Step 1: Zip the Project Folder
On your local computer, zip the entire `BCCPROJECT` folder (excluding the `.venv` folder to keep the size small).
* Name the zip file: `BCCPROJECT.zip`

---

## Step 2: Upload to Google Drive
Upload `BCCPROJECT.zip` and your dataset directory (if it's not already zipped inside) to your Google Drive. 
* Place them in a folder named `BCC_Skin_Cancer` in your Drive.

---

## Step 3: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Change the runtime type to GPU:
   * Click **Runtime** -> **Change runtime type** -> Select **T4 GPU** (or any available GPU) -> Click **Save**.

---

## Step 4: Run the Setup Cells
Copy and paste the following code blocks into cells in your Google Colab Notebook and run them in order.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Copy and Unzip Project
```python
# Create a folder in Colab
!mkdir -p /content/BCCPROJECT

# Copy the zip from Drive and extract it
!cp /content/drive/MyDrive/BCC_Skin_Cancer/BCCPROJECT.zip /content/
!unzip -q /content/BCCPROJECT.zip -d /content/BCCPROJECT

# Navigate into the project folder
%cd /content/BCCPROJECT
```

### Cell 3: Install Required Dependencies
Colab already has `torch`, `torchvision`, `pandas`, `numpy`, and `opencv` pre-installed. You only need to install `timm` and `albumentations`:
```python
!pip install timm albumentations
```

---

## Step 5: Run the Project Steps

### 1. Preprocess the Dataset Offline (Very Fast)
```python
!python scripts/preprocess_offline.py
```

### 2. Train the Model
To train the fast lightweight model:
```python
!python -u scripts/train_fast.py
```
*(Or to train the original configuration, run `!python -u src/train.py`)*

### 3. Generate Results and Comparison Reports
```python
!python scripts/evaluate_test_set.py
```

---

## Step 6: Download the Best Checkpoint and Reports
Once training is complete, download the saved model weights and reports back to your computer:
```python
from google.colab import files

# Download the model weights
files.download('outputs/best_model.pth')

# Download the reports
files.download('outputs/test_results.txt')
files.download('outputs/final_report.md')
```
