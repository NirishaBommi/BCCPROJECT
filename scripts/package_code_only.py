import os
import zipfile

def package_code_only():
    project_root = r"c:\Users\Hemanth\OneDrive\Desktop\BCCPROJECT"
    zip_path = os.path.join(project_root, "BCCPROJECT_code_only.zip")
    
    print(f"Creating compact code-only zip file at: {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Folders to add
        folders_to_add = ["models", "explainability", "datasets", "api", "utils", "configs", "dashboard", "outputs", "src", "scripts"]
        
        for folder in folders_to_add:
            folder_path = os.path.join(project_root, folder)
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        # Exclude big backups, logs, temp pyc files, and dataset images
                        if file.endswith(('.pyc', '.log', '_backup.pth')):
                            continue
                        # Ensure we do not pack any raw or preprocessed dataset files
                        file_path = os.path.join(root, file)
                        if "dataset" in file_path.lower() or "demo_data" in file_path.lower():
                            continue
                        
                        rel_path = os.path.relpath(file_path, project_root)
                        zipf.write(file_path, rel_path)
        
        # Add root script files
        for filename in ["app.py", "app_fastapi.py", "BCCPROJECT_Colab.ipynb", "BCCPROJECT_Kaggle.ipynb", "Dockerfile", "requirements.txt"]:
            file_path = os.path.join(project_root, filename)
            if os.path.exists(file_path):
                zipf.write(file_path, filename)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSUCCESS! Created code-only zip file: {zip_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    package_code_only()
