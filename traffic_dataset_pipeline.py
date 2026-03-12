import os
import sys
import subprocess
import requests
import shutil
import zipfile
import tarfile
import pathlib
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import yaml
import hashlib
from sklearn.model_selection import train_test_split
import glob

# Constants
BASE_DIR = "traffic_ai_dataset"
KAGGLE_DATASETS = [
    ("prateekbangia/indian-traffic-dataset", "images/indian_traffic"),
    ("abdallahwagih/emergency-vehicles", "images/emergency_vehicles"),
    ("sshikamaru/vehicle-detection-image-set", "images/vehicle_detection"),
    ("dtrnngc/ua-detrac-dataset", "images/ua_detrac"),
    ("arashnic/vehicle-dataset-from-open-images", "images/open_images")
]

GITHUB_REPOS = [
    ("https://github.com/iisc-aim/uvh26", "research_datasets/uvh26"),
    ("https://github.com/Abhijnan-Maji/JATAYUv1.0-vehicle-detection-UAV-Dataset", "research_datasets/jatayu"),
    ("https://github.com/SKKUAutoLab/TSBOW", "research_datasets/tsbow")
]

RESEARCH_DOWNLOADS = [
    ("http://detrac-db.rit.albany.edu/Data/DETRAC-images.zip", "raw_downloads"),
    ("https://bdd-data.berkeley.edu/bdd100k_images_100k.zip", "raw_downloads"),
    ("https://bdd-data.berkeley.edu/bdd100k_labels_release.zip", "raw_downloads")
]

def verify_requirements():
    print("STEP 1: Verifying System Requirements")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ required.")
        sys.exit(1)
    print("✓ Python 3.10+ OK")
    
    # Check git
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        print("✓ Git OK")
    except:
        print("Error: Git not installed.")
        sys.exit(1)
    
    # Check kaggle CLI
    try:
        subprocess.run([sys.executable, '-m', 'kaggle', '--version'], check=True, capture_output=True)
        print("✓ Kaggle CLI OK")
    except:
        print("Error: Kaggle CLI not installed.")
        sys.exit(1)
    
    # Check Kaggle API key
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        print("Error: Kaggle API key not found.")
        print("Please place your kaggle.json in ~/.kaggle/")
        print("You can download it from https://www.kaggle.com/account")
        sys.exit(1)
    print("✓ Kaggle API key OK")

def create_directory_structure():
    print("STEP 2: Creating Dataset Directory Structure")
    dirs = [
        "images/indian_traffic",
        "images/vehicle_detection",
        "images/ua_detrac",
        "images/emergency_vehicles",
        "videos/intersection_videos",
        "videos/highway_videos",
        "congestion/traffic_flow",
        "research_datasets/uvh26",
        "research_datasets/jatayu",
        "research_datasets/tsbow",
        "raw_downloads",
        "processed/yolo_format",
        "processed/train",
        "processed/val"
    ]
    for d in dirs:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    print("✓ Directory structure created")

def download_kaggle_datasets():
    print("STEP 3: Downloading Kaggle Datasets")
    commands = [
        [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'prateekbangia/indian-traffic-dataset', '-p', 'traffic_ai_dataset/images/indian_traffic', '--unzip'],
        [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'abdallahwagih/emergency-vehicles', '-p', 'traffic_ai_dataset/images/emergency_vehicles', '--unzip'],
        [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'sshikamaru/vehicle-detection-image-set', '-p', 'traffic_ai_dataset/images/vehicle_detection', '--unzip'],
        [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'dtrnngc/ua-detrac-dataset', '-p', 'traffic_ai_dataset/images/ua_detrac', '--unzip'],
        [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'arashnic/vehicle-dataset-from-open-images', '-p', 'traffic_ai_dataset/images/open_images', '--unzip']
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded {cmd[6]}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {cmd[6]}: {e}")

def clone_github_repos():
    print("STEP 4: Cloning GitHub Repositories")
    commands = [
        ['git', 'clone', 'https://github.com/iisc-aim/uvh26', 'traffic_ai_dataset/research_datasets/uvh26'],
        ['git', 'clone', 'https://github.com/Abhijnan-Maji/JATAYUv1.0-vehicle-detection-UAV-Dataset', 'traffic_ai_dataset/research_datasets/jatayu'],
        ['git', 'clone', 'https://github.com/SKKUAutoLab/TSBOW', 'traffic_ai_dataset/research_datasets/tsbow']
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Cloned {cmd[2]}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {cmd[2]}: {e}")

def download_research_datasets():
    print("STEP 5: Downloading Research Datasets")
    downloads = [
        ('http://detrac-db.rit.albany.edu/Data/DETRAC-images.zip', 'raw_downloads/ua_detrac_images.zip'),
        ('https://bdd-data.berkeley.edu/bdd100k_images_100k.zip', 'raw_downloads/bdd100k_images_100k.zip'),
        ('https://bdd-data.berkeley.edu/bdd100k_labels_release.zip', 'raw_downloads/bdd100k_labels_release.zip')
    ]
    for url, path in downloads:
        full_path = os.path.join(BASE_DIR, path)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(full_path, 'wb') as f, tqdm(desc=path, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print(f"✓ Downloaded {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def automatic_extraction():
    print("STEP 6: Automatic Extraction")
    raw_dir = "traffic_ai_dataset/raw_downloads"
    for file in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file)
        if file.endswith('.zip'):
            try:
                subprocess.run(['unzip', file_path, '-d', raw_dir], check=True)
                os.remove(file_path)
                print(f"✓ Extracted {file}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting {file}: {e}")
        elif file.endswith(('.tar', '.tar.gz')):
            try:
                subprocess.run(['tar', '-xvf', file_path, '-C', raw_dir], check=True)
                os.remove(file_path)
                print(f"✓ Extracted {file}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting {file}: {e}")
    print("✓ Extraction completed")

def dataset_cleaning():
    print("STEP 7: Dataset Cleaning")
    # Remove corrupt images
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()
                except:
                    os.remove(file_path)
                    print(f"Removed corrupt image: {file_path}")
    
    # Remove empty folders
    for root, dirs, files in os.walk(BASE_DIR, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty folder: {dir_path}")
    
    # Remove duplicates
    hashes = {}
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in hashes:
                os.remove(file_path)
                print(f"Removed duplicate: {file_path}")
            else:
                hashes[h] = file_path
    print("✓ Cleaning completed")

def convert_to_yolo():
    print("STEP 8: Converting to YOLO Format")
    # This is a placeholder; actual conversion depends on dataset formats
    # For each dataset, parse annotations and convert to YOLO txt files
    # Assuming some standard formats; in practice, need custom parsers
    yolo_dir = os.path.join(BASE_DIR, "processed/yolo_format")
    # Example: for UA-DETRAC or others, convert XML/VOC to YOLO
    # Implement specific conversions here
    print("✓ Conversion to YOLO format completed (placeholder)")

def train_val_split():
    print("STEP 9: Train/Validation Split")
    yolo_dir = os.path.join(BASE_DIR, "processed/yolo_format")
    train_dir = os.path.join(BASE_DIR, "processed/train")
    val_dir = os.path.join(BASE_DIR, "processed/val")
    
    images = glob.glob(os.path.join(yolo_dir, "*.jpg")) + glob.glob(os.path.join(yolo_dir, "*.png"))
    labels = [img.replace('.jpg', '.txt').replace('.png', '.txt') for img in images]
    
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    train_labels = [img.replace('.jpg', '.txt').replace('.png', '.txt') for img in train_imgs]
    val_labels = [img.replace('.jpg', '.txt').replace('.png', '.txt') for img in val_imgs]
    
    for img in train_imgs:
        shutil.copy(img, train_dir)
    for lbl in train_labels:
        if os.path.exists(lbl):
            shutil.copy(lbl, train_dir)
    for img in val_imgs:
        shutil.copy(img, val_dir)
    for lbl in val_labels:
        if os.path.exists(lbl):
            shutil.copy(lbl, val_dir)
    print("✓ Train/val split completed")

def dataset_statistics():
    print("STEP 10: Dataset Statistics")
    total_size = 0
    total_images = 0
    total_videos = 0
    total_annotations = 0
    
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
            elif file.lower().endswith(('.mp4', '.avi')):
                total_videos += 1
            elif file.endswith('.txt') and 'yolo' in root:
                total_annotations += 1
    
    print(f"Total dataset size: {total_size / (1024**3):.2f} GB")
    print(f"Total images: {total_images}")
    print(f"Total videos: {total_videos}")
    print(f"Total annotations: {total_annotations}")
    print("Vehicle classes detected: car, truck, bus, motorcycle, etc. (placeholder)")

def generate_requirements():
    print("STEP 11: Generating requirements.txt")
    reqs = [
        "kaggle",
        "requests",
        "opencv-python",
        "numpy",
        "pillow",
        "tqdm",
        "pyyaml",
        "ultralytics",
        "gdown",
        "wget"
    ]
    with open("requirements.txt", "w") as f:
        for req in reqs:
            f.write(req + "\n")
    print("✓ requirements.txt generated")

def execution_instructions():
    print("STEP 12: Execution Instructions")
    print("Run the following commands:")
    print("chmod +x setup.sh")
    print("./setup.sh")

def main():
    verify_requirements()
    create_directory_structure()
    download_kaggle_datasets()
    clone_github_repos()
    download_research_datasets()
    automatic_extraction()
    dataset_cleaning()
    convert_to_yolo()
    train_val_split()
    dataset_statistics()
    generate_requirements()
    execution_instructions()

if __name__ == "__main__":
    main()
