import os
import sys
import subprocess
import shutil
import zipfile
import json
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
import requests

# Constants
BASE_DIR = "datasets"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
YOLO_DIR = os.path.join(BASE_DIR, "yolo_format")
SCRIPTS_DIR = "scripts"
MODELS_DIR = "models"
NOTEBOOKS_DIR = "notebooks"
CONFIGS_DIR = "configs"

KAGGLE_DATASETS = [
    "piterfm/traffic-detection-project-dataset",
    "alincijov/self-driving-cars",
    "abdallahwagih/vehicle-detection-image-dataset",
    "sshikamaru/car-object-detection",
    "andrewmvd/road-traffic-images"
]

GITHUB_REPOS = [
    ("https://github.com/detectRecog/CCPD", "ua_detrac"),
    ("https://github.com/cityflow-project/CityFlow", "cityflow")
]

CLASS_NAMES = ["car", "bus", "truck", "motorcycle", "bicycle"]
CLASS_DICT = {name: idx for idx, name in enumerate(CLASS_NAMES)}

kaggle_exe = "C:/Users/LOQ/AppData/Roaming/Python/Python311/Scripts/kaggle.exe"

def create_folders():
    print("Creating folder structure...")
    dirs = [
        DOWNLOAD_DIR,
        RAW_DIR,
        os.path.join(RAW_DIR, "images"),
        os.path.join(RAW_DIR, "labels"),
        PROCESSED_DIR,
        YOLO_DIR,
        SCRIPTS_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        CONFIGS_DIR,
        os.path.join(YOLO_DIR, "train", "images"),
        os.path.join(YOLO_DIR, "train", "labels"),
        os.path.join(YOLO_DIR, "val", "images"),
        os.path.join(YOLO_DIR, "val", "labels"),
        os.path.join(YOLO_DIR, "test", "images"),
        os.path.join(YOLO_DIR, "test", "labels")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Folders created")

def download_kaggle():
    print("Downloading Kaggle datasets...")
    for dataset in KAGGLE_DATASETS:
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', DOWNLOAD_DIR, '--unzip']
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded {dataset}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                cmd[0] = kaggle_exe
                subprocess.run(cmd, check=True)
                print(f"✓ Downloaded {dataset} (using fallback path)")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {dataset}: {e}")

def clone_github():
    print("Cloning GitHub repositories...")
    for url, name in GITHUB_REPOS:
        dest = os.path.join(RAW_DIR, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        cmd = ['git', 'clone', url, dest]
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Cloned {url}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {url}: {e}")

def extract_and_organize():
    print("Extracting and organizing datasets...")
    # Extract zips in downloads
    for file in os.listdir(DOWNLOAD_DIR):
        file_path = os.path.join(DOWNLOAD_DIR, file)
        if file.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_DIR)
            os.remove(file_path)

    # Move images and labels to raw
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.move(os.path.join(root, file), os.path.join(RAW_DIR, "images", file))
            elif file.lower().endswith(('.xml', '.json')):
                shutil.move(os.path.join(root, file), os.path.join(RAW_DIR, "labels", file))
    print("✓ Extraction and organization completed")

def convert_to_yolo():
    print("Converting to YOLO format...")
    images_dir = os.path.join(RAW_DIR, "images")
    labels_dir = os.path.join(RAW_DIR, "labels")
    yolo_images = os.path.join(YOLO_DIR, "images")
    yolo_labels = os.path.join(YOLO_DIR, "labels")
    os.makedirs(yolo_images, exist_ok=True)
    os.makedirs(yolo_labels, exist_ok=True)

    # Copy images
    for img in glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")):
        shutil.copy(img, yolo_images)

    # Convert labels
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if label_file.endswith('.json'):
            convert_coco(label_path, yolo_labels, yolo_images)
        elif label_file.endswith('.xml'):
            convert_voc(label_path, yolo_labels, yolo_images)
    print("✓ Conversion completed")

def convert_coco(json_path, out_labels, out_images):
    with open(json_path) as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    for ann in annotations:
        img_id = ann['image_id']
        img = images[img_id]
        img_file = img['file_name']
        h, w = img['height'], img['width']
        bbox = ann['bbox']  # x, y, w, h
        category_id = ann['category_id']
        if category_id in CLASS_DICT.values():
            x_center = (bbox[0] + bbox[2]/2) / w
            y_center = (bbox[1] + bbox[3]/2) / h
            width = bbox[2] / w
            height = bbox[3] / h
            label = f"{category_id} {x_center} {y_center} {width} {height}\n"
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(out_labels, label_file), 'a') as f:
                f.write(label)

def convert_voc(xml_path, out_labels, out_images):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_file = root.find('filename').text
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in CLASS_DICT:
            class_id = CLASS_DICT[name]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    if labels:
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        with open(os.path.join(out_labels, label_file), 'w') as f:
            f.write('\n'.join(labels))

def split_dataset():
    print("Splitting dataset...")
    images = glob.glob(os.path.join(YOLO_DIR, "images", "*.jpg")) + glob.glob(os.path.join(YOLO_DIR, "images", "*.png"))
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split, imgs in splits.items():
        for img in imgs:
            shutil.move(img, os.path.join(YOLO_DIR, split, "images", os.path.basename(img)))
            label = img.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(label):
                shutil.move(label, os.path.join(YOLO_DIR, split, "labels", os.path.basename(label)))
    print("✓ Split completed")

def generate_config():
    print("Generating config file...")
    config = {
        'path': '../datasets/yolo_format',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(os.path.join(CONFIGS_DIR, "traffic_dataset.yaml"), 'w') as f:
        yaml.dump(config, f)
    print("✓ Config generated")

def print_stats():
    print("Dataset Statistics:")
    total_images = 0
    total_annotations = 0
    classes = set()
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(YOLO_DIR, split, "images")
        lbl_dir = os.path.join(YOLO_DIR, split, "labels")
        imgs = len(glob.glob(os.path.join(img_dir, "*.jpg"))) + len(glob.glob(os.path.join(img_dir, "*.png")))
        total_images += imgs
        for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.add(int(parts[0]))
                        total_annotations += 1
        print(f"{split}: {imgs} images")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Classes: {sorted(classes)}")

def main():
    create_folders()
    download_kaggle()
    clone_github()
    extract_and_organize()
    convert_to_yolo()
    split_dataset()
    generate_config()
    print_stats()

if __name__ == "__main__":
    main()
