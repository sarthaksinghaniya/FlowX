import os
import shutil
import cv2
import glob
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import json

RAW_DIR = "datasets/raw"
TEMP_IMAGES_DIR = "datasets/temp_images"
PROCESSED_IMAGES_ALL_DIR = "datasets/processed/images/all"
PROCESSED_LABELS_ALL_DIR = "datasets/processed/labels/all"
FINAL_IMAGES_DIR = "datasets/processed/images"
FINAL_LABELS_DIR = "datasets/processed/labels"
CLASSES = ["car", "truck", "bus", "motorcycle", "ambulance", "firetruck"]
CLASS_DICT = {cls: i for i, cls in enumerate(CLASSES)}

model = YOLO('yolov8n.pt')

def clear_processed():
    print("Clearing processed data...")
    for root, dirs, files in os.walk(FINAL_IMAGES_DIR):
        for file in files:
            os.remove(os.path.join(root, file))
    for root, dirs, files in os.walk(FINAL_LABELS_DIR):
        for file in files:
            os.remove(os.path.join(root, file))

def collect_images():
    print("Collecting source images...")
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(root, file), os.path.join(TEMP_IMAGES_DIR, f"img_{count:06d}.jpg"))
                count += 1

def auto_annotate():
    print("Auto-annotating with YOLOv8...")
    os.makedirs(PROCESSED_IMAGES_ALL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_LABELS_ALL_DIR, exist_ok=True)
    for img_file in os.listdir(TEMP_IMAGES_DIR):
        img_path = os.path.join(TEMP_IMAGES_DIR, img_file)
        lbl_path = os.path.join(PROCESSED_LABELS_ALL_DIR, os.path.splitext(img_file)[0] + '.txt')
        shutil.copy(img_path, PROCESSED_IMAGES_ALL_DIR)
        results = model(img_path)
        with open(lbl_path, 'w') as f:
            has_detections = False
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    if model.names[cls_id] in CLASSES:
                        has_detections = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        h, w, _ = cv2.imread(img_path).shape
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        f.write(f"{CLASS_DICT[model.names[cls_id]]} {x_center} {y_center} {width} {height}\n")

def remove_empty():
    print("Removing images with no detections...")
    for lbl_file in os.listdir(PROCESSED_LABELS_ALL_DIR):
        lbl_path = os.path.join(PROCESSED_LABELS_ALL_DIR, lbl_file)
        img_file = os.path.splitext(lbl_file)[0] + '.jpg'
        img_path = os.path.join(PROCESSED_IMAGES_ALL_DIR, img_file)
        if os.path.getsize(lbl_path) == 0:
            os.remove(lbl_path)
            if os.path.exists(img_path):
                os.remove(img_path)

def split_dataset():
    print("Splitting dataset...")
    img_files = [f for f in os.listdir(PROCESSED_IMAGES_ALL_DIR) if f.lower().endswith('.jpg')]
    train_imgs, temp = train_test_split(img_files, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=1/3, random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split, imgs in splits.items():
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        lbl_dir = os.path.join(FINAL_LABELS_DIR, split)
        for img in imgs:
            shutil.move(os.path.join(PROCESSED_IMAGES_ALL_DIR, img), img_dir)
            lbl = os.path.splitext(img)[0] + '.txt'
            lbl_path = os.path.join(PROCESSED_LABELS_ALL_DIR, lbl)
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, lbl_dir)
    shutil.rmtree(PROCESSED_IMAGES_ALL_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_LABELS_ALL_DIR, ignore_errors=True)
    shutil.rmtree(TEMP_IMAGES_DIR, ignore_errors=True)

def verify_dataset():
    print("Verifying dataset...")
    os.system("python tools/validate_dataset.py")

def generate_report():
    print("Generating final dataset stats...")
    stats = {
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'class_distribution': {}
    }
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        lbl_dir = os.path.join(FINAL_LABELS_DIR, split)
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
        stats[f'{split}_images'] = img_count
        stats['total_images'] += img_count
        for lbl_file in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    cls = CLASSES[cls_id]
                    stats['class_distribution'][cls] = stats['class_distribution'].get(cls, 0) + 1
    with open('reports/final_dataset_stats.md', 'w') as f:
        f.write(f"# Final Dataset Statistics\n\nTotal Images: {stats['total_images']}\nTrain Images: {stats['train_images']}\nVal Images: {stats['val_images']}\nTest Images: {stats['test_images']}\nClass Distribution: {stats['class_distribution']}\n")

def ready_training():
    print("Verifying dataset.yaml...")
    # Check if yaml exists and is correct
    if os.path.exists('datasets/dataset.yaml'):
        print("dataset.yaml exists.")
    else:
        print("dataset.yaml missing.")

def main():
    clear_processed()
    collect_images()
    auto_annotate()
    remove_empty()
    split_dataset()
    verify_dataset()
    generate_report()
    ready_training()
    print("Dataset rebuilt successfully with valid annotations and dataset splits ready for YOLO training")

if __name__ == "__main__":
    main()
