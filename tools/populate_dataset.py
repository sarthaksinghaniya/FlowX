import os
import shutil
import cv2
import glob
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import json

# Constants
RAW_DIR = "datasets/raw"
PROCESSED_IMAGES_DIR = "datasets/processed/images/train"
PROCESSED_LABELS_DIR = "datasets/processed/labels/train"
FINAL_IMAGES_DIR = "datasets/processed/images"
FINAL_LABELS_DIR = "datasets/processed/labels"
CLASSES = ["car", "truck", "bus", "motorcycle", "ambulance", "firetruck"]
CLASS_DICT = {cls: i for i, cls in enumerate(CLASSES)}

model = YOLO('yolov8n.pt')

def scan_raw():
    print("Scanning raw data...")
    images = []
    videos = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(root, file))
            elif file.lower().endswith(('.mp4', '.avi', '.mov')):
                videos.append(os.path.join(root, file))
    return images, videos

def extract_frames(videos):
    print("Extracting frames from videos...")
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    frame_count = 0
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 5 == 0:  # every 5 frames
                frame = cv2.resize(frame, (640, 640))
                frame_path = os.path.join(PROCESSED_IMAGES_DIR, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            count += 1
        cap.release()

def copy_images(images):
    print("Copying raw images...")
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    for img_path in images:
        shutil.copy(img_path, PROCESSED_IMAGES_DIR)

def class_detection():
    print("Detecting classes and generating labels...")
    os.makedirs(PROCESSED_LABELS_DIR, exist_ok=True)
    for img_file in os.listdir(PROCESSED_IMAGES_DIR):
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(PROCESSED_IMAGES_DIR, img_file)
        lbl_path = os.path.join(PROCESSED_LABELS_DIR, os.path.splitext(img_file)[0] + '.txt')
        results = model(img_path)
        with open(lbl_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    if model.names[cls_id] in CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        h, w, _ = cv2.imread(img_path).shape
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        f.write(f"{CLASS_DICT[model.names[cls_id]]} {x_center} {y_center} {width} {height}\n")

def dataset_split():
    print("Splitting dataset...")
    img_files = [f for f in os.listdir(PROCESSED_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    train_imgs, temp = train_test_split(img_files, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=1/3, random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split, imgs in splits.items():
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        lbl_dir = os.path.join(FINAL_LABELS_DIR, split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img in imgs:
            shutil.move(os.path.join(PROCESSED_IMAGES_DIR, img), img_dir)
            lbl = os.path.splitext(img)[0] + '.txt'
            lbl_path = os.path.join(PROCESSED_LABELS_DIR, lbl)
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, lbl_dir)
    # Remove empty train dirs
    shutil.rmtree(PROCESSED_IMAGES_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_LABELS_DIR, ignore_errors=True)

def generate_stats():
    print("Generating dataset stats...")
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'class_distribution': {},
        'train_val_test_counts': {}
    }
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        lbl_dir = os.path.join(FINAL_LABELS_DIR, split)
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        stats['total_images'] += img_count
        stats['total_labels'] += lbl_count
        stats['train_val_test_counts'][split] = img_count
        # Class dist from labels
        for lbl_file in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    cls = CLASSES[cls_id]
                    stats['class_distribution'][cls] = stats['class_distribution'].get(cls, 0) + 1
    with open('reports/dataset_stats.md', 'w') as f:
        f.write(f"# Dataset Statistics\n\nTotal Images: {stats['total_images']}\nTotal Labels: {stats['total_labels']}\nClass Distribution: {stats['class_distribution']}\nTrain/Val/Test Counts: {stats['train_val_test_counts']}\n")

def verify_dataset():
    print("Verifying dataset...")
    os.system("python tools/validate_dataset.py")

def main():
    images, videos = scan_raw()
    extract_frames(videos)
    copy_images(images)
    class_detection()
    dataset_split()
    generate_stats()
    verify_dataset()
    print("Training dataset successfully generated and ready for YOLO training")

if __name__ == "__main__":
    main()
