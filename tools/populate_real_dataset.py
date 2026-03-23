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

def verify_raw():
    print("Verifying raw data...")
    image_count = 0
    video_count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
            elif file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_count += 1
    print(f"Images: {image_count}, Videos: {video_count}")
    return image_count, video_count

def extract_frames():
    print("Extracting frames from videos...")
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    frame_count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                cap = cv2.VideoCapture(video_path)
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % 10 == 0:  # every 10 frames
                        frame = cv2.resize(frame, (640, 640))
                        frame_path = os.path.join(TEMP_IMAGES_DIR, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_count += 1
                    count += 1
                cap.release()

def copy_raw_images():
    print("Copying raw images...")
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(root, file), TEMP_IMAGES_DIR)

def remove_corrupt():
    print("Removing corrupt files...")
    for file in os.listdir(TEMP_IMAGES_DIR):
        file_path = os.path.join(TEMP_IMAGES_DIR, file)
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            continue
        try:
            img = cv2.imread(file_path)
            if img is None:
                os.remove(file_path)
        except:
            os.remove(file_path)

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

def filter_empty():
    print("Filtering empty images...")
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

def verify_counts():
    print("Verifying counts...")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        lbl_dir = os.path.join(FINAL_LABELS_DIR, split)
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        print(f"{split}: {img_count} images, {lbl_count} labels")
        # Check matching
        for img in os.listdir(img_dir):
            if img.lower().endswith('.jpg'):
                lbl = os.path.splitext(img)[0] + '.txt'
                if not os.path.exists(os.path.join(lbl_dir, lbl)):
                    print(f"Missing label for {img}")

def ready_training():
    print("Ready for training...")

def main():
    verify_raw()
    extract_frames()
    copy_raw_images()
    remove_corrupt()
    auto_annotate()
    filter_empty()
    split_dataset()
    verify_counts()
    ready_training()
    print("Dataset populated successfully with images and annotations ready for YOLO training")

if __name__ == "__main__":
    main()
