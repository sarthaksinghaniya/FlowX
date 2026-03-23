import os
import shutil
import cv2
import zipfile
import tarfile
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

def recursive_scan():
    print("Recursive scan of raw datasets...")
    image_exts = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
    video_exts = ['mp4', 'avi', 'mov', 'mkv']
    archive_exts = ['zip', 'rar', '7z']
    images = []
    videos = []
    archives = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            ext = file.lower().split('.')[-1]
            if ext in image_exts:
                images.append(os.path.join(root, file))
            elif ext in video_exts:
                videos.append(os.path.join(root, file))
            elif ext in archive_exts:
                archives.append(os.path.join(root, file))
    print(f"Images found: {len(images)}")
    if images:
        print("Examples:", images[:5])
    print(f"Videos found: {len(videos)}")
    if videos:
        print("Examples:", videos[:5])
    print(f"Archives found: {len(archives)}")
    if archives:
        print("Examples:", archives[:5])
    return images, videos, archives

def extract_zips(archives):
    print("Extracting archives...")
    for archive in archives:
        try:
            if archive.endswith('.zip'):
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(archive))
            elif archive.endswith('.tar.gz') or archive.endswith('.tar'):
                with tarfile.open(archive, 'r') as tar_ref:
                    tar_ref.extractall(os.path.dirname(archive))
            # For rar, need rarfile library, skip for now
        except:
            pass

def build_temp_pool(images):
    print("Building temp image pool...")
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    count = 0
    for img in images:
        base, ext = os.path.splitext(os.path.basename(img))
        dest = os.path.join(TEMP_IMAGES_DIR, f"{count:06d}_{base}{ext}")
        shutil.copy(img, dest)
        count += 1

def extract_video_frames(videos):
    print("Extracting video frames...")
    frame_count = 0
    for video in videos:
        cap = cv2.VideoCapture(video)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 10 == 0:
                frame = cv2.resize(frame, (640, 640))
                frame_path = os.path.join(TEMP_IMAGES_DIR, f"video_frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            count += 1
        cap.release()

def verify_pool():
    print("Verifying image pool...")
    images = [f for f in os.listdir(TEMP_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    total = len(images)
    print(f"Total images in temp_images: {total}")
    if total < 100:
        print("Warning: Less than 100 images, may not be sufficient for training.")
    return images

def auto_annotation(images):
    print("Auto-annotating...")
    os.makedirs(PROCESSED_IMAGES_ALL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_LABELS_ALL_DIR, exist_ok=True)
    for img_file in images:
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
    print("Filtering empty detections...")
    for lbl_file in os.listdir(PROCESSED_LABELS_ALL_DIR):
        lbl_path = os.path.join(PROCESSED_LABELS_ALL_DIR, lbl_file)
        img_file = os.path.splitext(lbl_file)[0] + '.jpg'
        img_path = os.path.join(PROCESSED_IMAGES_ALL_DIR, img_file)
        if os.path.getsize(lbl_path) == 0:
            os.remove(lbl_path)
            if os.path.exists(img_path):
                os.remove(img_path)

def dataset_split():
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

def final_report():
    print("Final report:")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(FINAL_IMAGES_DIR, split)
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        print(f"{split} images: {img_count}")

def main():
    images, videos, archives = recursive_scan()
    extract_zips(archives)
    # Re-scan after extract
    images2, videos2, _ = recursive_scan()
    images.extend(images2)
    videos.extend(videos2)
    build_temp_pool(images)
    extract_video_frames(videos)
    images = verify_pool()
    auto_annotation(images)
    filter_empty()
    dataset_split()
    final_report()
    print("Dataset successfully rebuilt from deep scan of raw datasets")

if __name__ == "__main__":
    main()
