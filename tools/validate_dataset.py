import os
import json
from ultralytics import YOLO

def validate_dataset():
    base_dir = "datasets/processed"
    splits = ['train', 'val', 'test']
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'class_distribution': {},
        'missing_labels': [],
        'invalid_annotations': []
    }

    for split in splits:
        img_dir = os.path.join(base_dir, 'images', split)
        lbl_dir = os.path.join(base_dir, 'labels', split)
        
        if not os.path.exists(img_dir):
            continue
        
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(img_dir, img_file)
            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            stats['total_images'] += 1
            
            if not os.path.exists(lbl_path):
                stats['missing_labels'].append(img_file)
                continue
            
            stats['total_labels'] += 1
            
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        stats['invalid_annotations'].append(lbl_file)
                        continue
                    try:
                        cls_id, x, y, w, h = map(float, parts)
                        if not (0 <= cls_id < 6 and all(0 <= v <= 1 for v in [x, y, w, h])):
                            stats['invalid_annotations'].append(lbl_file)
                            continue
                        cls_name = ['car', 'truck', 'bus', 'bike', 'ambulance', 'firetruck'][int(cls_id)]
                        stats['class_distribution'][cls_name] = stats['class_distribution'].get(cls_name, 0) + 1
                    except:
                        stats['invalid_annotations'].append(lbl_file)
    
    print("Dataset Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Labels: {stats['total_labels']}")
    print(f"Class Distribution: {stats['class_distribution']}")
    print(f"Missing Labels: {len(stats['missing_labels'])}")
    print(f"Invalid Annotations: {len(stats['invalid_annotations'])}")
    
    return stats

if __name__ == "__main__":
    validate_dataset()
