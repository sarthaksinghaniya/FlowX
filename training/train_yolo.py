import os
from ultralytics import YOLO

def main():
    # Load model
    model = YOLO('yolov8n.pt')

    # Train
    results = model.train(
        data='datasets/dataset.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        name='traffic_detector'
    )

    # Save best model
    if os.path.exists('runs/detect/traffic_detector/weights/best.pt'):
        os.makedirs('models', exist_ok=True)
        os.rename('runs/detect/traffic_detector/weights/best.pt', 'models/traffic_detector.pt')

if __name__ == "__main__":
    main()
