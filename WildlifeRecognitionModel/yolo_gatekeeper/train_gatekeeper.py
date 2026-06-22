## train_gatekeeper.py
from ultralytics import YOLO

def train_custom_gatekeeper():
    # Load the lightweight nano model to maintain pipeline speed
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data="/home/greatgilbertsoco/WolfDetect/data/yolo_gatekeeper/dataset.yaml",
        epochs=30,           # 20-30 epochs is usually sufficient for a binary canid switch
        imgsz=640,           # Native YOLO resolution
        batch=16,            # Adjust depending on your GPU memory
        device=0,            # CUDA device index
        workers=4,
        name="canid_gatekeeper"
    )

if __name__ == "__main__":
    train_custom_gatekeeper()