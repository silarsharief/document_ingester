import os
import requests
from ultralytics import YOLO
from PIL import Image

from src.core.config import settings

MODEL_NAME = "yolov8n-doclaynet.pt"


class VisualAuditor:
    def __init__(self):
        print("👁️ Initializing YOLO Visual Auditor...")
        # Use config path (e.g. /app/yolov8n-doclaynet.pt in Docker) if file exists, else resolve
        self.model_path = self._resolve_model_path()
        self._ensure_model_exists()
        self.model = YOLO(self.model_path)
        # DocLayNet: 4=Picture, 5=Section-header, 6=Caption, 7=Formula, 8=Table
        self.target_classes = [4, 8]

    def _resolve_model_path(self):
        """Prefer config YOLO path, then /app, then cwd."""
        if hasattr(settings, "YOLO_MODEL_PATH") and os.path.isfile(settings.YOLO_MODEL_PATH):
            return settings.YOLO_MODEL_PATH
        for base in ("/app", os.getcwd()):
            path = os.path.join(base, MODEL_NAME)
            if os.path.isfile(path):
                return path
        return MODEL_NAME

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"⬇️ Downloading {self.model_path}...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-doclaynet.pt"
            response = requests.get(url)
            with open(self.model_path, "wb") as f:
                f.write(response.content)

    def audit_page(self, image_path: str):
        """
        Returns a list of detected visual elements with their BBoxes.
        Format: [{'label': 'picture', 'bbox': [x1, y1, x2, y2], 'conf': 0.85}, ...]
        """
        results = self.model(image_path, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.target_classes:
                    label = "table" if cls_id == 8 else "object" # Generic 'object' for figures
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    detections.append({
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })
        return detections