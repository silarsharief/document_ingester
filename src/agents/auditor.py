import os
import requests
import certifi
import ssl
from ultralytics import YOLO
from src.core.schemas import DocElement, BoundingBox
from typing import List
import uuid
from PIL import Image, ImageDraw

# 1. SSL Fix for Mac
os.environ['SSL_CERT_FILE'] = certifi.where()

class VisualAuditor:
    def __init__(self):
        print("👁️ Loading Document-Specific YOLO (DocLayNet)...")
        self.model_filename = "yolov8n-doclaynet.pt"
        self.model_url = "https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8n-doclaynet.pt"
        
        self._ensure_model_exists()
        self.model = YOLO(self.model_filename)
        
        # DocLayNet Classes: 6=Picture, 8=Table
        self.target_classes = [6, 8] 

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_filename):
            print(f"⬇️ Downloading {self.model_filename}...")
            try:
                response = requests.get(self.model_url, stream=True, verify=certifi.where())
                response.raise_for_status()
                with open(self.model_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Model downloaded.")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                print("⚠️ Falling back to standard YOLOv8n")
                self.model_filename = "yolov8n.pt"

    def audit_page(self, image_path: str, page_num: int) -> List[DocElement]:
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            return []

        # Run Inference
        try:
            results = self.model.predict(image_path, conf=0.1, verbose=False) # Lowered conf to 0.1 for testing
            if not results:
                return []
            result = results[0]
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return []
        
        elements = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                label_map = {6: "figure", 8: "table"}
                
                element = DocElement(
                    id=f"p{page_num}_{str(uuid.uuid4())[:8]}",
                    type=label_map.get(class_id, "figure"),
                    content="[VISUAL_PLACEHOLDER]", 
                    confidence=confidence,
                    bbox=BoundingBox(page=page_num, x1=x1, y1=y1, x2=x2, y2=y2),
                    # --- THE FIX IS HERE: ---
                    source="yolo_audit"  # <--- Must match schema exactly
                    # ------------------------
                )
                elements.append(element)
                
        print(f"   ↳ DocLayNet Agent found {len(elements)} elements on Page {page_num}")
        return elements

# ... (Keep the VisualAuditor class exactly as it is) ...

auditor = VisualAuditor()

if __name__ == "__main__":
    # --- REAL IMAGE TEST MODE ---
    image_path = "data/test_chart.jpg"
    
    # 1. Check if YOU provided a file
    if not os.path.exists(image_path):
        print(f"⚠️ No file found at {image_path}")
        print("👉 Please drag a REAL chart image into the 'data' folder and name it 'test_chart.jpg'")
        
        # Optional: Try to download one automatically using system curl (more robust than Python)
        print("⬇️ Attempting to download a real sample via curl...")
        os.system(f"curl -L -o {image_path} https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Star_Wars_The_Last_Jedi_global_box_office_receipts_chart.png/640px-Star_Wars_The_Last_Jedi_global_box_office_receipts_chart.png")

    # 2. Run Audit on the file that exists
    if os.path.exists(image_path):
        print(f"🏃‍♂️ Auditing REAL image: {image_path}...")
        results = auditor.audit_page(image_path, 1)
        
        if not results:
            print("❌ No tables/figures detected. (Try a clearer image)")
        else:
            for r in results:
                print(f"✅ Found {r.type.upper()} | Confidence: {r.confidence:.2f}")