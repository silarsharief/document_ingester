from PIL import Image, ImageDraw
from src.agents.auditor import auditor
import os

# 1. Settings
image_path = "data/processed_images/page_1.png"
output_path = "data/debug_page_1_visualized.png"

if not os.path.exists(image_path):
    print(f"❌ Error: Run 'test_ingestion.py' first to generate {image_path}")
    exit()

# 2. Run the Auditor (YOLO)
print(f"👁️  Auditing {image_path}...")
elements = auditor.audit_page(image_path, 1)

if not elements:
    print("⚠️  No elements found. (Is the model loaded correctly?)")
    exit()

# 3. Draw the Boxes
original_img = Image.open(image_path)
draw = ImageDraw.Draw(original_img)

print(f"🎨 Drawing {len(elements)} boxes...")

for elem in elements:
    # Get coordinates
    x1, y1, x2, y2 = elem.bbox.x1, elem.bbox.y1, elem.bbox.x2, elem.bbox.y2
    
    # Choose color: Red for Table, Blue for Figure
    color = "red" if elem.type == "table" else "blue"
    
    # Draw thick box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
    
    # Draw label text (optional, small hack to draw text background)
    draw.text((x1 + 5, y1 + 5), f"{elem.type.upper()} ({elem.confidence:.2f})", fill=color)

# 4. Save and Show
original_img.save(output_path)
print(f"✅ Saved visualization to: {output_path}")
print("👉 Open that file to see the bounding boxes!")