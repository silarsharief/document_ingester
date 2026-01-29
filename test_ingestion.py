from src.ingestion.docling_wrapper import DoclingWrapper
from src.ingestion.page_processor import PageProcessor
import os
import requests

# 1. Setup Test Data
pdf_path = "data/test_doc.pdf"
os.makedirs("data", exist_ok=True)

if not os.path.exists(pdf_path):
    print("⬇️ Downloading test PDF...")
    url = "https://arxiv.org/pdf/2310.06825.pdf" # Good technical paper with tables
    response = requests.get(url)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

# 2. Run the Pipeline
try:
    # Step A: Convert
    wrapper = DoclingWrapper()
    result = wrapper.convert_pdf(pdf_path)
    
    # Step B: Process Images
    processor = PageProcessor()
    image_paths = processor.save_page_images(result)
    
    print("\n--- SUCCESS ---")
    print(f"Generated {len(image_paths)} images.")
    print(f"First image at: {image_paths[0]}")
    # Add this to the bottom of test_ingestion.py
    print(result.document.export_to_markdown())

except Exception as e:
    print(f"\n❌ Error: {e}")