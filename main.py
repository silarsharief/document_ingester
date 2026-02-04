"""
QuickSight AI - Root Entry Point

This is a convenience wrapper that imports and runs the main pipeline.
You can also run directly: python -m src.main
"""
import os
from src.main import AnalysisPipeline

if __name__ == "__main__":
    # Default test document path
    pdf_path = "data/test_doc.pdf"
    
    if os.path.exists(pdf_path):
        AnalysisPipeline().run(pdf_path, scanned_mode=False)
    else:
        print(f"❌ PDF not found at {pdf_path}")
        print("👉 Please provide a PDF file at data/test_doc.pdf")
