import os
from PIL import Image
from src.ingestion.docling_wrapper import DoclingWrapper
from src.ingestion.page_processor import PageProcessor
from src.agents.auditor import auditor
from src.agents.vision import vision_agent
from src.agents.validator import validator
from src.agents.fusion import fusion_agent

class AnalysisPipeline:
    def __init__(self):
        self.ingestor = DoclingWrapper()
        self.processor = PageProcessor()
    
    def run(self, pdf_path: str):
        print(f"🚀 Starting Context-Aware RAG Pipeline for: {pdf_path}")
        
        # 1. INGESTION
        print("\n" + "="*50)
        print(" PHASE 1: INGESTION & PARSING ")
        print("="*50)
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
            image_paths = self.processor.save_page_images(doc_result)
        except Exception as e:
            print(f"❌ Ingestion Failed: {e}")
            return

        final_document_markdown = ""

        # 2. PAGE LOOP
        print("\n" + "="*50)
        print(" PHASE 2: INTELLIGENT PROCESSING ")
        print("="*50)
        
        for page_num, img_path in enumerate(image_paths, start=1):
            print(f"\n📄 [PAGE {page_num}] Processing...")
            
            # --- A. Get Page Text ---
            page_text_content = self.ingestor.get_page_text(doc_result, page_num)
            text_len = len(page_text_content)
            print(f"   📝 Extracted {text_len} characters of text.")

            # --- B. Audit (YOLO) ---
            elements = auditor.audit_page(img_path, page_num)
            valid_visuals = []
            
            if elements:
                original_img = Image.open(img_path)
                
                for elem in elements:
                    # --- C. Crop & Analyze ---
                    bbox = (elem.bbox.x1, elem.bbox.y1, elem.bbox.x2, elem.bbox.y2)
                    crop_path = f"data/temp_crop_{elem.id}.png"
                    original_img.crop(bbox).save(crop_path)
                    
                    print(f"   👁️  Analyzing {elem.type.upper()} (Conf: {elem.confidence:.2f})...", end="\r")
                    
                    # Pass text context to Vision Agent
                    analysis = vision_agent.analyze_element(crop_path, elem.type, page_text_content)
                    
                    # --- D. Validate ---
                    check = validator.validate(elem.type, elem.confidence, analysis)
                    
                    if check["is_valid"]:
                        print(f"   ✅ Valid {elem.type.upper()}: {analysis[:60].replace(chr(10), ' ')}...")
                        valid_visuals.append({
                            "type": elem.type,
                            "analysis": analysis
                        })
                    else:
                        print(f"   🗑️  Dropped {elem.type.upper()}: {check['reason']}")
                    
                    if os.path.exists(crop_path): os.remove(crop_path)
            else:
                print("   (No visual elements found)")

            # --- E. Fusion ---
            print("   🔗 Fusing Text & Visuals...")
            fused_page = fusion_agent.fuse_page(page_num, page_text_content, valid_visuals)
            final_document_markdown += fused_page + "\n\n"

        # 3. SAVE FINAL OUTPUT
        output_file = "data/final_rag_document.md"
        with open(output_file, "w") as f:
            f.write(final_document_markdown)
            
        print("\n" + "="*60)
        print(f"✅ PIPELINE COMPLETE. Output saved to: {output_file}")
        print("="*60)

if __name__ == "__main__":
    if os.path.exists("data/test_doc.pdf"):
        AnalysisPipeline().run("data/test_doc.pdf")