import os
import time
import json
import shutil
from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
from src.ingestion.docling_wrapper import DoclingWrapper
from src.ingestion.visual_auditor import VisualAuditor
from src.agents.vision import vision_agent
from src.core.logger import setup_logger
from src.core.debug_reporter import DebugReporter
from src.core.final_pdf import FinalReportGenerator

log = setup_logger("pipeline")

class AnalysisPipeline:
    def __init__(self):
        self.ingestor = DoclingWrapper()
        self.auditor = VisualAuditor()
        self.debug_reporter = DebugReporter("pipeline_report.pdf")
        self.final_reporter = FinalReportGenerator("full_analysis.pdf")
        
        self.debug_dir = "data/crops"
        if os.path.exists(self.debug_dir): shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def run(self, pdf_path: str, scanned_mode: bool = False):
        start_time = time.time() # <--- Start Timer
        log.info(f"🚀 Starting Pipeline: {pdf_path} (Scanned Mode: {scanned_mode})")
        
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
        except Exception as e:
            log.critical(f"❌ Ingestion Failed: {e}")
            return

        # 1. PAGE BUCKETS: { 1: [], 2: [] }
        # We group everything by page first to prevent image drift
        pages_content = {} 
        yolo_cache = {}
        global_counter = 0 

        log.info("\n[bold yellow]🧠 PHASE 2: PROCESSING[/]")

        for item, text_before, text_after in self.ingestor.iterate_items(doc_result):
            global_counter += 1
            current_item = None
            page_no = item.prov[0].page_no
            
            # Initialize bucket
            if page_no not in pages_content:
                pages_content[page_no] = []

            # --- Text & Headers ---
            if isinstance(item, (TextItem, SectionHeaderItem)):
                if len(item.text.strip()) < 3: continue
                bbox = item.prov[0].bbox
                current_item = {
                    "type": "header" if isinstance(item, SectionHeaderItem) else "text", 
                    "content": item.text,
                    "page": page_no,
                    "bbox": [bbox.l, bbox.b, bbox.r, bbox.t],
                    # Store keys for sorting
                    "sort_y": bbox.t,          # Spatial (Top-Down)
                    "arrival_id": global_counter # Logical (Reading Order)
                }

            # --- Tables ---
            elif isinstance(item, TableItem):
                try: table_md = item.export_to_markdown(doc=doc_result.document)
                except: table_md = item.export_to_markdown()
                bbox = item.prov[0].bbox
                current_item = {
                    "type": "table", 
                    "content": table_md,
                    "page": page_no,
                    "bbox": [bbox.l, bbox.b, bbox.r, bbox.t],
                    "sort_y": bbox.t,
                    "arrival_id": global_counter
                }

            # --- Visuals ---
            elif isinstance(item, PictureItem):
                log.info(f"   🖼️  Processing Visual on Page {page_no}")
                
                try:
                    page_obj = doc_result.pages[page_no]
                    page_img = page_obj.image.pil_image if hasattr(page_obj.image, "pil_image") else page_obj.image
                    pg_w, pg_h = page_img.width, page_img.height

                    # 1. YOLO Detection
                    if page_no not in yolo_cache:
                        temp_path = f"data/temp_p{page_no}.png"
                        page_img.save(temp_path)
                        yolo_cache[page_no] = self.auditor.audit_page(temp_path)
                        if os.path.exists(temp_path): os.remove(temp_path)
                    
                    # 2. Coordinate Mapping
                    d_bbox = item.prov[0].bbox
                    # Normalize if needed
                    if d_bbox.l < 1.0: 
                        d_l, d_b, d_r, d_t = d_bbox.l*pg_w, d_bbox.b*pg_h, d_bbox.r*pg_w, d_bbox.t*pg_h
                    else: 
                        d_l, d_b, d_r, d_t = d_bbox.l, d_bbox.b, d_bbox.r, d_bbox.t
                    
                    # Flip Y for Cropping (Image uses Top-Left Origin)
                    d_top_img = pg_h - d_t
                    d_bot_img = pg_h - d_b
                    if d_top_img > d_bot_img: d_top_img, d_bot_img = d_bot_img, d_top_img

                    best_yolo_box = None
                    for y_obj in yolo_cache[page_no]:
                        y_box = y_obj['bbox']
                        x_overlap = max(0, min(d_r, y_box[2]) - max(d_l, y_box[0]))
                        y_overlap = max(0, min(d_bot_img, y_box[3]) - max(d_top_img, y_box[1]))
                        if x_overlap * y_overlap > 100: 
                            best_yolo_box = y_box
                            break
                    
                    base_box = tuple(map(int, best_yolo_box)) if best_yolo_box else (int(d_l), int(d_top_img), int(d_r), int(d_bot_img))

                    # Smart Crop Expansion
                    width = base_box[2] - base_box[0]
                    height = base_box[3] - base_box[1]
                    final_crop = (
                        max(0, base_box[0] - int(width * 0.1)),
                        max(0, base_box[1] - 50),
                        min(pg_w, base_box[2] + int(width * 0.1)),
                        min(pg_h, base_box[3] + max(150, int(height * 0.15)))
                    )

                    # Save Smart Crop (This is the Fixed Image)
                    crop_filename = f"p{page_no}_crop_{global_counter}.png"
                    crop_path = os.path.join(self.debug_dir, crop_filename)
                    page_img.crop(final_crop).save(crop_path)
                    
                    context_str = f"PREVIOUS TEXT:\n...{text_before[-1000:]}\nFOLLOWING TEXT:\n{text_after[:1000]}..."
                    analysis_result = vision_agent.analyze_element(crop_path, "visual", context_str)

                    # Debug report
                    docling_ref = os.path.join(self.debug_dir, f"p{page_no}_ref_{global_counter}.png")
                    page_img.crop((int(d_l), int(d_top_img), int(d_r), int(d_bot_img))).save(docling_ref)
                    self.debug_reporter.add_comparison(page_no, docling_ref, crop_path, str(analysis_result))

                    current_item = {
                        "type": "visual",
                        "bbox": list(final_crop), 
                        "analysis": analysis_result, 
                        "file_path": crop_path, # <--- Saving the YOLO Crop for Full Analysis
                        "page": page_no,
                        "sort_y": d_bbox.t, # Use Raw Docling Y for sorting
                        "arrival_id": global_counter
                    }

                except Exception as e:
                    log.error(f"      ⚠️ Visual Error: {e}")

            if current_item:
                pages_content[page_no].append(current_item)

        # --- 3. PAGE-LEVEL SORTING & FLATTENING ---
        log.info("🔄 Sorting elements per page...")
        final_output = []
        
        sorted_page_nums = sorted(pages_content.keys())
        
        for p_num in sorted_page_nums:
            items = pages_content[p_num]
            
            # CRITICAL FIX: Sort Logic
            if scanned_mode:
                # Scanned: Sort ASCENDING (0 -> 1000). 0 is Top.
                items.sort(key=lambda x: x['sort_y'])
            else:
                # Digital: Trust Arrival Order (Preserves Columns)
                items.sort(key=lambda x: x['arrival_id'])
            
            for item in items:
                item['order_id'] = len(final_output) + 1
                # Clean up temp keys
                del item['sort_y']
                del item['arrival_id']
                final_output.append(item)

        # Finalize
        with open("data/rag_dataset.json", "w") as f:
            json.dump(final_output, f, indent=2)
        
        self.debug_reporter.save()
        self.final_reporter.generate(final_output)
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        log.info(f"✅ Done.")
        log.info(f"⏱️  Total Processing Time: {minutes}m {seconds}s") # <--- Log it
        log.info(f"📄 Full Analysis: [underline]full_analysis.pdf[/]")
        log.info(f"✅ Done.")
        log.info(f"📄 Full Analysis: [underline]full_analysis.pdf[/]")

if __name__ == "__main__":
    if os.path.exists("data/test_doc.pdf"):
        # Digital PDF -> scanned_mode=False
        AnalysisPipeline().run("data/test_doc.pdf", scanned_mode=False)