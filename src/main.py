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
        
        # Clean debug directory for fresh crops
        self.debug_dir = "data/crops"
        if os.path.exists(self.debug_dir): shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def run(self, pdf_path: str, scanned_mode: bool = False):
        start_time = time.time() 
        log.info(f"🚀 Starting Pipeline: {pdf_path}")
        
        # --- 1. INGESTION ---
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
        except Exception as e:
            log.critical(f"❌ Ingestion Failed: {e}")
            return

        pages_content = {} 
        yolo_cache = {}
        global_counter = 0 

        log.info("\n[bold yellow]🧠 PHASE 2: PROCESSING[/]")

        # Iterate through Docling's structured output
        for item, text_before, text_after in self.ingestor.iterate_items(doc_result):
            global_counter += 1
            current_item = None
            
            # Safe Page Number Access
            try:
                page_no = item.prov[0].page_no
            except:
                page_no = 1
            
            if page_no not in pages_content:
                pages_content[page_no] = []

            # --- A. Text & Headers ---
            if isinstance(item, (TextItem, SectionHeaderItem)):
                if len(item.text.strip()) < 3: continue
                bbox = item.prov[0].bbox
                current_item = {
                    "type": "header" if isinstance(item, SectionHeaderItem) else "text", 
                    "content": item.text,
                    "page": page_no,
                    "bbox": [bbox.l, bbox.b, bbox.r, bbox.t],
                    "sort_y": bbox.t,
                    "arrival_id": global_counter 
                }

            # --- B. Tables ---
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

            # --- C. Visuals (Charts, Diagrams) ---
            elif isinstance(item, PictureItem):
                log.info(f"   🖼️  Processing Visual on Page {page_no}")
                
                try:
                    # 1. Get Page Image
                    page_idx = page_no - 1
                    if page_idx < 0 or page_idx >= len(doc_result.pages):
                        continue

                    page_obj = doc_result.pages[page_idx]
                    if hasattr(page_obj.image, "pil_image"):
                        page_img = page_obj.image.pil_image
                    else:
                        page_img = page_obj.image
                    
                    pg_w, pg_h = page_img.width, page_img.height

                    # 2. Run YOLO (Cached per page to avoid re-running)
                    if page_no not in yolo_cache:
                        temp_path = f"data/temp_p{page_no}.png"
                        page_img.save(temp_path)
                        yolo_cache[page_no] = self.auditor.audit_page(temp_path)
                        if os.path.exists(temp_path): os.remove(temp_path)
                    
                    # 3. Map Coordinates (Docling -> Image Pixels)
                    d_bbox = item.prov[0].bbox
                    # Normalize if Docling returns normalized 0-1 coords
                    if d_bbox.l < 1.0 and d_bbox.r <= 1.0: 
                        d_l, d_b, d_r, d_t = d_bbox.l*pg_w, d_bbox.b*pg_h, d_bbox.r*pg_w, d_bbox.t*pg_h
                    else: 
                        d_l, d_b, d_r, d_t = d_bbox.l, d_bbox.b, d_bbox.r, d_bbox.t
                    
                    # Convert Bottom-Left origin (PDF) to Top-Left origin (Image)
                    d_top_img = pg_h - d_t
                    d_bot_img = pg_h - d_b
                    # Swap if negative height
                    if d_top_img > d_bot_img: d_top_img, d_bot_img = d_bot_img, d_top_img

                    # 4. Find Best YOLO Match
                    best_yolo_box = None
                    for y_obj in yolo_cache[page_no]:
                        y_box = y_obj['bbox']
                        # Intersection over Union (IoU) simplified check
                        x_overlap = max(0, min(d_r, y_box[2]) - max(d_l, y_box[0]))
                        y_overlap = max(0, min(d_bot_img, y_box[3]) - max(d_top_img, y_box[1]))
                        if x_overlap * y_overlap > 100: # Threshold
                            best_yolo_box = y_box
                            break
                    
                    # Use YOLO box if found, otherwise fallback to Docling box
                    base_box = tuple(map(int, best_yolo_box)) if best_yolo_box else (int(d_l), int(d_top_img), int(d_r), int(d_bot_img))

                    # 5. Crop with Padding
                    width = base_box[2] - base_box[0]
                    height = base_box[3] - base_box[1]
                    final_crop = (
                        max(0, base_box[0] - int(width * 0.1)),
                        max(0, base_box[1] - 50),
                        min(pg_w, base_box[2] + int(width * 0.1)),
                        min(pg_h, base_box[3] + max(150, int(height * 0.15)))
                    )

                    # 6. Save & Analyze
                    crop_filename = f"p{page_no}_crop_{global_counter}.png"
                    crop_path = os.path.join(self.debug_dir, crop_filename)
                    page_img.crop(final_crop).save(crop_path)
                    
                    # Context for Vision Agent
                    context_str = f"PREVIOUS TEXT:\n...{text_before[-1000:]}\nFOLLOWING TEXT:\n{text_after[:1000]}..."
                    
                    # Call Vision Agent
                    analysis_result = vision_agent.analyze_element(crop_path, "visual", context_str)

                    # Debug comparison
                    docling_ref = os.path.join(self.debug_dir, f"p{page_no}_ref_{global_counter}.png")
                    page_img.crop((int(d_l), int(d_top_img), int(d_r), int(d_bot_img))).save(docling_ref)
                    self.debug_reporter.add_comparison(page_no, docling_ref, crop_path, str(analysis_result))

                    # 7. Create Visual Item
                    current_item = {
                        "type": "visual",
                        "bbox": list(final_crop), 
                        "analysis": analysis_result, 
                        "file_path": crop_path, # KEY: Swarm uses this path
                        "page": page_no,
                        "sort_y": d_bbox.t, 
                        "arrival_id": global_counter
                    }

                except Exception as e:
                    log.error(f"      ⚠️ Visual Error on Page {page_no}: {e}")

            if current_item:
                pages_content[page_no].append(current_item)

        log.info("🔄 Sorting elements per page...")
        final_output = []
        
        sorted_page_nums = sorted(pages_content.keys())
        
        for p_num in sorted_page_nums:
            items = pages_content[p_num]
            # Stable Sort by Arrival ID (Reading Order)
            items.sort(key=lambda x: x['arrival_id'])
            
            for item in items:
                item['order_id'] = len(final_output) + 1
                del item['sort_y']
                del item['arrival_id']
                final_output.append(item)

        # --- METRICS & SAVE ---
        total_pages = max([x.get('page', 1) for x in final_output]) if final_output else 0
        total_visuals = len([x for x in final_output if x['type'] == 'visual'])
        
        metrics = {
            "duration": f"{int(time.time() - start_time)}s",
            "total_pages": total_pages,
            "total_visuals": total_visuals,
            "total_tables": len([x for x in final_output if x['type'] == 'table']),
            "total_text": len([x for x in final_output if x['type'] == 'text']),
            "avg_confidence": 1.0
        }

        with open("data/rag_dataset.json", "w") as f:
            json.dump(final_output, f, indent=2)
        
        self.debug_reporter.save()
        self.final_reporter.generate(final_output, metrics=metrics)
        
        log.info(f"✅ Processing Complete. Found {total_visuals} Visuals.")
        log.info(f"📄 Report: [underline]full_analysis.pdf[/]")

if __name__ == "__main__":
    if os.path.exists("data/test_doc.pdf"):
        AnalysisPipeline().run("data/test_doc.pdf", scanned_mode=False)