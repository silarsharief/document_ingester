import os
import json
import shutil
from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
from src.ingestion.docling_wrapper import DoclingWrapper
from src.ingestion.visual_auditor import VisualAuditor
from src.agents.vision import vision_agent
from src.core.logger import setup_logger
from src.core.debug_reporter import DebugReporter
from src.core.final_pdf import FinalReportGenerator # <--- New Import

log = setup_logger("pipeline")

class AnalysisPipeline:
    def __init__(self):
        self.ingestor = DoclingWrapper()
        self.auditor = VisualAuditor()
        self.debug_reporter = DebugReporter("debug_pipeline_report.pdf") # Doc 1 (Images Only)
        self.final_reporter = FinalReportGenerator("full_analysis.pdf")  # Doc 2 (Full Output)
        
        self.debug_dir = "data/crops"
        if os.path.exists(self.debug_dir): shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def run(self, pdf_path: str):
        log.info(f"🚀 Starting Production Pipeline: {pdf_path}")
        
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
        except Exception as e:
            log.critical(f"❌ Ingestion Failed: {e}")
            return

        final_output = []
        yolo_cache = {}
        order_id = 1

        log.info("\n[bold yellow]🧠 PHASE 2: PROCESSING[/]")

        for item, text_before, text_after in self.ingestor.iterate_items(doc_result):
            
            # --- Text & Headers ---
            if isinstance(item, (TextItem, SectionHeaderItem)):
                if len(item.text.strip()) < 3: continue
                final_output.append({
                    "order_id": order_id, 
                    "page": item.prov[0].page_no, 
                    "type": "header" if isinstance(item, SectionHeaderItem) else "text", 
                    "content": item.text
                })
                order_id += 1
                continue

            # --- Tables ---
            if isinstance(item, TableItem):
                try: table_md = item.export_to_markdown(doc=doc_result.document)
                except: table_md = item.export_to_markdown()
                final_output.append({
                    "order_id": order_id, "page": item.prov[0].page_no, "type": "table", "content": table_md
                })
                order_id += 1
                continue

            # --- Visuals ---
            if isinstance(item, PictureItem):
                page_no = item.prov[0].page_no
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
                    
                    # 2. Coordinate Handshake
                    d_bbox = item.prov[0].bbox
                    if d_bbox.l < 1.0: 
                        d_l, d_b, d_r, d_t = d_bbox.l*pg_w, d_bbox.b*pg_h, d_bbox.r*pg_w, d_bbox.t*pg_h
                    else: 
                        d_l, d_b, d_r, d_t = d_bbox.l, d_bbox.b, d_bbox.r, d_bbox.t
                    
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

                    # 3. Aggressive Expansion (Padding)
                    width = base_box[2] - base_box[0]
                    height = base_box[3] - base_box[1]
                    final_crop = (
                        max(0, base_box[0] - int(width * 0.1)),
                        max(0, base_box[1] - 50),
                        min(pg_w, base_box[2] + int(width * 0.1)),
                        min(pg_h, base_box[3] + max(150, int(height * 0.15)))
                    )

                    # 4. Crop & Save
                    crop_filename = f"p{page_no}_fig_{order_id}.png"
                    crop_path = os.path.join(self.debug_dir, crop_filename)
                    page_img.crop(final_crop).save(crop_path)
                    
                    # 5. Gemini Analysis (With IMPROVED Context)
                    # FIX: Use [-1000:] to get the immediate text preceding the image
                    context_str = f"""
                    PREVIOUS TEXT CONTEXT:
                    ...{text_before[-1000:]}
                    
                    FOLLOWING TEXT CONTEXT:
                    {text_after[:1000]}...
                    """
                    
                    analysis_result = vision_agent.analyze_element(crop_path, "visual", context_str)

                    # 6. Add to Debug Report (Doc 1)
                    docling_debug_path = os.path.join(self.debug_dir, f"p{page_no}_ref.png")
                    page_img.crop((int(d_l), int(d_top_img), int(d_r), int(d_bot_img))).save(docling_debug_path)
                    
                    content_block = analysis_result.get('content', {})
                    report_text = f"<b>{analysis_result.get('heading', 'Visual')}</b><br/>{content_block.get('overview', '')}"
                    self.debug_reporter.add_comparison(page_no, docling_debug_path, crop_path, report_text)

                    # 7. Add to Final Output
                    final_output.append({
                        "order_id": order_id, "page": page_no, "type": "visual",
                        "bbox": list(final_crop), "analysis": analysis_result, "file_path": crop_path
                    })
                    order_id += 1

                except Exception as e:
                    log.error(f"      ⚠️ Error: {e}")

        # Finalize
        with open("data/rag_dataset.json", "w") as f:
            json.dump(final_output, f, indent=2)
        
        self.debug_reporter.save() # Save Doc 1 (Pipeline Debug)
        self.final_reporter.generate(final_output) # Save Doc 2 (Full Output)
        
        log.info(f"✅ Done.")
        log.info(f"📄 Debug Report: [underline]debug_pipeline_report.pdf[/]")
        log.info(f"📄 Full Analysis: [underline]full_analysis.pdf[/]")

if __name__ == "__main__":
    if os.path.exists("data/test_doc.pdf"): AnalysisPipeline().run("data/test_doc.pdf")