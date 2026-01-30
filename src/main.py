import os
import json
import shutil
from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
from src.ingestion.docling_wrapper import DoclingWrapper
from src.ingestion.visual_auditor import VisualAuditor
from src.agents.vision import vision_agent
from src.core.logger import setup_logger
from src.core.debug_reporter import DebugReporter

log = setup_logger("pipeline")

class AnalysisPipeline:
    def __init__(self):
        self.ingestor = DoclingWrapper()
        self.auditor = VisualAuditor()
        self.reporter = DebugReporter("pipeline_report.pdf")
        
        # Clean and recreate debug directory
        self.debug_dir = "data/crops"
        if os.path.exists(self.debug_dir): 
            shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def run(self, pdf_path: str):
        log.info(f"🚀 Starting Hybrid Pipeline: {pdf_path}")
        
        # 1. Ingestion (Docling)
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
        except Exception as e:
            log.critical(f"❌ Ingestion Failed: {e}")
            return

        final_output = []
        yolo_cache = {}
        order_id = 1

        log.info("\n[bold yellow]🧠 PHASE 2: PROCESSING[/]")

        # 2. Iteration Loop (Maintains Reading Order)
        for item, text_before, text_after in self.ingestor.iterate_items(doc_result):
            
            # --- CASE A: TEXT & HEADERS ---
            if isinstance(item, (TextItem, SectionHeaderItem)):
                text_content = item.text.strip()
                if len(text_content) < 3: continue # Skip artifacts
                
                final_output.append({
                    "order_id": order_id,
                    "page": item.prov[0].page_no,
                    "type": "header" if isinstance(item, SectionHeaderItem) else "text",
                    "content": text_content,
                    "bbox": [item.prov[0].bbox.l, item.prov[0].bbox.t, item.prov[0].bbox.r, item.prov[0].bbox.b]
                })
                order_id += 1

            # --- CASE B: TABLE ---
            elif isinstance(item, TableItem):
                log.info(f"   📊 Processing Table on Page {item.prov[0].page_no}")
                try: 
                    table_md = item.export_to_markdown(doc=doc_result.document)
                except: 
                    table_md = item.export_to_markdown()
                
                final_output.append({
                    "order_id": order_id,
                    "page": item.prov[0].page_no,
                    "type": "table",
                    "content": table_md,
                    "bbox": [item.prov[0].bbox.l, item.prov[0].bbox.t, item.prov[0].bbox.r, item.prov[0].bbox.b]
                })
                order_id += 1

            # --- CASE C: VISUAL (The Hybrid Logic) ---
            elif isinstance(item, PictureItem):
                page_no = item.prov[0].page_no
                log.info(f"   🖼️  Processing Visual on Page {page_no}")
                
                try:
                    # A. Get Page Image
                    page_obj = doc_result.pages[page_no]
                    if hasattr(page_obj.image, "pil_image"):
                        page_img = page_obj.image.pil_image
                    else:
                        page_img = page_obj.image
                    
                    pg_w, pg_h = page_img.width, page_img.height

                    # B. Run YOLO (Cached per page)
                    if page_no not in yolo_cache:
                        temp_path = f"data/temp_p{page_no}.png"
                        page_img.save(temp_path)
                        yolo_cache[page_no] = self.auditor.audit_page(temp_path)
                        if os.path.exists(temp_path): os.remove(temp_path)
                    
                    yolo_detections = yolo_cache[page_no]

                    # C. The Handshake (Match Docling position to YOLO Box)
                    d_bbox = item.prov[0].bbox
                    
                    # 1. Normalize Docling Box to Image Coordinates
                    if d_bbox.l < 1.0: 
                        d_l, d_b, d_r, d_t = d_bbox.l*pg_w, d_bbox.b*pg_h, d_bbox.r*pg_w, d_bbox.t*pg_h
                    else: 
                        d_l, d_b, d_r, d_t = d_bbox.l, d_bbox.b, d_bbox.r, d_bbox.t
                    
                    # 2. Flip Y-Axis (PDF has 0 at bottom, Image has 0 at top)
                    d_top_img = pg_h - d_t
                    d_bot_img = pg_h - d_b
                    
                    # 3. Find Overlap
                    best_yolo_box = None
                    for y_obj in yolo_detections:
                        y_box = y_obj['bbox'] # [x1, y1, x2, y2]
                        
                        # Calculate Intersection Area
                        x_overlap = max(0, min(d_r, y_box[2]) - max(d_l, y_box[0]))
                        y_overlap = max(0, min(d_bot_img, y_box[3]) - max(d_top_img, y_box[1]))
                        
                        # If significant overlap (>100px area), it's a match
                        if x_overlap * y_overlap > 100: 
                            best_yolo_box = y_box
                            break
                    
                    # 4. Decide Base Crop
                    if best_yolo_box:
                        base_box = tuple(map(int, best_yolo_box))
                        # log.info("      ✅ Matched with YOLO")
                    else:
                        base_box = (int(d_l), int(d_top_img), int(d_r), int(d_bot_img))
                        log.info("      ⚠️ No YOLO match, using fallback")

                    # D. Aggressive Expansion (The Context Fix)
                    # Add padding to catch Axis Labels, Legends, and Captions
                    width = base_box[2] - base_box[0]
                    height = base_box[3] - base_box[1]
                    
                    pad_x = int(width * 0.1)        # 10% width padding
                    pad_y_top = 50                  # 50px top padding (Title)
                    pad_y_bottom = max(150, int(height * 0.15)) # 150px bottom padding (Caption)

                    final_crop_box = (
                        max(0, base_box[0] - pad_x),
                        max(0, base_box[1] - pad_y_top),
                        min(pg_w, base_box[2] + pad_x),
                        min(pg_h, base_box[3] + pad_y_bottom)
                    )

                    # E. Crop & Save
                    crop_filename = f"p{page_no}_fig_{order_id}.png"
                    crop_path = os.path.join(self.debug_dir, crop_filename)
                    
                    # Save "Before" image for report (Docling original)
                    docling_debug_path = os.path.join(self.debug_dir, f"p{page_no}_docling_ref.png")
                    docling_box_safe = (int(d_l), int(d_top_img), int(d_r), int(d_bot_img))
                    # Clamp values for safety
                    docling_box_safe = (max(0,docling_box_safe[0]), max(0,docling_box_safe[1]), min(pg_w,docling_box_safe[2]), min(pg_h,docling_box_safe[3]))
                    page_img.crop(docling_box_safe).save(docling_debug_path)

                    # Save "After" image (Expanded Crop)
                    page_img.crop(final_crop_box).save(crop_path)

                    # F. Gemini Analysis
                    context_str = f"Prev text: {text_before[:200]}...\nNext text: {text_after[:200]}..."
                    analysis_result = vision_agent.analyze_element(crop_path, "visual", context_str)

                    # G. Format for PDF Report
                    content_block = analysis_result.get('content', {})
                    
                    # Helper to format bullets
                    findings_list = content_block.get('key_findings', [])
                    findings_html = "".join([f"&bull; {item}<br/>" for item in findings_list]) if findings_list else "No specific findings."
                    
                    # Helper to format data dict
                    raw_data = content_block.get('extracted_data', {})
                    data_html = json.dumps(raw_data, indent=2).replace('\n', '<br/>').replace(' ', '&nbsp;') if raw_data else "No raw data extracted."

                    formatted_report_text = f"""
                    <b>{analysis_result.get('heading', 'Visual Element')}</b><br/><br/>
                    <i>{content_block.get('overview', 'No overview provided.')}</i><br/><br/>
                    <b>Key Findings:</b><br/>
                    {findings_html}<br/><br/>
                    <b>Extracted Data:</b><br/>
                    <font name="Courier" size="8">{data_html}</font>
                    """
                    
                    self.reporter.add_comparison(page_no, docling_debug_path, crop_path, formatted_report_text) 

                    # H. Add to Final Output
                    final_output.append({
                        "order_id": order_id,
                        "page": page_no,
                        "type": "visual",
                        "bbox": list(final_crop_box),
                        "analysis": analysis_result, # The Structured JSON
                        "file_path": crop_path
                    })
                    order_id += 1

                except Exception as e:
                    log.error(f"      ⚠️ Processing Error on Page {page_no}: {e}")

        # 3. Finalize
        output_json_path = "data/rag_dataset.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        self.reporter.save()
        log.info(f"✅ Pipeline Complete.")
        log.info(f"📄 Report: [underline]pipeline_report.pdf[/]")
        log.info(f"📁 Data:   [underline]{output_json_path}[/]")

if __name__ == "__main__":
    # Change filename as needed
    if os.path.exists("data/test_doc.pdf"):
        AnalysisPipeline().run("data/test_doc.pdf")
    else:
        print("⚠️ Please place a file named 'test_doc.pdf' in the 'data/' folder.")