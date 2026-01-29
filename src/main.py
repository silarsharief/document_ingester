import os
import shutil
from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
from src.ingestion.docling_wrapper import DoclingWrapper
from src.agents.vision import vision_agent
from src.agents.validator import validator
from src.core.logger import setup_logger

log = setup_logger("main_pipeline")

class AnalysisPipeline:
    def __init__(self):
        self.ingestor = DoclingWrapper()
        
        # DEBUG SETUP
        self.debug_dir = "data/debug_crops"
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def run(self, pdf_path: str):
        log.info(f"🚀 [bold]Starting Pure Docling Pipeline (Debug Mode)[/]")
        log.info(f"📁 Debug images will be saved to: [underline]{self.debug_dir}[/]")
        
        # 1. INGESTION
        try:
            doc_result = self.ingestor.convert_pdf(pdf_path)
        except Exception as e:
            log.critical(f"❌ Ingestion Failed: {e}")
            return

        final_markdown = ""
        log.info("\n[bold yellow]🧠 PHASE 2: LINEAR ENHANCEMENT[/]")

        # 2. LINEAR LOOP
        for item, text_before, text_after in self.ingestor.iterate_items(doc_result):
            
            # --- HEADER ---
            if isinstance(item, SectionHeaderItem):
                log.info(f"   🔹 Header: [bold]{item.text[:50]}...[/]")
                final_markdown += f"\n## {item.text}\n\n"
                continue

            # --- TEXT ---
            if isinstance(item, TextItem):
                final_markdown += f"{item.text}\n\n"
                continue

            # --- TABLE ---
            if isinstance(item, TableItem):
                log.info(f"   📊 Found [bold]Table[/] on Page {item.prov[0].page_no}")
                try:
                    table_md = item.export_to_markdown(doc=doc_result.document)
                except:
                    table_md = item.export_to_markdown() 
                final_markdown += f"\n{table_md}\n\n"
                continue

            # --- FIGURE/IMAGE ---
            if isinstance(item, PictureItem):
                page_no = item.prov[0].page_no
                log.info(f"   🖼️  Found [bold magenta]Figure[/] on Page {page_no}")
                
                try:
                    # 1. Get Image
                    page_obj = doc_result.pages[page_no]
                    if hasattr(page_obj.image, "pil_image"):
                        page_img = page_obj.image.pil_image
                    else:
                        page_img = page_obj.image

                    # 2. Coordinate Maths
                    bbox = item.prov[0].bbox
                    page_h = page_img.height
                    page_w = page_img.width

                    # Normalize Check
                    if bbox.l < 1.0 and bbox.t < 1.0:
                        pdf_l, pdf_b, pdf_r, pdf_t = bbox.l * page_w, bbox.b * page_h, bbox.r * page_w, bbox.t * page_h
                    else:
                        pdf_l, pdf_b, pdf_r, pdf_t = bbox.l, bbox.b, bbox.r, bbox.t

                    # Flip Y-Axis Logic
                    crop_left = max(0, pdf_l)
                    crop_right = min(page_w, pdf_r)
                    crop_top = max(0, page_h - pdf_t)
                    crop_bottom = min(page_h, page_h - pdf_b)

                    if crop_top > crop_bottom:
                        crop_top, crop_bottom = crop_bottom, crop_top

                    crop_box = (crop_left, crop_top, crop_right, crop_bottom)
                    log.info(f"      ✂️  Cropping: {crop_box}")

                    # 3. Save Debug Crop
                    filename = f"page{page_no}_fig_{int(crop_top)}.png"
                    debug_path = os.path.join(self.debug_dir, filename)
                    
                    # Validate crop size
                    if (crop_right - crop_left) < 10 or (crop_bottom - crop_top) < 10:
                         log.warning("      ⚠️ Skipping tiny/invalid crop.")
                         continue

                    page_img.crop(crop_box).save(debug_path)
                    log.info(f"      💾 Saved crop to: [cyan]{debug_path}[/]")
                    
                    # 4. Prepare Context
                    context_str = f"Prev Paragraph: {text_before[:200]}...\nNext Paragraph: {text_after[:200]}..."
                    
                    # --- LOG INPUT ---
                    log.info(f"      📝 [bold]Input Context to Gemini:[/]\n[dim]\"{context_str.replace(chr(10), ' ')}\"[/dim]")
                    
                    log.info(f"      🤖 Sending to [blue]Gemini VLM[/]...")
                    
                    # 5. Call Gemini
                    analysis = vision_agent.analyze_element(debug_path, "figure", context_str)
                    
                    # --- LOG OUTPUT ---
                    log.info(f"      💡 [bold]Gemini Output:[/]\n[green]{analysis[:300]}... (truncated)[/]")

                    # 6. Validate
                    check = validator.validate("figure", 1.0, analysis)
                    
                    if check["is_valid"]:
                        final_markdown += f"\n\n> **[Figure Analysis]**\n> {analysis}\n\n"
                    else:
                        log.warning(f"      🗑️ [red]Dropped[/]: {check['reason']}")

                except Exception as e:
                    log.error(f"      ⚠️ Error processing image: {e}")
                continue

        # 3. SAVE
        output_path = "data/final_docling_pure.md"
        with open(output_path, "w") as f:
            f.write(final_markdown)
        
        log.info(f"✅ PIPELINE COMPLETE.")
        log.info(f"📁 Output saved to: [underline]{output_path}[/]")

if __name__ == "__main__":
    if os.path.exists("data/test_doc.pdf"):
        AnalysisPipeline().run("data/test_doc.pdf")