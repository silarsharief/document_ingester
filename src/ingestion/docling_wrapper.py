from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
import os
from src.core.logger import setup_logger

log = setup_logger("docling_wrapper")

class DoclingWrapper:
    def __init__(self):
        log.info("[bold green]📑 Initializing Docling Wrapper (Structure-First)...[/]")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = True 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            log.error(f"❌ PDF not found at {pdf_path}")
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
            
        log.info(f"   ↳ Converting [cyan]{os.path.basename(pdf_path)}[/]...")
        return self.converter.convert(pdf_path)

    def iterate_items(self, doc_result):
        """
        Yields every element in the document in strict reading order.
        Strategy: Collect ALL flat lists, combine, and sort by geometry.
        """
        log.info("   ⚙️  Flattening and Sorting Document Elements...")
        
        all_items = []
        
        # 1. Collect Items (Safely)
        if hasattr(doc_result.document, "texts"):
            all_items.extend(doc_result.document.texts)
        if hasattr(doc_result.document, "tables"):
            all_items.extend(doc_result.document.tables)
        if hasattr(doc_result.document, "pictures"):
            all_items.extend(doc_result.document.pictures)

        log.info(f"   📊 Found [bold cyan]{len(all_items)}[/] total elements.")

        # 2. Sort Function (Page -> Top -> Left)
        def sort_key(item):
            if not item.prov: return (9999, 9999, 9999)
            prov = item.prov[0]
            return (prov.page_no, prov.bbox.t, prov.bbox.l)

        all_items.sort(key=sort_key)

        # 3. Yield
        for i, item in enumerate(all_items):
            # Context Logic
            context_before = ""
            if i > 0 and hasattr(all_items[i-1], "text"):
                context_before = all_items[i-1].text

            context_after = ""
            if i < len(all_items) - 1 and hasattr(all_items[i+1], "text"):
                context_after = all_items[i+1].text

            yield item, context_before, context_after