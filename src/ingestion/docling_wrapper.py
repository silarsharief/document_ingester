from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os

class DoclingWrapper:
    def __init__(self):
        print("📑 Initializing Docling Wrapper...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = True # Crucial for Visual Audit

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        print(f"   ↳ Converting {os.path.basename(pdf_path)}...")
        return self.converter.convert(pdf_path)

    def get_page_text(self, doc_result, page_num):
        """
        Extracts text specifically for a single page.
        """
        try:
            # Iterate through text items to find those belonging to the specific page
            page_text_lines = []
            for item in doc_result.document.texts:
                # Check provenance to see if it belongs to the requested page
                if item.prov and item.prov[0].page_no == page_num:
                    page_text_lines.append(item.text)
            
            # If Docling didn't map it perfectly, try the page object directly (fallback)
            if not page_text_lines and page_num in doc_result.document.pages:
                page_obj = doc_result.document.pages[page_num]
                return getattr(page_obj, "text", "")
            
            return "\n".join(page_text_lines)
        except Exception as e:
            print(f"   ⚠️ Warning: Could not extract text for page {page_num}: {e}")
            return ""