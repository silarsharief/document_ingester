import json
from typing import List, Dict, Any
from src.core.logger import setup_logger

log = setup_logger("indexing")

class DocumentChunker:
    def __init__(self, input_path="data/rag_dataset.json"):
        self.input_path = input_path

    def load_and_chunk(self) -> List[Dict[str, Any]]:
        log.info(f"📂 Loading dataset from: {self.input_path}")
        
        try:
            with open(self.input_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            log.critical(f"❌ File not found: {self.input_path}")
            return []

        chunks = []
        log.info(f"📊 Found {len(data)} items. Starting semantic processing...")

        for i, item in enumerate(data):
            # Use 'order_id' to maintain the reading flow
            doc_id = f"doc_{item.get('order_id', i)}"
            page = item.get('page', 0)
            dtype = item.get('type', 'unknown')
            
            # Base Metadata - Vital for RAG Filtering
            metadata = {
                "source": "docling_pipeline",
                "page": page,
                "type": dtype,
                "doc_id": doc_id,
                "bbox": str(item.get('bbox', []))
            }

            text_to_embed = ""

            # --- STRATEGY: TEXT ---
            if dtype in ['text', 'header']:
                text_to_embed = item.get('content', '').strip()
                # Skip noise (page numbers, tiny artifacts)
                if len(text_to_embed) < 10: 
                    continue

            # --- STRATEGY: TABLE ---
            elif dtype == 'table':
                # Embed the Markdown structure so the LLM understands rows/cols
                text_to_embed = f"Table on Page {page}:\n" + item.get('content', '')
                log.info(f"   📅 Processed Table on Page {page}")

            # --- STRATEGY: VISUAL (The Multi-Modal Bridge) ---
            elif dtype == 'visual':
                analysis = item.get('analysis', {})
                content = analysis.get('content', {})
                
                # 1. We embed the SEMANTIC meaning (The Text Description)
                heading = analysis.get('heading', 'Visual Element')
                overview = content.get('overview', '')
                findings = " ".join(content.get('key_findings', []))
                
                # This text is what the Vector DB will "search" against
                text_to_embed = f"Visual Chart/Figure: {heading}.\nDescription: {overview}\nKey Insights: {findings}"
                
                # 2. We store the VISUAL PROOF (Image Path) in metadata
                metadata['image_path'] = item.get('file_path', '')
                metadata['confidence'] = analysis.get('confidence_score', 0.0)
                
                log.info(f"   🖼️  Processed Visual '{heading}' on Page {page} (Conf: {metadata['confidence']:.2f})")

            # Final check before adding
            if text_to_embed:
                chunks.append({
                    "id": doc_id,
                    "text": text_to_embed,
                    "metadata": metadata
                })

        log.info(f"✅ Chunking Complete. Created {len(chunks)} embeddable documents.")
        return chunks