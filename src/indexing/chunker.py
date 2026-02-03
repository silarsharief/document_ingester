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

        # --- PASS 1: AGGREGATION (Grouping small items) ---
        logical_blocks = []
        buffer_text = ""
        buffer_meta = None
        MIN_BLOCK_SIZE = 400
        
        log.info(f"📊 Starting Pass 1: Aggregation (Min Size: {MIN_BLOCK_SIZE})...")

        for i, item in enumerate(data):
            page = item.get('page', 0)
            dtype = item.get('type', 'unknown')
            content = item.get('content', '').strip()
            
            if not content: continue

            # A. VISUALS/TABLES (Hard Stops)
            if dtype in ['table', 'visual']:
                # 1. Flush any pending text buffer
                if buffer_text:
                    logical_blocks.append({"text": buffer_text, "meta": buffer_meta, "type": "text"})
                    buffer_text = ""
                    buffer_meta = None
                
                # 2. Process the Special Item
                special_text = ""
                special_meta = self._get_base_metadata(item, i)
                
                if dtype == 'table':
                    special_text = f"Table on Page {page}:\n{content}"
                    log.info(f"   📅 Processed Table on Page {page}")
                else:
                    # Visual Logic
                    analysis = item.get('analysis', {})
                    cont = analysis.get('content', {})
                    heading = analysis.get('heading', 'Visual Element')
                    
                    special_text = f"Visual Chart: {heading}.\nDesc: {cont.get('overview', '')}\nInsights: {' '.join(cont.get('key_findings', []))}"
                    
                    # Store valid image path
                    special_meta['image_path'] = item.get('file_path', "") 
                    special_meta['confidence'] = analysis.get('confidence_score', 0.0)
                    
                    log.info(f"   🖼️  Processed Visual '{heading}' on Page {page}")
                
                logical_blocks.append({"text": special_text, "meta": special_meta, "type": dtype})
                continue

            # B. TEXT AGGREGATION
            if not buffer_text:
                buffer_meta = self._get_base_metadata(item, i)
            
            separator = "\n" if buffer_text else ""
            buffer_text += f"{separator}{content}"

            is_header = (dtype == 'header')
            
            next_page_change = False
            if i + 1 < len(data):
                if data[i+1].get('page') != page:
                    next_page_change = True
            else:
                next_page_change = True 

            if (len(buffer_text) > MIN_BLOCK_SIZE and not is_header) or next_page_change:
                logical_blocks.append({"text": buffer_text, "meta": buffer_meta, "type": "text"})
                buffer_text = ""
                buffer_meta = None

        if buffer_text:
            logical_blocks.append({"text": buffer_text, "meta": buffer_meta, "type": "text"})

        # --- PASS 2: CONTEXT OVERLAP ---
        final_chunks = []
        log.info(f"🔗 Starting Pass 2: Context Overlap (Blocks: {len(logical_blocks)})...")

        for i, block in enumerate(logical_blocks):
            current_text = block['text']
            
            if block['type'] == 'text':
                prev_ctx = ""
                if i > 0 and logical_blocks[i-1]['meta']['page'] == block['meta']['page']:
                    prev_ctx = f"[Context Above]: ...{logical_blocks[i-1]['text'][-200:]}\n"
                
                next_ctx = ""
                if i < len(logical_blocks) - 1 and logical_blocks[i+1]['meta']['page'] == block['meta']['page']:
                    next_ctx = f"\n[Context Below]: {logical_blocks[i+1]['text'][:200]}..."

                final_text = f"{prev_ctx}*** {current_text} ***{next_ctx}"
            else:
                final_text = current_text

            final_chunks.append({
                "id": block['meta']['doc_id'],
                "text": final_text,
                "metadata": block['meta']
            })

        log.info(f"✅ Chunking Complete. Created {len(final_chunks)} Hybrid documents.")
        return final_chunks

    def _get_base_metadata(self, item, index):
        """
        Creates standard metadata.
        CRITICAL FIX: image_path defaults to "" (empty string), NOT None.
        ChromaDB crashes if metadata values are None.
        """
        return {
            "source": "docling_pipeline",
            "page": item.get('page', 0),
            "type": item.get('type', 'text'),
            "doc_id": f"doc_{item.get('order_id', index)}",
            "bbox": str(item.get('bbox', [])),
            "image_path": "" # <--- CHANGED FROM None TO ""
        }