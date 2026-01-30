from src.indexing.chunker import DocumentChunker
from src.indexing.vector_store import LocalVectorStore
from src.core.logger import setup_logger

log = setup_logger("builder")

def main():
    log.info("🧱 STARTING RAG INDEX BUILDER")
    
    # 1. Chunk
    chunker = DocumentChunker()
    chunks = chunker.load_and_chunk()
    
    if not chunks:
        log.error("❌ Aborting: No chunks found.")
        return

    # 2. Index
    vs = LocalVectorStore()
    vs.add_documents(chunks)
    
    log.info("🎉 RAG Brain is ready!")

if __name__ == "__main__":
    main()