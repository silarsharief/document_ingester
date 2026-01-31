import os
import chromadb
from chromadb.utils import embedding_functions
from src.core.logger import setup_logger
import gc

log = setup_logger("vector_store")

class LocalVectorStore:
    def __init__(self, collection_name="quicksight_rag"):
        # 1. ENSURE DIRECTORY EXISTS
        db_path = os.path.abspath("data/chroma_db")
        os.makedirs(db_path, exist_ok=True)
        
        # 2. INITIALIZE CLIENT
        # Using a persistent client that saves to disk
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 3. SETUP EMBEDDING FUNCTION
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection_name = collection_name
        self._get_or_create_collection()

    def _get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef
        )

    def add_documents(self, chunks):
        if not chunks: return
        log.info(f"🚀 Indexing {len(chunks)} documents...")
        
        batch_size = 40 # Increased batch size for speed
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                self.collection.upsert(
                    ids=[c['id'] for c in batch],
                    documents=[c['text'] for c in batch],
                    metadatas=[c['metadata'] for c in batch]
                )
            except Exception as e:
                log.error(f"❌ Batch Indexing Failed: {e}")
            
            del batch
            gc.collect()

    def query(self, query_text: str, n_results=25):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas'] 
        )