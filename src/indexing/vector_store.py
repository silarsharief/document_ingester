import chromadb
from chromadb.utils import embedding_functions
from src.core.logger import setup_logger
import gc  # <--- Add Garbage Collection

log = setup_logger("vector_store")

class LocalVectorStore:
    def __init__(self, collection_name="quicksight_rag"):
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        # Use a slightly lighter model configuration if possible, but default is usually fine
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection_name = collection_name
        self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
            log.info(f"💾 Vector Database Active: {self.collection_name}")
        except Exception as e:
            log.critical(f"❌ Failed to initialize ChromaDB: {e}")
            raise e

    def add_documents(self, chunks):
        if not chunks:
            log.warning("⚠️ No documents to add!")
            return

        log.info(f"🚀 Indexing {len(chunks)} documents...")
        
        # --- OPTIMIZED BATCHING FOR M1 RAM ---
        batch_size = 10  # Reduced from 50 to 10 to save RAM
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [c['id'] for c in batch]
            documents = [c['text'] for c in batch]
            
            metadatas = []
            for c in batch:
                m = c['metadata'].copy()
                for k, v in m.items():
                    if v is None: m[k] = ""
                metadatas.append(m)

            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                log.info(f"   ⏳ Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
                
                # --- MEMORY SAVER ---
                del batch, ids, documents, metadatas
                gc.collect()  # Force Python to release RAM immediately
                
            except Exception as e:
                log.error(f"❌ Batch Indexing Error: {e}")
        
        count = self.collection.count()
        log.info(f"✅ Success! Total Collection Size: {count} items")

    def query(self, query_text: str, n_results=5):
        log.info(f"🔍 Searching for: '{query_text}'")
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )