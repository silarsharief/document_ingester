import chromadb
from chromadb.utils import embedding_functions
from src.core.logger import setup_logger

log = setup_logger("vector_store")

class LocalVectorStore:
    def __init__(self, collection_name="quicksight_rag"):
        # Persistent storage (so you don't rebuild index every restart)
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        
        # Default = all-MiniLM-L6-v2 (Great balance of speed/accuracy)
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
        
        ids = [c['id'] for c in chunks]
        documents = [c['text'] for c in chunks]
        
        # Chroma metadata must be flat (str, int, float). Ensure list strings are handled.
        metadatas = []
        for c in chunks:
            m = c['metadata'].copy()
            # Safety check: Chroma doesn't like None types
            for k, v in m.items():
                if v is None: m[k] = ""
            metadatas.append(m)

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            count = self.collection.count()
            log.info(f"✅ Success! Total Collection Size: {count} items")
        except Exception as e:
            log.error(f"❌ Indexing Error: {e}")

    def query(self, query_text: str, n_results=5):
        log.info(f"🔍 Searching for: '{query_text}'")
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )