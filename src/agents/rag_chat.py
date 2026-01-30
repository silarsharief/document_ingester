import google.generativeai as genai
from src.indexing.vector_store import LocalVectorStore
from src.core.config import settings
from src.core.logger import setup_logger

log = setup_logger("rag_agent")

class RAGAgent:
    def __init__(self):
        # 1. Connect to the Brain (ChromaDB)
        self.vector_store = LocalVectorStore()
        
        # 2. Connect to the Mouth (Gemini 2.0 Flash)
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # UPDATED: Switched to 2.0 Flash per your request
        # If 'gemini-2.0-flash' fails, try 'gemini-2.0-flash-exp'
        self.model = genai.GenerativeModel('gemini-2.0-flash') 

    def ask(self, query: str) -> dict:
        log.info(f"🤔 User Query: '{query}'")
        
        # --- Step 1: Retrieval ---
        try:
            results = self.vector_store.query(query, n_results=5)
        except Exception as e:
            log.error(f"Vector Store Error: {e}")
            return {"answer": "Error accessing the knowledge base.", "sources": {}}
        
        if not results['documents'] or not results['documents'][0]:
            return {"answer": "I couldn't find any information about that in the document.", "sources": {}}

        # --- Step 2: Context Construction ---
        context_str = ""
        source_map = {} 
        
        for i, doc_text in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            source_id = f"[Source: Page {meta.get('page', '?')}]"
            
            # Label visuals clearly for the LLM
            if meta.get('type') == 'visual':
                prefix = f"VISUAL EVIDENCE (Confidence: {meta.get('confidence', 0):.2f}): "
            else:
                prefix = "TEXT SEGMENT: "
                
            context_str += f"{source_id} {prefix}{doc_text}\n\n"
            
            # Store metadata for the UI 
            source_map[source_id] = {
                "page": meta.get('page'),
                "type": meta.get('type'),
                "image_path": meta.get('image_path', None),
                "confidence": meta.get('confidence', None)
            }

        # --- Step 3: Generation ---
        prompt = f"""
        You are QuickSight AI, a multi-modal document assistant.
        
        USER QUESTION: "{query}"
        
        RETRIEVED CONTEXT:
        {context_str}
        
        INSTRUCTIONS:
        1. Answer the question using ONLY the provided context.
        2. If the answer comes from a "VISUAL EVIDENCE" block, explicitly mention the chart/figure trends.
        3. STRICT CITATION: Every fact must end with its source tag (e.g., [Source: Page 1]).
        4. If the context has low confidence (confidence < 0.5), mention that the data might be unclear.
        """
        
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
        except Exception as e:
            log.error(f"LLM Error: {e}")
            # Fallback message
            answer_text = f"I encountered an error generating the response with Gemini 2.0. Error: {e}"

        return {
            "answer": answer_text,
            "sources": source_map 
        }

if __name__ == "__main__":
    agent = RAGAgent()
    response = agent.ask("How does Mistral 7B compare to Llama 2?")
    print("\n🤖 ANSWER:\n", response['answer'])