import google.generativeai as genai
from sentence_transformers import CrossEncoder
from src.indexing.vector_store import LocalVectorStore
from src.core.config import settings
from src.core.logger import setup_logger
import time

log = setup_logger("rag_system")

class RAGAgent:
    def __init__(self):
        log.info("🔌 Initializing RAG Subsystems...")
        
        # 1. Retrieval (Recall)
        self.vector_store = LocalVectorStore()
        
        # 2. Reranking (Precision)
        log.info("🧠 Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 3. Generation (Synthesis)
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash') 

    def ask(self, query: str) -> dict:
        start_time = time.time()
        log.info(f"🔎 [QUERY] '{query}'")
        
        # --- STEP 1: BROAD RETRIEVAL ---
        try:
            results = self.vector_store.query(query, n_results=20) # Fetch 20 to be safe
        except Exception as e:
            return {"answer": "System Error: Database inaccessible.", "evidence": [], "rejected": []}

        if not results['documents'] or not results['documents'][0]:
            return {"answer": "I couldn't find any relevant information in the documents.", "evidence": [], "rejected": []}

        raw_docs = results['documents'][0]
        raw_metas = results['metadatas'][0]
        raw_ids = results['ids'][0]
        
        log.info(f"📥 Retrieved {len(raw_docs)} candidates from Vector DB.")

        # --- STEP 2: RERANKING ---
        pred_pairs = [[query, doc] for doc in raw_docs]
        scores = self.reranker.predict(pred_pairs)
        
        candidates = []
        for i, score in enumerate(scores):
            candidates.append({
                "id": raw_ids[i],
                "content": raw_docs[i],
                "meta": raw_metas[i],
                "score": float(score),
                "type": raw_metas[i].get("type", "text"),
                "page": raw_metas[i].get("page", "?"),
                "image_path": raw_metas[i].get("image_path", None)
            })

        # Sort & Filter
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        validated_evidence = []
        rejected_evidence = []
        THRESHOLD = 0.05 
        
        for cand in candidates:
            if cand['score'] >= THRESHOLD:
                validated_evidence.append(cand)
            else:
                rejected_evidence.append(cand)

        # --- SMART FALLBACK ---
        # If strict reranking kills ALL matches, fallback to top 5 but force the LLM to infer
        used_fallback = False
        if not validated_evidence:
            log.warning("⚠️ Strict Reranker killed all matches. Activating Smart Fallback.")
            validated_evidence = candidates[:5]
            used_fallback = True

        # --- LOGGING ---
        log.info(f"📊 [STATS] Total: {len(candidates)} | Kept: {len(validated_evidence)} | Rejected: {len(rejected_evidence)}")
        if validated_evidence:
            top = validated_evidence[0]
            log.info(f"✅ [TOP MATCH] Page {top['page']} (Score: {top['score']:.4f})")

        # Keep top 7 chunks for synthesis context
        final_context = validated_evidence[:7]

        # --- STEP 4: "DECISIVE ANALYST" SYNTHESIS ---
        context_block = ""
        for i, item in enumerate(final_context):
            context_block += f"""
            [SOURCE ID: {i+1}]
            Type: {item['type']}
            Page: {item['page']}
            Content: {item['content']}
            --------------------------------
            """

        # UPDATED PROMPT: Forces inference and removes hedging
        prompt = f"""
        You are a decisive Intelligence Analyst. Your goal is to answer the User Query directly using the provided fragments.

        USER QUERY: "{query}"

        SOURCES:
        {context_block}

        INSTRUCTIONS:
        1. **Answer First:** Start immediately with the answer. Do NOT say "The document mentions..." or "Based on the context...".
           - BAD: "The document contains a name P.J. Cross at the bottom."
           - GOOD: "The letter was written by P.J. Cross."
        
        2. **Connect the Dots:** The sources are fragmented (OCR chunks). You must logically infer the meaning.
           - If you see a name at the bottom of a letter, that is the Author.
           - If you see a date at the top, that is the Date.
           - If you see a list of items, summarize them.
        
        3. **Strict Citation:** Cite the Source ID used for every fact. e.g. [Source 1].
        
        4. **Handling Uncertainty:** Only say "I don't know" if the information is truly missing. If it's likely true based on document structure, state it.
        
        5. **Visuals:** If a source describes a chart or table, explain what the data shows.
        """
        
        try:
            response = self.model.generate_content(prompt)
            final_ans = response.text
            
            # Optional: Add a tiny marker if we forced fallback, but don't ruin the flow
            if used_fallback:
                final_ans += "\n\n*(Note: Answer inferred from low-confidence matches)*"
                
        except Exception as e:
            log.error(f"LLM Generation Error: {e}")
            final_ans = "Error generating synthesis response."

        return {
            "answer": final_ans,
            "evidence": final_context,
            "rejected": rejected_evidence[:10]
        }