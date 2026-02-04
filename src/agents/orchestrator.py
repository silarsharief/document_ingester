import time
import asyncio
from sentence_transformers import CrossEncoder
from src.indexing.vector_store import LocalVectorStore
from src.agents.specialists import TextAgent, VisionAgent, ValidationAgent
from src.core.logger import setup_logger
import google.generativeai as genai
from src.core.config import settings

log = setup_logger("orchestrator")

class MultiAgentOrchestrator:
    def __init__(self, vector_store=None):
        log.info("🤖 Initializing Formatted-Swarm Orchestrator...")
        
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = LocalVectorStore()
            
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.text_agent = TextAgent()
        self.vision_agent = VisionAgent()
        self.validator = ValidationAgent()
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.fusion_model = genai.GenerativeModel('gemini-2.0-flash')

    def retrieve_and_split(self, query: str):
        try:
            results = self.vector_store.query(query, n_results=50)
        except Exception as e:
            log.error(f"Vector Store Error: {e}")
            return [], [], []

        if not results['documents'] or not results['documents'][0]:
            return [], [], []

        raw_docs = results['documents'][0]
        raw_metas = results['metadatas'][0]
        
        # Rerank
        pred_pairs = [[query, doc] for doc in raw_docs]
        scores = self.reranker.predict(pred_pairs)
        
        text_candidates = []
        visual_candidates = []
        
        for i, score in enumerate(scores):
            item = {
                "content": raw_docs[i],
                "meta": raw_metas[i],
                "score": float(score),
                "type": raw_metas[i].get("type", "text"),
                "image_path": raw_metas[i].get("image_path", "")
            }
            
            if item['type'] == 'visual':
                visual_candidates.append(item)
            else:
                text_candidates.append(item)

        # Sort
        text_candidates.sort(key=lambda x: x['score'], reverse=True)
        visual_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # --- PERMISSIVE LOGIC ---
        # Show visuals if they exist (Score > -15.0)
        visual_threshold = -11.0
        
        final_visual = [v for v in visual_candidates if v['score'] > visual_threshold][:2]
        remaining_slots = 4 - len(final_visual)
        final_text = text_candidates[:remaining_slots]
        
        log.info(f"📊 Selection: {len(final_text)} Text + {len(final_visual)} Visual")
        return final_text, final_visual, final_text + final_visual

    async def _safe_generate(self, func, *args):
        retries = 3
        for i in range(retries):
            try:
                return await func(*args)
            except Exception as e:
                if "429" in str(e):
                    wait_time = (i + 1) * 2
                    log.warning(f"⚠️ Rate Limit. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    log.error(f"❌ API Error: {e}")
                    return "Error generating response."
        return "System Busy (Rate Limit)."

    async def ask(self, query: str) -> dict:
        start_time = time.time()
        log.info(f"🔎 Processing: '{query}'")
        
        # 1. Retrieval
        text_chunks, visual_chunks, all_evidence = self.retrieve_and_split(query)
        
        if not all_evidence:
            return {"answer": "I couldn't find any relevant information.", "evidence": []}

        # 2. Async Execution
        text_task = self._safe_generate(self.text_agent.analyze, query, text_chunks)
        
        if visual_chunks:
            vision_task = self._safe_generate(self.vision_agent.analyze, query, visual_chunks)
        else:
            async def dummy(): return "No visuals."
            vision_task = dummy()
            
        text_insight, visual_insight = await asyncio.gather(text_task, vision_task)

        # 3. Fusion (BETTER FORMATTING)
        log.info("   ... Fusing Insights")
        prompt = f"""
        You are the Fusion Agent. Combine the Text and Visual insights into a final answer.
        
        USER QUERY: "{query}"
        
        [TEXT INSIGHT]: {text_insight}
        
        [VISUAL INSIGHT]: {visual_insight}
        
        INSTRUCTIONS:
        1. **Format:** Use Markdown.
           - Use **Bold** for key terms or names.
           - Use **Bullet Points** for lists (authors, steps, comparisons).
           - Use short paragraphs for readability.
        2. **Visuals:** IF the Visual Insight contains useful info, mention it explicitly (e.g., "As shown in the chart...").
        3. **Cleanup:** IF the Visual Insight is irrelevant/broken, IGNORE IT.
        4. **Citations:** Cite sources as [Source 1].
        """
        
        fusion_response = await self._safe_generate(self.fusion_model.generate_content_async, prompt)
        try:
            final_answer = fusion_response.text
        except:
            final_answer = "Fusion Error."

        # 4. Validation
        try:
            validation = await self.validator.validate(query, final_answer, [c['content'] for c in all_evidence])
        except:
            validation = {'score': 1.0}

        duration = time.time() - start_time
        log.info(f"✅ Complete in {duration:.2f}s")

        return {
            "answer": final_answer,
            "evidence": all_evidence
        }