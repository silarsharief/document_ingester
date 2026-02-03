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
        log.info("🤖 Initializing Multi-Agent Swarm (Async Mode)...")
        
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
            # Fetch more candidates to ensure we capture visuals if they exist down the list
            results = self.vector_store.query(query, n_results=30)
        except Exception as e:
            log.error(f"Vector Store Error: {e}")
            return [], [], []

        if not results['documents'] or not results['documents'][0]:
            return [], [], []

        raw_docs = results['documents'][0]
        raw_metas = results['metadatas'][0]
        
        # Rerank everything
        pred_pairs = [[query, doc] for doc in raw_docs]
        scores = self.reranker.predict(pred_pairs)
        
        candidates = []
        for i, score in enumerate(scores):
            candidates.append({
                "content": raw_docs[i],
                "meta": raw_metas[i],
                "score": float(score),
                "type": raw_metas[i].get("type", "text"),
                "page": raw_metas[i].get("page", "?"),
                "image_path": raw_metas[i].get("image_path", None)
            })
        
        # Sort highest score first
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # --- DYNAMIC RELATIVE THRESHOLD ---
        # Instead of fixed 0.01, we look at the best score.
        if not candidates:
            return [], [], []

        best_score = candidates[0]['score']
        # We allow chunks that are within a reasonable range of the best match.
        # This prevents dropping good visuals just because they are slightly lower than text.
        acceptable_range = 7.0 
        threshold = best_score - acceptable_range
        
        final_selection = [c for c in candidates if c['score'] >= threshold]
        
        # Limit to Top 8 after filtering
        final_selection = final_selection[:8]
        
        # Split for agents
        text_chunks = [c for c in final_selection if c['type'] in ['text', 'header', 'table']]
        visual_chunks = [c for c in final_selection if c['type'] == 'visual']
        
        log.info(f"📊 Selection Strategy: Top Score {best_score:.2f} -> Threshold {threshold:.2f}")
        log.info(f"   Selected {len(text_chunks)} Text & {len(visual_chunks)} Visuals")
        
        return text_chunks, visual_chunks, final_selection

    async def ask(self, query: str) -> dict:
        start_time = time.time()
        log.info(f"🔎 Orchestrator processing: '{query}'")
        
        text_chunks, visual_chunks, all_evidence = self.retrieve_and_split(query)
        
        if not all_evidence:
            return {"answer": "I couldn't find any relevant information in the documents.", "evidence": []}

        # Dispatch Agents
        log.info(f"   ... ⚡ Dispatching Swarm: {len(text_chunks)} Text Docs + {len(visual_chunks)} Visual Docs")
        
        text_task = self.text_agent.analyze(query, text_chunks)
        vision_task = self.vision_agent.analyze(query, visual_chunks)
        
        text_insight, visual_insight = await asyncio.gather(text_task, vision_task)
        
        # Fusion
        log.info("   ... Fusing Insights")
        prompt = f"""
        You are the Fusion Agent. Combine the insights from the Text and Vision specialists.
        
        USER QUERY: "{query}"
        
        [TEXT SPECIALIST REPORT]:
        {text_insight}
        
        [VISION SPECIALIST REPORT]:
        {visual_insight}
        
        INSTRUCTIONS:
        1. Synthesize a single, natural answer.
        2. If the Vision Specialist found relevant charts/diagrams, EXPLICITLY reference them (e.g., "As seen in Figure on Page X...").
        3. If Visuals were analyzed but found irrelevant, ignore them.
        4. Cite sources as [Source 1, 2] based on the provided evidence list.
        """
        
        try:
            fusion_response = await self.fusion_model.generate_content_async(prompt)
            raw_answer = fusion_response.text
        except:
            raw_answer = "Error during fusion."

        # Validation
        log.info("   ... Validating Response")
        validation = await self.validator.validate(query, raw_answer, [c['content'] for c in all_evidence])
        
        final_answer = raw_answer
        if validation['confidence_score'] < 0.3:
            final_answer += f"\n\n*(Note: Low confidence. Critique: {validation['critique']})*"
            log.warning(f"⚠️ Validation Flag: {validation['critique']}")

        duration = time.time() - start_time
        log.info(f"✅ Request complete in {duration:.2f}s | Confidence: {validation.get('confidence_score', 0):.2f}")

        return {
            "answer": final_answer,
            "evidence": all_evidence
        }