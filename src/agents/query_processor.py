import google.generativeai as genai
from src.core.config import settings
from src.core.logger import setup_logger

log = setup_logger("query_processor")

class QueryProcessor:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def process(self, raw_query: str) -> dict:
        """
        Returns: {'valid': bool, 'clean_query': str, 'reason': str}
        """
        # 1. Hard Guardrails (Security)
        forbidden = ["ignore previous", "system prompt", "delete database", "drop table"]
        if any(bad in raw_query.lower() for bad in forbidden):
            log.warning(f"⛔ Blocked malicious query: {raw_query}")
            return {"valid": False, "clean_query": "", "reason": "I cannot process that request."}

        # 2. Query Enrichment (LLM)
        # UPGRADE: Now expands terms (e.g., "summary" -> "summary overview conclusion main points")
        prompt = f"""
        You are an expert Search Query Optimizer for a RAG system.
        
        GOAL: Rewrite the USER QUERY to maximize retrieval accuracy from a technical document.
        
        INSTRUCTIONS:
        1. **Expand, Don't Just Clean:** Keep the core meaning but add 2-3 relevant synonyms or related technical terms.
           - Example: "how does it work" -> "mechanism workflow process explanation"
           - Example: "who wrote it" -> "author creator signature writer"
        2. **Visuals:** If the user asks for "show me" or "chart", add: "visual chart diagram figure graph table".
        3. **No Conversational Filler:** Remove "hi", "please", "can you".
        4. **Output:** Return ONLY the optimized query string.

        USER QUERY: "{raw_query}"
        OPTIMIZED QUERY:
        """
        
        try:
            response = self.model.generate_content(prompt)
            clean_query = response.text.strip()
            log.info(f"✨ Optimized: '{raw_query}' -> '{clean_query}'")
            return {"valid": True, "clean_query": clean_query, "reason": "Success"}
        except Exception as e:
            log.error(f"⚠️ Optimization failed: {e}")
            return {"valid": True, "clean_query": raw_query, "reason": "Optimization failed, using raw."}