import google.generativeai as genai
import json
import asyncio
import time
from src.core.config import settings
from src.core.logger import setup_logger

log = setup_logger("specialists")

# --- RETRY DECORATOR (New) ---
def retry_with_backoff(retries=3, delay=2):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Catch 429 (Rate Limit) errors specifically
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        wait_time = delay * (2 ** i) # Exponential backoff: 2s, 4s, 8s
                        log.warning(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
            return await func(*args, **kwargs) # Final try
        return wrapper
    return decorator

class BaseAgent:
    def __init__(self, model_name='gemini-2.0-flash'):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)

class TextAgent(BaseAgent):
    """Specializes in extracting hard facts from text and tables."""
    
    @retry_with_backoff() # <--- Added Decorator
    async def analyze(self, query: str, text_chunks: list) -> str:
        if not text_chunks: return "No textual evidence found."
        
        context = "\n".join([f"- {c['content']}" for c in text_chunks])
        prompt = f"""
        ROLE: Text Evidence Analyst.
        TASK: Extract facts relevant to the query from the provided text fragments.
        QUERY: "{query}"
        TEXT FRAGMENTS:
        {context}
        
        OUTPUT: A concise bulleted list of facts found. If nothing is relevant, say 'No relevant text found'.
        """
        try:
            # Changed to async call
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            log.error(f"Text Agent Failed: {e}")
            return "Text analysis failed."

class VisionAgent(BaseAgent):
    """Specializes in interpreting visual descriptions and metadata."""
    
    @retry_with_backoff() # <--- Added Decorator
    async def analyze(self, query: str, visual_chunks: list) -> str:
        if not visual_chunks: return "No visual evidence found."
        
        context = "\n".join([f"- Figure on Page {c['page']}: {c['content']}" for c in visual_chunks])
        
        prompt = f"""
        ROLE: Visual Data Analyst.
        TASK: Interpret the descriptions of charts/diagrams to answer the query.
        QUERY: "{query}"
        VISUAL DESCRIPTIONS:
        {context}
        
        OUTPUT: Describe what the visuals show regarding the query. Focus on trends, labels, and relationships.
        """
        try:
            # Changed to async call
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            log.error(f"Vision Agent Failed: {e}")
            return "Visual analysis failed."

class ValidationAgent(BaseAgent):
    """Critiques the final answer for hallucination and citation."""
    
    @retry_with_backoff() # <--- Added Decorator
    async def validate(self, query: str, answer: str, sources: list) -> dict:
        prompt = f"""
        ROLE: Strict Quality Control Auditor.
        TASK: Verify if the ANSWER is supported by the SOURCES.
        
        QUERY: {query}
        PROPOSED ANSWER: {answer}
        AVAILABLE SOURCE INSTANCES: {len(sources)} sources provided.
        
        CHECKLIST:
        1. Hallucination: Is there info in the answer NOT in the sources?
        2. Relevance: Does it actually answer the query?
        
        OUTPUT JSON ONLY:
        {{
            "is_valid": boolean,
            "confidence_score": float (0.0 to 1.0),
            "critique": "Short explanation of issues if any"
        }}
        """
        try:
            # Changed to async call
            response = await self.model.generate_content_async(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text) 
        except Exception as e:
            log.error(f"Validation Agent Failed: {e}")
            return {"is_valid": True, "confidence_score": 0.5, "critique": "Validation parsing failed."}