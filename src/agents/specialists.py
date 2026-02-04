import google.generativeai as genai
from src.core.config import settings
from src.core.logger import setup_logger

log = setup_logger("specialists")

class BaseAgent:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

class TextAgent(BaseAgent):
    async def analyze(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return "No text context available."
            
        context_text = "\n\n".join([c['content'] for c in context_chunks])
        
        prompt = f"""
        You are a Text Analysis Specialist. Your job is to extract factual information relevant to the user's query.
        
        USER QUERY: "{query}"
        
        CONTEXT:
        {context_text}
        
        INSTRUCTIONS:
        1. Extract numbers, dates, definitions, and key explanations.
        2. Ignore descriptions of charts/images; focus on the main body text.
        3. Be concise and precise.
        """
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            log.error(f"Text Agent Error: {e}")
            return "Error analyzing text."

class VisionAgent(BaseAgent):
    async def analyze(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return "No visual context available."
            
        # We search the TEXT DESCRIPTIONS of the visuals (since we don't re-upload images to save latency)
        context_text = "\n\n".join([
            f"[Visual Source]: {c['content']}" 
            for c in context_chunks
        ])
        
        prompt = f"""
        You are a Data Visualization Specialist. Your job is to interpret the descriptions of charts and diagrams found in a document search.
        
        USER QUERY: "{query}"
        
        VISUAL DESCRIPTIONS FOUND:
        {context_text}
        
        INSTRUCTIONS:
        1. Identify if any chart contains data relevant to the query.
        2. Mention specific trends, numbers, or labels found in the visual descriptions.
        3. If the description says "Analysis failed" or is irrelevant, ignore it.
        """
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            log.error(f"Vision Agent Error: {e}")
            return "Error analyzing visuals."

class ValidationAgent(BaseAgent):
    async def validate(self, query: str, answer: str, evidence: list) -> dict:
        prompt = f"""
        You are a Fact-Checking Auditor. Rate the accuracy of the Proposed Answer based ONLY on the Evidence.
        
        Query: "{query}"
        Proposed Answer: "{answer}"
        Evidence: {str(evidence)[:2000]}...
        
        Output JSON: {{ "score": 0.0 to 1.0, "reason": "Short explanation", "critique": "Any specific errors" }}
        """
        try:
            response = await self.model.generate_content_async(prompt)
            import json
            clean = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean)
        except:
            return {"score": 0.5, "reason": "Validation failed", "critique": "None"}