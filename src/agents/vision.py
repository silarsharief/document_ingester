import google.generativeai as genai
import json
from PIL import Image
from src.core.config import settings
from src.core.utils import generate_content_with_retry

class VisionAgent:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_element(self, image_path: str, element_type: str, page_context: str = "") -> dict:
        if not image_path: return {"error": "No image path"}
        img = Image.open(image_path)
        
        prompt = f"""
        You are a Senior Data Analyst. Analyze this document element.
        
        CONTEXT SURROUNDING THIS IMAGE:
        "{page_context}"
        
        INSTRUCTIONS:
        1. **Identify**: Is this a Chart, Table, Diagram, or a Logo/Header?
        2. **Logos/Headers**: State what it represents.
        3. **Charts/Graphs**: Extract data and trends.
        4. **Confidence**: Rate your confidence (0.0 - 1.0) based on image clarity and ambiguity.
        
        OUTPUT FORMAT (JSON):
        {{
            "heading": "Short Title",
            "content": {{
                "overview": "Description...",
                "key_findings": ["Finding 1", "Finding 2"],
                "extracted_data": {{ "Metric": "Value" }}
            }},
            "confidence_score": 0.95,
            "confidence_reason": "Image is clear and legends are readable."
        }}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = generate_content_with_retry(self.model, [prompt, img])
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception:
            return {
                "heading": "Analysis Failed",
                "content": {"overview": "Could not analyze.", "key_findings": []},
                "confidence_score": 0.0,
                "confidence_reason": "Model error or invalid image."
            }

vision_agent = VisionAgent()