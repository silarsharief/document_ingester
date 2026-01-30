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
        2. **Logos/Headers**: If it is a company logo or graphical header, simply state what it says and represents. Do not hallucinate data.
        3. **Charts/Graphs**: Extract the real data and trends.
        
        OUTPUT FORMAT (JSON):
        {{
            "heading": "Short Title (e.g., 'Mistral AI Logo' or 'Performance Benchmark')",
            "content": {{
                "overview": "Clear description of the visual.",
                "key_findings": [
                    "Insight 1 (or 'Company branding detected' for logos)"
                ],
                "extracted_data": {{
                    // Only for charts/tables. Leave empty for logos.
                }}
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = generate_content_with_retry(self.model, [prompt, img])
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception:
            return {
                "heading": "Visual Element",
                "content": {
                    "overview": "Content could not be analyzed.",
                    "key_findings": [],
                    "extracted_data": {}
                }
            }

vision_agent = VisionAgent()