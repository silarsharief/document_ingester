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
        You are a Data Extraction Specialist. 
        Analyze this {element_type} image.
        
        CONTEXT: 
        {page_context[:500]}
        
        INSTRUCTIONS:
        1. Identify the core subject.
        2. If it is a graph/chart/table: Extract the raw numbers into a structured format.
        3. If it is a diagram/image: List the key components visible.
        4. Do NOT write paragraphs. Use lists and short sentences.
        
        OUTPUT FORMAT (JSON):
        {{
            "heading": "Short Title (3-5 words)",
            "content": {{
                "overview": "A single, concise sentence explaining what this is.",
                "key_findings": [
                    "Bullet point 1 (Insight or observation)",
                    "Bullet point 2"
                ],
                "extracted_data": {{
                    // IF CHART/TABLE: Extract X/Y values, Row/Cols here.
                    // IF DIAGRAM: Map labels to descriptions.
                    // Example: "Mistral": "60%", "Llama": "55%"
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
                "heading": "Processing Error",
                "content": {
                    "overview": "Could not analyze image.",
                    "key_findings": [],
                    "extracted_data": {}
                }
            }

vision_agent = VisionAgent()