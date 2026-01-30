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
        You are a Senior Data Analyst. Analyze this {element_type}.
        
        CONTEXT SURROUNDING THIS IMAGE:
        "{page_context}"
        
        INSTRUCTIONS:
        1. Contextualize: Use the provided text to understand *why* this figure exists.
        2. Interpret: Don't just read numbers. Explain the *trend*, *gap*, or *significance* (e.g., "Mistral outperforms Llama 2 significantly in Reasoning").
        3. Extract: Get the raw data into a clean structure.
        
        OUTPUT FORMAT (JSON):
        {{
            "heading": "Insightful Title (e.g., 'Performance Gap in Reasoning Tasks')",
            "content": {{
                "overview": "A detailed analytical summary. Explain what the data implies for the model's capabilities.",
                "key_findings": [
                    "Insight 1 (e.g., 'Model A is 2x faster than B')",
                    "Insight 2 (e.g., 'Accuracy drops as batch size increases')"
                ],
                "extracted_data": {{
                    // Structured data for charts/tables.
                    // Keep it clean: "Metric": "Value"
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
                "heading": "Analysis Error",
                "content": {
                    "overview": "Could not analyze image.",
                    "key_findings": [],
                    "extracted_data": {}
                }
            }

vision_agent = VisionAgent()