import google.generativeai as genai
from PIL import Image
from src.core.config import settings
from src.core.utils import generate_content_with_retry
import time

class VisionAgent:
    def __init__(self):
        print("🧠 Initializing Gemini Vision Agent (Context-Aware)...")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_element(self, image_path: str, element_type: str, page_context: str = "") -> str:
        if not image_path:
            return ""
            
        img = Image.open(image_path)
        
        prompt = f"""
        You are a Technical Data Analyst. 
        
        TASK:
        Analyze the provided {element_type} image.
        
        CONTEXT SURROUNDING THIS IMAGE (Text Before/After):
        \"\"\"{page_context}\"\"\"
        
        INSTRUCTIONS:
        1. Identify what this {element_type} represents based on the text context.
        2. Extract key data into a strict Markdown table.
        3. Summarize the insight.
        4. If it is a Logo, blank space, or decorative image, REPLY ONLY with: "irrelevant".
        """
        
        try:
            # Use the robust retry wrapper
            response = generate_content_with_retry(self.model, [prompt, img])
            return response.text
        except Exception as e:
            return f"❌ Analysis Failed (Max Retries Exceeded): {str(e)}"

vision_agent = VisionAgent()