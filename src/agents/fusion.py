import google.generativeai as genai
from src.core.config import settings
from src.core.utils import generate_content_with_retry

class FusionAgent:
    def __init__(self):
        print("🔗 Initializing Fusion Agent (Strict Layout Repair)...")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def fuse_page(self, page_num: int, text_content: str, visual_elements: list) -> str:
        # If we have nothing, return nothing
        if not visual_elements and not text_content.strip():
            return ""

        visual_block = ""
        for item in visual_elements:
            visual_block += f"\n>>> INSERT FIGURE ({item['type'].upper()}): \n{item['analysis']}\n"

        # If only text, return text
        if not visual_elements:
            return f"## Page {page_num}\n\n{text_content}"

        prompt = f"""
        You are a Document Layout Engineer.
        
        INPUT:
        1. RAW TEXT (Page {page_num})
        2. VISUAL DATA (Charts/Tables)

        TASK:
        Merge the text and visuals into a single clean Markdown stream.
        
        RULES:
        1. DO NOT chat (e.g., "Okay, I will do this"). Output ONLY Markdown.
        2. Insert visuals where they are semantically relevant.
        3. Repair any broken text columns into a single readable flow.
        
        --- RAW TEXT ---
        {text_content}
        
        --- VISUAL DATA ---
        {visual_block}
        """
        
        try:
            response = generate_content_with_retry(self.model, prompt)
            return response.text
        except Exception as e:
            print(f"   ⚠️ Fusion LLM Error: {e}. Falling back to append mode.")
            return f"## Page {page_num}\n\n{text_content}\n\n### Visual Data\n{visual_block}"

fusion_agent = FusionAgent()