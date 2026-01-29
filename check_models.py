import google.generativeai as genai
from src.core.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)

print("🔍 Listing available Gemini models for your API key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   ✅ {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")