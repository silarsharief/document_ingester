import sys
import os

# 1. Force Python to find the 'src' module
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# 2. Import the robust settings we wrote earlier
# This uses the logic in src/core/config.py to find the .env file correctly
try:
    from core.config import settings
    print("\n✅ SUCCESS! Pydantic found the config.")
    print(f"🔑 API Key: {settings.GOOGLE_API_KEY[:5]}...[Hidden]")
    print(f"📂 Project Root: {settings.BASE_DIR}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    # Debugging helper
    print(f"Current Working Directory: {os.getcwd()}")
    print("Listing files in current folder:")
    print(os.listdir("."))