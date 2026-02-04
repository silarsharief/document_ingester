# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies
# Added 'build-essential' again because some libraries in your TOML (like pymupdf) might need to compile C++ extensions.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Pre-Download YOLO Model (HuggingFace URL; validate size so bad download fails build)
RUN curl -L "https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8n-doclaynet.pt" -o yolov8n-doclaynet.pt \
    && if [ $(stat -c%s "yolov8n-doclaynet.pt") -lt 1000000 ]; then \
        echo "ERROR: Model download failed (file too small). Check internet."; \
        rm -f yolov8n-doclaynet.pt; \
        exit 1; \
    else \
        echo "Model downloaded successfully."; \
    fi

# 3. DIRECT PIP INSTALL (The Hybrid List)
# This includes everything from your TOML + the missing requirements for the code.
RUN pip install --no-cache-dir \
    # --- Core App ---
    streamlit \
    watchdog \
    python-dotenv \
    rich \

    # --- Ingestion & Parsing (From TOML) ---
    docling \
    pymupdf \
    pypdfium2 \
    pandas \

    # --- Computer Vision (From TOML) ---
    ultralytics \
    opencv-python-headless \
    pillow \
    timm \
    einops \

    # --- AI & Reasoning ---
    google-generativeai \
    sentence-transformers \

    # --- Database (CRITICAL FIX) ---
    # We install ChromaDB because your code uses it, even though TOML says Qdrant.
    chromadb \

    # --- Reporting ---
    reportlab

# 4. Copy Code
COPY src/ src/
COPY src/main.py .

# 5. Setup Data Folder
RUN mkdir -p data

# 6. Expose Port
EXPOSE 8501

# 7. Run Command
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]