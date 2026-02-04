# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (GL and Tesseract are CRITICAL for Vision/OCR)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 3. Copy Dependency Definitions
COPY pyproject.toml poetry.lock ./

# 4. Install Python Dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# --- 5. PRE-DOWNLOAD YOLO MODEL (The Fix) ---
# We download it now so it is BAKED into the image. 
# The user never has to wait for this download at runtime.
RUN curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-doclaynet.pt -o yolov8n-doclaynet.pt

# 6. Copy Project Code
COPY src/ src/
COPY src/main.py . 

# 7. Setup Data Directory
RUN mkdir -p data

# 8. Expose Streamlit Port
EXPOSE 8501

# 9. Run Application
CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
