##QuickSight AI: Neuro-Symbolic Multi-Agent RAG

**QuickSight AI** is a local Retrieval-Augmented Generation (RAG) system designed to process complex PDF documents containing charts, diagrams, and tables. Unlike standard text-only RAG, this system employs a **Neuro-Symbolic architecture** that treats visual elements as first-class citizens, using Computer Vision (YOLO) and Multi-Agent Orchestration to answer questions based on both text and visual evidence.

<img width="500" height="280" alt="Screenshot 2026-02-05 at 5 41 09 AM" src="https://github.com/user-attachments/assets/21994d72-bb10-43eb-9784-2f35dedc01a8" />

---

## 🏗️ Architecture & Pipeline

The system operates in three distinct phases: Ingestion, Retrieval, and Generation.

<img width="700" height="320" alt="Gemini_Generated_Image_aw3e7faw3e7faw3e" src="https://github.com/user-attachments/assets/94927059-e8ea-4d2c-b4fe-acc337c36064" />

### 1. Ingestion Layer (The "Eyes")

The ingestion pipeline processes raw PDFs into a semantic index. It runs two parallel extraction tracks:

* **Visual Track (YOLOv8):**
* **Model:** `yolov8n-doclaynet.pt` (Fine-tuned on DocLayNet).
* **Process:** Scans every page as an image to detect bounding boxes for Charts, Figures, and Tables. These elements are cropped, saved, and passed to a Vision Model (Gemini 2.0 Flash) to generate dense textual descriptions.


* **Structural Track (Docling):**
* **Tool:** `Docling` (IBM).
* **Process:** Parses the PDF's internal structure to extract hierarchical text, preserving headers and list formatting.


* **Indexing:**
* Text chunks and Visual descriptions are embedded using `all-MiniLM-L6-v2`.
* Data is stored in **ChromaDB** (Persistent Vector Store).



### 2. Retrieval Layer (The "Filter")

<img width="700" height="320" alt="Gemini_Generated_Image_88mota88mota88mo" src="https://github.com/user-attachments/assets/8293dd38-270d-45ef-930a-0efdf0ea5fe2" />

When a user asks a question, the system performs a multi-stage search:

* **Hybrid Search:** Retrieves the top 50 semantic matches from ChromaDB.
* **Re-Ranking:** A Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) re-scores every candidate pair `(Query, Document)` to filter out irrelevant matches with high precision.
* **Strict Filtering:** The system applies a quota logic (e.g., "Select top 3 Text chunks + Top 1 Visual chunk") to ensure the LLM receives multi-modal context.

### 3. Generation Layer (The "Swarm")

The query and selected evidence are passed to an asynchronous Multi-Agent System:

* **Orchestrator:** Manages the lifecycle of the request and handles rate limiting.
* **Text Agent:** Specialized prompt to extract hard facts/numbers from text chunks.
* **Vision Agent:** Specialized prompt to interpret trends and data points from the retrieved charts.
* **Fusion Agent:** Synthesizes the outputs from the Text and Vision agents into a final answer, citing sources (e.g., `[Source 1]`) and formatting the output in Markdown.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| --- | --- | --- |
| **LLM** | Google Gemini 2.0 Flash | Reasoning, Vision, & Synthesis |
| **Object Detection** | YOLOv8 (DocLayNet) | Identifying Charts/Tables in PDFs |
| **OCR & Parsing** | Docling | Text & Structure Extraction |
| **Vector Database** | ChromaDB | Local Persistence of Embeddings |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` |
| **Re-Ranker** | Cross-Encoder | `ms-marco-MiniLM-L-6-v2` |
| **Backend/UI** | Streamlit | Interface & Application Logic |
| **Containerization** | Docker | Deployment & Dependency Management |

---

## 🚀 Setup & Installation

### Prerequisites

1. **Docker Desktop** installed and running.
2. A **Google Gemini API Key** (Get it [here](https://aistudio.google.com/app/apikey)).

### Installation (Docker Method)

This is the recommended method as it handles all system dependencies (OCR, Vision libraries) automatically.

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/quicksight-ai.git
cd quicksight-ai
```

2. **Configure Environment:**
Create a `.env` file in the root directory:
```bash
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

3. **Build and Run:**
```bash
docker-compose up --build
```

*Note: The first build may take a few minutes to download the YOLO model and Python dependencies.*

4. **Access the App:**
Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 📖 Usage Guide

1. **Upload:**
* Use the sidebar to upload a PDF file (Text-based or Scanned).


2. **Ingestion:**
* Click the **"🚀 Process Document"** button.
* Monitor the logs in your terminal. You will see YOLO detecting objects (`Found 3 tables, 2 charts...`) and ChromaDB building the index.


3. **Query:**
* Type natural language questions into the chat interface.
* *Example:* "What are the sales trends shown in the chart on page 5?"
* *Example:* "Summarize the author's conclusion."


4. **Verification:**
* Expand the **"🔍 View Verified Sources"** dropdown under the answer to see exactly which text block or image crop was used.



---

## 📂 Project Structure

```text
.
├── Dockerfile              # Container definition (Install System Libs + Python)
├── docker-compose.yml      # Service config & Volume persistence
├── src/
│   ├── main.py             # Pipeline Entrypoint (Ingestion Logic)
│   ├── agents/
│   │   ├── orchestrator.py # Async Manager & Reranking Logic
│   │   ├── specialists.py  # Individual Agent Prompts (Text/Vision)
│   ├── core/
│   │   ├── config.py       # Configuration Settings
│   │   └── models.py       # Pydantic Data Models
│   ├── ingestion/
│   │   ├── docling_wrapper.py # Text Parser
│   │   └── visual_auditor.py  # YOLO Vision Pipeline
│   └── ui/
│       └── app.py          # Streamlit Frontend
└── data/                   # (Auto-generated) Local storage for DB & PDFs
```

---

## 🔧 Troubleshooting

**1. "Disk Full" or "No space left on device"**
Docker images for AI (PyTorch + OCR) are large.

* **Fix:** Run `docker system prune -a` to clear old cache, or increase Docker's disk limit in settings.

**2. "UnpicklingError: could not find MARK"**
The YOLO model file is corrupted due to an interrupted download.

* **Fix:** Force a clean rebuild:
```bash
docker-compose build --no-cache
docker-compose up
```

**3. "Port already in use"**
Another container is using port 8501.

* **Fix:** Stop all containers with `docker-compose down`, then start again.
