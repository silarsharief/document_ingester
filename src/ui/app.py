import sys
import os
import shutil
import datetime
import streamlit as st

# --- 1. FORCE PATH FIX (Critical for imports) ---
# Adds the project root to python path so src.main works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.main import AnalysisPipeline
from src.agents.rag_chat import RAGAgent
from src.agents.query_processor import QueryProcessor
from src.build_index import main as build_index

# --- 2. CONFIGURATION ---
st.set_page_config(
    page_title="QuickSight AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. STYLING ---
st.markdown("""
    <style>
    /* Chat Message Spacing */
    .stChatMessage { padding-top: 10px; padding-bottom: 10px; }
    
    /* Evidence Card Styling inside Expander */
    .evidence-item {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #007bff; /* Blue accent line */
    }
    .evidence-meta {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .score-badge {
        background-color: #e2e6ea;
        color: #212529;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .score-high { color: #008000; font-weight: bold; }
    .score-med { color: #ffa500; font-weight: bold; }
    
    .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "agent" not in st.session_state: st.session_state.agent = None
if "processor" not in st.session_state: st.session_state.processor = None

# --- 5. TOP BAR (HEADER & STATUS) ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("👁️ QuickSight AI")
    st.caption("Precision Multi-Modal RAG v3.0")

with col_status:
    # Status Indicators at the Top (Restored)
    db_exists = os.path.exists("data/chroma_db")
    agent_active = st.session_state.agent is not None
    
    if db_exists:
        st.markdown('<span class="status-badge status-ready">● Brain Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-off">● Brain Empty</span>', unsafe_allow_html=True)
        
    if agent_active:
        st.markdown('<span class="status-badge status-ready">● Agent Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-off">● Agent Offline</span>', unsafe_allow_html=True)

st.markdown("---")

# --- 6. SIDEBAR: CONTROLS & DATA ---
with st.sidebar:
    st.subheader("📄 Document Actions")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("🚀 Process Document", type="primary"):
            # CLEANUP
            if os.path.exists("data/chroma_db"): shutil.rmtree("data/chroma_db")
            if os.path.exists("data/rag_dataset.json"): os.remove("data/rag_dataset.json")
            
            with st.status("Ingesting & Indexing...", expanded=True) as status:
                save_path = f"data/input/{uploaded_file.name}"
                os.makedirs("data/input", exist_ok=True)
                with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                st.write("🔹 1. Vision & OCR Extraction (Docling)...")
                pipeline = AnalysisPipeline()
                pipeline.run(save_path)
                
                st.write("🔹 2. Building Semantic Index (ChromaDB)...")
                build_index()
                
                st.write("🔹 3. Loading Neural Agents (Gemini + Cross-Encoder)...")
                st.session_state.agent = RAGAgent()
                st.session_state.processor = QueryProcessor()
                
                status.update(label="System Ready!", state="complete", expanded=False)
            
            st.success("Processing Complete!")
            st.rerun()

    # --- DATA FOLDER SECTION (Restored) ---
    st.divider()
    st.header("📂 Data Folder")
    
    has_data = False
    
    if os.path.exists("full_analysis.pdf"):
        has_data = True
        with open("full_analysis.pdf", "rb") as f:
            st.download_button("📄 Download Report (PDF)", f, "analysis_report.pdf")
            
    if os.path.exists("data/rag_dataset.json"):
        has_data = True
        with open("data/rag_dataset.json", "rb") as f:
            st.download_button("💾 Download Data (JSON)", f, "extracted_data.json")
            
    if not has_data:
        st.caption("No processed data found.")

# --- 7. HELPER: EVIDENCE RENDERER ---
def render_evidence_ui(evidence_list):
    """
    Renders evidence in a collapsible bar below the message.
    """
    if not evidence_list:
        return

    # The Collapsible Bar
    with st.expander(f"🔍 View Verified Sources ({len(evidence_list)})", expanded=False):
        for i, item in enumerate(evidence_list):
            score = item.get('score', 0.0)
            page = item.get('page', '?')
            dtype = item.get('type', 'text').upper()
            content = item.get('content', '')
            image_path = item.get('image_path')
            
            # Score coloring
            score_class = "score-high" if score > 0 else "score-med"
            
            # Evidence Card HTML
            st.markdown(f"""
            <div class="evidence-item">
                <div class="evidence-meta">
                    <span><strong>Source {i+1}</strong> • Page {page} • {dtype}</span>
                    <span class="{score_class}">Relevance: {score:.3f}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Content Logic: Visual vs Text
            if dtype == 'VISUAL' and image_path:
                if os.path.exists(image_path):
                    # FIX: Use width="stretch" for images
                    st.image(image_path, caption=f"Figure extracted from Page {page}", width="stretch") 
                else:
                    st.warning(f"⚠️ Image file missing: {image_path}")
            else:
                # Text Content (Truncated for clean UI)
                display_text = content if len(content) < 500 else content[:500] + "..."
                st.info(f'"{display_text}"')
            
            st.markdown("</div>", unsafe_allow_html=True)

# --- 8. MAIN CHAT INTERFACE ---
chat_container = st.container()

with chat_container:
    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If this message has attached evidence, render it
            if "evidence" in msg and msg["evidence"]:
                render_evidence_ui(msg["evidence"])

# --- 9. INPUT HANDLING ---
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.agent:
        st.error("⚠️ Please process a document first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            
            # 1. Guardrails & Optimization
            processed_query = st.session_state.processor.process(prompt)
            
            if not processed_query['valid']:
                st.error(f"🛑 {processed_query['reason']}")
                st.stop()
            
            # 2. Retrieval & Synthesis
            response = st.session_state.agent.ask(processed_query['clean_query'])
            
            # 3. Display Answer
            st.markdown(response["answer"])
            
            # 4. Display Evidence (Collapsible)
            render_evidence_ui(response["evidence"])
            
            # 5. Save
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "evidence": response["evidence"]
            })