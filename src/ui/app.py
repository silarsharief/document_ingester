import sys
import os
import shutil
import datetime
import asyncio
import streamlit as st

# --- 1. FORCE PATH FIX ---
# Adds the project root to python path so src.main works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.main import AnalysisPipeline
from src.agents.orchestrator import MultiAgentOrchestrator 
from src.agents.query_processor import QueryProcessor
from src.build_index import main as build_index
from src.indexing.vector_store import LocalVectorStore # Needed for persistent store

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
    
    /* Status Badges */
    .status-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-right: 5px;
    }
    .status-ready { background-color: #d4edda; color: #155724; }
    .status-off { background-color: #f8d7da; color: #721c24; }
    
    /* Score Badges */
    .score-high { color: #008000; font-weight: bold; }
    .score-med { color: #ffa500; font-weight: bold; }
    
    /* Button Width */
    .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
# REPLACED 'agent' with 'vector_store' as the persistent object to fix Event Loop issues
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "processor" not in st.session_state: st.session_state.processor = None

# --- 5. TOP BAR (HEADER & STATUS) ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("👁️ QuickSight AI")
    st.caption("Neuro-Symbolic Multi-Agent Swarm v4.0") 

with col_status:
    # Status Indicators
    db_exists = os.path.exists("data/chroma_db")
    # Check if Vector Store is loaded in memory
    is_ready = st.session_state.vector_store is not None
    
    if db_exists:
        st.markdown('<span class="status-badge status-ready">● DB Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-off">● DB Offline</span>', unsafe_allow_html=True)
        
    if is_ready:
        st.markdown('<span class="status-badge status-ready">● Brain Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-off">● Brain Idle</span>', unsafe_allow_html=True)

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
            
            # --- ERROR HANDLING WRAPPER ---
            status_container = st.status("Ingesting & Indexing...", expanded=True)
            try:
                save_path = f"data/input/{uploaded_file.name}"
                os.makedirs("data/input", exist_ok=True)
                with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                st.write("🔹 1. Vision & OCR Extraction (Docling)...")
                pipeline = AnalysisPipeline()
                pipeline.run(save_path)
                
                st.write("🔹 2. Building Semantic Index (ChromaDB)...")
                build_index()
                
                st.write("🔹 3. Loading Neural Brain...")
                # Load Vector Store into Session State (Persistent)
                st.session_state.vector_store = LocalVectorStore()
                st.session_state.processor = QueryProcessor()
                
                status_container.update(label="System Ready!", state="complete", expanded=False)
                st.success("Processing Complete!")
                st.rerun()

            except Exception as e:
                # CATCH CRITICAL FAILURES AND SHOW UI ERROR
                status_container.update(label="Processing Failed!", state="error")
                st.error(f"❌ Critical Error: {str(e)}")
                st.markdown("Check the terminal logs for detailed traceback.")

    # --- DATA FOLDER SECTION ---
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
    Smart Renderer: Decides whether to show an Image or Text based on metadata.
    """
    if not evidence_list:
        return

    # The Collapsible Bar
    with st.expander(f"🔍 View Verified Sources ({len(evidence_list)})", expanded=False):
        for i, item in enumerate(evidence_list):
            # Extract Metadata
            score = item.get('score', 0.0)
            page = item.get('page', '?')
            dtype = item.get('type', 'text').upper()
            content = item.get('content', '') # This is the searchable text (Description/OCR)
            image_path = item.get('image_path') # This comes from our metadata map
            
            # Score coloring
            score_class = "score-high" if score > 0 else "score-med"
            
            # 1. Header Card
            st.markdown(f"""
            <div class="evidence-item">
                <div class="evidence-meta">
                    <span><strong>Source {i+1}</strong> • Page {page} • {dtype}</span>
                    <span class="{score_class}">Relevance: {score:.3f}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # 2. Smart Content Display
            # If it's a Visual Chunk AND we have the file, show the Image
            if dtype == 'VISUAL' and image_path and os.path.exists(image_path):
                st.image(image_path, caption=f"Figure extracted from Page {page}", width="stretch")
                # Optional: Show the description used for search in a small font
                with st.popover("See Search Context"):
                    st.caption(content)
            
            # Otherwise, show the Text
            else:
                # Truncate very long text for UI cleanliness
                display_text = content if len(content) < 600 else content[:600] + "..."
                st.info(f'"{display_text}"')
            
            st.markdown("</div>", unsafe_allow_html=True)

# --- 8. HELPER: ASYNC SWARM RUNNER ---
async def run_swarm_query(query, vector_store):
    """
    Creates a fresh Orchestrator instance inside the active event loop.
    This prevents 'Event loop is closed' errors.
    """
    # Instantiate agents FRESH every request so they bind to the CURRENT loop
    orchestrator = MultiAgentOrchestrator(vector_store=vector_store)
    return await orchestrator.ask(query)

# --- 9. MAIN CHAT INTERFACE ---
chat_container = st.container()

with chat_container:
    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If this message has attached evidence, render it
            if "evidence" in msg and msg["evidence"]:
                render_evidence_ui(msg["evidence"])

# --- 10. INPUT HANDLING ---
if prompt := st.chat_input("Ask a question about the document..."):
    # Check if Vector Store is loaded (means processing happened)
    if not st.session_state.vector_store:
        # Try to load if db exists but not in memory
        if os.path.exists("data/chroma_db"):
             st.session_state.vector_store = LocalVectorStore()
             st.session_state.processor = QueryProcessor()
        else:
            st.error("⚠️ Please process a document first!")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚡ Swarm Processing (Async): Text & Vision..."):
            
            # 1. Guardrails & Optimization
            processed_query = st.session_state.processor.process(prompt)
            if not processed_query['valid']:
                st.error(f"🛑 {processed_query['reason']}")
                st.stop()
            
            try:
                # RUN ASYNC SWARM SAFELY
                response = asyncio.run(run_swarm_query(
                    processed_query['clean_query'], 
                    st.session_state.vector_store
                ))
            except Exception as e:
                st.error(f"System Error: {e}")
                # Log to console for debugging
                print(f"ERROR DETAILS: {e}")
                st.stop()
            
            # 3. Display Answer
            st.markdown(response["answer"])
            
            # 4. Display Evidence (Collapsible)
            render_evidence_ui(response["evidence"])
            
            # 5. Save Assistant Message + Evidence to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "evidence": response["evidence"]
            })