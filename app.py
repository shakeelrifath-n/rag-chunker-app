import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from utils import RAGEvaluator as RealTimeRAGEvaluator


# Page configuration
st.set_page_config(
    page_title="Real-Time RAG Processing System",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'rt_evaluator' not in st.session_state:
    st.session_state.rt_evaluator = RealTimeRAGEvaluator()
if 'processed_methods' not in st.session_state:
    st.session_state.processed_methods = []

# Title and description
st.title("⚡ Real-Time RAG Processing System")
st.markdown("Upload documents and watch real-time chunking, embedding generation, and performance analysis")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Real-Time Processing", "🔍 Live Retrieval", "📊 Performance Analytics", "⚙️ System Monitoring"])

with tab1:
    st.header("📄 Real-Time Document Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload a text document", type=['txt'], key="realtime_upload")
        
        # Manual text input
        st.markdown("**Or paste text directly:**")
        manual_text = st.text_area("Paste your text here:", height=200, key="manual_text")
        
        # Processing parameters
        st.subheader("⚙️ Processing Parameters")
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            chunk_size = st.slider("Chunk Size (characters):", 200, 1000, 500)
        with col_param2:
            chunk_overlap = st.slider("Chunk Overlap (characters):", 0, 200, 50)
    
    with col2:
        st.subheader("📈 Live Statistics")
        stats_placeholder = st.empty()
        
        # Processing status
        st.subheader("🔄 Processing Status")
        status_placeholder = st.empty()
    
    # Get text content
    text_content = ""
    if uploaded_file is not None:
        text_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        st.success(f"📄 File uploaded: {len(text_content)} characters")
    elif manual_text.strip():
        text_content = manual_text.strip()
        st.info(f"📝 Manual text: {len(text_content)} characters")
    
    # Processing buttons
    if text_content:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔧 Process with Fixed Size Chunking", key="process_fixed"):
                process_realtime("Fixed Size", text_content, chunk_size, chunk_overlap, stats_placeholder, status_placeholder)
        
        with col_btn2:
            if st.button("🌲 Process with Recursive Chunking", key="process_recursive"):
                process_realtime("Recursive", text_content, chunk_size, chunk_overlap, stats_placeholder, status_placeholder)

def process_realtime(method_name, text_content, chunk_size, chunk_overlap, stats_placeholder, status_placeholder):
    """Process document in real-time with live updates"""
    progress_bar = st.progress(0)
    
    # Process with real-time updates
    for update in st.session_state.rt_evaluator.process_document_realtime(
        text_content, method_name, chunk_size, chunk_overlap
    ):
        # Update progress bar
        progress_bar.progress(int(update["progress"]))
        
        # Update status
        status_placeholder.info(f"**{update['step'].title()}**: {update['status']}")
        
        # Update statistics if available
        if "data" in update:
            if update["step"] == "analysis":
                stats_placeholder.metric("Document Analysis", 
                                        f"{update['data']['total_words']} words, {update['data']['estimated_chunks']} est. chunks")
            elif update["step"] == "chunking":
                stats_placeholder.metric("Chunks Created", len(update["data"]))
        
        # Update final stats
        if "stats" in update:
            stats_df = pd.DataFrame([update["stats"]]).T
            stats_df.columns = ["Value"]
            stats_placeholder.dataframe(stats_df)
        
        time.sleep(0.1)  # Small delay for visual effect
    
    # Add to processed methods
    if method_name not in st.session_state.processed_methods:
        st.session_state.processed_methods.append(method_name)
    
    st.success(f"✅ {method_name} processing completed!")

with tab2:
    st.header("🔍 Real-Time Retrieval Testing")
    
    if st.session_state.processed_methods:
        query = st.text_input("Enter your query:", value="What is the main topic discussed?")
        k_retrieve = st.slider("Number of chunks to retrieve:", 1, 5, 3)
        method_select = st.selectbox("Select processing method:", st.session_state.processed_methods)
        
        if st.button("🚀 Retrieve in Real-Time"):
            start_retrieval = time.time()
            
            with st.spinner("Searching through vectors..."):
                results = st.session_state.rt_evaluator.retrieve_chunks_realtime(
                    query, method_select, k_retrieve
                )
            
            end_retrieval = time.time()
            
            # Display timing
            col_timing1, col_timing2, col_timing3 = st.columns(3)
            col_timing1.metric("Retrieval Time", f"{results['retrieval_time']:.3f}s")
            col_timing2.metric("Total Time", f"{end_retrieval - start_retrieval:.3f}s")
            col_timing3.metric("Chunks Found", len(results['chunks']))
            
            # Display results
            st.subheader("📋 Retrieved Chunks")
            for i, (chunk, score) in enumerate(zip(results['chunks'], results['similarity_scores'])):
                with st.expander(f"Chunk {i+1} (Similarity: {score:.3f})"):
                    st.write(chunk)
    else:
        st.info("👆 Please process a document in the 'Real-Time Processing' tab first")

with tab3:
    st.header("📊 Real-Time Performance Analytics")
    
    if len(st.session_state.processed_methods) >= 2:
        comparison = st.session_state.rt_evaluator.get_processing_comparison()
        
        if comparison:
            # Processing time comparison
            fig_time = go.Figure(data=[
                go.Bar(name=method, x=['Processing Time'], y=[comparison['total_time'][method]])
                for method in comparison['total_time'].keys()
            ])
            fig_time.update_layout(title="Processing Time Comparison (seconds)")
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Chunks created comparison
            fig_chunks = go.Figure(data=[
                go.Bar(name=method, x=['Chunks Created'], y=[comparison['chunks_created'][method]])
                for method in comparison['chunks_created'].keys()
            ])
            fig_chunks.update_layout(title="Chunks Created Comparison")
            st.plotly_chart(fig_chunks, use_container_width=True)
            
            # Processing speed comparison
            fig_speed = go.Figure(data=[
                go.Bar(name=method, x=['Processing Speed'], y=[comparison['processing_speed'][method]])
                for method in comparison['processing_speed'].keys()
            ])
            fig_speed.update_layout(title="Processing Speed (chunks/second)")
            st.plotly_chart(fig_speed, use_container_width=True)
    else:
        st.info("Process documents with both chunking methods to see performance comparison")

with tab4:
    st.header("⚙️ System Monitoring")
    
    # Real-time system stats
    col_sys1, col_sys2, col_sys3 = st.columns(3)
    
    with col_sys1:
        st.metric("Active Methods", len(st.session_state.processed_methods))
    
    with col_sys2:
        total_chunks = sum(
            len(st.session_state.rt_evaluator.chunks_data.get(method, []))
            for method in st.session_state.processed_methods
        )
        st.metric("Total Chunks", total_chunks)
    
    with col_sys3:
        total_vectors = sum(
            st.session_state.rt_evaluator.vector_stores[method].ntotal
            for method in st.session_state.processed_methods
            if method in st.session_state.rt_evaluator.vector_stores
        )
        st.metric("Vectors Indexed", total_vectors)
    
    # Processing history
    if st.session_state.rt_evaluator.processing_stats:
        st.subheader("📈 Processing History")
        history_df = pd.DataFrame(st.session_state.rt_evaluator.processing_stats).T
        st.dataframe(history_df)
    
    # Reset button
    if st.button("🔄 Reset All Data"):
        st.session_state.rt_evaluator = RealTimeRAGEvaluator()
        st.session_state.processed_methods = []
        st.success("✅ System reset complete!")
        st.experimental_rerun()

# Sidebar information
st.sidebar.header("⚡ Real-Time Features")
st.sidebar.markdown("""
**New Capabilities:**
- 📄 Live document processing
- ⏱️ Real-time progress tracking
- 📊 Live performance metrics
- 🔍 Instant retrieval testing
- 📈 Dynamic comparisons

**Upload any text file and watch:**
- Chunking in real-time
- Embedding generation progress
- FAISS index creation
- Performance benchmarking
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Pro Tip:** Try uploading different document types to see how processing time varies!")
