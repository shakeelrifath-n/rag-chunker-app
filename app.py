import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Real-Time RAG Processing System",
    page_icon="⚡",
    layout="wide"
)

# ROBUST Session State Initialization
def initialize_session_state():
    """Initialize all session state variables with error handling"""
    try:
        # Only import after page config
        from utils import RealTimeRAGEvaluator
        
        if 'rt_evaluator' not in st.session_state:
            st.session_state.rt_evaluator = RealTimeRAGEvaluator()
        
        if 'processed_methods' not in st.session_state:
            st.session_state.processed_methods = []
        
        if 'current_document' not in st.session_state:
            st.session_state.current_document = ""
        
        if 'initialization_complete' not in st.session_state:
            st.session_state.initialization_complete = True
            
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.error("Please refresh the page or check your dependencies.")
        return False

# Initialize system
if not initialize_session_state():
    st.stop()

# Safety check function
def ensure_session_state():
    """Ensure session state variables exist before use"""
    if 'rt_evaluator' not in st.session_state:
        initialize_session_state()
    
    if not hasattr(st.session_state, 'rt_evaluator'):
        st.error("❌ System not properly initialized. Please refresh the page.")
        st.stop()

# Call safety check
ensure_session_state()

# Title and description
st.title("⚡ Real-Time RAG Processing System")
st.markdown("Upload documents and watch real-time chunking, embedding generation, and performance analysis")

def process_realtime(method_name, text_content, chunk_size, chunk_overlap):
    """Process document in real-time with live updates"""
    
    # Ensure session state is valid
    ensure_session_state()
    
    # Create containers for updates
    status_container = st.container()
    progress_container = st.container()
    stats_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with stats_container:
        stats_display = st.empty()
    
    try:
        # Process with real-time updates
        for update in st.session_state.rt_evaluator.process_document_realtime(
            text_content, method_name, chunk_size, chunk_overlap
        ):
            # Update progress bar
            progress_bar.progress(int(update["progress"]))
            
            # Update status
            status_text.info(f"**{update['step'].title()}**: {update['status']}")
            
            # Update statistics if available
            if "data" in update:
                if update["step"] == "analysis":
                    with stats_display.container():
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Characters", update['data']['total_chars'])
                        col2.metric("Total Words", update['data']['total_words'])
                        col3.metric("Estimated Chunks", update['data']['estimated_chunks'])
                elif update["step"] == "chunking":
                    with stats_display.container():
                        st.success(f"✅ Created {len(update['data'])} chunks")
            
            # Update final stats
            if "stats" in update:
                with stats_display.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Processing Time", f"{update['stats']['total_time']:.2f}s")
                    col2.metric("Chunks Created", update['stats']['chunks_created'])
                    col3.metric("Avg Chunk Length", f"{update['stats']['avg_chunk_length']:.0f}")
                    col4.metric("Processing Speed", f"{update['stats']['processing_speed']:.1f} chunks/s")
            
            # Small delay for visual effect
            time.sleep(0.1)
        
        # Add to processed methods
        if method_name not in st.session_state.processed_methods:
            st.session_state.processed_methods.append(method_name)
        
        st.success(f"✅ {method_name} processing completed successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        return False

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Real-Time Processing", "🔍 Live Retrieval", "🤖 Response Generation", "📊 Performance Analytics"])

with tab1:
    st.header("📄 Real-Time Document Processing")
    
    # Ensure session state at tab level
    ensure_session_state()
    
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
        st.subheader("📊 Document Info")
        info_placeholder = st.empty()
    
    # Get text content
    text_content = ""
    if uploaded_file is not None:
        try:
            text_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.session_state.current_document = text_content
            with info_placeholder.container():
                st.success(f"📄 File uploaded")
                st.metric("Characters", len(text_content))
                st.metric("Words", len(text_content.split()))
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    elif manual_text.strip():
        text_content = manual_text.strip()
        st.session_state.current_document = text_content
        with info_placeholder.container():
            st.info(f"📝 Manual text ready")
            st.metric("Characters", len(text_content))
            st.metric("Words", len(text_content.split()))
    
    # Processing buttons
    if text_content:
        st.subheader("🚀 Start Processing")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔧 Process with Fixed Size Chunking", key="process_fixed", type="primary"):
                with st.expander("🔄 Processing Progress", expanded=True):
                    process_realtime("Fixed Size", text_content, chunk_size, chunk_overlap)
        
        with col_btn2:
            if st.button("🌲 Process with Recursive Chunking", key="process_recursive", type="primary"):
                with st.expander("🔄 Processing Progress", expanded=True):
                    process_realtime("Recursive", text_content, chunk_size, chunk_overlap)
    else:
        st.info("👆 Please upload a document or paste text to start processing")

with tab2:
    st.header("🔍 Real-Time Retrieval Testing")
    
    # Ensure session state
    ensure_session_state()
    
    if st.session_state.processed_methods:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_input("Enter your query:", value="What is the main topic discussed?")
            
        with col2:
            k_retrieve = st.slider("Number of chunks to retrieve:", 1, 5, 3)
            method_select = st.selectbox("Select processing method:", st.session_state.processed_methods)
        
        if st.button("🚀 Search in Real-Time", type="primary"):
            start_time = time.time()
            
            try:
                with st.spinner("Searching through vectors..."):
                    results = st.session_state.rt_evaluator.retrieve_chunks_realtime(
                        query, method_select, k_retrieve
                    )
                
                end_time = time.time()
                
                # Display timing metrics
                col_timing1, col_timing2, col_timing3 = st.columns(3)
                col_timing1.metric("⚡ Retrieval Time", f"{results['retrieval_time']:.4f}s")
                col_timing2.metric("🕐 Total Time", f"{end_time - start_time:.4f}s")
                col_timing3.metric("📄 Chunks Found", len(results['chunks']))
                
                # Display results with similarity scores
                st.subheader("📋 Retrieved Chunks")
                for i, (chunk, score) in enumerate(zip(results['chunks'], results['similarity_scores'])):
                    similarity_percent = score * 100
                    with st.expander(f"📄 Chunk {i+1} - Similarity: {similarity_percent:.1f}%", expanded=True):
                        st.write(chunk)
                        st.caption(f"Score: {score:.4f}")
                        
            except Exception as e:
                st.error(f"❌ Error during retrieval: {str(e)}")
    else:
        st.info("👆 Please process a document in the 'Real-Time Processing' tab first")

with tab3:
    st.header("🤖 Response Generation")
    
    # Ensure session state
    ensure_session_state()
    
    if st.session_state.processed_methods:
        col1, col2 = st.columns(2)
        
        with col1:
            # Predefined test queries
            test_queries = [
                "What is the main topic of this document?",
                "Summarize the key points discussed",
                "What are the important details mentioned?",
                "Explain the main concepts",
                "What conclusions can be drawn?"
            ]
            
            selected_query = st.selectbox("Select a test query:", test_queries)
            custom_query = st.text_input("Or enter custom query:")
            
            final_query = custom_query if custom_query.strip() else selected_query
            
        with col2:
            chunking_method = st.selectbox("Chunking method:", st.session_state.processed_methods)
            prompting_technique = st.selectbox("Prompting technique:", 
                                             ["zero_shot", "few_shot", "chain_of_thought", "role_based"])
        
        if st.button("🤖 Generate Response", type="primary"):
            try:
                with st.spinner("Generating response..."):
                    # Retrieve chunks
                    retrieved_chunks = st.session_state.rt_evaluator.retrieve_chunks_realtime(
                        final_query, chunking_method, 3
                    )
                    
                    # Generate response
                    response = st.session_state.rt_evaluator.generate_response(
                        final_query, retrieved_chunks['chunks'], prompting_technique
                    )
                    
                    # Display results
                    st.subheader("🎯 Generated Response")
                    st.write(response)
                    
                    st.subheader("📚 Source Context")
                    for i, chunk in enumerate(retrieved_chunks['chunks'], 1):
                        with st.expander(f"Source {i}"):
                            st.write(chunk)
                            
            except Exception as e:
                st.error(f"❌ Error during response generation: {str(e)}")
    else:
        st.info("👆 Please process a document first")

with tab4:
    st.header("📊 Real-Time Performance Analytics")
    
    # Ensure session state
    ensure_session_state()
    
    if len(st.session_state.processed_methods) >= 2:
        try:
            comparison = st.session_state.rt_evaluator.get_processing_comparison()
            
            if comparison:
                st.subheader("⚡ Processing Performance Comparison")
                
                # Create performance comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Processing time comparison
                    methods = list(comparison['total_time'].keys())
                    times = list(comparison['total_time'].values())
                    
                    fig_time = go.Figure(data=[
                        go.Bar(x=methods, y=times, name="Processing Time",
                              marker_color=['#FF6B6B', '#4ECDC4'])
                    ])
                    fig_time.update_layout(
                        title="Processing Time Comparison (seconds)",
                        xaxis_title="Chunking Method",
                        yaxis_title="Time (seconds)"
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    # Chunks created comparison
                    chunks_counts = list(comparison['chunks_created'].values())
                    
                    fig_chunks = go.Figure(data=[
                        go.Bar(x=methods, y=chunks_counts, name="Chunks Created",
                              marker_color=['#FFE66D', '#A8E6CF'])
                    ])
                    fig_chunks.update_layout(
                        title="Chunks Created Comparison",
                        xaxis_title="Chunking Method",
                        yaxis_title="Number of Chunks"
                    )
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                # Processing speed comparison
                speeds = list(comparison['processing_speed'].values())
                fig_speed = go.Figure(data=[
                    go.Bar(x=methods, y=speeds, name="Processing Speed",
                          marker_color=['#95E1D3', '#F3D250'])
                ])
                fig_speed.update_layout(
                    title="Processing Speed (chunks/second)",
                    xaxis_title="Chunking Method",
                    yaxis_title="Chunks per Second"
                )
                st.plotly_chart(fig_speed, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("📋 Detailed Statistics")
                stats_df = pd.DataFrame(st.session_state.rt_evaluator.processing_stats).T
                st.dataframe(stats_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error in analytics: {str(e)}")
            
    elif len(st.session_state.processed_methods) == 1:
        st.info("📊 Process documents with both chunking methods to see comparative analytics")
        
        try:
            # Show single method stats
            method = st.session_state.processed_methods[0]
            stats = st.session_state.rt_evaluator.processing_stats[method]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Processing Time", f"{stats['total_time']:.2f}s")
            col2.metric("Chunks Created", stats['chunks_created'])
            col3.metric("Avg Chunk Length", f"{stats['avg_chunk_length']:.0f}")
            col4.metric("Processing Speed", f"{stats['processing_speed']:.1f} chunks/s")
        
        except Exception as e:
            st.error(f"❌ Error displaying stats: {str(e)}")
        
    else:
        st.info("👆 Please process documents to see performance analytics")
    
    # System monitoring section
    st.subheader("🖥️ System Status")
    col_sys1, col_sys2, col_sys3 = st.columns(3)
    
    try:
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
    
    except Exception as e:
        st.warning(f"System monitoring partially unavailable: {str(e)}")
    
    # Reset system button
    if st.button("🔄 Reset All Data", type="secondary"):
        try:
            from utils import RealTimeRAGEvaluator
            st.session_state.rt_evaluator = RealTimeRAGEvaluator()
            st.session_state.processed_methods = []
            st.session_state.current_document = ""
            st.success("✅ System reset complete!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Reset failed: {str(e)}")

# Sidebar information
st.sidebar.header("⚡ Real-Time RAG System")
st.sidebar.markdown("""
**Key Features:**
- 📄 Live document processing
- ⏱️ Real-time progress tracking
- 📊 Performance benchmarking
- 🔍 Instant vector search
- 🤖 Response generation
- 📈 Comparative analytics

**How to Use:**
1. Upload or paste your document
2. Choose processing parameters
3. Process with both chunking methods
4. Test retrieval and response generation
5. Compare performance metrics
""")

try:
    if st.session_state.processed_methods:
        st.sidebar.success(f"✅ {len(st.session_state.processed_methods)} method(s) processed")
        for method in st.session_state.processed_methods:
            chunks_count = len(st.session_state.rt_evaluator.chunks_data.get(method, []))
            st.sidebar.write(f"• {method}: {chunks_count} chunks")

    if st.session_state.current_document:
        st.sidebar.info(f"📄 Document loaded: {len(st.session_state.current_document)} characters")
        
except Exception as e:
    st.sidebar.warning("Sidebar status partially unavailable.")
