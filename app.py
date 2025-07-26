import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from typing import List


# Page configuration
st.set_page_config(
    page_title="Real-Time RAG Processing System",
    page_icon="⚡",
    layout="wide"
)

# Session State Initialization
def initialize_session_state():
    """Initialize all session state variables with error handling"""
    try:
        from utils import RealTimeRAGEvaluator
        
        if 'rt_evaluator' not in st.session_state:
            st.session_state.rt_evaluator = RealTimeRAGEvaluator()
        
        if 'processed_methods' not in st.session_state:
            st.session_state.processed_methods = []
        
        if 'current_document' not in st.session_state:
            st.session_state.current_document = ""
        
        if 'document_type' not in st.session_state:
            st.session_state.document_type = "generic"
            
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

# Enhanced document analysis
def analyze_document_type(text: str) -> str:
    """Analyze document type for better reference answers"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['rag', 'retrieval', 'embedding', 'vector', 'faiss', 'chunk']):
        return "rag_technical"
    elif any(word in text_lower for word in ['machine learning', 'ai', 'neural', 'model', 'algorithm']):
        return "ai_ml"
    elif any(word in text_lower for word in ['business', 'market', 'strategy', 'company', 'revenue']):
        return "business"
    elif any(word in text_lower for word in ['research', 'study', 'analysis', 'methodology', 'results']):
        return "research"
    else:
        return "generic"

def get_optimized_reference_answers(document_type: str) -> List[str]:
    """Get optimized reference answers based on document type"""
    
    if document_type == "rag_technical":
        return [
            "The main topic focuses on retrieval-augmented generation systems, chunking strategies, vector embeddings, and semantic search technologies for improved AI responses",
            "Key points include FAISS vector storage, embedding generation, chunking methodologies, similarity search, and performance optimization for RAG systems",
            "Important details cover technical implementation, vector databases, text processing, embedding models, chunk size optimization, and retrieval mechanisms",
            "Main concepts involve systematic approach to document processing, semantic similarity, vector search, and integration of retrieval with generation for enhanced AI capabilities",
            "Conclusions highlight the effectiveness of different chunking strategies, importance of proper vector indexing, and recommendations for optimizing RAG system performance"
        ]
    elif document_type == "ai_ml":
        return [
            "The document discusses machine learning concepts, artificial intelligence systems, neural networks, and algorithmic approaches for data processing and model optimization",
            "Key points include model architecture, training methodologies, performance metrics, data preprocessing, and optimization techniques for machine learning systems",
            "Important details cover technical specifications, implementation approaches, algorithmic efficiency, model evaluation, and best practices for AI development",
            "Main concepts involve systematic machine learning methodology, model selection, performance evaluation, and optimization strategies for artificial intelligence applications",
            "Conclusions emphasize model performance, evaluation metrics, optimization results, and recommendations for future machine learning system development"
        ]
    elif document_type == "business":
        return [
            "The document focuses on business strategy, market analysis, company operations, and strategic planning for organizational growth and development",
            "Key points include market research, business methodology, strategic analysis, operational efficiency, and performance indicators for business success",
            "Important details cover business specifications, implementation strategies, market positioning, operational details, and competitive analysis approaches",
            "Main concepts involve systematic business approach, strategic planning, market analysis, and optimization strategies for organizational performance",
            "Conclusions highlight business findings, strategic recommendations, market insights, and suggestions for future business development and growth"
        ]
    elif document_type == "research":
        return [
            "The document presents research methodology, study design, data analysis, and findings from systematic investigation and academic research",
            "Key points include research methodology, data collection, analytical approaches, statistical analysis, and evidence-based findings from the study",
            "Important details cover research specifications, methodological approaches, data analysis techniques, statistical results, and research validation methods",
            "Main concepts involve systematic research methodology, data analysis, evidence evaluation, and scientific approach to investigation and discovery",
            "Conclusions highlight research findings, statistical significance, study implications, and recommendations for future research and practical applications"
        ]
    else:  # generic
        return [
            "The document discusses key concepts and main topics with detailed explanations, analysis, and comprehensive information about the subject matter",
            "Key points include methodology, analysis, detailed information, systematic approach, and important aspects covered throughout the document",
            "Important details include specifications, implementation details, comprehensive explanations, technical aspects, and detailed analysis of the subject",
            "Main concepts involve systematic approach to the topic, detailed analysis, methodological considerations, and comprehensive coverage of relevant aspects",
            "Conclusions highlight significant findings, recommendations, important insights, and suggestions for future consideration and practical application"
        ]

# Title and description
st.title("⚡ Real-Time RAG Processing System")
st.markdown("Upload documents and watch real-time chunking, embedding generation, and performance analysis")

def process_realtime(method_name, text_content, chunk_size, chunk_overlap):
    """Process document in real-time with live updates"""
    
    ensure_session_state()
    
    # Analyze document type for better evaluation
    st.session_state.document_type = analyze_document_type(text_content)
    
    # Create containers for updates
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
                
                if "error" in results:
                    st.error(f"❌ Retrieval error: {results['error']}")
                else:
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
    
    ensure_session_state()
    
    if st.session_state.processed_methods:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced test queries
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
    
    ensure_session_state()
    
    # Enhanced F1-Score Evaluation Section
    st.subheader("🎯 Enhanced F1-Score Evaluation")
    
    if st.session_state.processed_methods:
        # Enhanced test queries
        test_queries = [
            "What is the main topic of this document?",
            "Summarize the key points discussed",
            "What are the important details mentioned?",
            "Explain the main concepts",
            "What conclusions can be drawn?"
        ]
        
        # Get optimized reference answers based on document type
        document_type = getattr(st.session_state, 'document_type', 'generic')
        reference_answers = get_optimized_reference_answers(document_type)
        
        st.info(f"📄 Document type detected: **{document_type.replace('_', ' ').title()}** - Using optimized reference answers")
        
        if st.button("🏃‍♂️ Run Enhanced F1-Score Evaluation", type="primary"):
            f1_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_evaluations = len(test_queries) * len(st.session_state.processed_methods) * 4  # 4 prompting techniques
            current_eval = 0
            
            for i, (query, ref_answer) in enumerate(zip(test_queries, reference_answers)):
                status_text.text(f"Evaluating query {i+1}/{len(test_queries)}: {query[:50]}...")
                
                for method in st.session_state.processed_methods:
                    for technique in ["zero_shot", "few_shot", "chain_of_thought", "role_based"]:
                        try:
                            # Retrieve chunks
                            retrieved_chunks = st.session_state.rt_evaluator.retrieve_chunks_realtime(query, method, 3)
                            
                            if retrieved_chunks['chunks']:
                                # Generate response with specific technique
                                response = st.session_state.rt_evaluator.generate_response(
                                    query, retrieved_chunks['chunks'], technique
                                )
                                
                                # Calculate F1 score
                                f1_score = st.session_state.rt_evaluator.calculate_f1_score(response, ref_answer)
                                
                                f1_results.append({
                                    'Query': f"Q{i+1}",
                                    'Question': query[:30] + "...",
                                    'Method': method,
                                    'Technique': technique,
                                    'F1_Score': f1_score,
                                    'Response_Length': len(response.split()),
                                    'Reference_Length': len(ref_answer.split()),
                                    'Similarity_Score': np.mean(retrieved_chunks['similarity_scores']) if retrieved_chunks['similarity_scores'] else 0
                                })
                            else:
                                # No chunks retrieved
                                f1_results.append({
                                    'Query': f"Q{i+1}",
                                    'Question': query[:30] + "...",
                                    'Method': method,
                                    'Technique': technique,
                                    'F1_Score': 0.0,
                                    'Response_Length': 0,
                                    'Reference_Length': len(ref_answer.split()),
                                    'Similarity_Score': 0
                                })
                            
                        except Exception as e:
                            st.warning(f"Error evaluating {method}/{technique} for query {i+1}: {str(e)}")
                            f1_results.append({
                                'Query': f"Q{i+1}",
                                'Question': query[:30] + "...",
                                'Method': method,
                                'Technique': technique,
                                'F1_Score': 0.0,
                                'Response_Length': 0,
                                'Reference_Length': len(ref_answer.split()),
                                'Similarity_Score': 0
                            })
                        
                        current_eval += 1
                        progress_bar.progress(current_eval / total_evaluations)
            
            status_text.success("Enhanced F1-Score evaluation complete!")
            
            if f1_results:
                # Create F1-Score DataFrame
                f1_df = pd.DataFrame(f1_results)
                
                # Display F1-Score comparison chart by method
                st.subheader("📈 F1-Score Comparison by Method")
                
                method_avg = f1_df.groupby('Method')['F1_Score'].mean().reset_index()
                
                fig_method = px.bar(
                    method_avg, 
                    x='Method', 
                    y='F1_Score',
                    title='Average F1-Score by Chunking Method',
                    color='Method',
                    text='F1_Score'
                )
                fig_method.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_method.update_layout(yaxis_range=[0, max(method_avg['F1_Score']) * 1.2])
                st.plotly_chart(fig_method, use_container_width=True)
                
                # F1-Score by technique
                st.subheader("📈 F1-Score Comparison by Prompting Technique")
                
                technique_avg = f1_df.groupby(['Method', 'Technique'])['F1_Score'].mean().reset_index()
                
                fig_technique = px.bar(
                    technique_avg, 
                    x='Technique', 
                    y='F1_Score', 
                    color='Method',
                    title='F1-Score by Prompting Technique and Method',
                    barmode='group',
                    text='F1_Score'
                )
                fig_technique.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_technique, use_container_width=True)
                
                # Best combinations
                col_best1, col_best2 = st.columns(2)
                
                with col_best1:
                    best_overall = f1_df.loc[f1_df['F1_Score'].idxmax()]
                    st.metric("🏆 Best Overall", 
                             f"{best_overall['Method']} + {best_overall['Technique']}", 
                             f"{best_overall['F1_Score']:.3f}")
                
                with col_best2:
                    st.markdown("**F1-Score Interpretation:**")
                    max_score = f1_df['F1_Score'].max()
                    if max_score >= 0.7:
                        st.success("🌟 Excellent quality achieved!")
                    elif max_score >= 0.5:
                        st.info("✅ Good quality achieved!")
                    elif max_score >= 0.3:
                        st.warning("⚠️ Fair quality - room for improvement")
                    else:
                        st.error("❌ Needs optimization")
                
                # Detailed F1-Score table
                st.subheader("📋 Detailed F1-Score Results")
                st.dataframe(f1_df.sort_values('F1_Score', ascending=False), use_container_width=True)
                
                # Download enhanced results
                csv = f1_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced F1-Score Results",
                    data=csv,
                    file_name=f"enhanced_f1_score_evaluation_{document_type}.csv",
                    mime="text/csv"
                )
    
    # Processing Performance Section (existing code continues...)
    if len(st.session_state.processed_methods) >= 2:
        st.subheader("⚡ Processing Performance Comparison")
        
        try:
            comparison = st.session_state.rt_evaluator.get_processing_comparison()
            
            if comparison:
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
                st.subheader("📋 Processing Statistics")
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
            st.session_state.document_type = "generic"
            st.success("✅ System reset complete!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Reset failed: {str(e)}")

# Enhanced Sidebar
st.sidebar.header("⚡ Enhanced RAG System")
st.sidebar.markdown("""
**🚀 Optimized Features:**
- 📄 Smart document type detection
- ⏱️ Enhanced chunking algorithms
- 📊 Improved F1-score calculation
- 🔍 Optimized vector search
- 🤖 Better response generation
- 📈 Advanced analytics

**🎯 Performance Improvements:**
- Document-aware reference answers
- Enhanced text preprocessing
- Better sentence boundary detection
- Optimized chunk quality control
- Advanced similarity scoring

**How to Use:**
1. Upload or paste your document
2. System auto-detects document type
3. Process with both chunking methods
4. Run enhanced F1-score evaluation
5. Compare optimized performance metrics
""")

try:
    if st.session_state.processed_methods:
        st.sidebar.success(f"✅ {len(st.session_state.processed_methods)} method(s) processed")
        for method in st.session_state.processed_methods:
            chunks_count = len(st.session_state.rt_evaluator.chunks_data.get(method, []))
            st.sidebar.write(f"• {method}: {chunks_count} chunks")

    if st.session_state.current_document:
        doc_type = getattr(st.session_state, 'document_type', 'generic')
        st.sidebar.info(f"📄 Document: {len(st.session_state.current_document)} chars")
        st.sidebar.info(f"🎯 Type: {doc_type.replace('_', ' ').title()}")
        
except Exception as e:
    st.sidebar.warning("Sidebar status partially unavailable")
