import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from utils import RAGEvaluator, load_chunks

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Evaluation System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'rag_evaluator' not in st.session_state:
    st.session_state.rag_evaluator = RAGEvaluator()
if 'faiss_initialized' not in st.session_state:
    st.session_state.faiss_initialized = False

# Title and description
st.title("🔍 Advanced RAG Evaluation System")
st.markdown("Compare chunking methods, test prompting techniques, and evaluate RAG performance with F1 scores")

# Sidebar configuration
st.sidebar.header("🛠️ Configuration")

# Test queries for evaluation
test_queries = [
    "What is retrieval-augmented generation and how does it work?",
    "How do embedding models convert text into vectors?",
    "What are the advantages of recursive chunking over fixed chunking?",
    "Why is text chunking important for RAG system performance?",
    "What infrastructure considerations are important for RAG systems?"
]

# Reference answers for F1 evaluation
reference_answers = [
    "RAG is an AI framework that enhances language models by incorporating external knowledge sources through retrieval and generation components.",
    "Embedding models use neural networks to convert text into high-dimensional numerical vectors that capture semantic meaning and enable similarity comparisons.",
    "Recursive chunking preserves semantic boundaries and document structure by splitting at natural boundaries like paragraphs and sentences.",
    "Text chunking is critical because it determines retrieval quality by breaking documents into manageable segments while preserving context and meaning.",
    "RAG systems require vector databases, scalable infrastructure, low latency search capabilities, and efficient index management for production deployment."
]

# Load and initialize FAISS indices
@st.cache_data
def initialize_faiss_indices():
    chunks1 = load_chunks("data/chunked_method1.json")
    chunks2 = load_chunks("data/chunked_method2.json")
    
    if chunks1 and chunks2:
        evaluator = RAGEvaluator()
        evaluator.create_faiss_index(chunks1, "Fixed Size")
        evaluator.create_faiss_index(chunks2, "Recursive")
        return evaluator, chunks1, chunks2
    return None, [], []

# Initialize FAISS
if not st.session_state.faiss_initialized:
    with st.spinner("Initializing FAISS vector stores..."):
        evaluator, chunks1, chunks2 = initialize_faiss_indices()
        if evaluator:
            st.session_state.rag_evaluator = evaluator
            st.session_state.faiss_initialized = True
            st.success("✅ FAISS indices created successfully!")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Chunk Analysis", "🔍 RAG Retrieval", "🤖 Response Generation", "📈 Performance Evaluation"])

with tab1:
    st.header("📊 Chunking Method Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Fixed Size Chunking")
        chunks1 = load_chunks("data/chunked_method1.json")
        if chunks1:
            st.info(f"Total chunks: {len(chunks1)}")
            st.write("**Sample chunk:**")
            st.text_area("", chunks1[0] if chunks1 else "", height=150, key="fixed_sample")
    
    with col2:
        st.subheader("🌲 Recursive Chunking")
        chunks2 = load_chunks("data/chunked_method2.json")
        if chunks2:
            st.info(f"Total chunks: {len(chunks2)}")
            st.write("**Sample chunk:**")
            st.text_area("", chunks2[0] if chunks2 else "", height=150, key="recursive_sample")

with tab2:
    st.header("🔍 FAISS-Powered Retrieval")
    
    query_input = st.text_input("Enter your query:", value="What is RAG?")
    k_value = st.slider("Number of chunks to retrieve:", 1, 5, 3)
    
    if st.button("🚀 Retrieve Chunks") and st.session_state.faiss_initialized:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fixed Size Results")
            fixed_results = st.session_state.rag_evaluator.retrieve_chunks(query_input, "Fixed Size", k_value)
            for i, chunk in enumerate(fixed_results, 1):
                st.write(f"**Chunk {i}:**")
                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                st.write("---")
        
        with col2:
            st.subheader("Recursive Results")
            recursive_results = st.session_state.rag_evaluator.retrieve_chunks(query_input, "Recursive", k_value)
            for i, chunk in enumerate(recursive_results, 1):
                st.write(f"**Chunk {i}:**")
                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                st.write("---")

with tab3:
    st.header("🤖 Response Generation with Prompting Techniques")
    
    selected_query = st.selectbox("Select test query:", test_queries)
    chunking_method = st.radio("Chunking method:", ["Fixed Size", "Recursive"])
    prompting_technique = st.selectbox("Prompting technique:", 
                                     ["zero_shot", "few_shot", "chain_of_thought", "role_based"])
    
    if st.button("Generate Response") and st.session_state.faiss_initialized:
        with st.spinner("Generating response..."):
            # Retrieve chunks
            retrieved_chunks = st.session_state.rag_evaluator.retrieve_chunks(
                selected_query, chunking_method, 3
            )
            
            # Generate response
            response = st.session_state.rag_evaluator.generate_response(
                selected_query, retrieved_chunks, prompting_technique
            )
            
            st.subheader("📝 Generated Response")
            st.write(response)
            
            st.subheader("📋 Retrieved Context")
            for i, chunk in enumerate(retrieved_chunks, 1):
                with st.expander(f"Context Chunk {i}"):
                    st.write(chunk)

with tab4:
    st.header("📈 F1 Score Performance Evaluation")
    
    if st.button("🏃‍♂️ Run Complete Evaluation") and st.session_state.faiss_initialized:
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (query, ref_answer) in enumerate(zip(test_queries, reference_answers)):
            status_text.text(f"Evaluating query {i+1}/{len(test_queries)}")
            
            for method in ["Fixed Size", "Recursive"]:
                for technique in ["zero_shot", "few_shot", "chain_of_thought", "role_based"]:
                    # Retrieve and generate
                    chunks = st.session_state.rag_evaluator.retrieve_chunks(query, method, 3)
                    response = st.session_state.rag_evaluator.generate_response(query, chunks, technique)
                    
                    # Calculate F1 score
                    f1 = st.session_state.rag_evaluator.calculate_f1_score(response, ref_answer)
                    
                    results.append({
                        'Query': f"Q{i+1}",
                        'Chunking Method': method,
                        'Prompting Technique': technique,
                        'F1 Score': f1
                    })
            
            progress_bar.progress((i + 1) / len(test_queries))
        
        status_text.text("Evaluation complete!")
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        # Display results
        st.subheader("📊 Results Summary")
        
        # Average F1 scores by method and technique
        summary = df_results.groupby(['Chunking Method', 'Prompting Technique'])['F1 Score'].mean().reset_index()
        
        # Create visualization
        fig = px.bar(summary, x='Prompting Technique', y='F1 Score', 
                    color='Chunking Method', barmode='group',
                    title='Average F1 Scores by Chunking Method and Prompting Technique')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        st.dataframe(df_results)
        
        # Best performing combinations
        st.subheader("🏆 Best Performing Combinations")
        best_overall = summary.loc[summary['F1 Score'].idxmax()]
        st.success(f"**Best Overall**: {best_overall['Chunking Method']} + {best_overall['Prompting Technique']} (F1: {best_overall['F1 Score']:.3f})")
        
        # Method comparison
        method_avg = df_results.groupby('Chunking Method')['F1 Score'].mean()
        st.write("**Average F1 by Chunking Method:**")
        for method, score in method_avg.items():
            st.write(f"- {method}: {score:.3f}")

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About This System")
st.sidebar.markdown("""
**Features:**
- FAISS vector storage for efficient retrieval
- Multiple prompting techniques
- F1-Score evaluation metrics
- Chunking method comparison
- Interactive response generation

**Technologies:**
- FAISS for vector search
- SentenceTransformers for embeddings
- Streamlit for web interface
- Plotly for visualizations
""")
