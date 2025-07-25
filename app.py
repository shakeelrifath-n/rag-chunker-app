import streamlit as st
import pandas as pd
import numpy as np
from utils import load_chunks, create_embeddings
import json

# Page configuration
st.set_page_config(
    page_title="RAG Chunking & Embedding Explorer by Shakeel",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 RAG Chunking & Embedding Explorer")
st.markdown("Compare different chunking technique and generate embeddings for text analysis")

# Sidebar for method selection
st.sidebar.header("Configuration")
method_option = st.sidebar.selectbox(
    "Select Chunking Method", 
    ("Fixed Size Chunking", "Recursive Chunking")
)

# Map selection to file
if method_option == "Fixed Size Chunking":
    filename = "data/chunked_method1.json"
    method_name = "Method 1 (Fixed Size)"
else:
    filename = "data/chunked_method2.json"
    method_name = "Method 2 (Recursive)"

# Load chunks
chunks = load_chunks(filename)

if not chunks:
    st.error(f"No data found in {filename}. Please check your data files.")
    st.stop()

# Display method information
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📄 {method_name}")
    st.info(f"Total chunks: {len(chunks)}")
    
    # Show chunk statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    st.write("**Chunk Statistics:**")
    stats_df = pd.DataFrame({
        'Metric': ['Average Length', 'Min Length', 'Max Length', 'Total Chunks'],
        'Value': [
            f"{np.mean(chunk_lengths):.1f} chars",
            f"{min(chunk_lengths)} chars",
            f"{max(chunk_lengths)} chars",
            f"{len(chunks)} chunks"
        ]
    })
    st.dataframe(stats_df, hide_index=True)

with col2:
    st.subheader("📝 Sample Chunks")
    for i, chunk in enumerate(chunks[:3], 1):
        st.write(f"**Chunk {i}:** {chunk[:100]}...")

# Embedding section
st.subheader("🧠 Generate Embeddings")

embedding_model = st.selectbox(
    "Choose Embedding Model",
    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"]
)

if st.button("🚀 Generate Embeddings", type="primary"):
    with st.spinner(f"Generating embeddings using {embedding_model}..."):
        try:
            embeddings = create_embeddings(chunks, embedding_model)
            
            if embeddings is not None:
                st.success(f"✅ Embeddings generated successfully!")
                
                # Display embedding information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Number of Embeddings", embeddings.shape[0])
                
                with col2:
                    st.metric("Embedding Dimensions", embeddings.shape[1])
                
                with col3:
                    st.metric("Total Parameters", f"{embeddings.shape[0] * embeddings.shape[1]:,}")
                
                # Show sample embeddings
                st.subheader("📊 Sample Embedding Vectors")
                
                # Create a dataframe with first few embeddings
                sample_embeddings = embeddings[:3]  # First 3 embeddings
                embedding_df = pd.DataFrame(
                    sample_embeddings,
                    columns=[f"Dim_{i+1}" for i in range(sample_embeddings.shape[1])],
                    index=[f"Chunk_{i+1}" for i in range(len(sample_embeddings))]
                )
                
                # Show only first 10 dimensions for display
                st.dataframe(embedding_df.iloc[:, :10])
                st.info("Showing first 10 dimensions of the embedding vectors")
                
                # Download option
                if st.button("💾 Download Embeddings"):
                    # Convert embeddings to JSON for download
                    embeddings_list = embeddings.tolist()
                    embeddings_json = json.dumps(embeddings_list)
                    
                    st.download_button(
                        label="📥 Download as JSON",
                        data=embeddings_json,
                        file_name=f"embeddings_{method_option.lower().replace(' ', '_')}.json",
                        mime="application/json"
                    )
                
            else:
                st.error("Failed to generate embeddings. Please check your data.")
                
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")

# Information section
st.subheader("ℹ️ About This App")
st.markdown("""
This application demonstrates:
- **Text Chunking**: Different methods to split large documents
- **Embeddings**: Converting text to numerical vectors for AI processing
- **Comparison**: Analyzing the impact of different chunking strategies

**Chunking Methods:**
- **Fixed Size**: Splits text into uniform character-based chunks
- **Recursive**: Respects document structure (paragraphs, sentences)
""")
