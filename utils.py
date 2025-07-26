from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Generator
import streamlit as st

class RealTimeRAGEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_stores = {}
        self.chunks_data = {}
        self.processing_stats = {}
        
    def process_document_realtime(self, text: str, method_name: str, 
                                chunk_size: int = 500, chunk_overlap: int = 50) -> Generator:
        """Process document in real-time with progress updates"""
        start_time = time.time()
        
        # Step 1: Text Analysis
        yield {"step": "analysis", "status": "Analyzing document...", "progress": 0}
        doc_stats = {
            "total_chars": len(text),
            "total_words": len(text.split()),
            "estimated_chunks": len(text) // chunk_size
        }
        yield {"step": "analysis", "status": "Document analyzed", "progress": 20, "data": doc_stats}
        
        # Step 2: Chunking
        yield {"step": "chunking", "status": f"Applying {method_name} chunking...", "progress": 30}
        
        if method_name == "Fixed Size":
            chunks = self._chunk_fixed_size_realtime(text, chunk_size, chunk_overlap)
        else:
            chunks = self._chunk_recursive_realtime(text, chunk_size, chunk_overlap)
            
        yield {"step": "chunking", "status": f"Created {len(chunks)} chunks", "progress": 50, "data": chunks}
        
        # Step 3: Embedding Generation
        yield {"step": "embedding", "status": "Generating embeddings...", "progress": 60}
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode([chunk])[0]
            embeddings.append(embedding)
            progress = 60 + (30 * (i + 1) / len(chunks))
            yield {"step": "embedding", "status": f"Processed chunk {i+1}/{len(chunks)}", "progress": progress}
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Step 4: FAISS Index Creation
        yield {"step": "indexing", "status": "Creating FAISS index...", "progress": 90}
        
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Store data
        self.vector_stores[method_name] = index
        self.chunks_data[method_name] = chunks
        
        # Final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.processing_stats[method_name] = {
            "total_time": processing_time,
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]),
            "processing_speed": len(chunks) / processing_time
        }
        
        yield {
            "step": "complete", 
            "status": "Processing complete!", 
            "progress": 100, 
            "stats": self.processing_stats[method_name]
        }
    
    def _chunk_fixed_size_realtime(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fixed size chunking with real-time processing"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > len(chunk) * 0.7:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def _chunk_recursive_realtime(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive chunking with real-time processing"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-chunk_overlap//10:]  # Approximate word overlap
                current_chunk = ' '.join(overlap_words) + ' ' + paragraph
            else:
                current_chunk += ('\n\n' if current_chunk else '') + paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def retrieve_chunks_realtime(self, query: str, method_name: str, k: int = 3) -> Dict:
        """Real-time retrieval with timing"""
        start_time = time.time()
        
        if method_name not in self.vector_stores:
            return {"chunks": [], "retrieval_time": 0, "similarity_scores": []}
        
        # Encode query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search with timing
        scores, indices = self.vector_stores[method_name].search(query_embedding, k)
        
        # Get results
        retrieved_chunks = [self.chunks_data[method_name][idx] for idx in indices[0]]
        retrieval_time = time.time() - start_time
        
        return {
            "chunks": retrieved_chunks,
            "retrieval_time": retrieval_time,
            "similarity_scores": scores[0].tolist(),
            "chunk_indices": indices[0].tolist()
        }
    
    def get_processing_comparison(self) -> Dict:
        """Compare processing statistics between methods"""
        if len(self.processing_stats) < 2:
            return {}
        
        comparison = {}
        methods = list(self.processing_stats.keys())
        
        for metric in ["total_time", "chunks_created", "avg_chunk_length", "processing_speed"]:
            comparison[metric] = {
                method: self.processing_stats[method][metric] 
                for method in methods
            }
        
        return comparison

# Backward compatibility functions
def load_chunks(filename):
    """Load chunks from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Create embeddings for text chunks"""
    if not chunks:
        return None
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings
