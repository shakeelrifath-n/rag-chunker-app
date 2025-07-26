from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Generator
import re

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
            "estimated_chunks": max(1, len(text) // chunk_size)
        }
        time.sleep(0.1)  # Simulate processing time
        yield {"step": "analysis", "status": "Document analyzed", "progress": 20, "data": doc_stats}
        
        # Step 2: Chunking
        yield {"step": "chunking", "status": f"Applying {method_name} chunking...", "progress": 30}
        
        if method_name == "Fixed Size":
            chunks = self._chunk_fixed_size_realtime(text, chunk_size, chunk_overlap)
        else:
            chunks = self._chunk_recursive_realtime(text, chunk_size, chunk_overlap)
            
        time.sleep(0.2)  # Simulate chunking time
        yield {"step": "chunking", "status": f"Created {len(chunks)} chunks", "progress": 50, "data": chunks}
        
        # Step 3: Embedding Generation
        yield {"step": "embedding", "status": "Generating embeddings...", "progress": 60}
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode([chunk])[0]
            embeddings.append(embedding)
            progress = 60 + (30 * (i + 1) / len(chunks))
            yield {"step": "embedding", "status": f"Processed chunk {i+1}/{len(chunks)}", "progress": progress}
            time.sleep(0.05)  # Small delay for real-time effect
        
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
            "processing_speed": len(chunks) / processing_time if processing_time > 0 else 0
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
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > len(chunk) * 0.7:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - chunk_overlap
            if start >= len(text):
                break
                
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def _chunk_recursive_realtime(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive chunking with real-time processing"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-max(1, chunk_overlap//10):]
                current_chunk = ' '.join(overlap_words) + ' ' + paragraph
            else:
                current_chunk += ('\n\n' if current_chunk else '') + paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no paragraphs found, split by sentences
        if len(chunks) <= 1 and len(text) > chunk_size:
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_words = current_chunk.split()[-max(1, chunk_overlap//10):]
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:
                    current_chunk += (' ' if current_chunk else '') + sentence
            
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
    
    def generate_response(self, query: str, context_chunks: List[str], 
                         prompting_technique: str = "zero_shot") -> str:
        """Generate response using different prompting techniques"""
        
        context = "\n\n".join(context_chunks)
        
        prompts = {
            "zero_shot": f"""Based on the following context, answer the question:

Context: {context}

Question: {query}

Answer:""",
            
            "few_shot": f"""Based on the following context, answer the question. Here are some examples:

Example 1:
Context: Electric vehicles use battery power for propulsion.
Question: How do electric vehicles work?
Answer: Electric vehicles work by using battery power to drive electric motors that propel the vehicle.

Example 2:
Context: RAG systems combine retrieval and generation for better AI responses.
Question: What is RAG?
Answer: RAG (Retrieval-Augmented Generation) is a system that combines document retrieval with text generation to produce more accurate AI responses.

Now answer this question:
Context: {context}

Question: {query}

Answer:""",
            
            "chain_of_thought": f"""Based on the following context, answer the question step by step:

Context: {context}

Question: {query}

Let me think through this step by step:
1. First, I'll identify the key information in the context
2. Then, I'll relate it to the question
3. Finally, I'll provide a comprehensive answer

Answer:""",
            
            "role_based": f"""You are an expert AI researcher specializing in RAG systems and text processing. Based on the following context, provide a detailed technical answer:

Context: {context}

Question: {query}

Expert Answer:"""
        }
        
        # For demo purposes, create a simple response
        return self._generate_simple_response(query, context_chunks)
    
    def _generate_simple_response(self, query: str, context_chunks: List[str]) -> str:
        """Simple response generation for demo"""
        response_parts = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for chunk in context_chunks:
            chunk_words = set(chunk.lower().split())
            # Check relevance
            if len(query_words.intersection(chunk_words)) > 0:
                response_parts.append(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "Based on the available context: " + " ".join([chunk[:150] + "..." for chunk in context_chunks[:2]])
    
    def calculate_f1_score(self, generated_response: str, reference_answer: str) -> float:
        """Calculate F1 score between generated and reference answers"""
        # Simple word tokenization
        generated_tokens = set(generated_response.lower().split())
        reference_tokens = set(reference_answer.lower().split())
        
        # Remove common punctuation
        punctuation = '.,!?;:"()[]{}\'`~@#$%^&*-_+=|\\/<>'
        
        # Clean tokens
        generated_clean = set()
        for token in generated_tokens:
            cleaned = token.strip(punctuation)
            if cleaned:
                generated_clean.add(cleaned)
        
        reference_clean = set()
        for token in reference_tokens:
            cleaned = token.strip(punctuation)
            if cleaned:
                reference_clean.add(cleaned)
        
        # Calculate precision and recall
        if len(generated_clean) == 0:
            return 0.0
        
        intersection = generated_clean.intersection(reference_clean)
        precision = len(intersection) / len(generated_clean)
        recall = len(intersection) / len(reference_clean) if len(reference_clean) > 0 else 0
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
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
