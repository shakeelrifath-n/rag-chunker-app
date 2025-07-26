from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
import openai
import os
from typing import List, Dict, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RAGEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_stores = {}
        self.chunks_data = {}
        
    def create_faiss_index(self, chunks: List[str], method_name: str) -> faiss.IndexFlatIP:
        """Create FAISS index for chunks"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)
        
        # Store for later use
        self.vector_stores[method_name] = index
        self.chunks_data[method_name] = chunks
        
        return index
    
    def retrieve_chunks(self, query: str, method_name: str, k: int = 3) -> List[str]:
        """Retrieve top-k chunks using FAISS"""
        if method_name not in self.vector_stores:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_stores[method_name].search(query_embedding, k)
        
        # Return retrieved chunks
        retrieved_chunks = [self.chunks_data[method_name][idx] for idx in indices[0]]
        return retrieved_chunks
    
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
        
        prompt = prompts.get(prompting_technique, prompts["zero_shot"])
        
        # For demo purposes, we'll create a simple response
        # In production, you'd use OpenAI API or another LLM
        return self._generate_simple_response(query, context_chunks)
    
    def _generate_simple_response(self, query: str, context_chunks: List[str]) -> str:
        """Simple response generation for demo (replace with actual LLM)"""
        # This is a simplified approach - combine relevant chunks
        response_parts = []
        query_lower = query.lower()
        
        for chunk in context_chunks:
            # Simple relevance check
            if any(word in chunk.lower() for word in query_lower.split()):
                response_parts.append(chunk[:200] + "...")
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "Based on the available context: " + " ".join([chunk[:100] + "..." for chunk in context_chunks[:2]])
    
    def calculate_f1_score(self, generated_response: str, reference_answer: str) -> float:
        """Calculate F1 score between generated and reference answers"""
        # Tokenize both responses
        generated_tokens = set(word_tokenize(generated_response.lower()))
        reference_tokens = set(word_tokenize(reference_answer.lower()))
        
        # Calculate precision and recall
        if len(generated_tokens) == 0:
            return 0.0
        
        intersection = generated_tokens.intersection(reference_tokens)
        precision = len(intersection) / len(generated_tokens)
        recall = len(intersection) / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

# Helper functions for backward compatibility
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
