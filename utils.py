from sentence_transformers import SentenceTransformer # type: ignore
import json

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

def save_chunks(chunks, filename):
    """Save chunks to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
