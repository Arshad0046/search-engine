import uvicorn
import os
import glob
import sqlite3
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import hashlib
import re
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="A lightweight embedding-based search engine with caching",
    version="1.0.0"
)

# Global search engine instance
search_engine = None

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of text for cache lookup"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text"""
        words = text.split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in words if len(w) > 2 and w not in stopwords]
        from collections import Counter
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)]

class Document:
    def __init__(self, doc_id: str, text: str, filename: str):
        self.doc_id = doc_id
        self.original_text = text
        self.cleaned_text = TextPreprocessor.clean_text(text)
        self.filename = filename
        self.hash = TextPreprocessor.compute_hash(self.cleaned_text)
        self.length = len(self.cleaned_text.split())
        self.keywords = TextPreprocessor.extract_keywords(self.cleaned_text)

class SimpleSearchEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = {}
        self.embeddings = None
        self.index = None
        self.index_doc_ids = []
        self._init_cache()
        print("Search engine initialized!")
    
    def _init_cache(self):
        """Initialize SQLite cache database"""
        self.conn = sqlite3.connect("embeddings_cache.db")
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                hash TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filename TEXT,
                doc_length INTEGER
            )
        ''')
        self.conn.commit()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to search engine with caching"""
        embeddings_list = []
        doc_ids = []
        
        print(f"Processing {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"Processed {i}/{len(documents)} documents...")
                
            # Check cache first
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE doc_id = ? AND hash = ?",
                (doc.doc_id, doc.hash)
            )
            result = cursor.fetchone()
            
            if result:
                # Use cached embedding
                embedding = np.frombuffer(result[0], dtype=np.float32)
                # print(f"Using cached embedding for {doc.doc_id}")
            else:
                # Generate new embedding
                embedding = self.model.encode([doc.cleaned_text])[0].astype(np.float32)
                # Store in cache
                cursor.execute(
                    "INSERT OR REPLACE INTO embeddings (doc_id, embedding, hash, updated_at, filename, doc_length) VALUES (?, ?, ?, datetime('now'), ?, ?)",
                    (doc.doc_id, embedding.tobytes(), doc.hash, doc.filename, doc.length)
                )
                self.conn.commit()
                # print(f"Generated new embedding for {doc.doc_id}")
            
            self.documents[doc.doc_id] = doc
            embeddings_list.append(embedding)
            doc_ids.append(doc.doc_id)
        
        # Build FAISS index
        if embeddings_list:
            print("Building search index...")
            embeddings_array = np.array(embeddings_list).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            embedding_dim = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(embeddings_array)
            self.index_doc_ids = doc_ids
            print(f"Index built with {len(doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("No documents indexed. Please index documents first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.index_doc_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.index_doc_ids):
                doc_id = self.index_doc_ids[idx]
                doc = self.documents[doc_id]
                
                # Generate explanation
                query_keywords = TextPreprocessor.extract_keywords(query)
                overlapping_keywords = set(query_keywords) & set(doc.keywords)
                overlap_ratio = len(overlapping_keywords) / max(len(query_keywords), 1)
                
                explanation = {
                    "matched_keywords": list(overlapping_keywords),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "document_length": doc.length,
                    "reasoning": f"Matched due to keywords: {list(overlapping_keywords)}"
                }
                
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "filename": doc.filename,
                    "preview": doc.cleaned_text[:200] + "..." if len(doc.cleaned_text) > 200 else doc.cleaned_text,
                    "explanation": explanation
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        cache_entries = cursor.fetchone()[0]
        
        return {
            "total_documents": len(self.documents),
            "index_built": self.index is not None,
            "index_size": len(self.index_doc_ids) if self.index else 0,
            "cache_entries": cache_entries
        }

# Initialize search engine
search_engine = SimpleSearchEngine()

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[dict]
    stats: dict

class IndexingResponse(BaseModel):
    message: str
    documents_indexed: int
    stats: dict

# API endpoints
@app.post("/index", response_model=IndexingResponse)
async def index_documents():
    """Index all documents from the data/docs folder"""
    try:
        data_folder = "./data/docs"
        if not os.path.exists(data_folder):
            raise HTTPException(status_code=404, detail="Data folder not found")
        
        # Find all text files
        txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
        if not txt_files:
            raise HTTPException(status_code=404, detail="No text files found in data/docs")
        
        documents = []
        print(f"Found {len(txt_files)} text files")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                filename = os.path.basename(file_path)
                doc_id = f"doc_{len(documents):03d}"
                
                document = Document(doc_id, content, filename)
                documents.append(document)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Add documents to search engine
        search_engine.add_documents(documents)
        
        stats = search_engine.get_stats()
        
        return IndexingResponse(
            message=f"Successfully indexed {len(documents)} documents",
            documents_indexed=len(documents),
            stats=stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents"""
    try:
        print(f"Searching for: '{request.query}'")
        results = search_engine.search(request.query, request.top_k)
        stats = search_engine.get_stats()
        
        return SearchResponse(
            results=results,
            stats=stats
        )
    
    except ValueError as e:
        if "No documents indexed" in str(e):
            raise HTTPException(status_code=400, detail="No documents indexed. Please index documents first.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get search engine statistics"""
    try:
        stats = search_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Multi-document Embedding Search Engine API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("./data/docs", exist_ok=True)
    
    print("Starting search engine server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    # Run the FastAPI application
    uvicorn.run(
        "simple_search:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )