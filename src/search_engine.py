import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from .embedder import EmbeddingGenerator
from .cache_manager import CacheManager
from .utils import Document, TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, cache_file: str = "embeddings_cache.db"):
        self.embedder = EmbeddingGenerator()
        self.cache_manager = CacheManager(cache_file)
        self.index = None
        self.documents = {}
        self.document_embeddings = {}
        self.is_index_built = False
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the search engine with caching"""
        embeddings_to_index = []
        docs_to_index = []
        
        for doc in documents:
            # Check cache first
            cached_embedding = self.cache_manager.get_cached_embedding(
                doc.doc_id, doc.hash
            )
            
            if cached_embedding is not None:
                logger.info(f"Using cached embedding for {doc.doc_id}")
                embedding = cached_embedding
            else:
                logger.info(f"Generating new embedding for {doc.doc_id}")
                embedding = self.embedder.generate_embedding(doc.cleaned_text)
                # Store in cache
                self.cache_manager.store_embedding(
                    doc.doc_id, embedding, doc.hash, doc.filename, doc.length
                )
            
            self.documents[doc.doc_id] = doc
            self.document_embeddings[doc.doc_id] = embedding
            embeddings_to_index.append(embedding)
            docs_to_index.append(doc.doc_id)
        
        # Build or update FAISS index
        if embeddings_to_index:
            self._build_index(embeddings_to_index, docs_to_index)
    
    def _build_index(self, embeddings: List[np.ndarray], doc_ids: List[str]):
        """Build FAISS index with embeddings"""
        if not embeddings:
            return
        
        embedding_dim = len(embeddings[0])
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        if self.index is None:
            # Create new index
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index_doc_ids = []
        else:
            # For simplicity, we'll rebuild the entire index
            # In production, you might want to use IndexIDMap for incremental updates
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index_doc_ids = []
        
        self.index.add(embeddings_array)
        self.index_doc_ids.extend(doc_ids)
        self.is_index_built = True
        
        logger.info(f"Index built with {len(doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.is_index_built:
            raise ValueError("Index not built. Please add documents first.")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.index_doc_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.index_doc_ids):  # Valid index
                doc_id = self.index_doc_ids[idx]
                doc = self.documents[doc_id]
                
                # Generate explanation
                explanation = self._generate_explanation(query, doc, score)
                
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "filename": doc.filename,
                    "preview": doc.cleaned_text[:200] + "..." if len(doc.cleaned_text) > 200 else doc.cleaned_text,
                    "explanation": explanation
                })
        
        return results
    
    def _generate_explanation(self, query: str, doc: Document, score: float) -> Dict[str, Any]:
        """Generate explanation for why document was matched"""
        query_keywords = TextPreprocessor.extract_keywords(query)
        doc_keywords = doc.keywords
        
        # Find overlapping keywords
        overlapping_keywords = set(query_keywords) & set(doc_keywords)
        
        # Simple overlap ratio
        overlap_ratio = len(overlapping_keywords) / max(len(query_keywords), 1)
        
        # Document length normalization (simple heuristic)
        length_score = 1.0 / (1.0 + 0.01 * max(0, doc.length - 100))
        
        return {
            "matched_keywords": list(overlapping_keywords),
            "overlap_ratio": round(overlap_ratio, 3),
            "document_length": doc.length,
            "length_normalization": round(length_score, 3),
            "reasoning": f"Document matched due to keyword overlap: {list(overlapping_keywords)}"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        cache_stats = self.cache_manager.get_cache_stats()
        return {
            "total_documents": len(self.documents),
            "index_built": self.is_index_built,
            "index_size": len(self.index_doc_ids) if self.is_index_built else 0,
            "cache_stats": cache_stats
        }