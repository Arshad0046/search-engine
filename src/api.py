from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import glob
from search_engine import SearchEngine
from utils import Document

app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="A lightweight embedding-based search engine with caching",
    version="1.0.0"
)

# Global search engine instance
search_engine = SearchEngine()

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

@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    # You can preload documents here if needed
    pass

@app.post("/index", response_model=IndexingResponse)
async def index_documents(data_folder: str = "./data/docs"):
    """Index all documents from a folder"""
    try:
        if not os.path.exists(data_folder):
            raise HTTPException(status_code=404, detail="Data folder not found")
        
        # Find all text files
        txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
        documents = []
        
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
        results = search_engine.search(request.query, request.top_k)
        stats = search_engine.get_stats()
        
        return SearchResponse(
            results=results,
            stats=stats
        )
    
    except ValueError as e:
        if "Index not built" in str(e):
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