givr readme
🔍 Multi-document Embedding Search Engine
A lightweight embedding-based search engine with caching built for AI Engineer Intern assignment. This system can efficiently search through 100-200 text documents using semantic similarity with intelligent caching to avoid recomputing embeddings.

🚀 Features
🤖 Smart Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 for high-quality embeddings

💾 Intelligent Caching: SQLite-based cache system prevents recomputing unchanged documents

🔍 Vector Search: FAISS for blazing-fast similarity search with cosine similarity

🌐 REST API: FastAPI with automatic Swagger documentation

📊 Result Explanations: Detailed matching reasons with keyword overlap analysis

⚡ Performance: Optimized for 100-200 documents with efficient batch processing

🛠️ Technology Stack
Embeddings: Sentence Transformers

Vector Database: FAISS (Facebook AI Similarity Search)

Cache: SQLite with SHA256 document hashing

API Framework: FastAPI

Language: Python 3.8+
