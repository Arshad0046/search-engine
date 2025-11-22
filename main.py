import uvicorn
import os
from src.api import app

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("./data/docs", exist_ok=True)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )