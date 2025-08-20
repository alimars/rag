from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag_tool.pipeline import FocusedRAGPipeline
import os
import uvicorn
import shutil
from pathlib import Path
import glob
import json
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Offline RAG Pipeline",
    description="Private RAG system with translation and advanced retrieval",
    version="1.0"
)

# Ensure UTF-8 encoding for all JSON responses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Offline RAG Pipeline API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }

class QueryRequest(BaseModel):
    text: str
    target_lang: str = None
    return_original: bool = False

class ToolInput(BaseModel):
    query: str
    target_lang: Optional[str] = None
    return_original: bool = False

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Initialize pipeline
PIPELINE = None
_PIPELINE_INITIALIZED = False

@app.on_event("startup")
async def startup_event():
    global PIPELINE, _PIPELINE_INITIALIZED
    # Check if pipeline is already initialized
    if _PIPELINE_INITIALIZED and PIPELINE is not None:
        print("⏭️ Pipeline already initialized, skipping initialization")
        return
        
    try:
        docs_path = os.getenv("DOCS_PATH", "./documents")
        docs_lang = os.getenv("DOCS_LANG", "ar")
        print(f"Starting RAG pipeline initialization with docs_path={docs_path}, docs_lang={docs_lang}")
        
        # Check Ollama connectivity before initializing pipeline
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Checking Ollama connectivity at {ollama_base_url}...")
        try:
            import requests
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama is accessible")
            else:
                print(f"❌ Ollama returned status code {response.status_code}")
                raise Exception(f"Ollama connectivity failed with status {response.status_code}")
        except Exception as ollama_error:
            print(f"❌ Ollama connectivity check failed: {str(ollama_error)}")
            raise Exception(f"Cannot connect to Ollama at {ollama_base_url}. Please ensure Ollama is running and accessible.")
        
        print("Creating pipeline...")
        PIPELINE = FocusedRAGPipeline(docs_path, docs_lang)
        print("Pipeline created, starting initialization...")
        
        # Add timeout to initialization
        import concurrent.futures
        import threading
        def initialize_pipeline():
            try:
                PIPELINE.initialize()
                return True
            except Exception as e:
                print(f"Pipeline initialization failed: {str(e)}")
                raise
        
        # Run initialization with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(initialize_pipeline)
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if result:
                    print("Pipeline initialization completed successfully")
                    _PIPELINE_INITIALIZED = True
            except concurrent.futures.TimeoutError:
                print("❌ Pipeline initialization timed out after 5 minutes")
                PIPELINE = None
                raise Exception("Pipeline initialization timed out. This might be due to large documents, missing models, or connectivity issues.")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # We'll set PIPELINE to None to indicate initialization failure
        PIPELINE = None
        # Note: In a production environment, you might want to exit here
        # but for debugging purposes, we'll continue to allow health checks

# @app.post("/query")
# async def query_endpoint(request: QueryRequest):
#     if PIPELINE is None:
#         raise HTTPException(status_code=500, detail="Pipeline failed to initialize")
#     try:
#         result = PIPELINE.query(request.text, request.target_lang, request.return_original)
#         response_data = {
#             "response": result["original_response"],
#             "translation": result.get("translation"),
#             "source_language": result["source_language"]
#         }
#         # Add target_language only if target_lang was requested
#         if request.target_lang is not None:
#             response_data["target_language"] = request.target_lang
#         # Ensure proper UTF-8 encoding for Arabic text
#         return JSONResponse(content=response_data, headers={"Content-Type": "application/json; charset=utf-8"})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/invoke")
async def invoke_endpoint(input: ToolInput):
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="Pipeline failed to initialize")
    try:
        result = PIPELINE.query(input.query, input.target_lang, input.return_original)
        
        # If return_original is True, return the new format directly
        if input.return_original:
            # Serialize JSON with ensure_ascii=False to prevent Unicode escaping
            #json_content = json.dumps(result, ensure_ascii=False)
            return JSONResponse(content=result, media_type="application/json; charset=utf-8")
        
        # Otherwise, return the original format with detailed results
        response_data = {
            "response": result["original_response"],
            "detailed_results": result.get("detailed_results", []),
            "translation": result.get("translation"),
            "source_language": result["source_language"]
        }
        # Add target_language only if target_lang was requested
        if input.target_lang is not None:
            response_data["target_language"] = input.target_lang
        # Serialize JSON with ensure_ascii=False to prevent Unicode escaping
        #json_content = json.dumps(response_data, ensure_ascii=False)
        #print(json_content)
        return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    if PIPELINE is None:
        return {"status": "unhealthy", "initialized": False, "error": "Pipeline failed to initialize"}
    
    # Check cache status
    cache_exists = os.path.exists(CACHE_DIR)
    cache_files = len(glob.glob(os.path.join(CACHE_DIR, "*.pkl"))) if cache_exists else 0
    
    return {
        "status": "healthy", 
        "initialized": PIPELINE.is_initialized if PIPELINE else False,
        "cache": {
            "exists": cache_exists,
            "file_count": cache_files
        }
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear all cached data"""
    if os.path.exists(CACHE_DIR):
        try:
            # Instead of removing the directory, remove only its contents
            # This avoids issues with Docker volume mounts
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            return {"message": "Cache cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
    return {"message": "Cache directory does not exist"}

@app.get("/cache/status")
def cache_status():
    """Get cache status and information"""
    if not os.path.exists(CACHE_DIR):
        return {"exists": False, "file_count": 0, "size": 0}
    
    try:
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
        total_size = sum(os.path.getsize(f) for f in cache_files)
        return {
            "exists": True,
            "file_count": len(cache_files),
            "size": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )
    
    openapi_schema["x-openwebui"] = {
        "tool_type": "rag",
        "enabled_by_default": True,
        "icon": "📚"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)