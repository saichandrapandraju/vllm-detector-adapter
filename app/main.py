from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

#FIXME: ModuleNotFoundError: No module named 'app'
from app.api.routes import evaluate, config
from app.core.errors import AdapterError

app = FastAPI(
    title="vLLM Detector Adapter",
    description="A model-agnostic adapter for enabling LLM-as-a-judge detections and evaluations with vLLM-hosted models",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(evaluate.router)
app.include_router(config.router)

# Error handling
@app.exception_handler(AdapterError)
async def adapter_error_handler(request: Request, exc: AdapterError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)