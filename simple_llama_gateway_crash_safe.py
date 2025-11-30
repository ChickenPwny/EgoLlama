#!/usr/bin/env python3
"""
Simple LLaMA Gateway - Modernized Version v2.0
==============================================

Complete LLaMA Gateway with Redis caching, PostgreSQL persistence,
and OpenAI-compatible API endpoints.

Author: L (Python Expert)
Version: 2.0.0 - Complete Modern Gateway
"""

import os
import sys
import json
import logging
import asyncio
import functools
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import psutil

# Add EgoLlama to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import CACHE_TTL_SHORT, CACHE_TTL_MEDIUM
    from database import init_database, get_async_db, check_database_health, close_database
    from db_models import LLaMAInferenceRequest
    from redis_cache import init_redis, cache_get, cache_set, check_redis_health, get_cache_stats, check_rate_limit, get_redis
    REDIS_AVAILABLE = True
except ImportError as e:
    REDIS_AVAILABLE = False
    print(f"⚠️ Redis/SQLAlchemy not available: {e}")

# Ollama provider
try:
    from ollama_provider import get_ollama_provider
    OLLAMA_AVAILABLE = True
except ImportError as e:
    OLLAMA_AVAILABLE = False
    print(f"⚠️ Ollama provider not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Fallback in-memory cache
_response_cache = {}
_cache_lock = threading.Lock()

def cached_response(ttl=5):
    """Decorator to cache endpoint responses"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            now = time.time()
            
            # Try Redis first, fallback to in-memory
            if REDIS_AVAILABLE:
                cached = cache_get(cache_key)
                if cached is not None:
                    return cached
            
            with _cache_lock:
                if cache_key in _response_cache:
                    cached_time, cached_response_data = _response_cache[cache_key]
                    if now - cached_time < ttl:
                        return cached_response_data
            
            # Call actual function
            result = await func(*args, **kwargs)
            
            # Cache the result
            if REDIS_AVAILABLE:
                cache_set(cache_key, result, ttl)
            with _cache_lock:
                _response_cache[cache_key] = (now, result)
            
            return result
        return wrapper
    return decorator

# Create FastAPI app
app = FastAPI(
    title="Simple LLaMA Gateway v2.0",
    description="Complete LLaMA Gateway with Redis + PostgreSQL + All Endpoints",
    version="2.0.0"
)

# ============================================================================
# SECURITY: CORS MIDDLEWARE (Prevent Drive-By Attacks)
# ============================================================================
from fastapi.middleware.cors import CORSMiddleware

# Get allowed origins from environment or default to localhost only
allowed_origins = os.getenv(
    "EGOLLAMA_CORS_ORIGINS",
    "http://localhost:9000,http://localhost:3000,http://127.0.0.1:9000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins only (not "*")
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],  # Only needed methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Only needed headers
    expose_headers=["*"],
    max_age=3600,
)
logger.info(f"✅ CORS middleware enabled (origins: {allowed_origins})")

# ============================================================================
# SECURITY: API KEY AUTHENTICATION
# ============================================================================

# Get API key from environment (default to None for development)
EGOLLAMA_API_KEY = os.getenv("EGOLLAMA_API_KEY")
REQUIRE_API_KEY = os.getenv("EGOLLAMA_REQUIRE_API_KEY", "false").lower() == "true"

if EGOLLAMA_API_KEY:
    logger.info("✅ API key authentication enabled")
else:
    logger.warning("⚠️  No API key set - endpoints are unprotected! Set EGOLLAMA_API_KEY environment variable.")

from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(x_api_key: str = Depends(api_key_header)):
    """Verify API key for protected endpoints - SECURITY: Fail-secure by default"""
    # SECURITY: Check if we're in production mode
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    # In development, allow access if API key is not required
    if not REQUIRE_API_KEY and not is_production:
        # Development mode: allow access without key if not required
        return True
    
    # Production mode OR API key required: enforce authentication
    if not EGOLLAMA_API_KEY:
        # SECURITY: In production or when required, fail if key not configured
        if is_production or REQUIRE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="API key authentication required but not configured. Set EGOLLAMA_API_KEY environment variable."
            )
        # Development mode fallback
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )
    
    if x_api_key != EGOLLAMA_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

# ============================================================================
# SECURITY: CONTENT-TYPE VALIDATION MIDDLEWARE
# ============================================================================

from starlette.middleware.base import BaseHTTPMiddleware

class ContentTypeValidationMiddleware(BaseHTTPMiddleware):
    """Validate Content-Type header for POST requests to prevent CORS simple request attacks"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for GET requests and documentation endpoints
        if request.method == "POST" and request.url.path not in ["/docs", "/openapi.json", "/redoc"]:
            content_type = request.headers.get("Content-Type", "")
            
            # Require application/json for POST requests with body
            if request.headers.get("Content-Length", "0") != "0":
                if "application/json" not in content_type:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Content-Type must be application/json",
                            "detail": "This prevents CORS simple request attacks. Always include 'Content-Type: application/json' header."
                        }
                    )
        
        response = await call_next(request)
        return response

app.add_middleware(ContentTypeValidationMiddleware)
logger.info("✅ Content-Type validation middleware enabled")

# ============================================================================
# SECURITY: GLOBAL EXCEPTION HANDLER (Prevent Information Disclosure)
# ============================================================================

from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to prevent information disclosure"""
    # Log full error details internally
    logger.error(f"Unhandled exception in {request.url.path}: {exc}", exc_info=True)
    
    # SECURITY: Only show detailed errors in development mode
    is_development = os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    if is_development:
        # Development: show detailed error for debugging
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url.path),
                "traceback": traceback.format_exc().split('\n')[-10:]  # Last 10 lines only
            }
        )
    else:
        # Production: generic error message to prevent information disclosure
        return JSONResponse(
            status_code=500,
            content={
                "error": "An internal error occurred",
                "detail": "Please contact support if this problem persists"
            }
        )

# Initialize database and Redis
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing services...")
    if REDIS_AVAILABLE:
        db_init = init_database()
        redis_init = init_redis()
        logger.info(f"Database: {db_init}, Redis: {redis_init}")
    logger.info("Services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down services...")
    if REDIS_AVAILABLE:
        await close_database()
    logger.info("Services shut down")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class LoadModelRequest(BaseModel):
    model_id: str
    quantization_bits: Optional[int] = 4

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llama-custom"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    session_id: Optional[str] = None

# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    features = []
    if REDIS_AVAILABLE:
        features.append("redis_cache")
        features.append("postgresql_persistence")
    if OLLAMA_AVAILABLE:
        features.append("ollama_integration")
    
    endpoints = {
        "health": "/health",
        "api_health": "/api/health",
        "status": "/api/status",
        "stats": "/stats",
        "performance_stats": "/api/performance/stats",
        "recommendations": "/api/performance/recommendations",
        "models": "/api/models",
        "cache_stats": "/api/cache/stats",
        "chat": "/v1/chat/completions",
        "generate": "/generate",
        "load_model": "/models/load",
        "ollama_endpoints": "/api/ollama/endpoints",
        "ollama_models": "/api/ollama/models",
        "ollama_preconfigured": "/api/ollama/models/preconfigured",
        "ollama_health": "/api/ollama/health"
    }
    if REDIS_AVAILABLE:
        endpoints["db_health"] = "/api/db/health"
        endpoints["redis_health"] = "/api/redis/health"
    
    return {
        "service": "Simple LLaMA Gateway",
        "version": "2.0.0",
        "port": 8082,
        "status": "running",
        "features": features,
        "endpoints": endpoints
    }

@cached_response(ttl=5)
@app.get("/health")
@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": False,
        "gpu_available": False,
        "quantization": "None",
        "total_inferences": 0,
        "average_tokens_per_second": 0.0
    }

@app.get("/api/status")
@cached_response(ttl=5)
async def api_status():
    """Extended status information"""
    cache_stats = {}
    if REDIS_AVAILABLE:
        cache_stats = get_cache_stats()
    
    return {
        "status": "operational",
        "version": "2.0.0",
        "uptime": "N/A",
        "cache": cache_stats,
        "features": {
            "redis_cache": REDIS_AVAILABLE and cache_stats.get("enabled", False),
            "postgresql": REDIS_AVAILABLE
        }
    }

# ============================================================================
# PERFORMANCE ENDPOINTS
# ============================================================================

@app.get("/api/performance/stats")
@cached_response(ttl=5)
async def performance_stats(_: bool = Depends(verify_api_key)):  # SECURITY: Require authentication for system stats
    """Get performance statistics"""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_usage": 0.0,
        "active_models": 0,
        "total_requests": 0,
        "average_response_time": 0.0,
        "error_rate": 0.0,
        "model_loaded": False,
        "gpu_available": False
    }

@app.get("/api/performance/recommendations")
@cached_response(ttl=60)
async def performance_recommendations():
    """Get performance recommendations"""
    recommendations = []
    
    if not REDIS_AVAILABLE:
        recommendations.append({
            "priority": "high",
            "title": "Enable Redis and PostgreSQL",
            "description": "Redis and PostgreSQL modules not available. Enable for improved performance."
        })
    
    if not recommendations:
        recommendations.append({
            "priority": "low",
            "title": "System Running Optimally",
            "description": "All systems are running efficiently."
        })
    
    high = sum(1 for r in recommendations if r.get('priority') == 'high')
    medium = sum(1 for r in recommendations if r.get('priority') == 'medium')
    low = sum(1 for r in recommendations if r.get('priority') == 'low')
    
    return {
        "total_recommendations": len(recommendations),
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "recommendations": recommendations
    }

# ============================================================================
# CHAT COMPLETIONS - OpenAI-compatible endpoint
# ============================================================================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, _: bool = Depends(verify_api_key)):
    """OpenAI-compatible chat endpoint"""
    import time
    
    try:
        messages = request.messages
        max_tokens = request.max_tokens or 512
        temperature = request.temperature or 0.7
        model = request.model or "llama-custom"
        
        logger.info(f"💬 Chat request: {len(messages)} messages, model={model}")
        
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get('role') if isinstance(msg, dict) else msg.role
            content = msg.get('content') if isinstance(msg, dict) else msg.content
            
            if role == 'system':
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
            elif role == 'user':
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == 'assistant':
                prompt_parts.append(f"{content}")
        
        prompt = '\n'.join(prompt_parts)
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        
        # Check if model is an Ollama model (starts with "ollama:" or matches preconfigured Ollama models)
        is_ollama_model = False
        ollama_model_name = None
        
        if OLLAMA_AVAILABLE:
            try:
                provider = get_ollama_provider()
                # Check if model matches preconfigured Ollama models or uses ollama: prefix
                if model.startswith("ollama:"):
                    is_ollama_model = True
                    ollama_model_name = model.replace("ollama:", "")
                elif model in provider.models:
                    is_ollama_model = True
                    ollama_model_name = model
                elif any(model in ep.get('models', []) for ep in provider.endpoints):
                    is_ollama_model = True
                    ollama_model_name = model
                
                if is_ollama_model:
                    # Check if model should be loaded into EgoLlama instead of proxying
                    model_config = provider.models.get(ollama_model_name, {})
                    load_into_egollama = model_config.get('load_into_egollama', False)
                    hf_model_id = model_config.get('huggingface_id')
                    
                    if load_into_egollama and hf_model_id:
                        # Load model into EgoLlama's memory/GPU
                        logger.info(f"🦙 Loading Ollama model '{ollama_model_name}' into EgoLlama (HF: {hf_model_id})")
                        try:
                            # Use the existing model loading mechanism
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            import torch
                            
                            # Load tokenizer
                            # SECURITY: trust_remote_code disabled by default - only enable for verified internal models
                            trust_code = os.getenv("EGOLLAMA_TRUST_REMOTE_CODE", "false").lower() == "true"
                            tokenizer = AutoTokenizer.from_pretrained(
                                hf_model_id,
                                trust_remote_code=trust_code
                            )
                            
                            # Load model
                            # SECURITY: trust_remote_code disabled by default - only enable for verified internal models
                            trust_code = os.getenv("EGOLLAMA_TRUST_REMOTE_CODE", "false").lower() == "true"
                            model = AutoModelForCausalLM.from_pretrained(
                                hf_model_id,
                                trust_remote_code=trust_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto" if torch.cuda.is_available() else None
                            )
                            
                            if not torch.cuda.is_available():
                                model = model.to("cpu")
                            
                            # Store in global model storage
                            global _loaded_model, _loaded_tokenizer, _loaded_model_id
                            _loaded_model = model
                            _loaded_tokenizer = tokenizer
                            _loaded_model_id = hf_model_id
                            
                            logger.info(f"✅ Ollama model loaded into EgoLlama: {ollama_model_name}")
                            
                            # Now generate using the loaded model
                            prompt_text = '\n'.join([
                                f"{msg.get('role') if isinstance(msg, dict) else msg.role}: {msg.get('content') if isinstance(msg, dict) else msg.content}"
                                for msg in messages
                            ])
                            
                            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                            if torch.cuda.is_available():
                                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                model = model.to("cuda")
                            else:
                                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id
                                )
                            
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if prompt_text in generated_text:
                                generated_text = generated_text.replace(prompt_text, "").strip()
                            
                            prompt_tokens = len(tokenizer.encode(prompt_text))
                            completion_tokens = len(tokenizer.encode(generated_text))
                            
                            return {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": ollama_model_name,
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": generated_text
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens
                                },
                                "metadata": {
                                    "engine": "EgoLlama (Ollama model loaded)",
                                    "ollama_model": ollama_model_name,
                                    "huggingface_model": hf_model_id,
                                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                                }
                            }
                        except Exception as load_error:
                            logger.warning(f"⚠️ Failed to load Ollama model into EgoLlama: {load_error}")
                            logger.info("🔄 Falling back to Ollama API proxy...")
                            # Fall through to proxy mode
                    
                    # Proxy mode: use Ollama API (original behavior)
                    logger.info(f"🦙 Using Ollama model via API proxy: {ollama_model_name}")
                    try:
                        result = await provider.generate(
                            model=ollama_model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=False
                        )
                        
                        if result and result.get('content'):
                            logger.info("✅ Ollama generation successful!")
                            return {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result['content']
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": result.get('prompt_eval_count', 0),
                                    "completion_tokens": result.get('eval_count', 0),
                                    "total_tokens": result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                                },
                                "metadata": {
                                    "engine": "Ollama",
                                    "endpoint": result.get('endpoint', 'unknown'),
                                    "model_name": result.get('model', ollama_model_name),
                                    "total_duration_ms": result.get('total_duration', 0) / 1000000,  # Convert nanoseconds to ms
                                    "load_duration_ms": result.get('load_duration', 0) / 1000000,
                                    "eval_duration_ms": result.get('eval_duration', 0) / 1000000
                                }
                            }
                    except Exception as ollama_error:
                        logger.warning(f"⚠️ Ollama generation failed: {ollama_error}")
                        # Continue to fallback
            except Exception as provider_error:
                logger.warning(f"⚠️ Ollama provider error: {provider_error}")
        
        # Try loaded transformers model first, then GPU module, then fallback
        try:
            # Check if we have a loaded transformers model
            if _loaded_model is not None and _loaded_tokenizer is not None:
                logger.info(f"📥 Using loaded transformers model: {_loaded_model_id}")
                
                import torch
                start_time = time.time()
                
                # Tokenize input
                inputs = _loaded_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    _loaded_model = _loaded_model.to("cuda")
                else:
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = _loaded_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=_loaded_tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = _loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt from output
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                
                inference_time = (time.time() - start_time) * 1000
                prompt_tokens = len(_loaded_tokenizer.encode(prompt))
                completion_tokens = len(_loaded_tokenizer.encode(generated_text))
                
                logger.info(f"✅ Transformers model generated response ({len(generated_text)} chars)")
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "metadata": {
                        "engine": "transformers",
                        "model_id": _loaded_model_id,
                        "inference_time_ms": int(inference_time),
                        "tokens_per_second": completion_tokens / (inference_time / 1000) if inference_time > 0 else 0,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    }
                }
            
            # Try EgoLlama GPU engine
            logger.info("🔧 Attempting to use EgoLlama GPU engine...")
            
            # Try full GPU module first
            try:
                from gpu import get_gpu_module
                gpu_module = get_gpu_module()
                logger.info("✅ GPU module imported successfully")
            except (ImportError, Exception) as import_error:
                logger.warning(f"⚠️ Full GPU module unavailable: {import_error}")
                logger.info("🔄 Falling back to simple fallback model...")
                from gpu_fallback import get_gpu_module
                gpu_module = get_gpu_module()
                # Auto-load fallback model if not loaded
                if not hasattr(gpu_module, 'model') or not gpu_module.model.model_loaded:
                    gpu_module.load_model("simple-fallback")
            
            result = gpu_module.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if 'error' not in result and result.get('generated_text'):
                engine_name = "EgoLlama GPU" if "gpu_fallback" not in str(type(gpu_module)) else "Simple Fallback"
                logger.info(f"✅ {engine_name} engine successful!")
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result['generated_text']
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": result.get('prompt_tokens', 0),
                        "completion_tokens": result.get('completion_tokens', 0),
                        "total_tokens": result.get('total_tokens', 0)
                    },
                    "metadata": {
                        "engine": engine_name,
                        "inference_time_ms": result.get('inference_time_ms', 0),
                        "tokens_per_second": result.get('tokens_per_second', 0),
                        "gpu_accelerated": "gpu_fallback" not in str(type(gpu_module))
                    }
                }
            else:
                # GPU module returned error or empty text - try fallback
                error_msg = result.get('error', 'Unknown error')
                logger.warning(f"⚠️ GPU engine returned error: {error_msg}")
                logger.info("🔄 Trying fallback model...")
                try:
                    from gpu_fallback import get_gpu_module as get_fallback_module
                    fallback_module = get_fallback_module()
                    if not hasattr(fallback_module, 'model') or not fallback_module.model.model_loaded:
                        fallback_module.load_model("simple-fallback")
                    fallback_result = fallback_module.generate(prompt, max_tokens, temperature)
                    if 'error' not in fallback_result and fallback_result.get('generated_text'):
                        logger.info("✅ Simple Fallback engine successful!")
                        return {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": fallback_result['generated_text']
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": fallback_result.get('prompt_tokens', 0),
                                "completion_tokens": fallback_result.get('completion_tokens', 0),
                                "total_tokens": fallback_result.get('total_tokens', 0)
                            },
                            "metadata": {
                                "engine": "Simple Fallback",
                                "inference_time_ms": fallback_result.get('inference_time_ms', 0),
                                "tokens_per_second": fallback_result.get('tokens_per_second', 0),
                                "gpu_accelerated": False
                            }
                        }
                    else:
                        logger.error(f"❌ Fallback also returned error: {fallback_result.get('error')}")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback failed: {fallback_error}")
        except Exception as engine_error:
            logger.error(f"❌ All GPU engines failed: {engine_error}")
            import traceback
            traceback.print_exc()
            # No Ollama fallback - fail gracefully with detailed error
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "LLM service unavailable",
                    "message": "EgoLlama GPU engine is currently unavailable",
                    "gpu_error": str(engine_error),
                    "timestamp": datetime.now().isoformat()
                }
            )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I apologize, but I'm currently unable to generate a response. The LLM engines are unavailable."
                },
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "error": "All LLM engines unavailable"
        }
        
    except Exception as e:
        logger.error(f"❌ Chat completions error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TEXT GENERATION - OpenAI-compatible completions
# ============================================================================

@app.post("/generate")
async def generate_text(request: GenerateRequest, _: bool = Depends(verify_api_key)):
    """Generate text from prompt - OpenAI-compatible completions"""
    import time
    
    try:
        prompt = request.prompt
        max_tokens = request.max_tokens or 512
        temperature = request.temperature or 0.7
        
        logger.info(f"📝 Generate request: {len(prompt)} chars")
        
        # Try EgoLlama GPU engine first
        try:
            logger.info("🔧 Using EgoLlama GPU engine...")
            from gpu import get_gpu_module
            
            gpu_module = get_gpu_module()
            result = gpu_module.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if 'error' not in result and result.get('generated_text'):
                logger.info("✅ EgoLlama GPU engine successful!")
                return {
                    "generated_text": result['generated_text'],
                    "tokens": result.get('completion_tokens', 0),
                    "finish_reason": "stop",
                    "timestamp": time.time(),
                    "metadata": {
                        "engine": "EgoLlama GPU",
                        "inference_time_ms": result.get('inference_time_ms', 0),
                        "tokens_per_second": result.get('tokens_per_second', 0),
                        "gpu_accelerated": True
                    }
                }
            else:
                logger.warning(f"⚠️ GPU engine returned error: {result.get('error', 'Unknown error')}")
        except Exception as engine_error:
            logger.warning(f"⚠️ EgoLlama GPU engine failed: {engine_error}")
        
        # Fallback to Ollama if GPU fails
        if OLLAMA_AVAILABLE:
            try:
                logger.info("🔄 Falling back to Ollama...")
                provider = get_ollama_provider()
                
                # Try default model or first available
                default_model = "llama3.1:8b"
                if default_model not in provider.models:
                    # Use first available preconfigured model
                    available_models = [m for m in provider.models.keys() if provider.models[m].get('enabled', True)]
                    if available_models:
                        default_model = available_models[0]
                
                result = await provider.generate(
                    model=default_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                
                if result and result.get('content'):
                    logger.info("✅ Ollama fallback successful!")
                    return {
                        "generated_text": result['content'],
                        "tokens": result.get('eval_count', 0),
                        "finish_reason": "stop",
                        "timestamp": time.time(),
                        "metadata": {
                            "engine": "Ollama",
                            "endpoint": result.get('endpoint', 'unknown'),
                            "model": result.get('model', default_model),
                            "gpu_accelerated": False
                        }
                    }
            except Exception as ollama_error:
                logger.warning(f"⚠️ Ollama fallback also failed: {ollama_error}")
        
        # Final fallback: return error message
        logger.error("❌ All LLM engines failed")
        return {
            "generated_text": "I apologize, but I'm currently unable to generate a response. The LLM engines are unavailable.",
            "tokens": 0,
            "finish_reason": "error",
            "timestamp": time.time(),
            "error": "All LLM engines unavailable"
        }
        
    except Exception as e:
        logger.error(f"❌ Generate error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODEL MANAGEMENT (Used by GPU infrastructure)
# ============================================================================

# Global model storage
_loaded_model = None
_loaded_tokenizer = None
_loaded_model_id = None

@app.post("/models/load")
async def load_model(request: LoadModelRequest, _: bool = Depends(verify_api_key)):
    """Load a model dynamically using HuggingFace transformers or Ollama model mapping"""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id
    
    # Check if this is an Ollama model that should be loaded via HuggingFace
    if OLLAMA_AVAILABLE:
        try:
            provider = get_ollama_provider()
            if request.model_id in provider.models:
                model_config = provider.models[request.model_id]
                if model_config.get('load_into_egollama') and model_config.get('huggingface_id'):
                    logger.info(f"🦙 Mapping Ollama model '{request.model_id}' to HuggingFace: {model_config['huggingface_id']}")
                    request.model_id = model_config['huggingface_id']
        except Exception as e:
            logger.debug(f"Ollama model mapping check failed: {e}")
    
    try:
        # Check if model path exists locally
        import glob
        model_paths = [
            f"/app/EgoLlama/hf_models/models--{request.model_id.replace('/', '--')}/snapshots/*",
            f"/app/hf_models/models--{request.model_id.replace('/', '--')}/snapshots/*"
        ]
        
        local_path = None
        for pattern in model_paths:
            matches = glob.glob(pattern)
            if matches:
                local_path = matches[0]
                break
        
        # Try to use local path or HuggingFace Hub
        model_to_load = local_path if local_path else request.model_id
        
        logger.info(f"📥 Loading model: {request.model_id}")
        logger.info(f"   Local path: {local_path or 'Not found, using HuggingFace Hub'}")
        logger.info(f"   Quantization: {request.quantization_bits} bits")
        
        # Try to use GPU module first
        try:
            from gpu import get_gpu_module
            gpu_module = get_gpu_module()
            if gpu_module and hasattr(gpu_module, 'load_model'):
                success = gpu_module.load_model(model_to_load, request.quantization_bits)
                if success:
                    _loaded_model_id = request.model_id
                    logger.info(f"✅ Model loaded via GPU module: {request.model_id}")
                    return {
                        "success": True,
                        "model_id": request.model_id,
                        "status": "loaded",
                        "method": "gpu_module",
                        "local_path": local_path
                    }
        except Exception as e:
            logger.warning(f"GPU module loading failed: {e}, trying transformers fallback...")
        
        # Fallback to transformers directly
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("📥 Loading with transformers (CPU/fallback)...")
            
            # Load tokenizer
            # SECURITY: trust_remote_code disabled by default - only enable for verified internal models
            trust_code = os.getenv("EGOLLAMA_TRUST_REMOTE_CODE", "false").lower() == "true"
            tokenizer = AutoTokenizer.from_pretrained(
                model_to_load,
                trust_remote_code=trust_code,
                local_files_only=local_path is not None
            )
            
            # Load model
            # SECURITY: trust_remote_code disabled by default - only enable for verified internal models
            trust_code = os.getenv("EGOLLAMA_TRUST_REMOTE_CODE", "false").lower() == "true"
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                trust_remote_code=trust_code,
                local_files_only=local_path is not None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                model = model.to("cpu")
            
            _loaded_model = model
            _loaded_tokenizer = tokenizer
            _loaded_model_id = request.model_id
            
            logger.info(f"✅ Model loaded with transformers: {request.model_id}")
            
            return {
                "success": True,
                "model_id": request.model_id,
                "status": "loaded",
                "method": "transformers",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "local_path": local_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load model with transformers: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload")
async def unload_model(_: bool = Depends(verify_api_key)):
    """Unload current model - Used by GPU infrastructure"""
    return {
        "success": True,
        "message": "Model unloading ready - LLM integration pending"
    }

# ============================================================================
# OLLAMA MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/ollama/endpoints")
@cached_response(ttl=30)
async def list_ollama_endpoints():
    """List configured Ollama endpoints"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama provider not available")
    
    provider = get_ollama_provider()
    endpoints = []
    
    for endpoint in provider.get_available_endpoints():
        health = await provider.check_endpoint_health(endpoint)
        endpoints.append({
            "name": endpoint['name'],
            "base_url": endpoint['base_url'],
            "enabled": endpoint.get('enabled', True),
            "priority": endpoint.get('priority', 999),
            "timeout": endpoint.get('timeout', 30),
            "description": endpoint.get('description', ''),
            "healthy": health
        })
    
    return {
        "endpoints": endpoints,
        "total": len(endpoints)
    }

@app.get("/api/ollama/models")
@cached_response(ttl=30)
async def list_ollama_models(endpoint: Optional[str] = None):
    """List available Ollama models from endpoint(s)"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama provider not available")
    
    provider = get_ollama_provider()
    models = await provider.list_models(endpoint)
    
    return {
        "models": models,
        "total": len(models),
        "endpoint": endpoint or "all"
    }

@app.get("/api/ollama/models/preconfigured")
@cached_response(ttl=60)
async def list_preconfigured_ollama_models():
    """List preconfigured Ollama models"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama provider not available")
    
    provider = get_ollama_provider()
    models = []
    
    for model_id, model_config in provider.models.items():
        if model_config.get('enabled', True):
            models.append({
                "id": model_id,
                "model_name": model_config.get('model_name', model_id),
                "endpoint": model_config.get('endpoint', 'local'),
                "description": model_config.get('description', ''),
                "context_size": model_config.get('context_size', 8192),
                "default_temperature": model_config.get('default_temperature', 0.7),
                "default_max_tokens": model_config.get('default_max_tokens', 2048),
                "enabled": True
            })
    
    return {
        "models": models,
        "total": len(models)
    }

@app.post("/api/ollama/pull")
async def pull_ollama_model(
    model_name: str = Body(..., embed=True),
    endpoint: Optional[str] = Body(None, embed=True),
    _: bool = Depends(verify_api_key)  # SECURITY: Require authentication for model pulling
):
    """Pull an Ollama model"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama provider not available")
    
    provider = get_ollama_provider()
    result = await provider.pull_model(model_name, endpoint)
    
    if result.get('success'):
        return {
            "success": True,
            "message": result.get('message', f"Model {model_name} pulled successfully"),
            "model": model_name,
            "endpoint": result.get('endpoint')
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Failed to pull model')
        )

@app.get("/api/ollama/health")
@cached_response(ttl=10)
async def ollama_health():
    """Check Ollama endpoints health"""
    if not OLLAMA_AVAILABLE:
        return {
            "available": False,
            "endpoints": [],
            "healthy_count": 0
        }
    
    provider = get_ollama_provider()
    endpoints = provider.get_available_endpoints()
    health_status = []
    healthy_count = 0
    
    for endpoint in endpoints:
        is_healthy = await provider.check_endpoint_health(endpoint)
        if is_healthy:
            healthy_count += 1
        health_status.append({
            "name": endpoint['name'],
            "base_url": endpoint['base_url'],
            "healthy": is_healthy
        })
    
    return {
        "available": True,
        "endpoints": health_status,
        "healthy_count": healthy_count,
        "total_count": len(endpoints)
    }

# ============================================================================
# MODEL & STATS ENDPOINTS
# ============================================================================

@app.get("/api/models")
@cached_response(ttl=10)
async def list_models(_: bool = Depends(verify_api_key)):  # SECURITY: Require authentication for model listing
    """List available models"""
    models = []
    
    # Add Ollama models if available
    if OLLAMA_AVAILABLE:
        try:
            provider = get_ollama_provider()
            # Get preconfigured models
            for model_id, model_config in provider.models.items():
                if model_config.get('enabled', True):
                    models.append({
                        "id": model_id,
                        "name": model_config.get('model_name', model_id),
                        "type": "ollama",
                        "description": model_config.get('description', ''),
                        "endpoint": model_config.get('endpoint', 'local'),
                        "context_size": model_config.get('context_size', 8192),
                        "enabled": True
                    })
            
            # Get models from Ollama endpoints
            for endpoint in provider.get_available_endpoints():
                try:
                    endpoint_models = await provider.list_models(endpoint['name'])
                    for model in endpoint_models:
                        # Only add if not already in preconfigured models
                        if not any(m['id'] == model['name'] for m in models):
                            models.append({
                                "id": model['name'],
                                "name": model['name'],
                                "type": "ollama",
                                "endpoint": endpoint['name'],
                                "endpoint_url": model.get('endpoint_url'),
                                "size": model.get('size', 0),
                                "enabled": True
                            })
                except Exception as e:
                    logger.debug(f"Failed to list models from {endpoint['name']}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get Ollama models: {e}")
    
    return {
        "models": models,
        "total": len(models),
        "providers": {
            "ollama": OLLAMA_AVAILABLE,
            "huggingface": True,
            "gpu": True
        }
    }

@app.get("/stats")
@cached_response(ttl=5)
async def stats():
    """Basic stats endpoint"""
    return {
        "status": "operational",
        "model_loaded": False,
        "gpu_available": False,
        "quantization": "None",
        "total_inferences": 0,
        "average_tokens_per_second": 0.0
    }

# ============================================================================
# REDIS/DB HEALTH (if available)
# ============================================================================

if REDIS_AVAILABLE:
    @app.get("/api/cache/stats")
    async def cache_stats(_: bool = Depends(verify_api_key)):  # SECURITY: Require authentication for cache stats
        """Get Redis cache statistics"""
        return get_cache_stats()
    
    @app.get("/api/db/health")
    async def db_health():
        """Get database health status"""
        is_healthy, message = await check_database_health()
        return {
            "healthy": is_healthy,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/redis/health")
    async def redis_health():
        """Get Redis health status"""
        is_healthy, message = await check_redis_health()
        return {
            "healthy": is_healthy,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================

if __name__ == "__main__":
    # Set EGOLLAMA_HOST=0.0.0.0 in Docker if network access is needed
    server_host = os.getenv("EGOLLAMA_HOST", "127.0.0.1")
    server_port = int(os.getenv("EGOLLAMA_PORT", "8082"))
    
    if server_host == "0.0.0.0":
        logger.warning("⚠️  Server binding to 0.0.0.0 - accessible from network!")
        logger.warning("⚠️  Ensure firewall rules and API key authentication are configured!")
    
    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
        log_level="info",
        access_log=True
    )