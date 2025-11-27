#!/usr/bin/env python3
"""
Ollama Provider for EgoLlama Gateway
=====================================

Provides integration with Ollama API endpoints, supporting multiple
preconfigured Ollama instances and models.

Features:
- Multiple Ollama endpoint support
- Automatic failover between endpoints
- Model discovery and health checking
- Preconfigured model mappings
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Manages Ollama API connections and model operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Ollama provider with configuration.
        
        Args:
            config_path: Path to ollama_config.json (default: ./ollama_config.json)
        """
        self.config_path = config_path or os.path.join(
            Path(__file__).parent, "ollama_config.json"
        )
        self.endpoints: List[Dict[str, Any]] = []
        self.models: Dict[str, Dict[str, Any]] = {}
        self.endpoint_health: Dict[str, bool] = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.endpoints = config.get('endpoints', [])
                    self.models = config.get('models', {})
                    logger.info(f"✅ Loaded Ollama config: {len(self.endpoints)} endpoints, {len(self.models)} models")
            else:
                # Create default configuration
                self._create_default_config()
                logger.info("📝 Created default Ollama configuration")
        except Exception as e:
            logger.error(f"❌ Failed to load Ollama config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration with local Ollama"""
        default_config = {
            "endpoints": [
                {
                    "name": "local",
                    "base_url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "timeout": 30,
                    "description": "Local Ollama instance"
                }
            ],
            "models": {
                "llama3.1:8b": {
                    "endpoint": "local",
                    "model_name": "llama3.1:8b",
                    "description": "Meta Llama 3.1 8B",
                    "context_size": 8192,
                    "enabled": True
                },
                "llama3.1:70b": {
                    "endpoint": "local",
                    "model_name": "llama3.1:70b",
                    "description": "Meta Llama 3.1 70B",
                    "context_size": 8192,
                    "enabled": True
                },
                "mistral": {
                    "endpoint": "local",
                    "model_name": "mistral",
                    "description": "Mistral 7B",
                    "context_size": 8192,
                    "enabled": True
                },
                "codellama": {
                    "endpoint": "local",
                    "model_name": "codellama",
                    "description": "Code Llama",
                    "context_size": 16384,
                    "enabled": True
                }
            }
        }
        
        # Save default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.endpoints = default_config['endpoints']
            self.models = default_config['models']
        except Exception as e:
            logger.error(f"❌ Failed to save default config: {e}")
    
    def get_endpoint(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get endpoint configuration by name"""
        for endpoint in self.endpoints:
            if endpoint['name'] == endpoint_name and endpoint.get('enabled', True):
                return endpoint
        return None
    
    def get_available_endpoints(self) -> List[Dict[str, Any]]:
        """Get list of enabled endpoints, sorted by priority"""
        enabled = [e for e in self.endpoints if e.get('enabled', True)]
        return sorted(enabled, key=lambda x: x.get('priority', 999))
    
    async def check_endpoint_health(self, endpoint: Dict[str, Any]) -> bool:
        """Check if an Ollama endpoint is healthy"""
        try:
            base_url = endpoint['base_url']
            timeout = endpoint.get('timeout', 30)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    self.endpoint_health[endpoint['name']] = True
                    return True
                else:
                    self.endpoint_health[endpoint['name']] = False
                    return False
        except Exception as e:
            logger.debug(f"Endpoint {endpoint['name']} health check failed: {e}")
            self.endpoint_health[endpoint['name']] = False
            return False
    
    async def list_models(self, endpoint_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models from Ollama endpoint(s).
        
        Args:
            endpoint_name: Specific endpoint to query (None = all endpoints)
        
        Returns:
            List of available models with endpoint info
        """
        models_list = []
        
        if endpoint_name:
            endpoints = [e for e in self.endpoints if e['name'] == endpoint_name]
        else:
            endpoints = self.get_available_endpoints()
        
        for endpoint in endpoints:
            try:
                base_url = endpoint['base_url']
                timeout = endpoint.get('timeout', 30)
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        for model in data.get('models', []):
                            models_list.append({
                                'name': model.get('name', 'unknown'),
                                'endpoint': endpoint['name'],
                                'endpoint_url': base_url,
                                'size': model.get('size', 0),
                                'modified_at': model.get('modified_at'),
                                'digest': model.get('digest')
                            })
            except Exception as e:
                logger.warning(f"Failed to list models from {endpoint['name']}: {e}")
        
        return models_list
    
    async def generate(
        self,
        model: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama API.
        
        Args:
            model: Model identifier (e.g., "llama3.1:8b" or "endpoint:model")
            prompt: Text prompt (for /api/generate)
            messages: Chat messages (for /api/chat)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            **kwargs: Additional Ollama API parameters
        
        Returns:
            Generated response with metadata
        """
        # Determine which endpoint and model to use
        endpoint_name, model_name = self._parse_model_identifier(model)
        
        if not endpoint_name or not model_name:
            # Try to find model in preconfigured models
            if model in self.models:
                model_config = self.models[model]
                endpoint_name = model_config['endpoint']
                model_name = model_config['model_name']
            else:
                # Default to first available endpoint
                available = self.get_available_endpoints()
                if not available:
                    raise ValueError("No Ollama endpoints available")
                endpoint_name = available[0]['name']
                model_name = model
        
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            raise ValueError(f"Endpoint '{endpoint_name}' not found or disabled")
        
        base_url = endpoint['base_url']
        timeout = endpoint.get('timeout', 30)
        
        # Try endpoint with failover
        available_endpoints = self.get_available_endpoints()
        errors = []
        
        for ep in available_endpoints:
            if ep['name'] != endpoint_name:
                continue
            
            try:
                ep_url = ep['base_url']
                
                # Use /api/chat if messages provided, otherwise /api/generate
                if messages:
                    api_url = f"{ep_url}/api/chat"
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "stream": stream,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            **kwargs.get('options', {})
                        }
                    }
                else:
                    api_url = f"{ep_url}/api/generate"
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": stream,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            **kwargs.get('options', {})
                        }
                    }
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(api_url, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if stream:
                            # Handle streaming response
                            return {
                                "stream": True,
                                "endpoint": ep['name'],
                                "model": model_name
                            }
                        else:
                            # Extract response content
                            if messages:
                                content = data.get('message', {}).get('content', '')
                            else:
                                content = data.get('response', '')
                            
                            return {
                                "content": content,
                                "model": model_name,
                                "endpoint": ep['name'],
                                "done": data.get('done', True),
                                "total_duration": data.get('total_duration', 0),
                                "load_duration": data.get('load_duration', 0),
                                "prompt_eval_count": data.get('prompt_eval_count', 0),
                                "eval_count": data.get('eval_count', 0),
                                "eval_duration": data.get('eval_duration', 0),
                                "context": data.get('context', [])
                            }
                    else:
                        error_text = response.text
                        errors.append(f"{ep['name']}: {response.status_code} - {error_text}")
                        continue
                        
            except Exception as e:
                errors.append(f"{ep['name']}: {str(e)}")
                continue
        
        # All endpoints failed
        raise Exception(f"All Ollama endpoints failed: {'; '.join(errors)}")
    
    def _parse_model_identifier(self, model: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse model identifier.
        
        Formats:
        - "endpoint:model" -> (endpoint, model)  (e.g., "local:llama3.1:8b")
        - "ollama:model" -> (None, model)  (special prefix for Ollama models)
        - "model" -> (None, model)  (e.g., "llama3.1:8b")
        
        Note: Model names can contain colons (e.g., "llama3.1:8b"), so we only
        split if it looks like an endpoint prefix (known endpoint name).
        """
        # Check for "ollama:" prefix (special case)
        if model.startswith("ollama:"):
            return None, model.replace("ollama:", "", 1)
        
        # Check if it starts with a known endpoint name
        if ':' in model and not model.startswith('http'):
            parts = model.split(':', 1)
            if len(parts) == 2:
                endpoint_name = parts[0]
                # Check if this is a known endpoint (not a model name with colon)
                if any(ep['name'] == endpoint_name for ep in self.endpoints):
                    return endpoint_name, parts[1]
        
        # Default: treat as model name (may contain colons)
        return None, model
    
    async def pull_model(self, model_name: str, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull a model from Ollama endpoint.
        
        Args:
            model_name: Name of model to pull
            endpoint_name: Specific endpoint (None = first available)
        
        Returns:
            Pull operation result
        """
        if not endpoint_name:
            available = self.get_available_endpoints()
            if not available:
                raise ValueError("No Ollama endpoints available")
            endpoint_name = available[0]['name']
        
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            raise ValueError(f"Endpoint '{endpoint_name}' not found")
        
        base_url = endpoint['base_url']
        timeout = endpoint.get('timeout', 300)  # Longer timeout for pulls
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/api/pull",
                    json={"name": model_name},
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "model": model_name,
                        "endpoint": endpoint_name,
                        "message": f"Model {model_name} pulled successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": response.text,
                        "status_code": response.status_code
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global provider instance
_ollama_provider: Optional[OllamaProvider] = None


def get_ollama_provider() -> OllamaProvider:
    """Get or create global Ollama provider instance"""
    global _ollama_provider
    if _ollama_provider is None:
        _ollama_provider = OllamaProvider()
    return _ollama_provider

