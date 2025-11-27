#!/usr/bin/env python3
"""
LLaMA Gateway CLI - Ollama-like interface
========================================

Provides Ollama-like CLI commands for the enhanced LLaMA Gateway:
- llama pull <model>     - Pull and load model from HuggingFace/Ollama
- llama list             - List available models  
- llama search <query>    - Search HuggingFace models
- llama rm <model>       - Remove/unload model
- llama run <model>      - Run model interactively
- llama chat             - Start chat session

Usage:
    python llama_cli.py pull meta-llama/Llama-2-7b-hf
    python llama_cli.py pull llama3.1:8b --source ollama
    python llama_cli.py list
    python llama_cli.py search llama
    python llama_cli.py run meta-llama/Llama-2-7b-hf
    python llama_cli.py chat
"""

import sys
import json
import os
import asyncio
import aiohttp
import argparse
from typing import Optional, Dict

class LLaMACLI:
    """CLI interface for LLaMA Gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:8082", api_key: Optional[str] = None):
        self.gateway_url = gateway_url.rstrip('/')
        self.api_key = api_key or os.getenv("EGOLLAMA_API_KEY")
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key if available"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def pull_model(self, model_name: str, source: str = "huggingface", quantize: bool = True, quantization_bits: Optional[int] = 4):
        """Pull and load model from various sources"""
        try:
            if source == "ollama":
                # For Ollama, pull using Ollama API first
                print(f"📥 Pulling {model_name} from Ollama...")
                ollama_success = await self._pull_ollama_model(model_name)
                if not ollama_success:
                    return False
                # Note: Ollama models are managed by Ollama, not the gateway
                print(f"✅ Model {model_name} pulled from Ollama")
                print(f"💡 Ollama models are managed separately. Use Ollama CLI to manage them.")
                return True
            
            elif source == "huggingface":
                # For HuggingFace, use gateway's /models/load endpoint
                # This will download automatically if not cached
                print(f"📥 Loading {model_name} from HuggingFace...")
                print(f"   (This will download the model if not already cached)")
                
                quantization_bits = quantization_bits if quantize else None
                
                if not self.session:
                    print("❌ Session not initialized")
                    return False
                async with self.session.post(
                    f"{self.gateway_url}/models/load",
                    json={
                        "model_id": model_name,
                        "quantization_bits": quantization_bits
                    },
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Successfully loaded {model_name}")
                        print(f"   Method: {result.get('method', 'unknown')}")
                        if result.get('local_path'):
                            print(f"   Local path: {result.get('local_path')}")
                        if result.get('device'):
                            print(f"   Device: {result.get('device')}")
                        if quantization_bits:
                            print(f"   Quantization: {quantization_bits} bits")
                        return True
                    else:
                        error_text = await response.text()
                        try:
                            error_json = await response.json()
                            error_msg = error_json.get('detail', error_text)
                        except:
                            error_msg = error_text
                        print(f"❌ Failed to load {model_name}: {error_msg}")
                        return False
            
            else:
                print(f"❌ Unknown source: {source}")
                return False
                
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Make sure the gateway is running at {self.gateway_url}")
            return False
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _pull_ollama_model(self, model_name: str) -> bool:
        """Pull model from Ollama using Ollama API"""
        try:
            import httpx
            client = httpx.AsyncClient(timeout=300.0)
            try:
                # Ollama pull endpoint
                response = await client.post(
                    "http://localhost:11434/api/pull",
                    json={"name": model_name},
                    timeout=300.0
                )
                if response.status_code == 200:
                    # Stream the response to show progress
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "status" in data:
                                    print(f"   {data['status']}")
                                if "completed" in data and data.get("completed") == data.get("total"):
                                    print(f"   ✅ Download complete")
                            except:
                                pass
                    return True
                else:
                    error_text = response.text
                    print(f"❌ Ollama pull failed: {error_text}")
                    return False
            finally:
                await client.aclose()
        except ImportError:
            print("❌ httpx not installed. Install with: pip install httpx")
            return False
        except Exception as e:
            print(f"❌ Error pulling from Ollama: {e}")
            print(f"   Make sure Ollama is running at http://localhost:11434")
            return False
    
    async def search_models(self, query: str, limit: int = 10):
        """Search HuggingFace models"""
        try:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                print(f"🔍 Searching HuggingFace for: {query}")
                print("   (This may take a moment...)")
                
                models = api.list_models(
                    search=query,
                    sort="downloads",
                    direction=-1,
                    limit=limit
                )
                
                print(f"\n📦 Found {len(list(models))} models:")
                print("-" * 80)
                for model in models:
                    print(f"  • {model.id}")
                    if hasattr(model, 'downloads') and model.downloads:
                        print(f"    Downloads: {model.downloads:,}")
                    print()
                
                return True
            except ImportError:
                print("❌ huggingface_hub not installed")
                print("   Install with: pip install huggingface_hub")
                print("\n💡 You can still load models directly:")
                print(f"   python llama_cli.py pull {query}")
                return False
        except Exception as e:
            print(f"❌ Error searching models: {e}")
            return False
    
    async def list_models(self):
        """List available models"""
        try:
            if not self.session:
                print("❌ Session not initialized")
                return
            async with self.session.get(
                f"{self.gateway_url}/api/models",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    total = data.get("total", len(models))
                    
                    if not models and total == 0:
                        print("📦 No models currently loaded")
                        print("\nTo load a model, use:")
                        print("  python llama_cli.py pull <model_name>")
                        return
                    
                    print(f"Available models ({total}):")
                    print("-" * 60)
                    for model in models:
                        if isinstance(model, dict):
                            name = model.get("name", "unknown")
                            source = model.get("source", "unknown")
                            quantized = model.get("quantized", False)
                            downloaded = model.get("downloaded_at", "unknown")
                            
                            print(f"📦 {name}")
                            print(f"   Source: {source}")
                            print(f"   Quantized: {'Yes' if quantized else 'No'}")
                            print(f"   Downloaded: {downloaded}")
                            print()
                        else:
                            print(f"📦 {model}")
                else:
                    print(f"❌ Failed to list models: HTTP {response.status}")
                    error_text = await response.text()
                    if error_text:
                        print(f"   Error: {error_text[:200]}")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    async def remove_model(self, model_name: str, source: str = "huggingface"):
        """Remove/unload model"""
        try:
            if source == "ollama":
                # Use Ollama API to remove
                import httpx
                client = httpx.AsyncClient()
                try:
                    response = await client.delete(
                        "http://localhost:11434/api/delete",
                        json={"name": model_name}
                    )
                    if response.status_code == 200:
                        print(f"✅ Successfully removed {model_name} from Ollama")
                        return True
                    else:
                        print(f"❌ Failed to remove {model_name}: {response.text}")
                        return False
                finally:
                    await client.aclose()
            
            # For gateway models, unload them
            if not self.session:
                print("❌ Session not initialized")
                return False
            async with self.session.post(
                f"{self.gateway_url}/models/unload",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully unloaded model")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to unload model: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error removing model: {e}")
            return False
    
    async def run_model(self, model_name: str, prompt: str, source: str = "huggingface"):
        """Run model with prompt"""
        try:
            if source == "huggingface":
                # First load the model if it's a HuggingFace model
                print(f"📥 Loading model {model_name}...")
                if not self.session:
                    print("❌ Session not initialized")
                    return False
                async with self.session.post(
                    f"{self.gateway_url}/models/load",
                    json={"model_id": model_name, "quantization_bits": 4},
                    headers=self._get_headers()
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ Failed to load model {model_name}: {error_text}")
                        return False
                    print("✅ Model loaded")
            
            # Generate response
            print(f"🤖 Generating response...")
            if not self.session:
                print("❌ Session not initialized")
                return False
            async with self.session.post(
                f"{self.gateway_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.7
                },
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"\n📝 Response from {model_name}:")
                    print("-" * 60)
                    print(result.get("generated_text", "No response"))
                    print("-" * 60)
                    if result.get("metadata"):
                        meta = result["metadata"]
                        print(f"Engine: {meta.get('engine', 'unknown')}")
                        if meta.get('tokens_per_second'):
                            print(f"Speed: {meta.get('tokens_per_second', 0):.2f} tokens/sec")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to generate: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error running model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def chat_session(self, model_name: str = "llama-custom"):
        """Start interactive chat session"""
        try:
            print(f"🤖 Starting chat with {model_name}")
            print("Type 'quit' to exit, 'clear' to clear context")
            print("-" * 50)
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        print("🧹 Context cleared")
                        continue
                    elif not user_input:
                        continue
                    
                    # Send chat request
                    if not self.session:
                        print("❌ Session not initialized")
                        continue
                    async with self.session.post(
                        f"{self.gateway_url}/v1/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "user", "content": user_input}
                            ],
                            "max_tokens": 512,
                            "temperature": 0.7
                        },
                        headers=self._get_headers()
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                            print(f"AI: {ai_response}")
                        else:
                            error_text = await response.text()
                            print(f"❌ Error: {error_text}")
                            
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Error starting chat: {e}")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="LLaMA Gateway CLI - Manage and use models from HuggingFace and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pull and load a HuggingFace model
  python llama_cli.py pull meta-llama/Llama-2-7b-hf
  
  # Pull from Ollama
  python llama_cli.py pull llama3.1:8b --source ollama
  
  # Search HuggingFace models
  python llama_cli.py search llama
  
  # List loaded models
  python llama_cli.py list
  
  # Run a model
  python llama_cli.py run meta-llama/Llama-2-7b-hf --prompt "Hello, world!"
  
  # Start interactive chat
  python llama_cli.py chat
        """
    )
    parser.add_argument("command", 
                       choices=["pull", "list", "search", "rm", "run", "chat"], 
                       help="Command to execute")
    parser.add_argument("model", nargs="?", help="Model name or search query")
    parser.add_argument("--source", default="huggingface", 
                       choices=["huggingface", "ollama"],
                       help="Source for pulling models (default: huggingface)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Don't quantize model (HuggingFace only)")
    parser.add_argument("--quantization-bits", type=int, default=4,
                       choices=[4, 8, 16],
                       help="Quantization bits (4, 8, or 16) - default: 4")
    parser.add_argument("--prompt", help="Prompt for run command")
    parser.add_argument("--gateway-url", default="http://localhost:8082",
                       help="LLaMA Gateway URL (default: http://localhost:8082)")
    parser.add_argument("--api-key", 
                       help="API key for authentication (or set EGOLLAMA_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("EGOLLAMA_API_KEY")
    
    async with LLaMACLI(args.gateway_url, api_key) as cli:
        if args.command == "pull":
            if not args.model:
                print("❌ Model name required for pull command")
                print("\nExamples:")
                print("  python llama_cli.py pull meta-llama/Llama-2-7b-hf")
                print("  python llama_cli.py pull llama3.1:8b --source ollama")
                sys.exit(1)
            quantization_bits: Optional[int] = None if args.no_quantize else args.quantization_bits
            await cli.pull_model(args.model, args.source, not args.no_quantize, quantization_bits)
            
        elif args.command == "list":
            await cli.list_models()
            
        elif args.command == "search":
            if not args.model:
                print("❌ Search query required")
                print("\nExample: python llama_cli.py search llama")
                sys.exit(1)
            await cli.search_models(args.model)
            
        elif args.command == "rm":
            if not args.model:
                print("❌ Model name required for rm command")
                sys.exit(1)
            await cli.remove_model(args.model, args.source)
            
        elif args.command == "run":
            if not args.model:
                print("❌ Model name required for run command")
                sys.exit(1)
            prompt = args.prompt or input("Enter prompt: ")
            await cli.run_model(args.model, prompt, args.source)
            
        elif args.command == "chat":
            model = args.model or "llama-custom"
            await cli.chat_session(model)

if __name__ == "__main__":
    asyncio.run(main())

