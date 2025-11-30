# EgoLlama-clean Wiki

Complete guide for using EgoLlama Gateway API and features.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
  - [Text Generation](#text-generation)
  - [Chat Completions](#chat-completions)
  - [Model Management](#model-management)
  - [Health & Status](#health--status)
- [Code Examples](#code-examples)
  - [Python](#python)
  - [JavaScript/Node.js](#javascriptnodejs)
  - [cURL](#curl)
- [Using Models](#using-models)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

EgoLlama Gateway is a production-ready API gateway for running Large Language Models (LLMs). It provides:

- 🤖 **OpenAI-compatible API** - Drop-in replacement for OpenAI API
- 🦙 **Multiple Model Support** - Ollama, HuggingFace, and custom models
- 🔒 **Enterprise Security** - API key authentication, CORS, rate limiting
- ⚡ **High Performance** - Redis caching, PostgreSQL persistence
- 🐳 **Docker Ready** - Easy deployment with Docker Compose

---

## Quick Start

### 1. Start the Gateway

```bash
# Using Docker (Recommended)
docker compose up -d

# Or standalone
./quick_start.sh
```

### 2. Verify Installation

```bash
curl http://localhost:8082/health
```

### 3. Generate Your First Response

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

---

## Authentication

EgoLlama Gateway uses API key authentication for protected endpoints.

### Getting Your API Key

Set the API key in your environment or `.env` file:

```bash
EGOLLAMA_API_KEY=your-strong-secret-key-here
EGOLLAMA_REQUIRE_API_KEY=true
```

### Using the API Key

Include the API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8082/api/models
```

**Note:** Public endpoints like `/health` don't require authentication.

---

## API Endpoints

### Text Generation

Generate text from a prompt using any available model.

**Endpoint:** `POST /generate`

**Request:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "max_tokens": 512,
  "temperature": 0.7,
  "model": "mistral:7b"
}
```

**Response:**
```json
{
  "generated_text": "Quantum computing is a revolutionary approach to computation...",
  "tokens": 150,
  "finish_reason": "stop",
  "timestamp": 1234567890.123,
  "metadata": {
    "engine": "Ollama",
    "model_name": "mistral:7b",
    "inference_time_ms": 2345,
    "tokens_per_second": 63.96
  }
}
```

**Parameters:**
- `prompt` (required): Text prompt to generate from
- `max_tokens` (optional, default: 512): Maximum tokens to generate
- `temperature` (optional, default: 0.7): Sampling temperature (0.0-2.0)
- `model` (optional): Model to use (defaults to configured model)

---

### Chat Completions

OpenAI-compatible chat completions endpoint.

**Endpoint:** `POST /v1/chat/completions`

**Request:**
```json
{
  "model": "mistral:7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mistral:7b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Python is a high-level programming language..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  }
}
```

**Message Roles:**
- `system`: System instructions (optional)
- `user`: User messages
- `assistant`: Assistant responses (for conversation history)

---

### Model Management

#### List Available Models

**Endpoint:** `GET /api/models`

**Response:**
```json
{
  "models": [
    {
      "id": "mistral:7b",
      "name": "mistral:7b",
      "type": "ollama",
      "description": "Mistral 7B - Balanced performance",
      "context_size": 8192,
      "enabled": true
    }
  ],
  "total": 10
}
```

#### Load a Model

**Endpoint:** `POST /models/load`

**Request:**
```json
{
  "model_id": "mistral:7b",
  "quantization_bits": 0
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "mistral:7b",
  "status": "loaded",
  "method": "gpu_module"
}
```

#### Unload Model

**Endpoint:** `POST /models/unload`

Frees memory by unloading the current model.

---

### Health & Status

#### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": false,
  "quantization": "None",
  "total_inferences": 150,
  "average_tokens_per_second": 45.2
}
```

#### API Status

**Endpoint:** `GET /api/status`

Returns available endpoints and features.

#### Performance Stats

**Endpoint:** `GET /api/performance/stats` (requires auth)

**Response:**
```json
{
  "timestamp": "2025-11-30T00:00:00",
  "cpu_usage": 34.7,
  "memory_usage": 19.6,
  "gpu_usage": 0.0,
  "active_models": 1,
  "total_requests": 150,
  "average_response_time": 2.3,
  "error_rate": 0.0
}
```

---

## Code Examples

### Python

#### Basic Text Generation

```python
import requests

url = "http://localhost:8082/generate"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "prompt": "Explain machine learning",
    "max_tokens": 200,
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

print(result["generated_text"])
```

#### Chat Completions

```python
import requests

url = "http://localhost:8082/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "model": "mistral:7b",
    "messages": [
        {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 100
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

message = result["choices"][0]["message"]["content"]
print(message)
```

#### Streaming Responses

```python
import requests
import json

url = "http://localhost:8082/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "model": "mistral:7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}

response = requests.post(url, json=data, headers=headers, stream=True)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode('utf-8'))
        if 'choices' in chunk:
            content = chunk['choices'][0].get('delta', {}).get('content', '')
            if content:
                print(content, end='', flush=True)
```

---

### JavaScript/Node.js

#### Basic Text Generation

```javascript
const fetch = require('node-fetch');

async function generateText(prompt) {
  const response = await fetch('http://localhost:8082/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key'
    },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 200,
      temperature: 0.7
    })
  });
  
  const result = await response.json();
  return result.generated_text;
}

// Usage
generateText('Explain quantum computing')
  .then(text => console.log(text))
  .catch(err => console.error(err));
```

#### Chat Completions

```javascript
async function chatCompletion(messages) {
  const response = await fetch('http://localhost:8082/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key'
    },
    body: JSON.stringify({
      model: 'mistral:7b',
      messages: messages,
      max_tokens: 256
    })
  });
  
  const result = await response.json();
  return result.choices[0].message.content;
}

// Usage
chatCompletion([
  { role: 'user', content: 'What is JavaScript?' }
]).then(response => console.log(response));
```

---

### cURL

#### Text Generation

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci",
    "max_tokens": 256,
    "temperature": 0.2,
    "model": "codellama:7b"
  }'
```

#### Chat with Conversation History

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "mistral:7b",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "How do I reverse a string in Python?"},
      {"role": "assistant", "content": "You can use slicing: string[::-1]"},
      {"role": "user", "content": "What about in JavaScript?"}
    ],
    "max_tokens": 150
  }'
```

#### List Models

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8082/api/models
```

---

## Using Models

### Ollama Models

Ollama models are referenced by their name:

```json
{
  "model": "mistral:7b"
}
```

Or with prefix:

```json
{
  "model": "ollama:mistral:7b"
}
```

**Available Ollama Models:**
- `mistral:7b` - Balanced performance
- `llama3.1:8b` - Fast and efficient
- `codellama:7b` - Code generation
- `phi3:3.8b` - Small and fast

### HuggingFace Models

Use the full HuggingFace model ID:

```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
```

Models are automatically downloaded on first use.

### Model Selection

If no model is specified, the gateway uses the configured default. Check available models:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8082/api/models
```

---

## Advanced Features

### Temperature Control

Control randomness in responses:

```json
{
  "temperature": 0.0   // Deterministic, focused responses
  "temperature": 0.7   // Balanced (default)
  "temperature": 1.5   // Creative, varied responses
}
```

### Token Limits

Control response length:

```json
{
  "max_tokens": 50     // Short responses
  "max_tokens": 512    // Medium (default)
  "max_tokens": 2048   // Long responses
}
```

### Multi-turn Conversations

Maintain context across messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What makes it popular?"}
  ]
}
```

### Code Generation

Use code-specific models for better results:

```json
{
  "model": "codellama:7b",
  "prompt": "Write a Python function to sort a list",
  "temperature": 0.2  // Lower temperature for code
}
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Problem:** `401 Unauthorized` or `403 Forbidden`

**Solution:**
- Verify API key is set: `echo $EGOLLAMA_API_KEY`
- Check `X-API-Key` header is included
- Ensure `EGOLLAMA_REQUIRE_API_KEY=true` is set

#### 2. Model Not Found

**Problem:** `Model not found` error

**Solution:**
- List available models: `GET /api/models`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Pull the model: `ollama pull mistral:7b`

#### 3. Connection Refused

**Problem:** Cannot connect to gateway

**Solution:**
- Check gateway is running: `docker compose ps`
- Verify port is correct: `curl http://localhost:8082/health`
- Check firewall settings

#### 4. Slow Responses

**Problem:** Generation is slow

**Solution:**
- Check GPU availability: `GET /api/performance/stats`
- Use smaller models for faster responses
- Reduce `max_tokens` parameter
- Check system resources

#### 5. Out of Memory

**Problem:** Model loading fails

**Solution:**
- Use smaller models (7B instead of 70B)
- Enable quantization: `quantization_bits: 4`
- Unload other models first
- Check available memory

---

## Best Practices

### 1. API Key Security

- ✅ Store API keys in environment variables, not code
- ✅ Use strong, randomly generated keys (32+ characters)
- ✅ Rotate keys regularly
- ✅ Never commit keys to version control

```bash
# Generate a strong API key
openssl rand -hex 32
```

### 2. Error Handling

Always handle errors gracefully:

```python
try:
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### 3. Rate Limiting

Respect rate limits:

- Default: 100 requests/hour
- Check remaining: Response headers may include rate limit info
- Implement exponential backoff for retries

### 4. Token Management

- Monitor token usage in responses
- Set appropriate `max_tokens` limits
- Cache responses when possible
- Use streaming for long responses

### 5. Model Selection

- Choose appropriate model size for your use case
- Use code-specific models for code generation
- Test different models for your task
- Monitor performance metrics

### 6. Production Deployment

- ✅ Enable SSL/TLS (HTTPS)
- ✅ Set `ENVIRONMENT=production`
- ✅ Configure CORS origins
- ✅ Set up monitoring and alerting
- ✅ Enable rate limiting
- ✅ Use reverse proxy (Nginx/Traefik)

---

## Integration Examples

### OpenAI SDK Compatibility

EgoLlama Gateway is compatible with OpenAI SDK:

```python
import openai

# Point OpenAI SDK to EgoLlama Gateway
openai.api_base = "http://localhost:8082/v1"
openai.api_key = "your-api-key"

# Use OpenAI SDK normally
response = openai.ChatCompletion.create(
    model="mistral:7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### LangChain Integration

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Use EgoLlama Gateway as OpenAI backend
llm = ChatOpenAI(
    openai_api_base="http://localhost:8082/v1",
    openai_api_key="your-api-key",
    model_name="mistral:7b"
)

response = llm.predict("What is AI?")
```

---

## API Reference

### Base URL

- Development: `http://localhost:8082`
- Production: `https://your-domain.com`

### Authentication

Protected endpoints require:
```
X-API-Key: your-api-key
```

### Response Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Missing or invalid API key
- `403 Forbidden` - Access denied
- `404 Not Found` - Endpoint not found
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service unavailable

---

## Support

- 📖 [Deployment Guide](DEPLOYMENT_GUIDE.md)
- 🐛 [Report Issues](https://github.com/ChickenPwny/EgoLlama/issues)
- 💬 [Discussions](https://github.com/ChickenPwny/EgoLlama/discussions)

---

**Happy coding! 🚀**

