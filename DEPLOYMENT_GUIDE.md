# EgoLlama-clean Deployment Guide

Complete guide for deploying EgoLlama Gateway in various environments.

## Quick Start Options

### Option 1: One-Command Quick Start (Recommended) ⭐

**Difficulty:** ⭐ Super Easy | **Time:** 5 minutes

```bash
./quick_start.sh
```

This script automatically:
- ✅ Detects your environment (Docker or standalone)
- ✅ Sets up all dependencies
- ✅ Configures environment
- ✅ Starts services
- ✅ Tests deployment

### Option 2: Docker Deployment

**Difficulty:** ⭐ Easy | **Time:** 5-10 minutes

```bash
./setup.sh
```

Or manually:
```bash
cp env.example .env
docker-compose up -d
```

### Option 3: Standalone Deployment

**Difficulty:** ⭐⭐ Moderate | **Time:** 10-15 minutes

```bash
./deploy_standalone.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Setup configuration
- ✅ Create startup script

Then start with:
```bash
./start_gateway.sh
```

## Deployment Methods

### 1. Docker Deployment (Production Recommended)

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

#### Steps

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd EgoLlama-clean
   cp env.example .env
   ```

2. **Edit .env for production:**
   ```bash
   # Security (REQUIRED for production)
   EGOLLAMA_API_KEY=your-strong-secret-key-here
   EGOLLAMA_REQUIRE_API_KEY=true
   EGOLLAMA_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
   
   # Database
   POSTGRES_PASSWORD=strong-database-password
   
   # Server
   EGOLLAMA_HOST=0.0.0.0  # For Docker networking
   EGOLLAMA_PORT=8082
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify:**
   ```bash
   curl http://localhost:8082/health
   ```

#### Docker Commands

```bash
# View logs
docker-compose logs -f gateway

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View status
docker-compose ps

# Update and restart
docker-compose pull
docker-compose up -d
```

### 2. Standalone Deployment

#### Prerequisites
- Python 3.11+
- pip
- (Optional) PostgreSQL
- (Optional) Redis

#### Steps

1. **Run deployment script:**
   ```bash
   ./deploy_standalone.sh
   ```

2. **Configure environment:**
   ```bash
   # Edit .env file
   nano .env
   ```

3. **Start gateway:**
   ```bash
   ./start_gateway.sh
   ```

   Or manually:
   ```bash
   source venv/bin/activate
   python simple_llama_gateway_crash_safe.py
   ```

#### Manual Standalone Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp env.example .env
# Edit .env as needed

# Start
python simple_llama_gateway_crash_safe.py
```

### 3. Systemd Service (Production)

For production deployments, create a systemd service:

#### Create service file

```bash
sudo nano /etc/systemd/system/egollama-gateway.service
```

#### Service configuration

```ini
[Unit]
Description=EgoLlama Gateway
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/EgoLlama-clean
Environment="PATH=/path/to/EgoLlama-clean/venv/bin"
ExecStart=/path/to/EgoLlama-clean/venv/bin/python simple_llama_gateway_crash_safe.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable egollama-gateway
sudo systemctl start egollama-gateway
sudo systemctl status egollama-gateway
```

## Model Setup

### Installing Ollama Models

EgoLlama Gateway supports Ollama for model inference. To set up Ollama:

1. **Automatic Setup (Recommended):**
   ```bash
   ./setup_ollama.sh
   ```
   
   This script will:
   - ✅ Install Ollama if not present
   - ✅ Start Ollama service
   - ✅ Pull recommended models (llama3.2:1b, mistral:7b, phi3:3.8b, codellama:7b)
   - ✅ Verify installation

2. **Manual Ollama Installation:**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service (in background)
   ollama serve &
   
   # Wait for service to start
   sleep 10
   
   # Pull models
   ollama pull llama3.2:1b
   ollama pull mistral:7b
   ollama pull phi3:3.8b
   ollama pull codellama:7b
   
   # Verify installation
   ollama list
   ```

3. **Configure Ollama Models:**
   
   Edit `ollama_config.json` to configure models and endpoints:
   ```json
   {
     "endpoints": [
       {
         "name": "local",
         "base_url": "http://localhost:11434",
         "enabled": true,
         "priority": 1,
         "timeout": 30
       }
     ],
     "models": {
       "llama3.1:8b": {
         "endpoint": "local",
         "model_name": "llama3.1:8b",
         "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
         "load_into_egollama": true,
         "enabled": true,
         "context_size": 8192,
         "default_temperature": 0.7,
         "default_max_tokens": 2048
       }
     }
   }
   ```

4. **Verify Ollama Setup:**
   ```bash
   # Check Ollama status (direct)
   curl http://localhost:11434/api/tags
   
   # Check gateway Ollama health
   curl http://localhost:8082/api/ollama/health
   
   # List available models via gateway
   curl http://localhost:8082/api/ollama/models
   
   # List preconfigured models
   curl http://localhost:8082/api/ollama/models/preconfigured
   ```

### Installing HuggingFace Models

EgoLlama Gateway can use HuggingFace models directly or via Ollama mappings:

1. **Via Ollama (Recommended):**
   
   Models are automatically mapped from Ollama to HuggingFace when `load_into_egollama: true` is set in `ollama_config.json`. The gateway will download HuggingFace models on first use from the `huggingface_id` specified in the config.

2. **Direct HuggingFace Usage:**
   
   Models are automatically downloaded from HuggingFace Hub when requested using the full model ID (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). No manual installation required.

3. **Configure HuggingFace:**
   
   Set HuggingFace token (optional, for private models):
   ```bash
   # In .env file
   HF_TOKEN=your-huggingface-token-here
   ```
   
   Or export as environment variable:
   ```bash
   export HF_TOKEN=your-huggingface-token-here
   ```

4. **Verify HuggingFace Integration:**
   ```bash
   # Check available models (includes both Ollama and HuggingFace)
   curl http://localhost:8082/models
   ```

### Model Management

**Pull new Ollama model via API:**
```bash
curl -X POST http://localhost:8082/api/ollama/pull \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"model": "llama3.1:70b"}'
```

**Pull Ollama model via CLI:**
```bash
ollama pull llama3.1:70b
```

**List all available models:**
```bash
# Via Ollama CLI
ollama list

# Via Gateway API
curl http://localhost:8082/models
```

## Production Configuration

### Security Settings

**REQUIRED for production:**

1. **Environment Mode:**
   ```bash
   ENVIRONMENT=production
   ```
   - Production mode enforces stricter security defaults
   - Generic error messages (no stack traces)
   - Requires API key if configured

2. **API Key Authentication:**
   ```bash
   EGOLLAMA_API_KEY=your-strong-random-key-here
   EGOLLAMA_REQUIRE_API_KEY=true
   ```
   - Generate a strong, random API key (min 32 characters)
   - Store securely (use secrets management in production)

3. **CORS Configuration:**
   ```bash
   EGOLLAMA_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
   ```
   - Restrict to your actual domains
   - Never use `*` in production

4. **Remote Code Execution (CRITICAL):**
   ```bash
   EGOLLAMA_TRUST_REMOTE_CODE=false
   ```
   - **ALWAYS keep this `false` unless you fully trust the model source**
   - Setting to `true` allows models to execute arbitrary Python code
   - Only enable for verified internal models

5. **Host Binding:**
   ```bash
   # For Docker
   EGOLLAMA_HOST=0.0.0.0
   
   # For standalone (behind reverse proxy)
   EGOLLAMA_HOST=127.0.0.1
   ```

### Security Checklist

Before deploying to production, ensure:
- [ ] `ENVIRONMENT=production` is set
- [ ] Strong `EGOLLAMA_API_KEY` is configured
- [ ] `EGOLLAMA_REQUIRE_API_KEY=true`
- [ ] `EGOLLAMA_TRUST_REMOTE_CODE=false` (unless absolutely necessary)
- [ ] CORS origins restricted to your domains
- [ ] SSL/TLS enabled (via reverse proxy)
- [ ] Review `SECURITY_FIXES_APPLIED.md` for recent security updates

### Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8082;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### SSL/TLS (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## Environment Variables

### Required (Production)

| Variable | Description | Example |
|----------|-------------|---------|
| `EGOLLAMA_API_KEY` | API key for authentication | `your-secret-key` |
| `EGOLLAMA_REQUIRE_API_KEY` | Enforce API key | `true` |
| `EGOLLAMA_CORS_ORIGINS` | Allowed CORS origins | `https://yourdomain.com` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `EGOLLAMA_HOST` | Server host | `127.0.0.1` |
| `EGOLLAMA_PORT` | Server port | `8082` |
| `EGOLLAMA_DATABASE_URL` | PostgreSQL connection | (optional) |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `ENVIRONMENT` | Environment mode (`development`/`production`) | `development` |
| `EGOLLAMA_TRUST_REMOTE_CODE` | Enable remote code execution for models | `false` |

**Security Note:**
- `ENVIRONMENT=production` enforces stricter security defaults
- `EGOLLAMA_TRUST_REMOTE_CODE=true` should ONLY be used for verified internal models
- See `SECURITY_FIXES_APPLIED.md` for security configuration details

See `env.example` for complete list.

## Using the Server

### API Endpoints

#### Text Generation

**Endpoint:** `POST /generate`

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 512,
    "temperature": 0.7,
    "model": "llama3.1:8b"
  }'
```

#### Chat Completions (OpenAI-compatible)

**Endpoint:** `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "model": "mistral:7b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

#### List Available Models

```bash
# List all models (Ollama + HuggingFace)
curl http://localhost:8082/models

# List Ollama models from specific endpoint
curl http://localhost:8082/api/ollama/models?endpoint=local

# List all Ollama models
curl http://localhost:8082/api/ollama/models

# List preconfigured Ollama models
curl http://localhost:8082/api/ollama/models/preconfigured
```

#### Pull Ollama Model via API

```bash
curl -X POST http://localhost:8082/api/ollama/pull \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"model": "llama3.1:70b"}'
```

#### Health Check

```bash
# Overall health
curl http://localhost:8082/health

# Ollama health (checks all endpoints)
curl http://localhost:8082/api/ollama/health

# Database health
curl http://localhost:8082/api/db/health

# Redis health
curl http://localhost:8082/api/redis/health
```

### Using Different Models

1. **Ollama Models:**
   - Use model name from `ollama list` (e.g., `mistral:7b`)
   - Or use `ollama:model-name` prefix (e.g., `ollama:mistral:7b`)
   - Models are configured in `ollama_config.json`

2. **HuggingFace Models:**
   - Use full HuggingFace model ID (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`)
   - Or use Ollama model name that has `load_into_egollama: true` configured
   - Models are automatically downloaded on first use

3. **Default Models:**
   - If no model specified, uses configured default
   - Check `/models` endpoint for available options

### Example Usage Scenarios

**Basic Text Generation:**
```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world! Explain what AI is.",
    "max_tokens": 100
  }'
```

**Chat with System Message:**
```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Code Generation with CodeLlama:**
```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "model": "codellama:7b",
    "max_tokens": 256,
    "temperature": 0.2
  }'
```

**Multi-turn Conversation:**
```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [
      {"role": "user", "content": "What is recursion?"},
      {"role": "assistant", "content": "Recursion is a programming technique..."},
      {"role": "user", "content": "Can you give me an example?"}
    ],
    "max_tokens": 512
  }'
```

**Using HuggingFace Model Directly:**
```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to French: Hello, how are you?",
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "max_tokens": 100
  }'
```

### API Authentication

If `EGOLLAMA_REQUIRE_API_KEY=true`, include the API key in requests:

```bash
# Set your API key
export API_KEY="your-api-key-here"

# Use in requests
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Monitoring

### Health Checks

```bash
# Overall health
curl http://localhost:8082/health

# Database health
curl http://localhost:8082/api/db/health

# Redis health
curl http://localhost:8082/api/redis/health
```

### Logs

**Docker:**
```bash
docker-compose logs -f gateway
```

**Standalone:**
```bash
tail -f logs/*.log
```

### Metrics

```bash
# Performance stats
curl http://localhost:8082/api/performance/stats

# Cache stats
curl http://localhost:8082/api/cache/stats
```

## Troubleshooting

### Gateway won't start

1. **Check logs:**
   ```bash
   # Docker
   docker-compose logs gateway
   
   # Standalone
   python simple_llama_gateway_crash_safe.py
   ```

2. **Check ports:**
   ```bash
   sudo lsof -i :8082
   ```

3. **Verify environment:**
   ```bash
   cat .env
   ```

### Database connection errors

1. **Check PostgreSQL:**
   ```bash
   # Docker
   docker-compose exec postgres psql -U postgres -d ego
   
   # Standalone
   psql -h localhost -U postgres -d ego
   ```

2. **Verify connection string:**
   ```bash
   echo $EGOLLAMA_DATABASE_URL
   ```

### API key authentication issues

1. **Check API key is set:**
   ```bash
   echo $EGOLLAMA_API_KEY
   ```

2. **Test with API key:**
   ```bash
   curl -X POST http://localhost:8082/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-key" \
     -d '{"prompt":"test","max_tokens":10}'
   ```

## Deployment Checklist

### Pre-Deployment

- [ ] Review security settings in `.env`
- [ ] Set strong `EGOLLAMA_API_KEY`
- [ ] Configure `EGOLLAMA_CORS_ORIGINS`
- [ ] Set `EGOLLAMA_REQUIRE_API_KEY=true` for production
- [ ] Configure database passwords
- [ ] Review firewall rules
- [ ] Setup SSL/TLS certificates

### Post-Deployment

- [ ] Verify health endpoint responds
- [ ] Test API key authentication
- [ ] Verify CORS restrictions
- [ ] Check logs for errors
- [ ] Monitor resource usage
- [ ] Setup monitoring/alerting
- [ ] Document deployment details

## Updating

### Docker

```bash
git pull
docker-compose pull
docker-compose up -d
```

### Standalone

```bash
git pull
source venv/bin/activate
pip install -r requirements.txt
# Restart service
```

## Backup

### Database

```bash
# Docker
docker-compose exec postgres pg_dump -U postgres ego > backup.sql

# Restore
docker-compose exec -T postgres psql -U postgres ego < backup.sql
```

### Configuration

```bash
cp .env .env.backup
```

## Support

For issues:
1. Check [README.md](README.md)
2. Check [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. Check [SECURITY_FIXES_APPLIED.md](SECURITY_FIXES_APPLIED.md)
4. Review logs
5. Open GitHub issue

