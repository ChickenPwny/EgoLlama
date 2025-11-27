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

## Production Configuration

### Security Settings

**REQUIRED for production:**

1. **API Key Authentication:**
   ```bash
   EGOLLAMA_API_KEY=your-strong-random-key-here
   EGOLLAMA_REQUIRE_API_KEY=true
   ```

2. **CORS Configuration:**
   ```bash
   EGOLLAMA_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
   ```

3. **Host Binding:**
   ```bash
   # For Docker
   EGOLLAMA_HOST=0.0.0.0
   
   # For standalone (behind reverse proxy)
   EGOLLAMA_HOST=127.0.0.1
   ```

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

See `env.example` for complete list.

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

