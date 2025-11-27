#!/bin/bash
# ==============================================================================
# EgoLlama-clean Standalone Deployment Script
# ==============================================================================
# Deploys EgoLlama Gateway without Docker
# Perfect for development or systems without Docker
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✅${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}❌${NC} $1"; }
title() { echo -e "${CYAN}$1${NC}"; }

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    title "Checking prerequisites..."
    
    local missing=0
    
    # Python
    if ! command_exists python3; then
        error "Python 3 is not installed"
        echo "  Install: https://www.python.org/downloads/"
        missing=1
    else
        local python_version=$(python3 --version 2>&1 | awk '{print $2}')
        success "Python found: $python_version"
        
        # Check version (3.11+)
        local major=$(echo $python_version | cut -d. -f1)
        local minor=$(echo $python_version | cut -d. -f2)
        if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 11 ]); then
            warning "Python 3.11+ recommended (found $python_version)"
        fi
    fi
    
    # pip
    if ! command_exists pip3 && ! python3 -m pip --version >/dev/null 2>&1; then
        error "pip is not installed"
        missing=1
    else
        success "pip found"
    fi
    
    if [ $missing -eq 1 ]; then
        error "Missing prerequisites. Please install them and try again."
        exit 1
    fi
    
    success "All prerequisites met!"
}

# Setup virtual environment
setup_venv() {
    title "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        info "Virtual environment already exists"
    else
        info "Creating virtual environment..."
        python3 -m venv venv
        success "Virtual environment created"
    fi
    
    info "Activating virtual environment..."
    source venv/bin/activate
    
    success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    title "Installing dependencies..."
    
    source venv/bin/activate
    
    info "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    info "Installing Python packages..."
    pip install -r requirements.txt
    
    success "Dependencies installed"
}

# Setup environment
setup_environment() {
    title "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            success "Created .env from env.example"
            info "💡 Edit .env to configure settings"
        else
            warning "env.example not found, creating basic .env"
            cat > .env << EOF
# Database (optional - gateway works without it)
EGOLLAMA_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ego?sslmode=disable

# Redis (optional - gateway works without it)
REDIS_HOST=localhost
REDIS_PORT=6379

# Gateway
EGOLLAMA_HOST=127.0.0.1
EGOLLAMA_PORT=8082

# Security (recommended for production)
# EGOLLAMA_API_KEY=your-secret-api-key-here
# EGOLLAMA_REQUIRE_API_KEY=false
# EGOLLAMA_CORS_ORIGINS=http://localhost:9000,http://localhost:3000
EOF
            success "Created basic .env"
        fi
    else
        success ".env already exists"
    fi
}

    # Check optional services
    check_optional_services() {
    title "Checking optional services..."
    
    # PostgreSQL
    if command_exists psql; then
        if psql -h localhost -U postgres -d ego -c "SELECT 1;" >/dev/null 2>&1; then
            success "PostgreSQL is available"
            # Run database migrations
            info "Running database migrations..."
            if [ -d "venv" ]; then
                source venv/bin/activate
                if alembic upgrade head >/dev/null 2>&1; then
                    success "Database migrations completed"
                else
                    warning "Migration failed (may already be up to date)"
                fi
            fi
        else
            warning "PostgreSQL client found but connection failed"
            info "Gateway will work without PostgreSQL (graceful degradation)"
        fi
    else
        warning "PostgreSQL not found (optional)"
        info "Gateway will work without PostgreSQL (graceful degradation)"
    fi
    
    # Redis
    if command_exists redis-cli; then
        if redis-cli ping >/dev/null 2>&1; then
            success "Redis is available"
        else
            warning "Redis client found but connection failed"
            info "Gateway will work without Redis (graceful degradation)"
        fi
    else
        warning "Redis not found (optional)"
        info "Gateway will work without Redis (graceful degradation)"
    fi
}

# Create startup script
create_startup_script() {
    title "Creating startup script..."
    
    cat > start_gateway.sh << 'EOF'
#!/bin/bash
# Start EgoLlama Gateway (Standalone)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run deploy_standalone.sh first."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start gateway
echo "🚀 Starting EgoLlama Gateway..."
python simple_llama_gateway_crash_safe.py
EOF
    
    chmod +x start_gateway.sh
    success "Startup script created: ./start_gateway.sh"
}

# Show completion message
show_completion() {
    echo ""
    title "============================================================================"
    success "Standalone Deployment Complete!"
    title "============================================================================"
    echo ""
    echo "To start the gateway:"
    echo "  ./start_gateway.sh"
    echo ""
    echo "Or manually:"
    echo "  source venv/bin/activate"
    echo "  python simple_llama_gateway_crash_safe.py"
    echo ""
    echo "Gateway will be available at:"
    echo "  http://localhost:8082"
    echo ""
    echo "Health check:"
    echo "  curl http://localhost:8082/health"
    echo ""
    echo "📚 Documentation:"
    echo "  README.md - Full documentation"
    echo "  SETUP_GUIDE.md - Detailed setup guide"
    echo "  SECURITY_FIXES_APPLIED.md - Security configuration"
    echo ""
}

# Main execution
main() {
    echo ""
    title "============================================================================"
    title "  EgoLlama-clean - Standalone Deployment"
    title "============================================================================"
    echo ""
    
    check_prerequisites
    echo ""
    
    setup_venv
    echo ""
    
    install_dependencies
    echo ""
    
    setup_environment
    echo ""
    
    check_optional_services
    echo ""
    
    create_startup_script
    echo ""
    
    show_completion
}

# Run main function
main

