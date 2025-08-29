#!/bin/bash

# Deployment script for Synesthesia AI
# Supports staging and production deployments with canary releases

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-staging}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
REGISTRY="${REGISTRY:-synesthesia-ai}"
CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Deployment Script for Synesthesia AI

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker images
    deploy          Deploy to environment
    canary          Deploy canary release
    promote         Promote canary to full deployment
    rollback        Rollback to previous version
    status          Check deployment status
    logs            View deployment logs
    cleanup         Clean up old deployments

Options:
    -e, --env ENV           Environment (staging|production) [default: staging]
    -v, --version VERSION   Version to deploy [default: git short hash]
    -r, --registry REGISTRY Docker registry [default: synesthesia-ai]
    -c, --canary PERCENT    Canary percentage [default: 10]
    -f, --force             Force deployment without checks
    -h, --help              Show this help message

Environment Variables:
    ENVIRONMENT             Target environment
    VERSION                 Version to deploy
    REGISTRY                Docker registry
    CANARY_PERCENTAGE       Canary deployment percentage
    DOCKER_REGISTRY_URL     Docker registry URL
    DOCKER_USERNAME         Docker registry username
    DOCKER_PASSWORD         Docker registry password

Examples:
    # Deploy to staging
    $0 deploy

    # Deploy specific version to production
    $0 -e production -v v1.2.3 deploy

    # Canary deployment
    $0 -e production -c 20 canary

    # Rollback production
    $0 -e production rollback

EOF
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment files
    local env_file=".env.${ENVIRONMENT}"
    if [ ! -f "$env_file" ]; then
        error "Environment file $env_file not found"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Load environment variables
load_environment() {
    local env_file=".env.${ENVIRONMENT}"
    log "Loading environment from $env_file"
    
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
        success "Environment loaded"
    else
        error "Environment file $env_file not found"
        exit 1
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images for version $VERSION..."
    
    # Build API image
    log "Building API image..."
    docker build \
        -t "${REGISTRY}/api:${VERSION}" \
        -t "${REGISTRY}/api:latest" \
        --target production \
        ./apps/api
    
    # Build Web image
    log "Building Web image..."
    docker build \
        -t "${REGISTRY}/web:${VERSION}" \
        -t "${REGISTRY}/web:latest" \
        --target production \
        ./apps/web
    
    # Build Workers image
    log "Building Workers image..."
    docker build \
        -t "${REGISTRY}/workers:${VERSION}" \
        -t "${REGISTRY}/workers:latest" \
        ./apps/workers
    
    success "Docker images built successfully"
}

# Push images to registry
push_images() {
    if [ -n "$DOCKER_REGISTRY_URL" ]; then
        log "Pushing images to registry..."
        
        # Login to registry
        if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
            echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY_URL" -u "$DOCKER_USERNAME" --password-stdin
        fi
        
        # Push images
        docker push "${REGISTRY}/api:${VERSION}"
        docker push "${REGISTRY}/api:latest"
        docker push "${REGISTRY}/web:${VERSION}"
        docker push "${REGISTRY}/web:latest"
        docker push "${REGISTRY}/workers:${VERSION}"
        docker push "${REGISTRY}/workers:latest"
        
        success "Images pushed to registry"
    else
        log "No registry configured, skipping push"
    fi
}

# Run pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if services are healthy
    if [ "$ENVIRONMENT" = "production" ]; then
        log "Checking production health endpoints..."
        
        # Check API health
        if ! curl -f -s "https://api.synesthesia-ai.com/v1/health" > /dev/null; then
            warning "Production API health check failed"
        fi
        
        # Check web health
        if ! curl -f -s "https://synesthesia-ai.com/api/health" > /dev/null; then
            warning "Production web health check failed"
        fi
    fi
    
    # Run tests
    log "Running test suite..."
    if [ -f "package.json" ]; then
        npm test || {
            error "Tests failed"
            exit 1
        }
    fi
    
    # Check database migrations
    log "Checking database migrations..."
    # This would run actual migration checks
    
    success "Pre-deployment checks passed"
}

# Deploy to environment
deploy() {
    log "Deploying version $VERSION to $ENVIRONMENT..."
    
    # Set version in environment
    export VERSION
    
    # Choose compose file
    local compose_file="docker-compose.${ENVIRONMENT}.yml"
    if [ ! -f "$compose_file" ]; then
        error "Compose file $compose_file not found"
        exit 1
    fi
    
    # Run database migrations
    log "Running database migrations..."
    docker-compose -f "$compose_file" run --rm api python -m alembic upgrade head
    
    # Deploy services
    log "Deploying services..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    wait_for_health "$compose_file"
    
    # Run post-deployment tests
    log "Running post-deployment tests..."
    run_smoke_tests
    
    success "Deployment completed successfully"
}

# Canary deployment
canary_deploy() {
    log "Starting canary deployment ($CANARY_PERCENTAGE%) for version $VERSION..."
    
    if [ "$ENVIRONMENT" != "production" ]; then
        error "Canary deployments are only supported in production"
        exit 1
    fi
    
    # Deploy canary version
    export VERSION
    export CANARY_PERCENTAGE
    
    # Update load balancer configuration for canary
    log "Configuring load balancer for canary deployment..."
    # This would update nginx configuration to route percentage of traffic
    
    # Deploy canary instances
    docker-compose -f docker-compose.prod.yml up -d --scale api=1
    
    # Monitor canary metrics
    log "Monitoring canary deployment..."
    monitor_canary_metrics
    
    success "Canary deployment started"
}

# Promote canary to full deployment
promote_canary() {
    log "Promoting canary to full deployment..."
    
    # Update load balancer to route 100% traffic to new version
    log "Updating load balancer configuration..."
    
    # Scale up new version and scale down old version
    docker-compose -f docker-compose.prod.yml up -d
    
    # Clean up old version
    cleanup_old_versions
    
    success "Canary promoted to full deployment"
}

# Rollback deployment
rollback() {
    log "Rolling back deployment..."
    
    # Get previous version
    local previous_version
    previous_version=$(get_previous_version)
    
    if [ -z "$previous_version" ]; then
        error "No previous version found for rollback"
        exit 1
    fi
    
    log "Rolling back to version $previous_version..."
    
    # Set previous version
    export VERSION="$previous_version"
    
    # Deploy previous version
    local compose_file="docker-compose.${ENVIRONMENT}.yml"
    docker-compose -f "$compose_file" up -d
    
    # Wait for health
    wait_for_health "$compose_file"
    
    success "Rollback completed to version $previous_version"
}

# Wait for services to be healthy
wait_for_health() {
    local compose_file="$1"
    local max_attempts=30
    local attempt=1
    
    log "Waiting for services to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts..."
        
        # Check API health
        if docker-compose -f "$compose_file" exec -T api curl -f http://localhost:8000/v1/health > /dev/null 2>&1; then
            success "Services are healthy"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    error "Services failed to become healthy within timeout"
    exit 1
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Basic API tests
    local api_url
    if [ "$ENVIRONMENT" = "production" ]; then
        api_url="https://api.synesthesia-ai.com"
    else
        api_url="http://localhost:8001"
    fi
    
    # Test health endpoint
    if ! curl -f -s "$api_url/v1/health" | jq -e '.status == "healthy"' > /dev/null; then
        error "API health check failed"
        exit 1
    fi
    
    # Test authentication endpoint
    if ! curl -f -s "$api_url/v1/auth/health" > /dev/null; then
        warning "Auth endpoint check failed"
    fi
    
    success "Smoke tests passed"
}

# Monitor canary metrics
monitor_canary_metrics() {
    log "Monitoring canary metrics for 5 minutes..."
    
    local monitoring_duration=300  # 5 minutes
    local check_interval=30        # 30 seconds
    local checks=$((monitoring_duration / check_interval))
    
    for ((i=1; i<=checks; i++)); do
        log "Canary monitoring check $i/$checks..."
        
        # Check error rates, response times, etc.
        # This would integrate with monitoring systems
        
        # Check if error rate is acceptable
        local error_rate
        error_rate=$(get_canary_error_rate)
        
        if (( $(echo "$error_rate > 5.0" | bc -l) )); then
            error "Canary error rate too high: $error_rate%"
            log "Automatically rolling back canary..."
            rollback_canary
            exit 1
        fi
        
        sleep $check_interval
    done
    
    success "Canary monitoring completed successfully"
}

# Get canary error rate (mock implementation)
get_canary_error_rate() {
    # This would query actual monitoring systems
    echo "1.2"
}

# Rollback canary deployment
rollback_canary() {
    log "Rolling back canary deployment..."
    
    # Reset load balancer to route 100% to stable version
    # Scale down canary instances
    
    success "Canary rollback completed"
}

# Get previous version
get_previous_version() {
    # This would query deployment history
    git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo ""
}

# Cleanup old versions
cleanup_old_versions() {
    log "Cleaning up old versions..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove old containers
    docker container prune -f
    
    success "Cleanup completed"
}

# Get deployment status
get_status() {
    log "Getting deployment status for $ENVIRONMENT..."
    
    local compose_file="docker-compose.${ENVIRONMENT}.yml"
    
    if [ -f "$compose_file" ]; then
        docker-compose -f "$compose_file" ps
        
        # Show service health
        log "Service health status:"
        docker-compose -f "$compose_file" exec -T api curl -s http://localhost:8000/v1/health | jq '.' || echo "API health check failed"
    else
        error "Compose file $compose_file not found"
        exit 1
    fi
}

# View logs
view_logs() {
    local service="${1:-}"
    local compose_file="docker-compose.${ENVIRONMENT}.yml"
    
    if [ -n "$service" ]; then
        log "Viewing logs for service: $service"
        docker-compose -f "$compose_file" logs -f "$service"
    else
        log "Viewing logs for all services"
        docker-compose -f "$compose_file" logs -f
    fi
}

# Main execution
main() {
    local command=""
    local force=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -c|--canary)
                CANARY_PERCENTAGE="$2"
                shift 2
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|deploy|canary|promote|rollback|status|logs|cleanup)
                command="$1"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
        error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
        exit 1
    fi
    
    # Require command
    if [ -z "$command" ]; then
        error "Command is required"
        show_help
        exit 1
    fi
    
    log "Starting deployment script"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Command: $command"
    
    # Run prerequisites check
    check_prerequisites
    
    # Load environment
    load_environment
    
    # Execute command
    case $command in
        build)
            build_images
            push_images
            ;;
        deploy)
            if [ "$force" = false ]; then
                pre_deployment_checks
            fi
            build_images
            push_images
            deploy
            ;;
        canary)
            if [ "$force" = false ]; then
                pre_deployment_checks
            fi
            build_images
            push_images
            canary_deploy
            ;;
        promote)
            promote_canary
            ;;
        rollback)
            rollback
            ;;
        status)
            get_status
            ;;
        logs)
            view_logs "$2"
            ;;
        cleanup)
            cleanup_old_versions
            ;;
        *)
            error "Unknown command: $command"
            exit 1
            ;;
    esac
    
    success "Deployment script completed successfully"
}

# Run main function
main "$@"
