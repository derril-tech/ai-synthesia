#!/bin/bash

# Load testing script for Synesthesia AI
# Comprehensive load testing with multiple scenarios

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"
CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
TEST_DURATION="${TEST_DURATION:-300}"
RAMP_UP_TIME="${RAMP_UP_TIME:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-./load-test-results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if required Python packages are available
    python3 -c "import aiohttp, psutil" 2>/dev/null || {
        error "Required Python packages not installed. Run: pip install aiohttp psutil"
        exit 1
    }
    
    # Check if curl is available for health checks
    if ! command -v curl &> /dev/null; then
        error "curl is required but not installed"
        exit 1
    fi
    
    success "All dependencies are available"
}

# Health check
health_check() {
    log "Performing health check on $BASE_URL..."
    
    if curl -s -f "$BASE_URL/v1/health" > /dev/null; then
        success "Service is healthy"
        return 0
    else
        error "Service health check failed"
        return 1
    fi
}

# Create output directory
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    log "Output directory: $OUTPUT_DIR"
}

# Run basic load test
run_basic_load_test() {
    log "Running basic load test..."
    
    local output_file="$OUTPUT_DIR/basic_load_test_$TIMESTAMP.json"
    
    python3 scripts/performance-test.py \
        --url "$BASE_URL" \
        --users "$CONCURRENT_USERS" \
        --duration "$TEST_DURATION" \
        --ramp-up "$RAMP_UP_TIME" \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        success "Basic load test completed successfully"
        return 0
    else
        error "Basic load test failed"
        return 1
    fi
}

# Run stress test with increasing load
run_stress_test() {
    log "Running stress test with increasing load..."
    
    local stress_users=(10 25 50 100 200)
    local stress_duration=120
    
    for users in "${stress_users[@]}"; do
        log "Testing with $users concurrent users..."
        
        local output_file="$OUTPUT_DIR/stress_test_${users}users_$TIMESTAMP.json"
        
        python3 scripts/performance-test.py \
            --url "$BASE_URL" \
            --users "$users" \
            --duration "$stress_duration" \
            --ramp-up 30 \
            --output "$output_file"
        
        if [ $? -ne 0 ]; then
            error "Stress test failed at $users users"
            return 1
        fi
        
        # Brief pause between stress levels
        sleep 10
    done
    
    success "Stress test completed successfully"
    return 0
}

# Run spike test
run_spike_test() {
    log "Running spike test..."
    
    local spike_users=500
    local spike_duration=60
    local output_file="$OUTPUT_DIR/spike_test_$TIMESTAMP.json"
    
    python3 scripts/performance-test.py \
        --url "$BASE_URL" \
        --users "$spike_users" \
        --duration "$spike_duration" \
        --ramp-up 5 \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        success "Spike test completed successfully"
        return 0
    else
        warning "Spike test may have revealed performance issues"
        return 1
    fi
}

# Run endurance test
run_endurance_test() {
    log "Running endurance test..."
    
    local endurance_users=20
    local endurance_duration=1800  # 30 minutes
    local output_file="$OUTPUT_DIR/endurance_test_$TIMESTAMP.json"
    
    python3 scripts/performance-test.py \
        --url "$BASE_URL" \
        --users "$endurance_users" \
        --duration "$endurance_duration" \
        --ramp-up 60 \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        success "Endurance test completed successfully"
        return 0
    else
        error "Endurance test failed"
        return 1
    fi
}

# Run chaos engineering test
run_chaos_test() {
    log "Running chaos engineering test..."
    
    local chaos_users=30
    local chaos_duration=600  # 10 minutes
    local output_file="$OUTPUT_DIR/chaos_test_$TIMESTAMP.json"
    
    python3 scripts/performance-test.py \
        --url "$BASE_URL" \
        --users "$chaos_users" \
        --duration "$chaos_duration" \
        --ramp-up 30 \
        --chaos \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        success "Chaos engineering test completed successfully"
        return 0
    else
        warning "Chaos engineering test revealed potential resilience issues"
        return 1
    fi
}

# Generate summary report
generate_summary_report() {
    log "Generating summary report..."
    
    local summary_file="$OUTPUT_DIR/load_test_summary_$TIMESTAMP.md"
    
    cat > "$summary_file" << EOF
# Load Test Summary Report

**Test Date:** $(date)
**Base URL:** $BASE_URL
**Test Duration:** $TEST_DURATION seconds
**Max Concurrent Users:** $CONCURRENT_USERS

## Test Results

EOF
    
    # Add results from each test file
    for result_file in "$OUTPUT_DIR"/*_$TIMESTAMP.json; do
        if [ -f "$result_file" ]; then
            local test_name=$(basename "$result_file" .json)
            echo "### $test_name" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Extract key metrics using Python
            python3 -c "
import json
import sys

try:
    with open('$result_file', 'r') as f:
        data = json.load(f)
    
    results = data.get('test_results', {})
    system = data.get('system_summary', {})
    
    print(f'- **Total Requests:** {results.get(\"total_requests\", 0)}')
    print(f'- **Success Rate:** {results.get(\"success_rate\", 0):.2f}%')
    print(f'- **Requests/Second:** {results.get(\"requests_per_second\", 0):.2f}')
    print(f'- **Average Response Time:** {results.get(\"avg_response_time\", 0)*1000:.1f}ms')
    print(f'- **95th Percentile:** {results.get(\"p95_response_time\", 0)*1000:.1f}ms')
    print(f'- **Peak CPU Usage:** {system.get(\"max_cpu_percent\", 0):.1f}%')
    print(f'- **Peak Memory Usage:** {system.get(\"max_memory_percent\", 0):.1f}%')
    print('')
    
    if results.get('errors'):
        print('**Errors:**')
        for error, count in results['errors'].items():
            print(f'- {error}: {count}')
        print('')

except Exception as e:
    print(f'Error processing {result_file}: {e}')
" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Recommendations

Based on the test results:

1. **Performance Optimization:**
   - Monitor response times under load
   - Consider horizontal scaling if needed
   - Optimize database queries for high concurrency

2. **Resource Management:**
   - Monitor CPU and memory usage patterns
   - Set up appropriate resource limits
   - Consider auto-scaling policies

3. **Error Handling:**
   - Review any errors that occurred during testing
   - Implement proper circuit breakers
   - Add comprehensive monitoring and alerting

4. **Capacity Planning:**
   - Use these results to plan production capacity
   - Set up load balancing if needed
   - Consider CDN for static assets

## Next Steps

- Review detailed results in individual JSON files
- Set up continuous performance monitoring
- Implement performance regression testing in CI/CD
- Plan for regular load testing schedule

EOF
    
    success "Summary report generated: $summary_file"
}

# Monitor system resources during tests
monitor_system() {
    local duration=$1
    local output_file="$OUTPUT_DIR/system_monitor_$TIMESTAMP.log"
    
    log "Starting system monitoring for ${duration}s..."
    
    # Monitor system resources in background
    (
        end_time=$(($(date +%s) + duration))
        while [ $(date +%s) -lt $end_time ]; do
            echo "$(date '+%Y-%m-%d %H:%M:%S') $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1) $(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')" >> "$output_file"
            sleep 5
        done
    ) &
    
    local monitor_pid=$!
    echo $monitor_pid > "$OUTPUT_DIR/monitor.pid"
}

# Stop system monitoring
stop_monitoring() {
    if [ -f "$OUTPUT_DIR/monitor.pid" ]; then
        local monitor_pid=$(cat "$OUTPUT_DIR/monitor.pid")
        kill $monitor_pid 2>/dev/null || true
        rm -f "$OUTPUT_DIR/monitor.pid"
        log "System monitoring stopped"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    stop_monitoring
}

# Set up cleanup trap
trap cleanup EXIT

# Main execution
main() {
    log "Starting comprehensive load testing for Synesthesia AI"
    log "Configuration: $CONCURRENT_USERS users, ${TEST_DURATION}s duration"
    
    # Check dependencies
    check_dependencies
    
    # Setup
    setup_output_dir
    
    # Health check
    if ! health_check; then
        error "Service is not healthy. Aborting load test."
        exit 1
    fi
    
    # Start system monitoring
    monitor_system $((TEST_DURATION * 5))  # Monitor for longer than tests
    
    local failed_tests=0
    
    # Run test suite
    log "=== Starting Load Test Suite ==="
    
    # Basic load test
    if ! run_basic_load_test; then
        ((failed_tests++))
    fi
    
    # Stress test
    if ! run_stress_test; then
        ((failed_tests++))
    fi
    
    # Spike test
    if ! run_spike_test; then
        ((failed_tests++))
    fi
    
    # Endurance test (optional, takes long time)
    if [ "${RUN_ENDURANCE:-false}" = "true" ]; then
        if ! run_endurance_test; then
            ((failed_tests++))
        fi
    fi
    
    # Chaos engineering test (optional)
    if [ "${RUN_CHAOS:-false}" = "true" ]; then
        if ! run_chaos_test; then
            ((failed_tests++))
        fi
    fi
    
    # Generate summary
    generate_summary_report
    
    # Final results
    log "=== Load Test Suite Complete ==="
    
    if [ $failed_tests -eq 0 ]; then
        success "All load tests passed successfully!"
        log "Results available in: $OUTPUT_DIR"
        exit 0
    else
        error "$failed_tests test(s) failed"
        log "Check detailed results in: $OUTPUT_DIR"
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
Load Testing Script for Synesthesia AI

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -u, --url URL       Base URL for testing (default: http://localhost:8000)
    -c, --users NUM     Number of concurrent users (default: 50)
    -d, --duration SEC  Test duration in seconds (default: 300)
    -r, --ramp-up SEC   Ramp up time in seconds (default: 60)
    -o, --output DIR    Output directory (default: ./load-test-results)
    --endurance         Include endurance test (30 minutes)
    --chaos             Include chaos engineering test

Environment Variables:
    BASE_URL            Base URL for testing
    CONCURRENT_USERS    Number of concurrent users
    TEST_DURATION       Test duration in seconds
    RAMP_UP_TIME        Ramp up time in seconds
    OUTPUT_DIR          Output directory
    RUN_ENDURANCE       Set to 'true' to run endurance test
    RUN_CHAOS           Set to 'true' to run chaos engineering test

Examples:
    # Basic load test
    $0

    # Custom configuration
    $0 --url http://staging.example.com --users 100 --duration 600

    # Full test suite including endurance and chaos
    RUN_ENDURANCE=true RUN_CHAOS=true $0

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -c|--users)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        -d|--duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        -r|--ramp-up)
            RAMP_UP_TIME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --endurance)
            RUN_ENDURANCE=true
            shift
            ;;
        --chaos)
            RUN_CHAOS=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main
