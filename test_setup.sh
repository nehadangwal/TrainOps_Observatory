#!/bin/bash

# TrainOps Observatory - Setup Test Script
# Tests that all components are working correctly

set -e  # Exit on error

echo "=================================================="
echo "TrainOps Observatory - Setup Test"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check Docker is running
echo "Test 1: Docker daemon..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker is running"
else
    echo -e "${RED}✗${NC} Docker is not running. Please start Docker."
    exit 1
fi

# Test 2: Start database
echo ""
echo "Test 2: Starting database..."
docker-compose up -d db
sleep 5  # Wait for DB to initialize

if docker-compose ps db | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Database is running"
else
    echo -e "${RED}✗${NC} Database failed to start"
    docker-compose logs db
    exit 1
fi

# Test 3: Check database health
echo ""
echo "Test 3: Database health check..."
MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if docker-compose exec -T db pg_isready -U trainops > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Database is healthy"
        break
    fi
    RETRY=$((RETRY+1))
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo -e "${RED}✗${NC} Database health check failed"
        exit 1
    fi
    echo -e "${YELLOW}⏳${NC} Waiting for database... ($RETRY/$MAX_RETRIES)"
    sleep 2
done

# Test 4: Start backend
echo ""
echo "Test 4: Starting backend..."
docker-compose up -d backend
sleep 5

if docker-compose ps backend | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Backend is running"
else
    echo -e "${RED}✗${NC} Backend failed to start"
    docker-compose logs backend
    exit 1
fi

# Test 5: Backend health check
echo ""
echo "Test 5: Backend API health..."
MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5000/health | grep -q "healthy"; then
        echo -e "${GREEN}✓${NC} Backend API is responding"
        break
    fi
    RETRY=$((RETRY+1))
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo -e "${RED}✗${NC} Backend API health check failed"
        docker-compose logs backend
        exit 1
    fi
    echo -e "${YELLOW}⏳${NC} Waiting for backend... ($RETRY/$MAX_RETRIES)"
    sleep 2
done

# Test 6: API endpoints
echo ""
echo "Test 6: Testing API endpoints..."

# Test stats endpoint
if curl -s http://localhost:5000/api/stats | grep -q "total_runs"; then
    echo -e "${GREEN}✓${NC} /api/stats endpoint working"
else
    echo -e "${RED}✗${NC} /api/stats endpoint failed"
    exit 1
fi

# Test projects endpoint
if curl -s http://localhost:5000/api/projects | grep -q "projects"; then
    echo -e "${GREEN}✓${NC} /api/projects endpoint working"
else
    echo -e "${RED}✗${NC} /api/projects endpoint failed"
    exit 1
fi

# Test runs endpoint
if curl -s http://localhost:5000/api/runs | grep -q "runs"; then
    echo -e "${GREEN}✓${NC} /api/runs endpoint working"
else
    echo -e "${RED}✗${NC} /api/runs endpoint failed"
    exit 1
fi

# Test 7: SDK installation
echo ""
echo "Test 7: Checking SDK installation..."
cd sdk
if pip show trainops > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  SDK already installed, skipping..."
else
    echo "Installing SDK..."
    pip install -e . > /dev/null 2>&1
fi

if python -c "from trainops import TrainOpsMonitor" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} SDK imports successfully"
else
    echo -e "${RED}✗${NC} SDK import failed"
    exit 1
fi

cd ..

# Test 8: Create a test run
echo ""
echo "Test 8: Creating test run via API..."
RESPONSE=$(curl -s -X POST http://localhost:5000/api/runs \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test_setup_run",
        "project": "test",
        "instance_type": "test",
        "tags": {"source": "setup_test"}
    }')

if echo "$RESPONSE" | grep -q "run_id"; then
    RUN_ID=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['run_id'])")
    echo -e "${GREEN}✓${NC} Test run created: $RUN_ID"
else
    echo -e "${RED}✗${NC} Failed to create test run"
    echo "$RESPONSE"
    exit 1
fi

# Test 9: Ingest test metrics
echo ""
echo "Test 9: Ingesting test metrics..."
METRICS_RESPONSE=$(curl -s -X POST "http://localhost:5000/api/runs/$RUN_ID/metrics" \
    -H "Content-Type: application/json" \
    -d '{
        "metrics": [
            {
                "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%S")'",
                "gpu_util": 75.5,
                "cpu_util": 45.2,
                "throughput": 123.4
            }
        ]
    }')

if echo "$METRICS_RESPONSE" | grep -q "ingested successfully"; then
    echo -e "${GREEN}✓${NC} Test metrics ingested"
else
    echo -e "${RED}✗${NC} Failed to ingest metrics"
    echo "$METRICS_RESPONSE"
    exit 1
fi

# Test 10: Retrieve metrics
echo ""
echo "Test 10: Retrieving metrics..."
METRICS=$(curl -s "http://localhost:5000/api/runs/$RUN_ID/metrics?limit=10")
if echo "$METRICS" | grep -q "gpu_util"; then
    echo -e "${GREEN}✓${NC} Metrics retrieved successfully"
else
    echo -e "${RED}✗${NC} Failed to retrieve metrics"
    exit 1
fi

# Test 11: Delete test run
echo ""
echo "Test 11: Cleaning up test run..."
DELETE_RESPONSE=$(curl -s -X DELETE "http://localhost:5000/api/runs/$RUN_ID")
if echo "$DELETE_RESPONSE" | grep -q "deleted successfully"; then
    echo -e "${GREEN}✓${NC} Test run deleted"
else
    echo -e "${YELLOW}⚠${NC}  Could not delete test run (non-critical)"
fi

# Summary
echo ""
echo "=================================================="
echo -e "${GREEN}All tests passed! ✓${NC}"
echo "=================================================="
echo ""
echo "Services running:"
echo "  • Database:  localhost:5432"
echo "  • Backend:   http://localhost:5000"
echo ""
echo "Next steps:"
echo "  1. Run example: cd examples && python mnist_simple.py"
echo "  2. View runs:   trainops runs list"
echo "  3. Check docs:  cat QUICKSTART.md"
echo ""
