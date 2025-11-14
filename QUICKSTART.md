# TrainOps Observatory - Quick Start

Get up and running in 5 minutes!

## Prerequisites

- Docker & Docker Compose
- Python 3.8+
- PyTorch (for examples)

## Step 1: Start the Backend (30 seconds)

```bash
# Clone or navigate to project
cd trainops-observatory

# Start services (database + API)
docker-compose up -d db backend

# Wait for services to be healthy
docker-compose ps

# Check health
curl http://localhost:5000/health
# Should return: {"status": "healthy"}
```

## Step 2: Install SDK (10 seconds)

```bash
# Install in development mode
cd sdk
pip install -e .
```

## Step 3: Run Example (2 minutes)

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Run MNIST example
cd ../examples
python mnist_simple.py
```

You should see:
```
Starting training with TrainOps monitoring
Run ID: <uuid>
...
Training complete!
View results at: http://localhost:3000/runs/<uuid>
```

## Step 4: Check API

```bash
# List all runs
curl http://localhost:5000/api/runs

# Get specific run
curl http://localhost:5000/api/runs/<run-id>

# Get metrics
curl http://localhost:5000/api/runs/<run-id>/metrics?limit=10
```

## Troubleshooting

### Database connection failed
```bash
# Check if database is running
docker-compose ps db

# View logs
docker-compose logs db

# Restart
docker-compose restart db
```

### SDK not found
```bash
# Make sure you installed it
cd sdk
pip install -e .

# Verify
python -c "import trainops; print(trainops.__version__)"
```

### Port already in use
```bash
# Change ports in docker-compose.yml
# Default: 5432 (postgres), 5000 (backend), 3000 (frontend)
```

## Next Steps

1. **Add to your training script:**
   ```python
   from trainops import TrainOpsMonitor
   
   monitor = TrainOpsMonitor(
       run_name="my_experiment",
       project="my_project"
   )
   
   @monitor.track_training
   def train_epoch(model, dataloader):
       for batch in dataloader:
           # ... your training code
           monitor.log_step(loss=loss.item())
   ```

2. **View in dashboard** (Week 2): http://localhost:3000

3. **See docs**: [User Guide](docs/user-guide.md)

## Development

```bash
# Run backend locally (without Docker)
cd backend
pip install -r requirements.txt
export DATABASE_URL=postgresql://trainops:trainops_dev_password@localhost:5432/trainops
python run.py

# Run tests (coming soon)
pytest
```

