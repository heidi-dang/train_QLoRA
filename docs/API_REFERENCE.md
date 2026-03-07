# QLoRA Training Pipeline - API Reference

## 🎯 Overview

The QLoRA Training Pipeline provides a comprehensive REST API for training control, monitoring, and management. This document covers all available endpoints, request/response formats, and usage examples.

## 🌐 Base URL

```
http://localhost:8000
```

## 🔐 Authentication

Currently, the API runs without authentication. For production use, configure API keys or OAuth in `server/train_api.py`.

## 📚 Endpoints

### Training Control

#### Start Training
```http
POST /train/start
```

**Response**:
```json
{
  "status": "success",
  "message": "Training started",
  "pid": 12345
}
```

#### Stop Training
```http
POST /train/stop
```

**Response**:
```json
{
  "status": "success",
  "message": "Training stopped"
}
```

#### Get Training Status
```http
GET /train/status
```

**Response**:
```json
{
  "running": true,
  "current_round": 3,
  "total_rounds": 10,
  "current_step": 234,
  "total_steps": 500,
  "progress_percent": 46.8,
  "training_loss": 0.3421,
  "learning_rate": 2e-4,
  "start_time": "2026-03-07T15:30:00Z",
  "estimated_completion": "2026-03-07T16:45:00Z"
}
```

#### Get Training Metrics
```http
GET /train/metrics
```

**Response**:
```json
{
  "resource_utilization": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "gpu_utilization": 78.0,
    "disk_usage": 23.4
  },
  "training_metrics": {
    "current_loss": 0.3421,
    "learning_rate": 2e-4,
    "training_speed": 2.3,
    "samples_processed": 1567,
    "total_samples": 5000
  },
  "lora_metrics": {
    "lora_r": 16,
    "lora_alpha": 32,
    "adapter_size_mb": 45.2,
    "trainable_params": 4200000,
    "total_params": 7000000000
  }
}
```

### Configuration Management

#### Get Configuration
```http
GET /config
```

**Response**:
```json
{
  "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "languages": ["python", "javascript"],
  "rounds": 10,
  "samples_per_round": 100,
  "train_steps": 500,
  "batch_size": 50,
  "learning_rate": 2e-4,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "enable_mlflow": true,
  "enable_tensorboard": true,
  "enable_dashboard": true
}
```

#### Update Configuration
```http
POST /config
Content-Type: application/json

{
  "learning_rate": 1e-4,
  "batch_size": 32,
  "lora_r": 32
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Configuration updated",
  "updated_fields": ["learning_rate", "batch_size", "lora_r"]
}
```

### Data Management

#### Get Dataset Info
```http
GET /data/info
```

**Response**:
```json
{
  "raw_samples": 1000,
  "clean_samples": 850,
  "training_samples": 800,
  "validation_samples": 50,
  "languages": {
    "python": 600,
    "javascript": 200,
    "cpp": 50
  },
  "last_updated": "2026-03-07T15:30:00Z"
}
```

#### Clean Dataset
```http
POST /data/clean
```

**Response**:
```json
{
  "status": "success",
  "message": "Dataset cleaning started",
  "job_id": "clean_20260307_153000"
}
```

#### Get Data Generation Status
```http
GET /data/generation/status
```

**Response**:
```json
{
  "stage": "generating",
  "total_samples": 100,
  "processed_samples": 67,
  "progress_percent": 67.0,
  "generation_rate": 12.3,
  "estimated_completion": "2026-03-07T15:45:00Z"
}
```

### Model Management

#### Get Model Info
```http
GET /model/info
```

**Response**:
```json
{
  "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "current_checkpoint": "adapter_round_3",
  "available_checkpoints": [
    "adapter_round_1",
    "adapter_round_2",
    "adapter_round_3"
  ],
  "model_size_gb": 14.2,
  "adapter_size_mb": 45.2,
  "last_training": "2026-03-07T15:30:00Z"
}
```

#### Load Checkpoint
```http
POST /model/load
Content-Type: application/json

{
  "checkpoint": "adapter_round_2"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Checkpoint loaded successfully",
  "checkpoint": "adapter_round_2"
}
```

### Evaluation

#### Run Evaluation
```http
POST /evaluation/run
Content-Type: application/json

{
  "checkpoint": "adapter_round_3",
  "metrics": ["python", "cpp", "instruction_following"]
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Evaluation started",
  "job_id": "eval_20260307_153000",
  "estimated_duration": "00:15:00"
}
```

#### Get Evaluation Results
```http
GET /evaluation/results/{round_number}
```

**Response**:
```json
{
  "round": 3,
  "status": "completed",
  "evaluation_time_seconds": 892,
  "python_total": 20,
  "python_passed": 16,
  "cpp_total": 15,
  "cpp_passed": 12,
  "instruction_total": 10,
  "instruction_passed": 8,
  "overall_score": 0.789,
  "python_pass_rate": 0.8,
  "cpp_pass_rate": 0.8,
  "instruction_pass_rate": 0.8,
  "detailed_results": {
    "python_tasks": [
      {
        "task_id": 1,
        "passed": true,
        "execution_time": 2.3,
        "error": null
      }
    ]
  }
}
```

### GitHub Integration

#### Search Repositories
```http
POST /github/search
Content-Type: application/json

{
  "query": "machine learning",
  "language": "python",
  "min_stars": 100,
  "min_forks": 50,
  "max_results": 50
}
```

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "name": "awesome-ml",
      "full_name": "user/awesome-ml",
      "stars": 1234,
      "forks": 567,
      "language": "Python",
      "clone_url": "https://github.com/user/awesome-ml.git"
    }
  ],
  "total_found": 50
}
```

#### Get Repository Status
```http
GET /github/repo/status
```

**Response**:
```json
{
  "total_repos": 50,
  "cloned_repos": 45,
  "processed_repos": 42,
  "failed_repos": 3,
  "last_search": "2026-03-07T15:30:00Z"
}
```

### System Health

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-07T15:30:00Z",
  "services": {
    "api": "running",
    "training": "running",
    "mlflow": "running",
    "tensorboard": "running",
    "dashboard": "running"
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_usage": 23.4,
    "uptime_hours": 24.5
  }
}
```

#### System Metrics
```http
GET /metrics
```

**Response**:
```json
{
  "timestamp": "2026-03-07T15:30:00Z",
  "cpu_percent": 45.2,
  "memory_percent": 67.8,
  "memory_used_gb": 8.1,
  "memory_total_gb": 16.0,
  "disk_usage": 23.4,
  "network_sent_mb": 1024.5,
  "network_recv_mb": 2048.7,
  "process_count": 156,
  "gpu_utilization": 78.0,
  "gpu_memory_used": 8.2,
  "gpu_memory_total": 12.0
}
```

#### Logs
```http
GET /logs/{service}
```

**Parameters**:
- `service`: `api`, `training`, `mlflow`, `tensorboard`, `dashboard`
- `lines`: Number of lines to return (default: 50)

**Response**:
```json
{
  "service": "training",
  "lines": [
    {
      "timestamp": "2026-03-07T15:30:00Z",
      "level": "INFO",
      "message": "Training round 3 started"
    }
  ]
}
```

## 📊 WebSocket Events

### Real-time Updates

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

#### Event Types

**Training Progress**:
```json
{
  "type": "training_progress",
  "data": {
    "round": 3,
    "step": 234,
    "loss": 0.3421,
    "progress_percent": 46.8
  }
}
```

**Resource Update**:
```json
{
  "type": "resource_update",
  "data": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "gpu_utilization": 78.0
  }
}
```

**Log Message**:
```json
{
  "type": "log_message",
  "data": {
    "service": "training",
    "level": "INFO",
    "message": "Training step completed",
    "timestamp": "2026-03-07T15:30:00Z"
  }
}
```

## 🚨 Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid configuration parameter",
    "details": {
      "field": "learning_rate",
      "value": "invalid_value",
      "expected": "numeric"
    }
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid request parameters |
| `TRAINING_RUNNING` | Training already in progress |
| `TRAINING_NOT_RUNNING` | Training not active |
| `CHECKPOINT_NOT_FOUND` | Specified checkpoint doesn't exist |
| `CONFIG_ERROR` | Configuration error |
| `SYSTEM_ERROR` | Internal system error |
| `RESOURCE_LIMIT` | Resource limit exceeded |

## 🔧 SDK Examples

### Python SDK

```python
import requests

class QLoRAClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def start_training(self):
        response = requests.post(f"{self.base_url}/train/start")
        return response.json()
    
    def get_status(self):
        response = requests.get(f"{self.base_url}/train/status")
        return response.json()
    
    def update_config(self, config):
        response = requests.post(f"{self.base_url}/config", json=config)
        return response.json()

# Usage
client = QLoRAClient()
client.start_training()
status = client.get_status()
print(f"Training status: {status['running']}")
```

### JavaScript SDK

```javascript
class QLoRAClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async startTraining() {
        const response = await fetch(`${this.baseUrl}/train/start`, {
            method: 'POST'
        });
        return await response.json();
    }
    
    async getStatus() {
        const response = await fetch(`${this.baseUrl}/train/status`);
        return await response.json();
    }
    
    connectWebSocket() {
        const ws = new WebSocket(`ws://${this.baseUrl}/ws`);
        return ws;
    }
}

// Usage
const client = new QLoRAClient();
await client.startTraining();
const status = await client.getStatus();
console.log(`Training status: ${status.running}`);
```

### cURL Examples

```bash
# Start training
curl -X POST http://localhost:8000/train/start

# Get status
curl http://localhost:8000/train/status

# Update configuration
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"learning_rate": 1e-4, "batch_size": 32}'

# Get metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

## 📈 Rate Limiting

- **Default Limit**: 100 requests per minute
- **Burst Limit**: 10 requests per second
- **WebSocket Limit**: 5 connections per IP

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1646820000
```

## 🔒 Security Considerations

### Production Deployment

1. **Enable Authentication**
   ```python
   # In server/train_api.py
   from fastapi import Depends, HTTPException
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   async def verify_token(token: str = Depends(security)):
       # Validate token
       pass
   ```

2. **Use HTTPS**
   ```bash
   # Configure SSL certificates
   uvicorn server.train_api:app --ssl-keyfile key.pem --ssl-certfile cert.pem
   ```

3. **Network Security**
   ```bash
   # Firewall rules
   ufw allow 8000/tcp
   ufw enable
   ```

### Input Validation

All inputs are validated using Pydantic models:
```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    learning_rate: float = Field(gt=0, le=1)
    batch_size: int = Field(gt=0, le=1024)
    rounds: int = Field(gt=0, le=100)
```

## 📝 Monitoring & Logging

### Request Logging

All API requests are logged with:
- Timestamp
- Method
- Endpoint
- Response code
- Processing time
- Client IP

### Performance Metrics

Track API performance:
- Request count
- Response times
- Error rates
- Resource usage

### Health Monitoring

Automated health checks:
- Service availability
- Resource thresholds
- Error rate monitoring

---

**Last Updated**: 2026-03-07  
**Version**: 1.0.0  
**Author**: QLoRA Development Team
