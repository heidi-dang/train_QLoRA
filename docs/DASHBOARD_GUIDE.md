# QLoRA Training Dashboard - Complete User Guide

## 🎯 Overview

The QLoRA Training Dashboard provides real-time monitoring and visualization of your training pipeline. It offers comprehensive insights into system resources, training progress, LoRA specifics, data generation, and live logs.

## 🚀 Quick Start

### Starting the Dashboard

```bash
# Method 1: Start with all services
./run.sh up

# Method 2: Standalone dashboard
./run.sh dashboard

# Method 3: Interactive setup (includes dashboard option)
./run.sh setup
```

### Access Points

- **Dashboard**: Terminal UI (starts automatically)
- **API Documentation**: http://localhost:8000/docs
- **MLflow Tracking**: http://localhost:5000
- **TensorBoard**: http://localhost:6006

## 📊 Dashboard Features

### 1. Resource Utilization Panel
**Location**: Top-left panel

**Metrics Displayed**:
- **CPU Usage**: Real-time CPU percentage
- **Memory Usage**: RAM usage with used/total breakdown
- **Disk Usage**: Storage space consumption
- **Uptime**: System uptime in hours
- **Processes**: Total number of running processes
- **GPU Usage**: GPU utilization (if available)
- **GPU Memory**: GPU memory usage (if available)
- **Network**: Upload/download statistics

**Color Indicators**:
- 🟢 Green: Normal usage (< 70%)
- 🟡 Yellow: Moderate usage (70-90%)
- 🔴 Red: High usage (> 90%)

### 2. Training Metrics Panel
**Location**: Bottom-left panel

**Metrics Displayed**:
- **Current Round**: Round number (current/total)
- **Training Step**: Step progress (current/total)
- **Step Progress**: Visual progress percentage
- **Round Progress**: Overall training round progress
- **Current Loss**: Latest training loss value
- **Learning Rate**: Active learning rate
- **Training Speed**: Steps per second
- **ETA**: Estimated time to completion
- **Samples**: Samples processed (current/total)
- **Status**: Training state (🟢 Training / 🔴 Idle)

### 3. LoRA Specifics Panel
**Location**: Top-right panel

**Metrics Displayed**:
- **LoRA Rank**: Adapter rank parameter
- **LoRA Alpha**: Scaling factor
- **LoRA Dropout**: Regularization dropout rate
- **Adapter Size**: Memory footprint in MB
- **Trainable Params**: Number of trainable parameters
- **Total Params**: Total model parameters
- **Trainable %**: Percentage of parameters being trained

### 4. Data Generation Panel
**Location**: Middle-right panel

**Metrics Displayed**:
- **Stage**: Current generation stage (idle/generating)
- **Samples**: Sample count (processed/total)
- **Progress**: Visual completion percentage
- **Rate**: Samples per minute generation rate

### 5. Overall Progress Panel
**Location**: Bottom-right panel

**Features**:
- **Training Progress Bar**: Visual step-by-step progress
- **Data Generation Progress**: Sample generation progress
- **Time Remaining**: ETA calculations
- **Progress Percentage**: Overall completion metrics

### 6. Real-time Logs Panel
**Location**: Bottom panel

**Features**:
- **Live Log Streaming**: Real-time log updates
- **Color Coding**:
  - 🔴 Red: Error messages
  - 🟡 Yellow: Warning messages
  - ⚪ White: Information messages
  - 🔘 Gray: Other messages
- **Auto-scroll**: Latest logs always visible
- **Log Sources**: Supports multiple log files

## ⚙️ Configuration

### Interactive Setup

```bash
./run.sh setup
```

During setup, you'll be prompted for:
- Dashboard enable/disable
- Auto-start preferences
- Refresh rate settings

### Manual Configuration

Edit `config.json` to customize dashboard settings:

```json
{
  "enable_dashboard": true,
  "dashboard_refresh_rate": 1.0,
  "dashboard_log_lines": 20,
  "enable_mlflow": true,
  "enable_tensorboard": true
}
```

### Environment Variables

```bash
# Enable/disable dashboard
export ENABLE_DASHBOARD=true

# Set refresh rate (seconds)
export DASHBOARD_REFRESH_RATE=1.0

# Log verbosity
export DASHBOARD_LOG_LEVEL=INFO
```

## 🔧 Advanced Features

### Performance Optimization

The dashboard includes built-in performance optimizations:
- **Data Caching**: 0.5-second cache TTL
- **Limited Log Lines**: Shows last 15 log entries
- **Efficient Updates**: Only updates when data changes
- **Configurable Refresh Rate**: Adjust update frequency

### Custom Monitoring

Add custom metrics by extending monitor classes:

```python
# Example: Custom metric
class CustomMonitor:
    def get_metrics(self):
        return {
            'custom_metric': self.calculate_metric()
        }
```

### Log Filtering

Filter logs by severity or content:

```python
# In dashboard/app.py
def filter_logs(self, logs, level='INFO'):
    return [log for log in logs if level in log]
```

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Not Starting
```bash
# Check dependencies
pip install rich psutil

# Check logs
tail -f logs/dashboard.log

# Restart services
./run.sh stop && ./run.sh up
```

#### High CPU Usage
```bash
# Reduce refresh rate
export DASHBOARD_REFRESH_RATE=2.0

# Limit log lines
export DASHBOARD_LOG_LINES=10
```

#### Missing GPU Metrics
```bash
# Check nvidia-smi
nvidia-smi

# Install GPU drivers
sudo apt install nvidia-driver-470
```

#### Log Display Issues
```bash
# Check log permissions
ls -la logs/

# Create missing directories
mkdir -p logs
```

### Performance Tuning

#### High Resource Usage
- Reduce refresh rate to 2-5 seconds
- Limit log lines to 10-20
- Disable unused panels

#### Memory Leaks
- Restart dashboard periodically
- Monitor memory usage
- Check for log file growth

#### Network Issues
- Check firewall settings
- Verify port availability
- Test API endpoints

## 📈 Monitoring Best Practices

### Production Monitoring

1. **Set Appropriate Refresh Rates**
   - Production: 2-5 seconds
   - Development: 1 second
   - Testing: 0.5 seconds

2. **Monitor Resource Usage**
   - Keep CPU < 80%
   - Monitor memory growth
   - Watch disk space

3. **Log Management**
   - Rotate logs regularly
   - Monitor log file sizes
   - Filter critical messages

### Alert Thresholds

Configure alerts for:
- CPU usage > 90%
- Memory usage > 85%
- Disk usage > 80%
- Training failures
- API errors

### Backup and Recovery

```bash
# Backup configuration
cp config.json config.json.backup

# Export metrics
curl http://localhost:8000/metrics > metrics.log

# Restore dashboard
./run.sh stop && ./run.sh up
```

## 🔗 Integration

### API Integration

```python
import requests

# Get training status
response = requests.get('http://localhost:8000/train/status')
status = response.json()

# Start training
requests.post('http://localhost:8000/train/start')
```

### MLflow Integration

```python
import mlflow

# Log custom metrics
mlflow.log_metric("custom_metric", value)

# Track experiments
mlflow.set_experiment("qlora-dashboard")
```

### External Monitoring

```bash
# Export metrics to Prometheus
curl http://localhost:8000/metrics

# Grafana dashboard setup
# Add data source: http://localhost:8000
# Import dashboard template
```

## 📚 API Reference

### Dashboard Endpoints

```bash
# Get dashboard status
GET /dashboard/status

# Update configuration
POST /dashboard/config

# Export metrics
GET /dashboard/metrics
```

### Training Endpoints

```bash
# Start training
POST /train/start

# Stop training
POST /train/stop

# Get status
GET /train/status

# Get metrics
GET /train/metrics
```

## 🎨 Customization

### Themes and Colors

```python
# Custom color scheme
CUSTOM_COLORS = {
    'success': 'green',
    'warning': 'yellow',
    'error': 'red',
    'info': 'blue'
}
```

### Layout Customization

```python
# Custom layout
def create_custom_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="custom_panel", size=10),
        Layout(name="main")
    )
    return layout
```

### Metric Extensions

```python
# Add custom metrics
class ExtendedResourceMonitor(ResourceMonitor):
    def get_extended_stats(self):
        stats = super().get_current_stats()
        stats['custom_metric'] = self.calculate_custom()
        return stats
```

## 🚀 Performance Tips

### Optimization Strategies

1. **Reduce Update Frequency**
   ```bash
   export DASHBOARD_REFRESH_RATE=2.0
   ```

2. **Limit Data Collection**
   ```python
   # Limit log lines
   self.max_log_lines = 10
   
   # Reduce history size
   self.max_history = 30
   ```

3. **Use Efficient Data Structures**
   ```python
   # Use deque for rolling windows
   from collections import deque
   self.history = deque(maxlen=60)
   ```

### Resource Management

```bash
# Monitor dashboard resources
ps aux | grep dashboard

# Limit memory usage
ulimit -v 1048576  # 1GB limit

# Set process priority
nice -n 10 ./run.sh dashboard
```

## 📞 Support

### Getting Help

1. **Check Logs**: `tail -f logs/dashboard.log`
2. **Run Doctor**: `./run.sh doctor`
3. **Check Status**: `./run.sh status`
4. **Review Configuration**: `cat config.json`

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Full API reference
- **Examples**: Sample configurations and scripts

---

**Last Updated**: 2026-03-07  
**Version**: 1.0.0  
**Author**: QLoRA Development Team
