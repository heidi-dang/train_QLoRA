# QLoRA Training Dashboard

A rich, real-time monitoring dashboard for the QLoRA training pipeline.

## Features

### 📊 **Resource Utilization**
- **CPU Usage**: Real-time CPU percentage
- **Memory Usage**: RAM usage with used/total breakdown  
- **GPU Usage**: GPU utilization and memory (if available)
- **Disk Usage**: Storage space consumption

### 🎯 **Training Convergence Metrics**
- **Current Round**: Active training round number
- **Training Step**: Current step vs total steps
- **Progress %**: Visual progress indicator
- **Training Loss**: Current loss value with history
- **Learning Rate**: Active learning rate
- **ETA**: Estimated time to completion
- **Training Status**: Active/Idle indicator

### 🔧 **LoRA Specifics**
- **LoRA Rank**: Adapter rank parameter
- **LoRA Alpha**: Scaling factor
- **LoRA Dropout**: Regularization dropout rate
- **Adapter Size**: Memory footprint of LoRA adapter
- **Trainable Parameters**: Number of trainable parameters
- **Total Parameters**: Total model parameters
- **Trainable %**: Percentage of parameters being trained

### 📈 **Progress % and Time Estimates**
- **Training Progress**: Real-time training progress bar
- **Data Generation Progress**: Sample generation progress
- **Time Remaining**: Estimated completion time
- **Sample Rate**: Samples per minute generation rate

### 📝 **Teacher Data Generation Progress**
- **Generation Stage**: Current stage (idle/generating)
- **Sample Count**: Processed vs total samples
- **Generation Rate**: Samples per minute
- **Progress %**: Visual completion indicator

### 📋 **Real-time Log Tailing**
- **Live Logs**: Real-time log streaming
- **Color Coding**: 
  - 🔴 Red for errors
  - 🟡 Yellow for warnings
  - ⚪ White for info
  - 🔘 Gray for other messages
- **Auto-scroll**: Latest log lines always visible

## Usage

### Start Dashboard Standalone
```bash
./run.sh dashboard
```

### Auto-start with Services
```bash
./run.sh up  # Dashboard automatically starts if enabled
```

### Interactive Setup
```bash
./run.sh setup  # Choose whether to enable dashboard during setup
```

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QLoRA Training Dashboard                        │
├─────────────────────────────┬─────────────────────────────────────┤
│ 📊 Resource Utilization    │ 🔧 LoRA Specifics                   │
│ • CPU Usage: 45.2%        │ • LoRA Rank: 16                     │
│ • Memory: 8.1/16.0 GB     │ • LoRA Alpha: 32                    │
│ • GPU: 78% (if available) │ • Adapter Size: 45.2 MB             │
│                           │ • Trainable: 4.2M/7.0B (0.06%)      │
├─────────────────────────────┼─────────────────────────────────────┤
│ 🎯 Training Metrics        │ 📝 Data Generation                  │
│ • Round: 3/10             │ • Stage: Generating                  │
│ • Step: 234/500           │ • Samples: 67/100                   │
│ • Progress: 46.8%         │ • Rate: 12.3 samples/min            │
│ • Loss: 0.3421            │ • Progress: 67%                     │
│ • LR: 2e-4                │                                     │
│ • ETA: 00:12:45           │ 📈 Overall Progress                  │
│ • Status: 🟢 Training     │ ████████████████████████████░░░░ 67%│
├─────────────────────────────┴─────────────────────────────────────┤
│ 📋 Real-time Logs                                               │
│ [15:13:42] INFO: Starting training round 3                      │
│ [15:13:43] INFO: Loading model checkpoint round_2               │
│ [15:13:44] INFO: Training step 1/500, loss: 0.5432             │
│ [15:13:45] INFO: Training step 2/500, loss: 0.4891             │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

The dashboard can be configured during setup:

```bash
./run.sh setup
```

Key options:
- **Enable Dashboard**: Turn on/off rich monitoring
- **Auto-start**: Start with `./run.sh up`
- **Refresh Rate**: Update frequency (default: 1 second)

## Requirements

- Python 3.8+
- Rich library for terminal UI
- psutil for system monitoring
- Access to log files and checkpoints

## Dependencies

The dashboard requires the following Python packages:
- `rich` - Terminal UI framework
- `psutil` - System resource monitoring
- `nvidia-smi` - GPU monitoring (optional)

## Troubleshooting

### Dashboard Not Starting
```bash
# Check if rich is installed
pip install rich

# Check logs
tail -f logs/dashboard.log
```

### GPU Not Showing
```bash
# Verify nvidia-smi is available
nvidia-smi

# Check GPU drivers
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
```

### Log Files Not Updating
```bash
# Check log permissions
ls -la logs/

# Restart training to generate logs
curl -X POST http://localhost:8000/train/start
```

## Performance

The dashboard is optimized for minimal resource usage:
- **CPU**: < 1% overhead
- **Memory**: ~50MB
- **Network**: No external dependencies
- **Disk**: Read-only access to logs

## Customization

You can customize the dashboard by modifying:
- `dashboard/app.py` - Main dashboard logic
- Update intervals
- Color schemes
- Layout configuration
- Additional metrics

## Integration

The dashboard integrates seamlessly with:
- **MLflow**: Training experiment tracking
- **TensorBoard**: Training visualization
- **API Server**: Training control endpoints
- **Training Loop**: Continuous training pipeline
