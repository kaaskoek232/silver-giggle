# Enhanced Memory Management for ComfyUI Reactor Node

## Overview

This guide explains how to integrate the enhanced memory management system with your custom Reactor face swapping node to achieve optimal performance and prevent memory leaks.

## Key Improvements

### 1. **Advanced GPU Memory Monitoring**
- **pynvml Integration**: Replaces basic `torch.cuda` monitoring with comprehensive NVIDIA Management Library
- **Process-Level Tracking**: Monitors individual processes using GPU memory
- **Real-time Utilization**: Tracks both memory and GPU utilization percentages
- **Container Awareness**: Properly handles Docker/Runpod environments

### 2. **VRAM/RAM Separation**
- **Dual Monitoring**: Separate thresholds and alerts for VRAM and RAM
- **Intelligent Cleanup**: Different strategies for GPU vs system memory
- **Process Filtering**: Focus monitoring on relevant processes (comfyui, python)

### 3. **Specialized VRAM Optimization**
- **Conservative Mode**: Light cache clearing for minimal disruption
- **Balanced Mode**: Standard optimization with peak stats reset
- **Aggressive Mode**: Multiple cleanup passes for maximum memory recovery
- **Cooldown System**: Prevents excessive optimization calls

## Integration Workflow

### Basic Setup

1. **Install Dependencies**:
```bash
pip install pynvml>=11.5.0
```

2. **Add Memory Nodes to Workflow**:
   - `EnhancedMemoryMonitorNode`: Primary monitoring
   - `VRAMOptimizerNode`: Specialized GPU optimization
   - `SmartMemoryManagerNode`: Automated management

### Recommended Workflow Structure

```
Input Image → EnhancedMemoryMonitor → VRAMOptimizer → ReActor Node → Output
                ↓
         SmartMemoryManager (Background)
```

### Node Configuration

#### EnhancedMemoryMonitorNode
```python
# Recommended settings for face swapping
{
    "enable_monitoring": True,
    "monitoring_mode": "Detailed",
    "ram_warning_threshold": 80.0,
    "vram_warning_threshold": 85.0,
    "auto_cleanup_enabled": False,  # Manual control preferred
    "cleanup_aggressive": False,
    "check_interval": 30.0,
    "enable_vram_optimization": True,
    "process_filter": "comfyui,python,reactor"
}
```

#### VRAMOptimizerNode
```python
# Optimized for face swapping workloads
{
    "enable_optimization": True,
    "optimization_mode": "Balanced",
    "vram_threshold": 85.0,
    "auto_optimize": True,
    "clear_cache": True,
    "reset_peak_stats": True,
    "synchronize_gpu": True
}
```

#### SmartMemoryManagerNode
```python
# Automated background management
{
    "enable_smart_management": True,
    "management_mode": "Balanced",
    "memory_warning_threshold": 75.0,
    "memory_critical_threshold": 85.0,
    "check_interval": 45.0,
    "enable_leak_detection": True,
    "enable_vram_optimization": True,
    "auto_cleanup_aggressive": False,
    "leak_detection_interval": 300.0
}
```

## Reactor-Specific Optimizations

### 1. **Batch Processing Optimization**
For large batch processing in Reactor:

```python
# Pre-batch optimization
VRAMOptimizerNode(
    optimization_mode="Aggressive",
    vram_threshold=80.0,
    auto_optimize=True
)

# Post-batch cleanup
MemoryCleanupNode(
    aggressive_cleanup=True,
    include_vram=True
)
```

### 2. **Video Processing Pipeline**
For video face swapping workflows:

```python
# Before video processing
EnhancedMemoryMonitorNode(
    monitoring_mode="Advanced",
    vram_warning_threshold=80.0,
    auto_cleanup_enabled=True
)

# During video processing (every N frames)
VRAMOptimizerNode(
    optimization_mode="Conservative",
    vram_threshold=85.0,
    auto_optimize=True
)
```

### 3. **Model Loading Optimization**
Before loading large face models:

```python
# Pre-model loading
VRAMOptimizerNode(
    optimization_mode="Aggressive",
    vram_threshold=70.0,
    auto_optimize=True
)

# Post-model loading
EnhancedMemoryMonitorNode(
    monitoring_mode="Detailed",
    vram_warning_threshold=90.0
)
```

## Memory Leak Prevention

### 1. **Regular Monitoring**
- Use `EnhancedMemoryMonitorNode` to track memory usage patterns
- Set appropriate warning thresholds (80% RAM, 85% VRAM)
- Monitor for unusual memory growth patterns

### 2. **Proactive Cleanup**
- Implement cleanup nodes at strategic points in your workflow
- Use `SmartMemoryManagerNode` for automated background management
- Set up periodic VRAM optimization for long-running processes

### 3. **Process Isolation**
- Use process filtering to focus on relevant applications
- Monitor specific GPU processes for memory leaks
- Implement cleanup triggers based on memory thresholds

## Performance Monitoring

### Key Metrics to Track

1. **VRAM Utilization**:
   - Target: <85% for optimal performance
   - Warning: >90% requires immediate attention
   - Critical: >95% may cause OOM errors

2. **RAM Usage**:
   - Target: <80% for stable operation
   - Warning: >85% may impact performance
   - Critical: >90% may cause system instability

3. **Process Memory**:
   - Monitor Reactor process memory growth
   - Check for memory leaks in face detection models
   - Track GPU process memory allocation

### Recommended Alerts

```python
# High VRAM usage
if vram_usage > 90:
    trigger_aggressive_cleanup()
    log_warning("Critical VRAM usage detected")

# Memory leak detection
if process_memory_growth > 50:  # 50% growth
    trigger_memory_cleanup()
    log_warning("Potential memory leak detected")

# GPU utilization mismatch
if gpu_utilization < 10 and memory_utilization > 70:
    log_warning("High memory usage with low GPU utilization")
```

## Troubleshooting

### Common Issues

1. **High VRAM Usage**:
   - Reduce batch size in Reactor node
   - Use `--lowvram` flag for ComfyUI
   - Implement aggressive VRAM optimization

2. **Memory Leaks**:
   - Enable leak detection in SmartMemoryManager
   - Use aggressive cleanup mode
   - Monitor process memory growth

3. **Container Memory Issues**:
   - Verify container memory limits
   - Use container-aware memory monitoring
   - Implement proper cleanup in containerized environments

### Debug Information

Enable detailed logging to track memory usage:

```python
import logging
logging.getLogger('memory_management').setLevel(logging.DEBUG)
```

## Best Practices

1. **Workflow Design**:
   - Place memory monitoring nodes early in the workflow
   - Use VRAM optimization before heavy operations
   - Implement cleanup nodes at logical breakpoints

2. **Threshold Management**:
   - Start with conservative thresholds
   - Adjust based on your specific hardware
   - Monitor and tune over time

3. **Automation**:
   - Use SmartMemoryManager for background monitoring
   - Implement automatic cleanup triggers
   - Set up periodic optimization schedules

4. **Monitoring**:
   - Track memory usage patterns over time
   - Identify memory leak sources
   - Optimize workflow based on usage data

## Advanced Configuration

### Custom Memory Thresholds
```python
# For high-end GPUs (24GB+)
vram_warning_threshold = 90.0
vram_critical_threshold = 95.0

# For mid-range GPUs (8-16GB)
vram_warning_threshold = 85.0
vram_critical_threshold = 90.0

# For low-end GPUs (<8GB)
vram_warning_threshold = 80.0
vram_critical_threshold = 85.0
```

### Process-Specific Monitoring
```python
# Focus on Reactor-specific processes
process_filter = "reactor,comfyui,python,face_swapping"

# Monitor specific GPU processes
gpu_process_monitoring = True
process_memory_tracking = True
```

This enhanced memory management system should significantly improve the stability and performance of your Reactor face swapping workflows while preventing the memory leak issues you've experienced. 