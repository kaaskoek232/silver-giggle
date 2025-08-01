# Quick Start: Unified Memory Manager for Reactor

## 🚀 One Node to Rule Them All

Instead of 7 different memory nodes, you now have **ONE** node that does everything:

### **UnifiedMemoryManager** 🧠

This single node replaces all the others with toggleable features:

## Basic Usage

### 1. **Just Monitoring** (No Optimization)
```python
UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=False,
    mode="Monitor Only"
)
```

### 2. **Auto-Optimization** (When Thresholds Hit)
```python
UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=True,
    auto_optimize=True,
    ram_threshold=80.0,
    vram_threshold=85.0,
    mode="Balanced"
)
```

### 3. **Aggressive Cleanup** (For Memory Leaks)
```python
UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=True,
    enable_auto_cleanup=True,
    mode="Aggressive"
)
```

## Reactor Workflow Integration

### **Before Face Swapping**
```python
# Pre-optimization
UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=True,
    mode="Conservative",
    auto_optimize=True
) → ReActor Node
```

### **After Face Swapping**
```python
# Post-cleanup
ReActor Node → UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=True,
    enable_auto_cleanup=True,
    mode="Balanced"
)
```

### **Video Processing Pipeline**
```python
# Before video processing
UnifiedMemoryManager(
    enable_monitoring=True,
    enable_optimization=True,
    mode="Aggressive",
    auto_optimize=True
) → VHS_LoadVideo → ReActor Node → VHS_VideoCombine
```

## Modes Explained

| Mode | Use Case | Impact |
|------|----------|---------|
| **Monitor Only** | Just watching memory | No performance impact |
| **Conservative** | Light cleanup | Minimal disruption |
| **Balanced** | Standard optimization | Moderate impact |
| **Aggressive** | Maximum cleanup | May pause briefly |

## Key Features

✅ **Container-Aware**: Works in Docker/Runpod environments  
✅ **VRAM/RAM Separation**: Monitors both separately  
✅ **Process Tracking**: Shows which processes use GPU memory  
✅ **Cooldown System**: Prevents spam optimization  
✅ **History Tracking**: Last 10 optimizations with timestamps  
✅ **Threshold Triggers**: Auto-optimize when memory hits limits  
✅ **Image Pass-through**: Can trigger on image input  

## Outputs

The node returns:
- **Status Summary**: "RAM: 8.5GB/16GB (53%) | VRAM: 12.3GB/24GB (51%)"
- **Detailed Report**: Full memory breakdown (optional)
- **RAM Usage %**: System memory percentage
- **VRAM Usage %**: GPU memory percentage  
- **Memory Freed GB**: Amount freed by optimization
- **Optimization Performed**: Boolean flag
- **Trigger Image**: Pass-through image (if provided)

## Troubleshooting

### **High VRAM Usage**
```python
UnifiedMemoryManager(
    enable_optimization=True,
    mode="Aggressive",
    vram_threshold=80.0,
    auto_optimize=True
)
```

### **Memory Leaks**
```python
UnifiedMemoryManager(
    enable_optimization=True,
    enable_auto_cleanup=True,
    mode="Aggressive",
    ram_threshold=70.0
)
```

### **Container Issues**
The node automatically detects container limits and uses them instead of host memory.

## That's It! 

No more juggling 7 different nodes. One node, all the features, toggleable as needed. 🎉 