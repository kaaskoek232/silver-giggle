# ComfyUI Memory Management - Deployment Guide

## 🚀 **Production Deployment Instructions**

This guide provides step-by-step instructions for deploying the ComfyUI Memory Management custom nodes in production environments.

## 📋 **Prerequisites**

### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 1.9.0 or higher  
- **ComfyUI**: Latest stable version
- **RAM**: Minimum 8GB (16GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Required Dependencies
- `psutil>=5.9.0` - System monitoring
- `torch>=1.9.0` - GPU operations

## 🔧 **Installation Methods**

### Method 1: ComfyUI Manager (Recommended)

1. **Install ComfyUI Manager** (if not already installed):
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   ```

2. **Use Manager to Install**:
   - Launch ComfyUI
   - Open ComfyUI Manager
   - Search for "Memory Management"
   - Click Install
   - Restart ComfyUI

### Method 2: Manual Git Installation

1. **Navigate to ComfyUI custom nodes directory**:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/ComfyUI/ComfyUI-MemoryManagement.git
   ```

3. **Install dependencies**:
   ```bash
   # For ComfyUI Portable (Windows)
   ../python_embeded/python.exe -m pip install -r ComfyUI-MemoryManagement/requirements.txt
   
   # For standard Python installation
   pip install -r ComfyUI-MemoryManagement/requirements.txt
   ```

4. **Restart ComfyUI**

### Method 3: Manual Download

1. **Download ZIP** from GitHub repository
2. **Extract** to `ComfyUI/custom_nodes/ComfyUI-MemoryManagement`
3. **Install dependencies** (see Method 2, step 3)
4. **Restart ComfyUI**

## ⚙️ **Configuration**

### Environment Variables (Optional)
```bash
# Set logging level
export COMFY_MEMORY_LOG_LEVEL=INFO

# Set memory thresholds
export COMFY_MEMORY_WARNING_THRESHOLD=75
export COMFY_MEMORY_CRITICAL_THRESHOLD=85
```

### ComfyUI Launch Parameters
For systems with limited VRAM, use these launch parameters:
```bash
# Low VRAM mode
python main.py --lowvram

# CPU fallback
python main.py --cpu

# Memory optimization
python main.py --reserve-vram 2 --disable-smart-memory
```

## 🔍 **Verification**

### Check Installation
1. **Launch ComfyUI**
2. **Look for memory management nodes** in the node menu under "Memory Management"
3. **Check console** for initialization messages:
   ```
   [ComfyUI-MemoryMgmt] INFO: ComfyUI integration enabled
   [ComfyUI-MemoryMgmt] INFO: Successfully loaded 7 memory management nodes
   ```

### Test Basic Functionality
```python
# Add a Memory Monitor node to your workflow
# Set refresh_trigger to 1
# Execute the workflow
# Check output for memory statistics
```

## 🏭 **Production Settings**

### Recommended Node Configurations

#### **Memory Monitor Node**
- `show_detailed`: True (for debugging)
- `show_gpu_info`: True (if GPU available)
- `memory_threshold_warning`: 75.0

#### **Smart Memory Manager**
- `management_mode`: "Balanced"
- `memory_warning_threshold`: 75.0
- `memory_critical_threshold`: 85.0
- `check_interval`: 45.0
- `enable_leak_detection`: True

#### **VRAM Optimizer**
- `optimization_level`: "Moderate" (production)
- `reset_peak_stats`: True
- `target_device`: -1 (all devices)

### Performance Optimization
```python
# For high-throughput environments
Smart_Memory_Manager_Settings = {
    "check_interval": 30.0,  # More frequent checks
    "management_mode": "Aggressive",  # Maximum optimization
    "enable_vram_optimization": True,
    "auto_cleanup_aggressive": True
}

# For stability-focused environments  
Conservative_Settings = {
    "check_interval": 60.0,  # Less frequent checks
    "management_mode": "Conservative",
    "enable_vram_optimization": False,
    "auto_cleanup_aggressive": False
}
```

## 🔒 **Security Considerations**

### File Permissions
```bash
# Ensure proper permissions
chmod 755 /path/to/ComfyUI/custom_nodes/ComfyUI-MemoryManagement
chmod 644 /path/to/ComfyUI/custom_nodes/ComfyUI-MemoryManagement/*.py
```

### Network Security
- Memory management nodes operate locally only
- No network connections or external data transmission
- All operations are confined to the ComfyUI process

## 📊 **Monitoring & Maintenance**

### Log Files
Check ComfyUI console output for memory management logs:
```
[ComfyUI-MemoryMgmt] INFO: Memory cleanup completed: 1543 objects collected
[ComfyUI-MemoryMgmt] WARNING: Critical memory usage: 87.3%
[ComfyUI-MemoryMgmt] INFO: VRAM optimization completed: 2.1GB freed
```

### Performance Metrics
Monitor these key metrics:
- **Memory Usage**: Should stay below 85%
- **Cleanup Frequency**: Should not be too frequent (indicates issues)
- **Leak Detection**: Should report minimal leaks
- **VRAM Utilization**: Optimal usage without overflow

### Maintenance Tasks
```bash
# Weekly: Check for updates
cd ComfyUI/custom_nodes/ComfyUI-MemoryManagement
git pull origin main

# Monthly: Verify dependencies
pip check

# As needed: Clear old snapshots (handled automatically)
```

## 🚨 **Troubleshooting**

### Common Issues

#### **Nodes Not Appearing**
```bash
# Check dependencies
pip install psutil torch

# Check console for errors
# Look for import errors or missing dependencies
```

#### **Memory Management Not Working** 
```bash
# Verify CUDA availability (for VRAM optimization)
python -c "import torch; print(torch.cuda.is_available())"

# Check system permissions
# Ensure ComfyUI has sufficient privileges
```

#### **Performance Issues**
```bash
# Reduce monitoring frequency
# Use Conservative mode instead of Aggressive
# Disable leak detection if not needed
```

### Error Codes
- `CUDA not available`: Install CUDA toolkit
- `Missing dependencies`: Run `pip install -r requirements.txt`
- `Permission denied`: Check file permissions
- `Memory monitoring error`: Check system resources

## 🔧 **Advanced Configuration**

### Custom Thresholds
```python
# In your workflow, configure Smart Memory Manager:
warning_threshold = 70.0  # Start cleanup at 70%
critical_threshold = 80.0  # Aggressive cleanup at 80%
check_interval = 20.0     # Check every 20 seconds
```

### Integration with Other Tools
```python
# For use with other monitoring tools
# Memory data can be accessed via node outputs
memory_data = memory_monitor_output
# Process with external monitoring systems
```

## 📞 **Support**

### Getting Help
1. **Check logs** in ComfyUI console
2. **Review this documentation**
3. **Search existing issues** on GitHub
4. **Create detailed bug report** with:
   - System specifications
   - ComfyUI version
   - Error messages
   - Reproduction steps

### Community Resources
- ComfyUI Discord: Memory Management channel
- GitHub Issues: Bug reports and feature requests
- ComfyUI Forum: Community discussions

---

## ✅ **Deployment Checklist**

- [ ] ComfyUI installed and working
- [ ] Dependencies installed (`psutil`, `torch`)
- [ ] Memory management nodes visible in ComfyUI
- [ ] Console shows successful initialization
- [ ] Basic memory monitoring working
- [ ] VRAM optimization functional (if GPU available)
- [ ] Smart memory manager configured
- [ ] Performance thresholds set appropriately
- [ ] Monitoring and logging configured
- [ ] Backup and update procedures established

**🎉 Your ComfyUI Memory Management system is now production-ready!** 