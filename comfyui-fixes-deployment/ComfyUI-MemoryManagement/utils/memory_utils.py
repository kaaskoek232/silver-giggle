"""
Memory Management Utilities for ComfyUI Custom Nodes

This module provides utility functions for memory monitoring, cleanup, and optimization.
"""

import gc
import os
import psutil
import torch
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import weakref
from collections import defaultdict
import tracemalloc

# Configure logging for ComfyUI
logger = logging.getLogger(__name__)

# ComfyUI-specific imports with fallback
try:
    import comfy.model_management
    COMFY_AVAILABLE = True
    logger.info("ComfyUI model management available")
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("ComfyUI model management not available - using fallback methods")

class MemoryTracker:
    """Advanced memory tracking and leak detection utility"""
    
    def __init__(self):
        self.snapshots = []
        self.baseline_memory = None
        self.tracked_objects = weakref.WeakSet()
        self.memory_history = []
        self.leak_threshold = 100 * 1024 * 1024  # 100MB threshold
        self.max_snapshots = 100  # Prevent memory leaks in the detector itself
        
    def start_tracking(self):
        """Start memory tracking"""
        try:
            tracemalloc.start()
            self.baseline_memory = get_memory_info()
            logger.info("Memory tracking started successfully")
        except Exception as e:
            logger.error(f"Failed to start memory tracking: {e}")
            raise
        
    def stop_tracking(self):
        """Stop memory tracking"""
        try:
            tracemalloc.stop()
            logger.info("Memory tracking stopped")
        except Exception as e:
            logger.warning(f"Error stopping memory tracking: {e}")
        
    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot with automatic cleanup"""
        if not tracemalloc.is_tracing():
            logger.warning("Cannot take snapshot - tracking not active")
            return None
            
        try:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot, time.time()))
            
            # Automatic cleanup to prevent memory leaks
            if len(self.snapshots) > self.max_snapshots:
                snapshots_to_remove = len(self.snapshots) - self.max_snapshots
                removed = self.snapshots[:snapshots_to_remove]
                self.snapshots = self.snapshots[snapshots_to_remove:]
                logger.info(f"Cleaned up {len(removed)} old snapshots")
            
            logger.debug(f"Memory snapshot '{label}' taken")
            return snapshot
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return None
        
    def detect_leaks(self) -> List[Dict]:
        """Detect potential memory leaks"""
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots to detect leaks")
            return []
            
        try:
            current = self.snapshots[-1][1]
            previous = self.snapshots[-2][1]
            
            top_stats = current.compare_to(previous, 'lineno')
            leaks = []
            
            for stat in top_stats[:10]:  # Top 10 differences
                if stat.size_diff > self.leak_threshold:
                    leak_info = {
                        'filename': stat.traceback.format()[0],
                        'size_diff': stat.size_diff,
                        'count_diff': stat.count_diff,
                        'size_diff_mb': stat.size_diff / (1024 * 1024)
                    }
                    leaks.append(leak_info)
                    logger.warning(f"Potential memory leak detected: {leak_info['filename']} (+{leak_info['size_diff_mb']:.1f}MB)")
                    
            return leaks
        except Exception as e:
            logger.error(f"Error detecting memory leaks: {e}")
            return []

def get_memory_info() -> Dict[str, Union[int, float]]:
    """Get comprehensive memory information with container/pod awareness and enhanced GPU monitoring"""
    try:
        # Check for container memory limits first (Docker/Runpod)
        container_memory_limit = None
        container_memory_usage = None
        
        # Method 1: Check cgroup memory limits (Docker/Kubernetes)
        try:
            if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    limit = int(f.read().strip())
                    # Check if it's a real limit (not the massive default)
                    if limit < (1 << 62):  # Less than ~4 exabytes (realistic container limit)
                        container_memory_limit = limit
                        
            if os.path.exists('/sys/fs/cgroup/memory/memory.usage_in_bytes'):
                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                    container_memory_usage = int(f.read().strip())
        except Exception as e:
            logger.debug(f"Could not read cgroup memory info: {e}")
            
        # Method 2: Check cgroup v2 (newer systems)
        if container_memory_limit is None:
            try:
                if os.path.exists('/sys/fs/cgroup/memory.max'):
                    with open('/sys/fs/cgroup/memory.max', 'r') as f:
                        limit_str = f.read().strip()
                        if limit_str != 'max':
                            container_memory_limit = int(limit_str)
                            
                if os.path.exists('/sys/fs/cgroup/memory.current'):
                    with open('/sys/fs/cgroup/memory.current', 'r') as f:
                        container_memory_usage = int(f.read().strip())
            except Exception as e:
                logger.debug(f"Could not read cgroup v2 memory info: {e}")
        
        # Get system memory as fallback
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Use container limits if available, otherwise fall back to system
        if container_memory_limit is not None:
            total_memory = container_memory_limit
            used_memory = container_memory_usage if container_memory_usage else process_memory.rss
            available_memory = total_memory - used_memory
            memory_percent = (used_memory / total_memory) * 100
            logger.info(f"Using container memory limits: {format_bytes(total_memory)} total")
        else:
            total_memory = memory.total
            used_memory = memory.used
            available_memory = memory.available
            memory_percent = memory.percent
            logger.info(f"Using system memory info: {format_bytes(total_memory)} total")
        
        info = {
            'system_total': total_memory,
            'system_available': available_memory,
            'system_used': used_memory,
            'system_percent': memory_percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent,
            'process_rss': process_memory.rss,
            'process_vms': process_memory.vms,
            'process_percent': process.memory_percent(),
            'container_aware': container_memory_limit is not None
        }
        
        # Enhanced GPU memory monitoring with pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get device name
                    device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Get running processes on this GPU
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        process_info = [
                            {'pid': proc.pid, 'memory': proc.usedGpuMemory} 
                            for proc in processes
                        ]
                    except pynvml.NVMLError:
                        process_info = []
                    
                    # Get utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        memory_util = util.memory
                    except pynvml.NVMLError:
                        gpu_util = 0
                        memory_util = 0
                    
                    info[f'gpu_{i}'] = {
                        'name': device_name,
                        'total_memory': mem_info.total,
                        'free_memory': mem_info.free,
                        'used_memory': mem_info.used,
                        'memory_utilization': (mem_info.used / mem_info.total) * 100,
                        'gpu_utilization': gpu_util,
                        'processes': process_info
                    }
                    
                    logger.debug(f"GPU {i} ({device_name}): {format_bytes(mem_info.used)}/{format_bytes(mem_info.total)} ({info[f'gpu_{i}']['memory_utilization']:.1f}%)")
                    
                except Exception as e:
                    logger.warning(f"Error getting detailed info for GPU {i}: {e}")
                    
        except ImportError:
            logger.warning("pynvml not available, falling back to torch.cuda")
            # Fallback to torch.cuda if pynvml not available
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        info[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i)
                        info[f'gpu_{i}_reserved'] = torch.cuda.memory_reserved(i)
                        info[f'gpu_{i}_max_allocated'] = torch.cuda.max_memory_allocated(i)
                        info[f'gpu_{i}_max_reserved'] = torch.cuda.max_memory_reserved(i)
                except Exception as e:
                    logger.warning(f"Error getting GPU memory info: {e}")
        except Exception as e:
            logger.warning(f"Error initializing pynvml: {e}")
            # Fallback to torch.cuda
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        info[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i)
                        info[f'gpu_{i}_reserved'] = torch.cuda.memory_reserved(i)
                        info[f'gpu_{i}_max_allocated'] = torch.cuda.max_memory_allocated(i)
                        info[f'gpu_{i}_max_reserved'] = torch.cuda.max_memory_reserved(i)
                except Exception as e:
                    logger.warning(f"Error getting GPU memory info: {e}")
                
        return info
        
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {'error': str(e)}

def cleanup_memory(aggressive: bool = False) -> Dict[str, int]:
    """Perform memory cleanup operations with ComfyUI integration"""
    try:
        initial_memory = get_memory_info()
        
        # ComfyUI-specific cleanup
        if COMFY_AVAILABLE:
            try:
                if aggressive:
                    logger.info("Performing aggressive ComfyUI cleanup")
                    comfy.model_management.unload_all_models()
                    comfy.model_management.soft_empty_cache()
                else:
                    logger.debug("Performing standard ComfyUI cleanup")
                    comfy.model_management.soft_empty_cache()
            except Exception as e:
                logger.warning(f"ComfyUI cleanup error: {e}")
        
        # Standard cleanup
        collected = gc.collect()
        
        if aggressive:
            # Force garbage collection multiple times
            for i in range(3):
                collected += gc.collect()
                logger.debug(f"Aggressive GC pass {i+1}: {gc.collect()} objects")
                
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("PyTorch CUDA cache cleared")
                except Exception as e:
                    logger.warning(f"CUDA cache clear error: {e}")
                
        final_memory = get_memory_info()
        
        result = {
            'objects_collected': collected,
            'memory_freed': initial_memory.get('process_rss', 0) - final_memory.get('process_rss', 0),
            'initial_rss': initial_memory.get('process_rss', 0),
            'final_rss': final_memory.get('process_rss', 0)
        }
        
        logger.info(f"Memory cleanup completed: {collected} objects collected, {result['memory_freed']} bytes freed")
        return result
        
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return {'error': str(e), 'objects_collected': 0, 'memory_freed': 0}

def optimize_vram() -> Dict[str, Union[str, int]]:
    """Optimize VRAM usage with enhanced error handling"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available for VRAM optimization")
        return {'status': 'CUDA not available'}
    
    try:
        result = {}
        
        for device_id in range(torch.cuda.device_count()):
            try:
                initial_allocated = torch.cuda.memory_allocated(device_id)
                initial_reserved = torch.cuda.memory_reserved(device_id)
                
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(device_id)
                
                final_allocated = torch.cuda.memory_allocated(device_id)
                final_reserved = torch.cuda.memory_reserved(device_id)
                
                device_result = {
                    'initial_allocated': initial_allocated,
                    'final_allocated': final_allocated,
                    'initial_reserved': initial_reserved,
                    'final_reserved': final_reserved,
                    'freed_allocated': initial_allocated - final_allocated,
                    'freed_reserved': initial_reserved - final_reserved
                }
                
                result[f'device_{device_id}'] = device_result
                logger.debug(f"VRAM optimized for device {device_id}: {device_result['freed_allocated']} bytes freed")
                
            except Exception as e:
                logger.error(f"VRAM optimization failed for device {device_id}: {e}")
                result[f'device_{device_id}'] = {'error': str(e)}
        
        return result
        
    except Exception as e:
        logger.error(f"VRAM optimization failed: {e}")
        return {'error': str(e)}

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def check_memory_pressure() -> Dict[str, Union[bool, float, str]]:
    """Check if system is under memory pressure"""
    memory = psutil.virtual_memory()
    
    # Define thresholds
    warning_threshold = 80.0  # 80% usage
    critical_threshold = 90.0  # 90% usage
    
    status = 'normal'
    if memory.percent >= critical_threshold:
        status = 'critical'
    elif memory.percent >= warning_threshold:
        status = 'warning'
        
    return {
        'under_pressure': memory.percent >= warning_threshold,
        'status': status,
        'memory_percent': memory.percent,
        'available_gb': memory.available / (1024**3),
        'recommendation': get_memory_recommendation(memory.percent)
    }

def get_memory_recommendation(memory_percent: float) -> str:
    """Get memory usage recommendations"""
    if memory_percent < 70:
        return "Memory usage is normal"
    elif memory_percent < 80:
        return "Consider closing unused applications"
    elif memory_percent < 90:
        return "High memory usage - cleanup recommended"
    else:
        return "Critical memory usage - immediate cleanup required"

class AutoMemoryManager:
    """Automatic memory management with configurable thresholds"""
    
    def __init__(self, 
                 warning_threshold: float = 80.0,
                 critical_threshold: float = 90.0,  
                 check_interval: float = 30.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.callbacks = defaultdict(list)
        self._lock = threading.Lock()  # Thread safety lock
        
        # Validation
        if warning_threshold >= critical_threshold:
            raise ValueError("Warning threshold must be less than critical threshold")
        if check_interval <= 0:
            raise ValueError("Check interval must be positive")
            
        logger.info(f"AutoMemoryManager initialized: warning={warning_threshold}%, critical={critical_threshold}%")
        
    def add_callback(self, event_type: str, callback):
        """Add callback for memory events"""
        with self._lock:
            self.callbacks[event_type].append(callback)
            logger.debug(f"Callback added for event type: {event_type}")
        
    def start_monitoring(self):
        """Start automatic memory monitoring"""
        with self._lock:
            if self.running:
                logger.warning("Memory monitoring already running")
                return
                
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop automatic memory monitoring"""
        with self._lock:
            if not self.running:
                return
            self.running = False
            thread_to_join = self.thread
            
        # Join thread outside lock to avoid deadlock
        if thread_to_join:
            thread_to_join.join(timeout=5.0)
            if thread_to_join.is_alive():
                logger.warning("Memory monitoring thread did not stop cleanly")
            else:
                logger.info("Memory monitoring stopped")
            
    def _monitor_loop(self):
        """Main monitoring loop with enhanced error handling"""
        logger.info("Memory monitoring loop started")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                memory_info = get_memory_info()
                
                if 'error' in memory_info:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, stopping monitoring")
                        break
                    time.sleep(self.check_interval)
                    continue
                
                consecutive_errors = 0  # Reset error counter
                memory_percent = memory_info.get('system_percent', 0)
                
                if memory_percent >= self.critical_threshold:
                    logger.warning(f"Critical memory usage: {memory_percent:.1f}%")
                    self._trigger_callbacks('critical', memory_info)
                    cleanup_memory(aggressive=True)
                elif memory_percent >= self.warning_threshold:
                    logger.info(f"Warning memory usage: {memory_percent:.1f}%")
                    self._trigger_callbacks('warning', memory_info)
                    cleanup_memory(aggressive=False)
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Memory monitoring error: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors, stopping monitoring")
                    break
                time.sleep(self.check_interval)
                
    def _trigger_callbacks(self, event_type: str, memory_info: Dict):
        """Trigger registered callbacks with error handling"""
        callbacks_to_call = []
        with self._lock:
            callbacks_to_call = self.callbacks[event_type].copy()
            
        for callback in callbacks_to_call:
            try:
                callback(memory_info)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")

# Global memory manager instance with proper initialization
try:
    global_memory_manager = AutoMemoryManager()
    logger.info("Global memory manager initialized")
except Exception as e:
    logger.error(f"Failed to initialize global memory manager: {e}")
    global_memory_manager = None 