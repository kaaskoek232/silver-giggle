"""
ComfyUI Memory Management Custom Nodes Package

This package provides comprehensive memory management solutions for ComfyUI including:
- Memory leak detection and prevention
- RAM overflow management
- VRAM optimization
- Automatic cleanup utilities
- Memory monitoring and reporting

Compatible with ComfyUI custom node standards and requirements.

Author: AI Coding Assistant
Version: 1.2.0
License: MIT
"""

import logging
import sys
from typing import Dict, Any

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='[ComfyUI-MemoryMgmt] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Compatibility checks
def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import psutil
        logger.debug("psutil available")
    except ImportError:
        missing_deps.append("psutil>=5.9.0")
    
    try:
        import torch
        logger.debug("torch available")
    except ImportError:
        missing_deps.append("torch>=1.9.0")
    
    # Check for pynvml (optional but recommended)
    try:
        import pynvml
        logger.debug("pynvml available for enhanced GPU monitoring")
    except ImportError:
        logger.warning("pynvml not available - will use torch.cuda fallback")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

# Check ComfyUI availability
try:
    import comfy.model_management
    COMFY_INTEGRATION = True
    logger.info("ComfyUI integration enabled")
except ImportError:
    COMFY_INTEGRATION = False
    logger.warning("ComfyUI not detected - using standalone mode")

# Initialize nodes only if dependencies are met
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if check_dependencies():
    try:
        # Import our new unified node
        from .nodes.unified_memory_manager import UnifiedMemoryManagerNode
        
        # Import legacy nodes for backward compatibility
        try:
            from .nodes.memory_monitor import MemoryMonitorNode
            from .nodes.memory_cleanup import MemoryCleanupNode
            from .nodes.smart_memory_manager import SmartMemoryManagerNode
            from .nodes.enhanced_memory_monitor import EnhancedMemoryMonitorNode
            from .nodes.vram_optimizer import VRAMOptimizerNode
            
            # Node mappings for ComfyUI
            NODE_CLASS_MAPPINGS = {
                # Primary unified node
                "UnifiedMemoryManager": UnifiedMemoryManagerNode,
                
                # Legacy nodes for backward compatibility
                "MemoryMonitor": MemoryMonitorNode,
                "MemoryCleanup": MemoryCleanupNode,
                "SmartMemoryManager": SmartMemoryManagerNode,
                "EnhancedMemoryMonitor": EnhancedMemoryMonitorNode,
                "VRAMOptimizer": VRAMOptimizerNode,
            }

            # Display names for the nodes in ComfyUI interface
            NODE_DISPLAY_NAME_MAPPINGS = {
                # Primary unified node
                "UnifiedMemoryManager": "Unified Memory Manager 🧠",
                
                # Legacy nodes
                "MemoryMonitor": "Memory Monitor 📊",
                "MemoryCleanup": "Memory Cleanup 🧹",
                "SmartMemoryManager": "Smart Memory Manager 🔄",
                "EnhancedMemoryMonitor": "Enhanced Memory Monitor 📈",
                "VRAMOptimizer": "VRAM Optimizer ⚡",
            }
            
            logger.info(f"Successfully loaded {len(NODE_CLASS_MAPPINGS)} memory management nodes")
            logger.info("Primary node: UnifiedMemoryManager - All-in-one memory management")
            
        except ImportError as e:
            logger.warning(f"Some legacy nodes not available: {e}")
            # Fallback to just the unified node
            NODE_CLASS_MAPPINGS = {
                "UnifiedMemoryManager": UnifiedMemoryManagerNode,
            }
            NODE_DISPLAY_NAME_MAPPINGS = {
                "UnifiedMemoryManager": "Unified Memory Manager 🧠",
            }
            logger.info("Loaded UnifiedMemoryManager node only")
        
    except Exception as e:
        logger.error(f"Failed to load memory management nodes: {e}")
        # Provide empty mappings to prevent ComfyUI errors
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    logger.error("Dependencies not met - nodes will not be available")

# Package metadata
__version__ = "1.2.0"
__author__ = "AI Coding Assistant"
__license__ = "MIT"
__description__ = "Advanced memory management custom nodes for ComfyUI"

# Export required ComfyUI variables
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "__author__",
    "__license__",
    "__description__"
]

# Package initialization complete
logger.info("ComfyUI Memory Management package initialized successfully") 