"""
Unified Memory Control Node for ComfyUI

A comprehensive memory management node that combines monitoring, leak detection,
VRAM optimization, and cleanup with toggle switches for all features.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
import logging
import torch

try:
    from ..utils.memory_utils import (
        get_memory_info, cleanup_memory, optimize_vram, check_memory_pressure,
        format_bytes, MemoryTracker
    )
except ImportError:
    # Fallback for direct testing
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.memory_utils import (
        get_memory_info, cleanup_memory, optimize_vram, check_memory_pressure,
        format_bytes, MemoryTracker
    )

logger = logging.getLogger(__name__)

class UnifiedMemoryControlNode:
    """
    Unified Memory Control Node - All memory management features in one node
    """
    
    def __init__(self):
        self.tracker = MemoryTracker()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.leak_history = []
        self.optimization_history = []
        self.last_vram_warning = 0
        self.last_ram_warning = 0
        self.last_optimization = 0
        self.warning_cooldown = 30  # seconds
        self.optimization_cooldown = 60  # seconds
        self.max_snapshots = 100
        self.max_leak_history = 50
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Main Control
                "enable_monitoring": ("BOOLEAN", {"default": True}),
                "enable_leak_detection": ("BOOLEAN", {"default": True}),
                "enable_vram_optimization": ("BOOLEAN", {"default": True}),
                "enable_auto_cleanup": ("BOOLEAN", {"default": False}),
                
                # Monitoring Settings
                "monitoring_mode": (["Basic", "Detailed", "Advanced"], {
                    "default": "Detailed"
                }),
                "ram_warning_threshold": ("FLOAT", {
                    "default": 80.0,
                    "min": 60.0,
                    "max": 95.0,
                    "step": 5.0,
                    "display": "slider"
                }),
                "vram_warning_threshold": ("FLOAT", {
                    "default": 85.0,
                    "min": 70.0,
                    "max": 95.0,
                    "step": 5.0,
                    "display": "slider"
                }),
                
                # Leak Detection Settings
                "leak_threshold_mb": ("FLOAT", {
                    "default": 100.0,
                    "min": 10.0,
                    "max": 1000.0,
                    "step": 10.0,
                    "display": "slider"
                }),
                "auto_snapshot_interval": ("FLOAT", {
                    "default": 60.0,
                    "min": 10.0,
                    "max": 600.0,
                    "step": 10.0,
                    "display": "slider"
                }),
                "enable_auto_snapshots": ("BOOLEAN", {"default": False}),
                
                # VRAM Optimization Settings
                "optimization_mode": (["Conservative", "Balanced", "Aggressive"], {
                    "default": "Balanced"
                }),
                "vram_optimization_threshold": ("FLOAT", {
                    "default": 85.0,
                    "min": 70.0,
                    "max": 95.0,
                    "step": 5.0,
                    "display": "slider"
                }),
                "auto_optimize": ("BOOLEAN", {"default": False}),
                
                # Cleanup Settings
                "cleanup_aggressive": ("BOOLEAN", {"default": False}),
                "clear_cache": ("BOOLEAN", {"default": True}),
                "reset_peak_stats": ("BOOLEAN", {"default": True}),
                "synchronize_gpu": ("BOOLEAN", {"default": True}),
                
                # Action Control
                "action": (["Monitor Only", "Take Snapshot", "Detect Leaks", "Optimize VRAM", "Cleanup Memory", "Full Analysis"], {
                    "default": "Monitor Only"
                }),
                "snapshot_label": ("STRING", {
                    "default": "Snapshot",
                    "multiline": False
                }),
            },
            "optional": {
                "check_interval": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 300.0,
                    "step": 10.0,
                    "display": "slider"
                }),
                "process_filter": ("STRING", {
                    "default": "comfyui,python",
                    "multiline": False
                }),
                "trigger_image": ("IMAGE",),  # Trigger optimization on image input
            }
        }
    
    RETURN_TYPES = (
        "STRING",      # Status summary
        "STRING",      # Detailed report
        "BOOLEAN",     # Monitoring active
        "FLOAT",       # RAM usage percentage
        "FLOAT",       # VRAM usage percentage
        "STRING",      # Leak report
        "INT",         # Number of leaks detected
        "BOOLEAN",     # Leaks found
        "STRING",      # Optimization status
        "BOOLEAN",     # Optimization performed
        "FLOAT",       # VRAM freed (GB)
        "STRING",      # Recommendations
        "STRING",      # GPU process info
        "IMAGE",       # Pass through trigger image
    )
    
    RETURN_NAMES = (
        "status_summary",
        "detailed_report",
        "monitoring_active",
        "ram_usage_percent",
        "vram_usage_percent",
        "leak_report",
        "leak_count",
        "leaks_detected",
        "optimization_status",
        "optimization_performed",
        "vram_freed_gb",
        "recommendations",
        "gpu_processes",
        "trigger_image",
    )
    
    FUNCTION = "unified_memory_control"
    CATEGORY = "Memory Management"
    OUTPUT_NODE = True
    
    def unified_memory_control(self, enable_monitoring, enable_leak_detection, 
                              enable_vram_optimization, enable_auto_cleanup,
                              monitoring_mode, ram_warning_threshold, vram_warning_threshold,
                              leak_threshold_mb, auto_snapshot_interval, enable_auto_snapshots,
                              optimization_mode, vram_optimization_threshold, auto_optimize,
                              cleanup_aggressive, clear_cache, reset_peak_stats, synchronize_gpu,
                              action, snapshot_label="Snapshot", check_interval=30.0,
                              process_filter="comfyui,python", trigger_image=None):
        """Unified memory control with all features"""
        
        try:
            # Initialize return values
            status_summary = "Ready"
            detailed_report = ""
            monitoring_active = False
            ram_usage_percent = 0.0
            vram_usage_percent = 0.0
            leak_report = "No leak detection performed"
            leak_count = 0
            leaks_detected = False
            optimization_status = "No optimization performed"
            optimization_performed = False
            vram_freed_gb = 0.0
            recommendations = ""
            gpu_processes = ""
            
            # Get current memory info
            memory_info = get_memory_info()
            ram_percent = self._calculate_ram_usage(memory_info)
            vram_percent = self._calculate_vram_usage(memory_info)
            
            # Update tracker threshold
            self.tracker.leak_threshold = leak_threshold_mb * 1024 * 1024
            
            # Handle different actions
            if action == "Monitor Only":
                return self._handle_monitoring_only(enable_monitoring, monitoring_mode,
                                                  ram_warning_threshold, vram_warning_threshold,
                                                  enable_auto_cleanup, cleanup_aggressive,
                                                  check_interval, process_filter, memory_info,
                                                  ram_percent, vram_percent)
            
            elif action == "Take Snapshot":
                return self._handle_snapshot(snapshot_label, memory_info, ram_percent, vram_percent)
            
            elif action == "Detect Leaks":
                return self._handle_leak_detection(memory_info, ram_percent, vram_percent)
            
            elif action == "Optimize VRAM":
                return self._handle_vram_optimization(enable_vram_optimization, optimization_mode,
                                                    vram_optimization_threshold, auto_optimize,
                                                    clear_cache, reset_peak_stats, synchronize_gpu,
                                                    memory_info, ram_percent, vram_percent, trigger_image)
            
            elif action == "Cleanup Memory":
                return self._handle_memory_cleanup(enable_auto_cleanup, cleanup_aggressive,
                                                 memory_info, ram_percent, vram_percent)
            
            elif action == "Full Analysis":
                return self._handle_full_analysis(enable_monitoring, enable_leak_detection,
                                                enable_vram_optimization, enable_auto_cleanup,
                                                monitoring_mode, ram_warning_threshold, vram_warning_threshold,
                                                leak_threshold_mb, optimization_mode, vram_optimization_threshold,
                                                cleanup_aggressive, check_interval, process_filter,
                                                memory_info, ram_percent, vram_percent, trigger_image)
            
            else:
                return self._create_error_response("Invalid action specified")
                
        except Exception as e:
            logger.error(f"Error in unified memory control: {e}")
            return self._create_error_response(f"Error: {str(e)}")
    
    def _handle_monitoring_only(self, enable_monitoring, monitoring_mode,
                               ram_warning_threshold, vram_warning_threshold,
                               enable_auto_cleanup, cleanup_aggressive,
                               check_interval, process_filter, memory_info,
                               ram_percent, vram_percent):
        """Handle monitoring-only action"""
        
        # Check warnings
        ram_warning = ram_percent >= ram_warning_threshold
        vram_warning = vram_percent >= vram_warning_threshold
        
        # Perform auto-cleanup if enabled
        cleanup_performed = False
        if enable_auto_cleanup and (ram_warning or vram_warning):
            cleanup_performed = self._perform_auto_cleanup(cleanup_aggressive, True)
        
        # Create reports
        status_summary = self._create_status_summary(memory_info, ram_percent, vram_percent,
                                                   ram_warning, vram_warning, cleanup_performed)
        detailed_report = self._create_detailed_report(memory_info, monitoring_mode, process_filter)
        recommendations = self._generate_recommendations(memory_info, ram_warning, vram_warning, cleanup_performed)
        gpu_processes = self._get_gpu_process_info(memory_info, process_filter)
        
        return (status_summary, detailed_report, enable_monitoring, ram_percent, vram_percent,
                "No leak detection performed", 0, False, "No optimization performed", False, 0.0,
                recommendations, gpu_processes, None)
    
    def _handle_snapshot(self, snapshot_label, memory_info, ram_percent, vram_percent):
        """Handle snapshot action"""
        
        # Take snapshot
        snapshot = self.tracker.take_snapshot(snapshot_label)
        
        # Create reports
        status_summary = f"Snapshot '{snapshot_label}' taken successfully"
        detailed_report = self._create_snapshot_report(snapshot_label, memory_info, len(self.tracker.snapshots))
        leak_report = "Snapshot taken - use 'Detect Leaks' to analyze"
        
        return (status_summary, detailed_report, False, ram_percent, vram_percent,
                leak_report, 0, False, "No optimization performed", False, 0.0,
                "Snapshot saved for leak analysis", "", None)
    
    def _handle_leak_detection(self, memory_info, ram_percent, vram_percent):
        """Handle leak detection action"""
        
        # Detect leaks
        leaks = self.tracker.detect_leaks()
        
        # Create reports
        status_summary = f"Leak detection completed - {len(leaks)} potential leaks found"
        detailed_report = self._create_leak_report(leaks)
        leak_report = detailed_report
        leak_count = len(leaks)
        leaks_detected = len(leaks) > 0
        recommendations = self._generate_leak_recommendations(leaks)
        
        return (status_summary, detailed_report, False, ram_percent, vram_percent,
                leak_report, leak_count, leaks_detected, "No optimization performed", False, 0.0,
                recommendations, "", None)
    
    def _handle_vram_optimization(self, enable_vram_optimization, optimization_mode,
                                 vram_optimization_threshold, auto_optimize,
                                 clear_cache, reset_peak_stats, synchronize_gpu,
                                 memory_info, ram_percent, vram_percent, trigger_image):
        """Handle VRAM optimization action"""
        
        optimization_performed = False
        vram_freed_gb = 0.0
        optimization_status = "No optimization needed"
        
        if enable_vram_optimization and (auto_optimize or vram_percent >= vram_optimization_threshold):
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_optimization < self.optimization_cooldown:
                optimization_status = f"Optimization on cooldown ({self.optimization_cooldown - (current_time - self.last_optimization):.0f}s remaining)"
            else:
                # Perform optimization
                result = self._perform_vram_optimization(optimization_mode, clear_cache,
                                                       reset_peak_stats, synchronize_gpu)
                optimization_performed = result["success"]
                vram_freed_gb = result["vram_freed_gb"]
                optimization_status = result["status"]
                self.last_optimization = current_time
        
        # Create reports
        status_summary = optimization_status
        detailed_report = self._create_optimization_report(memory_info, vram_percent,
                                                         optimization_performed, vram_freed_gb)
        recommendations = self._generate_vram_recommendations(memory_info, vram_percent, optimization_performed)
        
        return (status_summary, detailed_report, False, ram_percent, vram_percent,
                "No leak detection performed", 0, False, optimization_status, optimization_performed, vram_freed_gb,
                recommendations, "", trigger_image)
    
    def _handle_memory_cleanup(self, enable_auto_cleanup, cleanup_aggressive,
                              memory_info, ram_percent, vram_percent):
        """Handle memory cleanup action"""
        
        cleanup_performed = False
        if enable_auto_cleanup:
            cleanup_performed = self._perform_auto_cleanup(cleanup_aggressive, True)
        
        # Create reports
        status_summary = f"Memory cleanup {'performed' if cleanup_performed else 'not needed'}"
        detailed_report = self._create_cleanup_report(memory_info, cleanup_performed)
        recommendations = self._generate_cleanup_recommendations(memory_info, cleanup_performed)
        
        return (status_summary, detailed_report, False, ram_percent, vram_percent,
                "No leak detection performed", 0, False, "No optimization performed", False, 0.0,
                recommendations, "", None)
    
    def _handle_full_analysis(self, enable_monitoring, enable_leak_detection,
                             enable_vram_optimization, enable_auto_cleanup,
                             monitoring_mode, ram_warning_threshold, vram_warning_threshold,
                             leak_threshold_mb, optimization_mode, vram_optimization_threshold,
                             cleanup_aggressive, check_interval, process_filter,
                             memory_info, ram_percent, vram_percent, trigger_image):
        """Handle full analysis action"""
        
        # Perform all operations
        ram_warning = ram_percent >= ram_warning_threshold
        vram_warning = vram_percent >= vram_warning_threshold
        
        # Cleanup if needed
        cleanup_performed = False
        if enable_auto_cleanup and (ram_warning or vram_warning):
            cleanup_performed = self._perform_auto_cleanup(cleanup_aggressive, True)
        
        # Leak detection
        leaks = []
        if enable_leak_detection:
            leaks = self.tracker.detect_leaks()
        
        # VRAM optimization
        optimization_performed = False
        vram_freed_gb = 0.0
        if enable_vram_optimization and (vram_percent >= vram_optimization_threshold):
            result = self._perform_vram_optimization(optimization_mode, True, True, True)
            optimization_performed = result["success"]
            vram_freed_gb = result["vram_freed_gb"]
        
        # Create comprehensive reports
        status_summary = self._create_comprehensive_status(memory_info, ram_percent, vram_percent,
                                                         ram_warning, vram_warning, cleanup_performed,
                                                         len(leaks), optimization_performed)
        detailed_report = self._create_comprehensive_report(memory_info, monitoring_mode, process_filter,
                                                          leaks, optimization_performed, vram_freed_gb)
        recommendations = self._generate_comprehensive_recommendations(memory_info, ram_warning, vram_warning,
                                                                     cleanup_performed, leaks, optimization_performed)
        gpu_processes = self._get_gpu_process_info(memory_info, process_filter)
        
        return (status_summary, detailed_report, enable_monitoring, ram_percent, vram_percent,
                self._create_leak_report(leaks), len(leaks), len(leaks) > 0,
                f"Optimization {'performed' if optimization_performed else 'not needed'}", 
                optimization_performed, vram_freed_gb, recommendations, gpu_processes, trigger_image)
    
    # Helper methods (implemented from existing nodes)
    def _calculate_ram_usage(self, memory_info: Dict) -> float:
        """Calculate RAM usage percentage"""
        if "ram" in memory_info and "total" in memory_info["ram"]:
            return (memory_info["ram"]["used"] / memory_info["ram"]["total"]) * 100
        return 0.0
    
    def _calculate_vram_usage(self, memory_info: Dict) -> float:
        """Calculate VRAM usage percentage"""
        if "vram" in memory_info and "total" in memory_info["vram"]:
            return (memory_info["vram"]["used"] / memory_info["vram"]["total"]) * 100
        return 0.0
    
    def _perform_auto_cleanup(self, aggressive: bool, optimize_vram: bool) -> bool:
        """Perform automatic memory cleanup"""
        try:
            cleanup_memory(aggressive=aggressive)
            if optimize_vram and torch.cuda.is_available():
                optimize_vram()
            return True
        except Exception as e:
            logger.error(f"Auto cleanup failed: {e}")
            return False
    
    def _perform_vram_optimization(self, mode: str, clear_cache: bool,
                                  reset_peak_stats: bool, synchronize_gpu: bool) -> Dict:
        """Perform VRAM optimization"""
        try:
            result = optimize_vram(mode=mode, clear_cache=clear_cache,
                                 reset_peak_stats=reset_peak_stats, synchronize_gpu=synchronize_gpu)
            return {
                "success": True,
                "status": f"VRAM optimization ({mode}) completed",
                "vram_freed_gb": result.get("vram_freed_gb", 0.0)
            }
        except Exception as e:
            logger.error(f"VRAM optimization failed: {e}")
            return {
                "success": False,
                "status": f"VRAM optimization failed: {str(e)}",
                "vram_freed_gb": 0.0
            }
    
    # Report creation methods (implemented from existing nodes)
    def _create_status_summary(self, memory_info: Dict, ram_percent: float, vram_percent: float,
                              ram_warning: bool, vram_warning: bool, cleanup_performed: bool) -> str:
        """Create status summary"""
        status = f"RAM: {ram_percent:.1f}% | VRAM: {vram_percent:.1f}%"
        if ram_warning or vram_warning:
            status += " ⚠️"
        if cleanup_performed:
            status += " | Cleanup performed"
        return status
    
    def _create_detailed_report(self, memory_info: Dict, mode: str, process_filter: str) -> str:
        """Create detailed memory report"""
        report = f"Memory Report ({mode} Mode)\n"
        report += f"RAM: {format_bytes(memory_info['ram']['used'])} / {format_bytes(memory_info['ram']['total'])}\n"
        if 'vram' in memory_info:
            report += f"VRAM: {format_bytes(memory_info['vram']['used'])} / {format_bytes(memory_info['vram']['total'])}\n"
        return report
    
    def _create_snapshot_report(self, label: str, memory_info: Dict, snapshot_count: int) -> str:
        """Create snapshot report"""
        return f"Snapshot '{label}' saved. Total snapshots: {snapshot_count}"
    
    def _create_leak_report(self, leaks: List) -> str:
        """Create leak detection report"""
        if not leaks:
            return "No memory leaks detected"
        
        report = f"Memory Leaks Detected: {len(leaks)}\n"
        for i, leak in enumerate(leaks[:5]):  # Show first 5 leaks
            report += f"{i+1}. {leak.get('description', 'Unknown leak')}\n"
        if len(leaks) > 5:
            report += f"... and {len(leaks) - 5} more leaks"
        return report
    
    def _create_optimization_report(self, memory_info: Dict, current_usage: float,
                                   optimization_performed: bool, vram_freed_gb: float) -> str:
        """Create VRAM optimization report"""
        report = f"VRAM Optimization Report\n"
        report += f"Current VRAM usage: {current_usage:.1f}%\n"
        if optimization_performed:
            report += f"VRAM freed: {vram_freed_gb:.2f} GB\n"
        else:
            report += "No optimization performed\n"
        return report
    
    def _create_cleanup_report(self, memory_info: Dict, cleanup_performed: bool) -> str:
        """Create memory cleanup report"""
        if cleanup_performed:
            return "Memory cleanup completed successfully"
        else:
            return "Memory cleanup not needed"
    
    def _create_comprehensive_status(self, memory_info: Dict, ram_percent: float, vram_percent: float,
                                   ram_warning: bool, vram_warning: bool, cleanup_performed: bool,
                                   leak_count: int, optimization_performed: bool) -> str:
        """Create comprehensive status summary"""
        status = f"RAM: {ram_percent:.1f}% | VRAM: {vram_percent:.1f}%"
        if ram_warning or vram_warning:
            status += " ⚠️"
        if cleanup_performed:
            status += " | Cleanup ✓"
        if leak_count > 0:
            status += f" | Leaks: {leak_count}"
        if optimization_performed:
            status += " | Optimized ✓"
        return status
    
    def _create_comprehensive_report(self, memory_info: Dict, mode: str, process_filter: str,
                                   leaks: List, optimization_performed: bool, vram_freed_gb: float) -> str:
        """Create comprehensive analysis report"""
        report = f"Comprehensive Memory Analysis ({mode} Mode)\n"
        report += f"RAM: {format_bytes(memory_info['ram']['used'])} / {format_bytes(memory_info['ram']['total'])}\n"
        if 'vram' in memory_info:
            report += f"VRAM: {format_bytes(memory_info['vram']['used'])} / {format_bytes(memory_info['vram']['total'])}\n"
        report += f"Memory leaks detected: {len(leaks)}\n"
        if optimization_performed:
            report += f"VRAM optimization performed: {vram_freed_gb:.2f} GB freed\n"
        return report
    
    def _get_gpu_process_info(self, memory_info: Dict, process_filter: str) -> str:
        """Get GPU process information"""
        if 'gpu_processes' in memory_info:
            processes = memory_info['gpu_processes']
            filtered = [p for p in processes if any(f in p.lower() for f in process_filter.split(','))]
            if filtered:
                return f"GPU Processes: {len(filtered)} found"
        return "No GPU process info available"
    
    # Recommendation generation methods
    def _generate_recommendations(self, memory_info: Dict, ram_warning: bool,
                                vram_warning: bool, cleanup_performed: bool) -> str:
        """Generate memory recommendations"""
        recommendations = []
        if ram_warning:
            recommendations.append("Consider closing unused applications")
        if vram_warning:
            recommendations.append("Consider VRAM optimization")
        if not cleanup_performed and (ram_warning or vram_warning):
            recommendations.append("Consider memory cleanup")
        return "; ".join(recommendations) if recommendations else "Memory usage is normal"
    
    def _generate_leak_recommendations(self, leaks: List) -> str:
        """Generate leak-specific recommendations"""
        if not leaks:
            return "No memory leaks detected"
        
        recommendations = []
        if len(leaks) > 10:
            recommendations.append("Multiple leaks detected - consider restarting ComfyUI")
        else:
            recommendations.append("Monitor for increasing memory usage")
        
        return "; ".join(recommendations)
    
    def _generate_vram_recommendations(self, memory_info: Dict, current_usage: float,
                                     optimization_performed: bool) -> str:
        """Generate VRAM-specific recommendations"""
        if current_usage > 90:
            return "VRAM usage critical - consider reducing batch size or model resolution"
        elif current_usage > 80:
            return "VRAM usage high - monitor for potential issues"
        elif optimization_performed:
            return "VRAM optimization completed successfully"
        else:
            return "VRAM usage is normal"
    
    def _generate_cleanup_recommendations(self, memory_info: Dict, cleanup_performed: bool) -> str:
        """Generate cleanup-specific recommendations"""
        if cleanup_performed:
            return "Memory cleanup completed - monitor for improved performance"
        else:
            return "Memory usage is normal - no cleanup needed"
    
    def _generate_comprehensive_recommendations(self, memory_info: Dict, ram_warning: bool,
                                              vram_warning: bool, cleanup_performed: bool,
                                              leaks: List, optimization_performed: bool) -> str:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        if ram_warning:
            recommendations.append("High RAM usage - consider cleanup")
        if vram_warning:
            recommendations.append("High VRAM usage - consider optimization")
        if leaks:
            recommendations.append(f"Memory leaks detected ({len(leaks)}) - monitor closely")
        if cleanup_performed:
            recommendations.append("Cleanup performed - should improve performance")
        if optimization_performed:
            recommendations.append("VRAM optimization completed")
        
        return "; ".join(recommendations) if recommendations else "All systems normal"
    
    def _create_error_response(self, error_message: str) -> Tuple:
        """Create error response"""
        return (f"Error: {error_message}", "Error occurred", False, 0.0, 0.0,
                "Error", 0, False, "Error", False, 0.0, "Check logs for details", "", None) 