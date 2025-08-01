#!/usr/bin/env python3
"""
Test script for ComfyUI Memory Management Custom Nodes

This script tests the basic functionality of each memory management node
to ensure they work correctly before installation.
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils.memory_utils import (
        get_memory_info, cleanup_memory, optimize_vram, 
        check_memory_pressure, format_bytes, MemoryTracker
    )
    from nodes.memory_monitor import MemoryMonitorNode
    from nodes.memory_cleanup import MemoryCleanupNode, AutoMemoryCleanupNode
    from nodes.vram_optimizer import VRAMOptimizerNode, VRAMUnloadNode
    from nodes.memory_leak_detector import MemoryLeakDetectorNode
    from nodes.smart_memory_manager import SmartMemoryManagerNode
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def test_memory_utils():
    """Test memory utility functions"""
    print("🧪 Testing Memory Utilities...")
    
    try:
        # Test memory info
        memory_info = get_memory_info()
        assert 'system_total' in memory_info
        assert 'system_used' in memory_info
        print(f"✅ Memory Info: {format_bytes(memory_info['system_used'])} / {format_bytes(memory_info['system_total'])}")
        
        # Test memory pressure check
        pressure_info = check_memory_pressure()
        assert 'under_pressure' in pressure_info
        print(f"✅ Memory Pressure: {pressure_info['status']}")
        
        # Test cleanup
        cleanup_result = cleanup_memory()
        print(f"✅ Memory Cleanup: {cleanup_result['objects_collected']} objects collected")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Utils Test Failed: {e}")
        return False

def test_memory_monitor_node():
    """Test Memory Monitor Node"""
    print("\n🧪 Testing Memory Monitor Node...")
    
    try:
        node = MemoryMonitorNode()
        result = node.monitor_memory(
            refresh_trigger=1,
            show_detailed=True,
            show_gpu_info=True,
            memory_threshold_warning=80.0
        )
        
        assert len(result) == 5
        summary, detailed_report, memory_percent, under_pressure, recommendation = result
        
        assert isinstance(summary, str)
        assert isinstance(memory_percent, float)
        assert isinstance(under_pressure, bool)
        
        print(f"✅ Memory Monitor: {summary}")
        return True
        
    except Exception as e:
        print(f"❌ Memory Monitor Test Failed: {e}")
        traceback.print_exc()
        return False

def test_memory_cleanup_node():
    """Test Memory Cleanup Node"""
    print("\n🧪 Testing Memory Cleanup Node...")
    
    try:
        node = MemoryCleanupNode()
        result = node.cleanup_memory(
            trigger=1,
            aggressive_cleanup=False,
            include_vram=True
        )
        
        assert len(result) == 4
        cleanup_report, objects_collected, memory_freed, success = result
        
        assert isinstance(cleanup_report, str)
        assert isinstance(objects_collected, int)
        assert isinstance(success, bool)
        
        print(f"✅ Memory Cleanup: {objects_collected} objects collected")
        return True
        
    except Exception as e:
        print(f"❌ Memory Cleanup Test Failed: {e}")
        traceback.print_exc()
        return False

def test_vram_optimizer_node():
    """Test VRAM Optimizer Node"""
    print("\n🧪 Testing VRAM Optimizer Node...")
    
    try:
        node = VRAMOptimizerNode()
        result = node.optimize_vram(
            trigger=1,
            optimization_level="Moderate",
            reset_peak_stats=True,
            target_device=-1
        )
        
        assert len(result) == 4
        optimization_report, memory_freed, success, recommendations = result
        
        assert isinstance(optimization_report, str)
        assert isinstance(memory_freed, int)
        assert isinstance(success, bool)
        
        print(f"✅ VRAM Optimizer: {format_bytes(memory_freed)} freed")
        return True
        
    except Exception as e:
        print(f"❌ VRAM Optimizer Test Failed: {e}")
        traceback.print_exc()
        return False

def test_leak_detector_node():
    """Test Memory Leak Detector Node"""
    print("\n🧪 Testing Memory Leak Detector Node...")
    
    try:
        node = MemoryLeakDetectorNode()
        
        # Start tracking
        result = node.detect_memory_leaks(
            action="Start Tracking",
            snapshot_label="Test_Start",
            leak_threshold_mb=50.0
        )
        
        assert len(result) == 5
        action_result, leak_report, leak_count, leaks_detected, recommendations = result
        
        print(f"✅ Leak Detector Start: {action_result}")
        
        # Take snapshot
        time.sleep(1)  # Brief delay
        result = node.detect_memory_leaks(
            action="Take Snapshot",
            snapshot_label="Test_Snapshot"
        )
        
        action_result = result[0]
        print(f"✅ Leak Detector Snapshot: {action_result}")
        
        # Stop tracking
        result = node.detect_memory_leaks(
            action="Stop Tracking",
            snapshot_label="Test_Stop"
        )
        
        action_result = result[0]
        print(f"✅ Leak Detector Stop: {action_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Leak Detector Test Failed: {e}")
        traceback.print_exc()
        return False

def test_smart_memory_manager_node():
    """Test Smart Memory Manager Node"""
    print("\n🧪 Testing Smart Memory Manager Node...")
    
    try:
        node = SmartMemoryManagerNode()
        result = node.manage_memory_smart(
            enable_smart_management=False,  # Don't actually start monitoring in test
            management_mode="Balanced",
            memory_warning_threshold=75.0,
            memory_critical_threshold=85.0,
            check_interval=45.0,
            enable_leak_detection=False,
            enable_vram_optimization=True,
            auto_cleanup_aggressive=False,
            leak_detection_interval=300.0
        )
        
        assert len(result) == 6
        management_status, detailed_report, management_active, current_usage, last_action, performance_stats = result
        
        assert isinstance(management_status, str)
        assert isinstance(management_active, bool)
        assert isinstance(current_usage, float)
        
        print(f"✅ Smart Memory Manager: {management_status}")
        return True
        
    except Exception as e:
        print(f"❌ Smart Memory Manager Test Failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 ComfyUI Memory Management Nodes - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Memory Utilities", test_memory_utils),
        ("Memory Monitor Node", test_memory_monitor_node),
        ("Memory Cleanup Node", test_memory_cleanup_node),
        ("VRAM Optimizer Node", test_vram_optimizer_node),
        ("Memory Leak Detector Node", test_leak_detector_node),
        ("Smart Memory Manager Node", test_smart_memory_manager_node),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - Unexpected Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The memory management nodes are ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 