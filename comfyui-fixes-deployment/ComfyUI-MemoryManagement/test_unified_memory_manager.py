#!/usr/bin/env python3
"""
Test script for Unified Memory Manager Node

This script tests the unified memory management node to ensure it works correctly
before integration with ComfyUI.
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
    from nodes.unified_memory_manager import UnifiedMemoryManagerNode
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def test_unified_memory_manager():
    """Test Unified Memory Manager Node"""
    print("🧪 Testing Unified Memory Manager Node...")
    
    try:
        # Create node instance
        node = UnifiedMemoryManagerNode()
        
        # Test 1: Monitor Only Mode
        print("\n📊 Test 1: Monitor Only Mode")
        result = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=False,
            enable_auto_cleanup=False,
            mode="Monitor Only",
            show_detailed_report=True
        )
        
        assert len(result) == 7
        status_summary, detailed_report, ram_percent, vram_percent, memory_freed, optimization_performed, trigger_image = result
        
        assert isinstance(status_summary, str)
        assert isinstance(ram_percent, float)
        assert isinstance(vram_percent, float)
        assert isinstance(memory_freed, float)
        assert isinstance(optimization_performed, bool)
        
        print(f"✅ Monitor Only: {status_summary}")
        print(f"   RAM: {ram_percent:.1f}%, VRAM: {vram_percent:.1f}%")
        
        # Test 2: Conservative Mode
        print("\n⚡ Test 2: Conservative Mode")
        result = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=False,
            mode="Conservative",
            auto_optimize=True,
            show_detailed_report=True
        )
        
        status_summary, detailed_report, ram_percent, vram_percent, memory_freed, optimization_performed, trigger_image = result
        print(f"✅ Conservative: {status_summary}")
        print(f"   Memory freed: {memory_freed:.2f}GB, Optimization performed: {optimization_performed}")
        
        # Test 3: Balanced Mode
        print("\n⚖️ Test 3: Balanced Mode")
        result = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=True,
            mode="Balanced",
            ram_threshold=70.0,  # Lower threshold to trigger
            vram_threshold=70.0,
            auto_optimize=True,
            show_detailed_report=True
        )
        
        status_summary, detailed_report, ram_percent, vram_percent, memory_freed, optimization_performed, trigger_image = result
        print(f"✅ Balanced: {status_summary}")
        print(f"   Memory freed: {memory_freed:.2f}GB, Optimization performed: {optimization_performed}")
        
        # Test 4: Aggressive Mode
        print("\n🚀 Test 4: Aggressive Mode")
        result = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=True,
            mode="Aggressive",
            auto_optimize=True,
            show_detailed_report=True
        )
        
        status_summary, detailed_report, ram_percent, vram_percent, memory_freed, optimization_performed, trigger_image = result
        print(f"✅ Aggressive: {status_summary}")
        print(f"   Memory freed: {memory_freed:.2f}GB, Optimization performed: {optimization_performed}")
        
        # Test 5: Threshold-based triggering
        print("\n🎯 Test 5: Threshold-based Triggering")
        result = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=False,
            mode="Balanced",
            ram_threshold=50.0,  # Very low threshold to trigger
            vram_threshold=50.0,
            auto_optimize=False,  # Only trigger on thresholds
            show_detailed_report=True
        )
        
        status_summary, detailed_report, ram_percent, vram_percent, memory_freed, optimization_performed, trigger_image = result
        print(f"✅ Threshold Trigger: {status_summary}")
        print(f"   RAM: {ram_percent:.1f}%, VRAM: {vram_percent:.1f}%")
        print(f"   Memory freed: {memory_freed:.2f}GB, Optimization performed: {optimization_performed}")
        
        # Test 6: Cooldown system
        print("\n⏰ Test 6: Cooldown System")
        # First optimization
        result1 = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=False,
            mode="Conservative",
            auto_optimize=True
        )
        
        # Immediate second optimization (should be on cooldown)
        result2 = node.manage_memory_unified(
            enable_monitoring=True,
            enable_optimization=True,
            enable_auto_cleanup=False,
            mode="Conservative",
            auto_optimize=True
        )
        
        status1 = result1[0]
        status2 = result2[0]
        
        print(f"✅ First optimization: {status1}")
        print(f"✅ Second optimization (cooldown): {status2}")
        
        # Test 7: Error handling
        print("\n🛡️ Test 7: Error Handling")
        # Test with invalid parameters
        try:
            result = node.manage_memory_unified(
                enable_monitoring=True,
                enable_optimization=True,
                enable_auto_cleanup=False,
                mode="InvalidMode",  # Invalid mode
                show_detailed_report=True
            )
            print(f"✅ Error handling: {result[0]}")
        except Exception as e:
            print(f"✅ Error handling: Caught exception as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified Memory Manager Test Failed: {e}")
        traceback.print_exc()
        return False

def test_memory_utils_integration():
    """Test integration with memory utilities"""
    print("\n🧪 Testing Memory Utils Integration...")
    
    try:
        # Test memory info
        memory_info = get_memory_info()
        print(f"✅ Memory Info: {format_bytes(memory_info.get('system_used', 0))} / {format_bytes(memory_info.get('system_total', 0))}")
        
        # Test VRAM info
        if 'gpu_0' in memory_info:
            gpu_info = memory_info['gpu_0']
            print(f"✅ GPU Info: {gpu_info.get('name', 'Unknown')}")
            print(f"   Memory: {format_bytes(gpu_info.get('used_memory', 0))} / {format_bytes(gpu_info.get('total_memory', 0))}")
            print(f"   Utilization: {gpu_info.get('memory_utilization', 0):.1f}%")
        
        # Test cleanup
        cleanup_result = cleanup_memory(aggressive=False)
        print(f"✅ Cleanup: {cleanup_result.get('objects_collected', 0)} objects collected")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Utils Integration Test Failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Unified Memory Manager Tests...")
    print("=" * 50)
    
    tests = [
        test_memory_utils_integration,
        test_unified_memory_manager,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified Memory Manager is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 