#!/usr/bin/env python3
"""
Comprehensive test script for batch generation performance
"""

import requests
import json
import time
import numpy as np

def generate_test_frames(num_frames):
    """Generate test frame data for animation"""
    frames = []
    
    for i in range(num_frames):
        # Create a simple animation - apple size changing
        t = i / (num_frames - 1)  # 0 to 1
        apple_size_delta = -0.5 + t * 1.0  # -0.5 to 0.5
        
        frames.append({
            "picbreeder": {4140: apple_size_delta},  # Apple size control
            "sgd": {135: apple_size_delta}           # Approximate apple size control  
        })
    
    return frames

def test_batch_performance(num_frames, batch_size):
    """Test batch generation performance"""
    frames = generate_test_frames(num_frames)
    
    # Test batch approach
    print(f"Testing batch approach: {num_frames} frames, batch size {batch_size}")
    
    batch_start = time.time()
    try:
        # Process in batches
        all_picbreeder_frames = []
        all_sgd_frames = []
        
        for i in range(0, num_frames, batch_size):
            batch_frames = frames[i:i + batch_size]
            
            url = "http://localhost:8000/api/generate_batch/apple"
            payload = {
                "frames": batch_frames,
                "img_size": 128
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                all_picbreeder_frames.extend(data['picbreeder_frames'])
                all_sgd_frames.extend(data['sgd_frames'])
            else:
                print(f"❌ Batch request failed: {response.status_code}")
                return None, None
        
        batch_time = time.time() - batch_start
        print(f"   Batch time: {batch_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Batch error: {e}")
        return None, None
    
    # Test individual approach
    print(f"Testing individual approach: {num_frames} frames")
    
    individual_start = time.time()
    try:
        for frame in frames:
            url = "http://localhost:8000/api/generate_comparison/apple"
            payload = {
                "weight_deltas": {**frame["picbreeder"], **frame["sgd"]},
                "img_size": 128
            }
            
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                print(f"❌ Individual request failed: {response.status_code}")
                return None, None
        
        individual_time = time.time() - individual_start
        print(f"   Individual time: {individual_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Individual error: {e}")
        return None, None
    
    return batch_time, individual_time

def main():
    """Run comprehensive performance tests"""
    print("🚀 Comprehensive Batch Performance Test")
    print("=" * 50)
    
    # Test different scenarios
    test_scenarios = [
        (10, 5),    # 10 frames, batch size 5
        (20, 10),   # 20 frames, batch size 10  
        (30, 15),   # 30 frames, batch size 15
        (60, 20),   # 60 frames, batch size 20 (default)
        (60, 30),   # 60 frames, batch size 30
        (60, 60),   # 60 frames, batch size 60 (all at once)
    ]
    
    results = []
    
    for num_frames, batch_size in test_scenarios:
        print(f"\n📊 Test: {num_frames} frames, batch size {batch_size}")
        print("-" * 40)
        
        batch_time, individual_time = test_batch_performance(num_frames, batch_size)
        
        if batch_time and individual_time:
            speedup = individual_time / batch_time
            efficiency = "✅ Batch faster" if speedup > 1 else "⚠️  Individual faster"
            
            print(f"   Speedup: {speedup:.2f}x {efficiency}")
            
            results.append({
                'frames': num_frames,
                'batch_size': batch_size,
                'batch_time': batch_time,
                'individual_time': individual_time,
                'speedup': speedup
            })
        else:
            print("   ❌ Test failed")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 SUMMARY")
    print("=" * 50)
    
    for result in results:
        efficiency_icon = "✅" if result['speedup'] > 1 else "⚠️"
        print(f"{efficiency_icon} {result['frames']} frames (batch {result['batch_size']}): "
              f"{result['speedup']:.2f}x speedup")
    
    # Find optimal batch size
    if results:
        best_result = max(results, key=lambda x: x['speedup'])
        print(f"\n🏆 Best performance: {best_result['frames']} frames with batch size {best_result['batch_size']}")
        print(f"   Speedup: {best_result['speedup']:.2f}x")
        
        # Recommendation
        if best_result['speedup'] > 1:
            print(f"\n💡 Recommendation: Use batch size {best_result['batch_size']} for ~{best_result['frames']} frame animations")
        else:
            print(f"\n💡 Recommendation: Consider individual requests for small animations")

if __name__ == "__main__":
    main() 