#!/usr/bin/env python3
"""
Test script for the new batch generation endpoint
"""

import requests
import json
import time

def test_batch_endpoint():
    """Test the batch generation endpoint"""
    
    # Test data - simple weight changes for apple model
    test_frames = [
        {
            "picbreeder": {4140: -0.5},  # Apple size control
            "sgd": {135: -0.5}           # Approximate apple size control
        },
        {
            "picbreeder": {4140: 0.0},
            "sgd": {135: 0.0}
        },
        {
            "picbreeder": {4140: 0.5},
            "sgd": {135: 0.5}
        }
    ]
    
    print("Testing batch endpoint with 3 frames...")
    
    # Test the batch endpoint
    url = "http://localhost:8000/api/generate_batch/apple"
    payload = {
        "frames": test_frames,
        "img_size": 128  # Smaller size for faster testing
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        batch_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch request successful!")
            print(f"   Time: {batch_time:.2f} seconds")
            print(f"   Picbreeder frames: {len(data['picbreeder_frames'])}")
            print(f"   SGD frames: {len(data['sgd_frames'])}")
            
            # Test individual requests for comparison
            print("\nTesting individual requests for comparison...")
            individual_start = time.time()
            
            for i, frame in enumerate(test_frames):
                individual_url = "http://localhost:8000/api/generate_comparison/apple"
                individual_payload = {
                    "weight_deltas": {**frame["picbreeder"], **frame["sgd"]},
                    "img_size": 128
                }
                
                individual_response = requests.post(individual_url, json=individual_payload)
                if individual_response.status_code != 200:
                    print(f"❌ Individual request {i+1} failed")
                    return
            
            individual_time = time.time() - individual_start
            
            print(f"✅ Individual requests successful!")
            print(f"   Time: {individual_time:.2f} seconds")
            print(f"   Speedup: {individual_time/batch_time:.1f}x faster with batching")
            
        else:
            print(f"❌ Batch request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_batch_endpoint() 