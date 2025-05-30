#!/usr/bin/env python3
"""
Test script for the new progress and fullscreen features
"""

import requests
import json
import time

def test_batch_with_progress():
    """Test the batch endpoint and verify it works for progress tracking"""
    
    # Test data for a quick animation - just a few frames
    test_frames = []
    for i in range(5):  # Just 5 frames for quick testing
        t = i / 4  # 0 to 1
        apple_size_delta = -0.5 + t * 1.0  # -0.5 to 0.5
        
        test_frames.append({
            "picbreeder": {4140: apple_size_delta},  # Apple size control
            "sgd": {135: apple_size_delta}           # SGD equivalent
        })
    
    print("🧪 Testing batch endpoint for progress bar functionality...")
    print(f"   Generating {len(test_frames)} test frames")
    
    # Test the batch endpoint
    url = "http://localhost:8000/api/generate_batch/apple"
    payload = {
        "frames": test_frames,
        "img_size": 128  # Small size for speed
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
            print(f"   Ready for frontend progress bar testing!")
            
            return True
        else:
            print(f"❌ Batch request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_server_status():
    """Test if server is running and models are loaded"""
    try:
        response = requests.get("http://localhost:8000/api/models")
        if response.status_code == 200:
            data = response.json()
            print("🚀 Server is running!")
            print(f"   Models loaded: {len(data['models'])}")
            for model_name in data['models'].keys():
                print(f"   - {model_name}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not reachable on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing New Features")
    print("=" * 40)
    
    # Test 1: Server status
    print("\n1. Checking server status...")
    if not test_server_status():
        print("   Please start the server first: python app.py")
        exit(1)
    
    # Test 2: Batch endpoint for progress bars
    print("\n2. Testing batch generation for progress bars...")
    if test_batch_with_progress():
        print("   ✅ Backend ready for progress bar testing!")
    else:
        print("   ❌ Batch generation failed")
        exit(1)
    
    print("\n🎉 All tests passed!")
    print("\nNow test the frontend features:")
    print("✓ Visit http://localhost:8000")
    print("✓ Set keyframes and test progress bars during animation generation")
    print("✓ Test Space key behavior (start/restart)")
    print("✓ Test fullscreen buttons on images")
    print("✓ Test split fullscreen button")
    print("✓ Test Escape key to exit fullscreen") 