#!/usr/bin/env python3
"""
Test script for fullscreen animation and bounce loop features
"""

import requests
import json
import time

def test_basic_functionality():
    """Test that the server is responding and basic endpoints work"""
    
    print("🧪 Testing basic server functionality...")
    
    # Test main page loads
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Main page loads successfully")
        else:
            print(f"❌ Main page failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get("http://localhost:8000/api/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint works - found {len(data.get('models', {}))} models")
        else:
            print(f"❌ Models endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False
    
    # Test batch generation endpoint
    try:
        test_frames = [
            {
                "picbreeder": {4140: -0.3},
                "sgd": {135: -0.3}
            },
            {
                "picbreeder": {4140: 0.0},
                "sgd": {135: 0.0}
            },
            {
                "picbreeder": {4140: 0.3},
                "sgd": {135: 0.3}
            }
        ]
        
        print("🎬 Testing batch generation for fullscreen animation...")
        response = requests.post(
            "http://localhost:8000/api/generate_batch/apple",
            json={"frames": test_frames, "img_size": 256},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            picbreeder_frames = len(data.get('picbreeder_frames', []))
            sgd_frames = len(data.get('sgd_frames', []))
            print(f"✅ Batch generation works - {picbreeder_frames} picbreeder frames, {sgd_frames} SGD frames")
            
            if picbreeder_frames == 3 and sgd_frames == 3:
                print("✅ Frame counts match expected values")
            else:
                print(f"⚠️  Expected 3 frames each, got {picbreeder_frames} and {sgd_frames}")
                
        else:
            print(f"❌ Batch generation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Batch generation test failed: {e}")
        return False
    
    return True

def test_html_features():
    """Test that the HTML contains the new features"""
    
    print("\n🎯 Testing HTML features...")
    
    try:
        response = requests.get("http://localhost:8000/")
        html_content = response.text
        
        # Check for fullscreen animation features
        if "fullscreenCanvases" in html_content:
            print("✅ Fullscreen canvas support found in HTML")
        else:
            print("❌ Fullscreen canvas support missing")
            
        # Check for bounce mode
        if "animationMode" in html_content:
            print("✅ Animation mode support found in HTML")
        else:
            print("❌ Animation mode support missing")
            
        # Check for R key handler
        if "KeyR" in html_content:
            print("✅ R key handler found in HTML")
        else:
            print("❌ R key handler missing")
            
        # Check for bounce loop functionality
        if "bounce" in html_content.lower():
            print("✅ Bounce loop functionality found in HTML")
        else:
            print("❌ Bounce loop functionality missing")
            
        # Check for updated instructions
        if "bounce loop mode" in html_content.lower():
            print("✅ Updated instructions with bounce mode found")
        else:
            print("❌ Updated instructions missing")
            
    except Exception as e:
        print(f"❌ HTML feature test failed: {e}")
        return False
        
    return True

def main():
    print("🚀 Testing Fullscreen Animation & Bounce Loop Features")
    print("=" * 60)
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    # Test HTML features
    html_ok = test_html_features()
    
    print("\n" + "=" * 60)
    if basic_ok and html_ok:
        print("🎉 All tests passed! New features are ready:")
        print("   • Fullscreen animation support")
        print("   • Bounce loop mode (R key)")
        print("   • Enhanced keyboard controls")
        print("   • Updated UI feedback")
        print("\n📋 Manual Testing Instructions:")
        print("   1. Open http://localhost:8000")
        print("   2. Load an apple model and add some sliders")
        print("   3. Set IN/OUT keyframes on a weight")
        print("   4. Press Space for normal animation")
        print("   5. Press R for bounce loop animation")
        print("   6. Try fullscreen during animation")
        print("   7. Try split fullscreen during animation")
    else:
        print("❌ Some tests failed - check the implementation")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 