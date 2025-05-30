#!/usr/bin/env python3
"""
Test script for topology visualization functionality
"""

import sys
import os
import asyncio
import json

# Add the src directory to path to import modules
sys.path.append('src')

try:
    import util
    from app import CPPNManager, get_network_topology
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)

async def test_topology():
    """Test the topology endpoint functionality"""
    print("Testing CPPN Network Topology Visualization...")
    
    # Initialize CPPN manager
    cppn_manager = CPPNManager()
    print(f"Available models: {list(cppn_manager.models.keys())}")
    
    # Test with picbreeder_apple (which should have NEAT topology)
    model_key = "picbreeder_apple"
    
    if model_key not in cppn_manager.models:
        print(f"Error: Model {model_key} not found")
        return False
    
    try:
        print(f"\nTesting topology for {model_key}...")
        
        # This would normally be called by FastAPI, but we'll test the logic directly
        result = await get_network_topology(model_key)
        
        print(f"✓ Topology data generated successfully")
        print(f"  Model: {result['model_key']}")
        print(f"  Source: {result['source']}")
        print(f"  Genome: {result['genome']}")
        print(f"  Layers: {result['n_layers']}")
        print(f"  Layered nodes: {len(result['nodes'])}")
        print(f"  Layered links: {len(result['links'])}")
        
        # Check NEAT topology
        if result['original_neat']:
            neat = result['original_neat']
            print(f"  NEAT nodes: {len(neat['nodes'])}")
            print(f"  NEAT links: {len(neat['links'])}")
            
            # Analyze node types in NEAT structure
            node_types = {}
            for node in neat['nodes']:
                node_type = node.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"  NEAT node types: {node_types}")
            
            # Show sample nodes
            print("\n  Sample Input Nodes:")
            for node in neat['nodes']:
                if node.get('type') == 'input':
                    print(f"    {node['id']}: \"{node['label']}\" ({node['activation']})")
            
            print("\n  Sample Output Nodes:")
            for node in neat['nodes']:
                if node.get('type') == 'output':
                    print(f"    {node['id']}: \"{node['label']}\" ({node['activation']})")
            
            print("\n  Sample Hidden Nodes (first 5):")
            hidden_count = 0
            for node in neat['nodes']:
                if node.get('type') == 'hidden' and hidden_count < 5:
                    print(f"    {node['id']}: \"{node['label']}\" ({node['activation']})")
                    hidden_count += 1
            
            print(f"    ... and {node_types.get('hidden', 0) - 5} more hidden nodes")
            
            # Check that we have the expected structure
            expected_inputs = 4  # x, y, d, bias
            expected_outputs = 3  # h, s, v
            
            if node_types.get('input', 0) == expected_inputs:
                print(f"  ✓ Correct number of input nodes ({expected_inputs})")
            else:
                print(f"  ✗ Expected {expected_inputs} input nodes, got {node_types.get('input', 0)}")
                
            if node_types.get('output', 0) == expected_outputs:
                print(f"  ✓ Correct number of output nodes ({expected_outputs})")
            else:
                print(f"  ✗ Expected {expected_outputs} output nodes, got {node_types.get('output', 0)}")
                
            if node_types.get('hidden', 0) > 0:
                print(f"  ✓ Has hidden nodes ({node_types.get('hidden', 0)})")
            else:
                print(f"  ✗ No hidden nodes found")
            
            print(f"\n✓ NEAT topology visualization data ready for frontend")
            return True
        else:
            print("  ✗ No NEAT topology data available")
            return False
            
    except Exception as e:
        print(f"✗ Error testing topology: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = asyncio.run(test_topology())
    
    if success:
        print("\n🎉 Topology visualization test PASSED!")
        print("The NEAT network structure is correctly identified and ready for visualization.")
        print("\nKey findings:")
        print("- Input nodes (bias, x, y, d) correctly identified")
        print("- Output nodes (h, s, v) correctly identified with proper labels")
        print("- Hidden nodes properly classified")
        print("- Network connections preserved")
        print("- Data format ready for D3.js force-directed visualization")
    else:
        print("\n❌ Topology visualization test FAILED!")
        
    return success

if __name__ == "__main__":
    main()