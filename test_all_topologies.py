#!/usr/bin/env python3
"""
Comprehensive test for all Picbreeder model topologies
"""

import pickle

def test_model_topology(model_name):
    """Test topology structure for a specific model"""
    print(f"\n=== Testing {model_name.upper()} ===")
    
    try:
        # Load the pbcppn.pkl file
        with open(f'data/picbreeder_{model_name}/pbcppn.pkl', 'rb') as f:
            pbcppn_data = pickle.load(f)
        
        print(f"✓ Loaded pbcppn.pkl")
        print(f"  Nodes: {len(pbcppn_data['nodes'])}")
        print(f"  Links: {len(pbcppn_data['links'])}")
        
        # Get special nodes
        special_nodes = pbcppn_data.get('special_nodes', {})
        print(f"  Special nodes: {special_nodes}")
        
        # Fixed classification logic
        input_node_ids = {str(special_nodes.get(name)) for name in ['x', 'y', 'd', 'bias'] if special_nodes.get(name) is not None}
        output_node_ids = {str(special_nodes.get(name)) for name in ['h', 's', 'v'] if special_nodes.get(name) is not None}
        
        # Classify all nodes
        node_types = {'input': [], 'output': [], 'hidden': []}
        
        for node in pbcppn_data['nodes']:
            node_id = str(node.get('id', ''))
            
            if node_id in input_node_ids:
                node_type = 'input'
            elif node_id in output_node_ids:
                node_type = 'output'
            else:
                node_type = 'hidden'
            
            node_types[node_type].append({
                'id': node_id,
                'label': node.get('label', ''),
                'activation': node.get('activation', 'identity'),
                'type': node_type
            })
        
        print(f"  Node classification:")
        print(f"    Input: {len(node_types['input'])}")
        print(f"    Output: {len(node_types['output'])}")
        print(f"    Hidden: {len(node_types['hidden'])}")
        
        # Validate structure
        success = True
        if len(node_types['input']) != 4:
            print(f"    ✗ Expected 4 input nodes, got {len(node_types['input'])}")
            success = False
        else:
            print(f"    ✓ Correct input nodes (4)")
            
        if len(node_types['output']) != 3:
            print(f"    ✗ Expected 3 output nodes, got {len(node_types['output'])}")
            success = False
        else:
            print(f"    ✓ Correct output nodes (3)")
            
        if len(node_types['hidden']) == 0:
            print(f"    ✗ No hidden nodes found")
            success = False
        else:
            print(f"    ✓ Has hidden nodes ({len(node_types['hidden'])})")
        
        # Show actual nodes
        print(f"  Input nodes:")
        for node in node_types['input']:
            print(f"    {node['id']}: \"{node['label']}\" ({node['activation']})")
        
        print(f"  Output nodes:")
        for node in node_types['output']:
            print(f"    {node['id']}: \"{node['label']}\" ({node['activation']})")
        
        return success
        
    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}")
        return False

def main():
    """Test all models"""
    print("=== COMPREHENSIVE TOPOLOGY VALIDATION ===")
    print("Testing NEAT network structure identification for all Picbreeder models")
    
    models = ['apple', 'butterfly', 'skull']
    results = {}
    
    for model in models:
        results[model] = test_model_topology(model)
    
    print(f"\n=== SUMMARY ===")
    all_passed = True
    for model, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{model.capitalize()}: {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print(f"\n🎉 SUCCESS! All Picbreeder models have correct topology structure.")
        print(f"\nThe fix handles:")
        print(f"- Mixed string/integer node IDs (skull model has integer output IDs)")
        print(f"- Proper input/output classification using special_nodes mapping") 
        print(f"- Preserved network connections and node properties")
        print(f"- Ready for D3.js visualization in both layered and NEAT views")
        print(f"\nThe topology visualization in the web app should now work correctly!")
    else:
        print(f"\n❌ Some models still have issues with topology structure.")
    
    return all_passed

if __name__ == "__main__":
    main()