#!/usr/bin/env python3
"""
FastAPI Backend for FER Paper - CPPN Weight Sweep Web App
Serves the Picbreeder and SGD CPPN models with real-time weight manipulation
"""

import os
import sys
from functools import partial
import numpy as np
import base64
import io
from PIL import Image
from typing import List, Dict, Optional, Any
import asyncio

# Add the src directory to path to import modules
sys.path.append('src')

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import jax
    import jax.numpy as jnp
    from einops import repeat
    from cppn import CPPN, FlattenCPPNParameters
    import util
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you have all dependencies installed.")
    sys.exit(1)

app = FastAPI(title="FER CPPN Explorer", description="Interactive exploration of Fractured vs Unified representations")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CPPNManager:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load both Picbreeder and SGD models"""
        for source in ["picbreeder", "sgd"]:
            for genome in ["apple", "skull", "butterfly"]:
                try:
                    save_dir = f"data/{source}_{genome}"
                    if os.path.exists(save_dir):
                        arch = util.load_pkl(save_dir, "arch")
                        params = util.load_pkl(save_dir, "params")
                        cppn = FlattenCPPNParameters(CPPN(arch))
                        
                        # Get default weight suggestions (but not mandatory)
                        weight_suggestions = self.get_weight_suggestions(source, genome)
                        
                        self.models[f"{source}_{genome}"] = {
                            "arch": arch,
                            "params": params,
                            "cppn": cppn,
                            "weight_suggestions": weight_suggestions,
                            "n_params": len(params)
                        }
                        print(f"Loaded {source} {genome} model ({len(params)} parameters)")
                except Exception as e:
                    print(f"Warning: Could not load {source} {genome}: {e}")
    
    def get_weight_suggestions(self, source: str, genome: str) -> Dict:
        """Get suggested interesting weights for a model (for reference, not mandatory)"""
        weight_data = {
            ("picbreeder", "apple"): {
                "weight_ids": [42178, 4140, 34459, 17131],
                "descriptions": ["Controls Stem Angle", "Controls Apple Size", "Cleans Background", "Removes Stem"],
            },
            ("picbreeder", "skull"): {
                "weight_ids": [4371, 5009, 5097, 37],
                "descriptions": ["Controls Mouth Opening", "Controls Eye Winking", "Controls Eye Width", "Controls Jaw Width"],
            },
            ("picbreeder", "butterfly"): {
                "weight_ids": [1949, 3702, 17, 133],
                "descriptions": ["Controls Wing Area", "Controls Color", "Converts Butterfly to Fly", "Controls Vertical Shape"],
            }
        }
        
        return weight_data.get((source, genome), {
            "weight_ids": [],
            "descriptions": [],
        })
    
    def generate_image(self, model_key: str, weight_deltas: Dict[int, float], img_size: int = 256) -> np.ndarray:
        """Generate image with modified weights"""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        model = self.models[model_key]
        params = model["params"].copy()
        
        # Apply weight deltas
        for weight_id, delta in weight_deltas.items():
            if 0 <= weight_id < len(params):
                params = params.at[weight_id].set(params[weight_id] + delta)
        
        # Generate image
        img = model["cppn"].generate_image(params, img_size=img_size)
        return np.array(img)
    
    def generate_feature_maps(self, model_key: str, img_size: int = 128) -> List[np.ndarray]:
        """Generate feature maps for all layers of a model"""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        model = self.models[model_key]
        params = model["params"]
        
        # Generate image with features
        img, features = model["cppn"].generate_image(params, img_size=img_size, return_features=True)
        
        # Convert JAX arrays to numpy and process features
        features = [np.array(f) for f in features]
        
        # The last feature is the final output (h,s,v), we don't need it for weight selection
        # The first feature is the input (x,y,d,b), also not useful for weight selection
        # Return intermediate features that correspond to actual network weights
        intermediate_features = features[1:-1]  # Skip input and output
        
        return intermediate_features, self._get_weight_mapping(model_key)
    
    def _get_weight_mapping(self, model_key: str) -> Dict:
        """
        Get mapping between neuron positions and weight parameter indices.
        This helps map feature map clicks to actual weight parameters.
        """
        if model_key not in self.models:
            return {}
        
        model = self.models[model_key]
        cppn = model["cppn"]
        
        # Parse architecture to understand weight structure
        arch_str = model["arch"]
        try:
            n_layers_str, activation_neurons_str = arch_str.split(";")
            n_layers = int(n_layers_str)
            
            # Parse layer structure
            activations = []
            d_hidden = []
            for layer_spec in activation_neurons_str.split(","):
                activation, count = layer_spec.split(":")
                activations.append(activation.strip())
                d_hidden.append(int(count))
            
            layer_size = sum(d_hidden)  # Total neurons per hidden layer
            
            # Create mapping from (layer, neuron) to weight indices
            weight_mapping = {}
            current_weight_idx = 0
            
            # Input size is determined by the CPPN inputs (x, y, d, b typically)
            input_size = len(cppn.cppn.inputs.split(","))
            
            for layer_idx in range(n_layers + 1):  # +1 for output layer
                if layer_idx == 0:
                    # First hidden layer: weights from input
                    for neuron_idx in range(layer_size):
                        weight_start = current_weight_idx + (neuron_idx * input_size)
                        weight_end = weight_start + input_size
                        weight_mapping[f"layer_{layer_idx}_neuron_{neuron_idx}"] = {
                            "weight_range": [weight_start, weight_end],
                            "primary_weight": weight_start  # Use first weight as representative
                        }
                    current_weight_idx += layer_size * input_size
                    
                elif layer_idx < n_layers:
                    # Hidden layers: weights from previous layer
                    for neuron_idx in range(layer_size):
                        weight_start = current_weight_idx + (neuron_idx * layer_size)
                        weight_end = weight_start + layer_size
                        weight_mapping[f"layer_{layer_idx}_neuron_{neuron_idx}"] = {
                            "weight_range": [weight_start, weight_end],
                            "primary_weight": weight_start
                        }
                    current_weight_idx += layer_size * layer_size
                    
                else:
                    # Output layer: 3 outputs (h, s, v)
                    for neuron_idx in range(3):
                        weight_start = current_weight_idx + (neuron_idx * layer_size)
                        weight_end = weight_start + layer_size
                        weight_mapping[f"layer_{layer_idx}_neuron_{neuron_idx}"] = {
                            "weight_range": [weight_start, weight_end],
                            "primary_weight": weight_start
                        }
                    current_weight_idx += 3 * layer_size
            
            return weight_mapping
            
        except Exception as e:
            print(f"Warning: Could not parse architecture for {model_key}: {e}")
            # Fallback: simple linear mapping
            return {f"weight_{i}": {"primary_weight": i} for i in range(model["n_params"])}

# Global CPPN manager
cppn_manager = CPPNManager()

# Pydantic models for API
class WeightUpdate(BaseModel):
    weight_deltas: Dict[int, float]
    img_size: Optional[int] = 256

class SliderConfig(BaseModel):
    weight_id: int
    description: Optional[str] = None
    min_val: Optional[float] = -1.0
    max_val: Optional[float] = 1.0

class ModelInfo(BaseModel):
    models: Dict[str, Dict[str, Any]]

class BatchFrameRequest(BaseModel):
    frames: List[Dict[str, Dict[int, float]]]  # List of {picbreeder_weights, sgd_weights} for each frame
    img_size: Optional[int] = 256

class BatchFrameResponse(BaseModel):
    picbreeder_frames: List[str]  # Base64 encoded images
    sgd_frames: List[str]         # Base64 encoded images

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG"""
    # Ensure image is in [0, 1] range and convert to uint8
    img_array = np.clip(img_array, 0, 1)
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_uint8)
    
    # Save to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def feature_map_to_base64(feature_map: np.ndarray) -> str:
    """Convert feature map to base64 encoded PNG with proper normalization"""
    # Normalize to [-1, 1] -> [0, 1] for visualization
    normalized = (feature_map + 1) / 2
    normalized = np.clip(normalized, 0, 1)
    
    # Convert to uint8
    img_uint8 = (normalized * 255).astype(np.uint8)
    
    # Convert to PIL Image (grayscale)
    img_pil = Image.fromarray(img_uint8, mode='L')
    
    # Save to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def read_root():
    """Serve the main page"""
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/api/models", response_model=ModelInfo)
async def get_models():
    """Get information about all loaded models"""
    model_info = {}
    for key, model in cppn_manager.models.items():
        model_info[key] = {
            "n_params": model["n_params"],
            "arch": model["arch"],
            "weight_suggestions": model["weight_suggestions"]
        }
    return ModelInfo(models=model_info)

@app.post("/api/generate/{model_key}")
async def generate_image(model_key: str, weight_update: WeightUpdate):
    """Generate image with modified weights"""
    try:
        img_array = cppn_manager.generate_image(
            model_key, 
            weight_update.weight_deltas, 
            weight_update.img_size
        )
        img_base64 = numpy_to_base64(img_array)
        return {"image": img_base64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate_comparison/{genome}")
async def generate_comparison(genome: str, weight_update: WeightUpdate):
    """Generate side-by-side comparison of Picbreeder vs SGD"""
    try:
        picbreeder_key = f"picbreeder_{genome}"
        sgd_key = f"sgd_{genome}"
        
        if picbreeder_key not in cppn_manager.models or sgd_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Models for {genome} not found")
        
        # Generate both images
        picbreeder_img = cppn_manager.generate_image(
            picbreeder_key, 
            weight_update.weight_deltas, 
            weight_update.img_size
        )
        sgd_img = cppn_manager.generate_image(
            sgd_key, 
            weight_update.weight_deltas, 
            weight_update.img_size
        )
        
        return {
            "picbreeder": numpy_to_base64(picbreeder_img),
            "sgd": numpy_to_base64(sgd_img)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate_batch/{genome}", response_model=BatchFrameResponse)
async def generate_batch(genome: str, batch_request: BatchFrameRequest):
    """Generate batch of animation frames efficiently"""
    try:
        picbreeder_key = f"picbreeder_{genome}"
        sgd_key = f"sgd_{genome}"
        
        if picbreeder_key not in cppn_manager.models or sgd_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Models for {genome} not found")
        
        picbreeder_frames = []
        sgd_frames = []
        
        # Process each frame in the batch
        for frame_data in batch_request.frames:
            # Generate picbreeder image
            picbreeder_weights = frame_data.get("picbreeder", {})
            picbreeder_img = cppn_manager.generate_image(
                picbreeder_key, 
                picbreeder_weights, 
                batch_request.img_size
            )
            picbreeder_frames.append(numpy_to_base64(picbreeder_img))
            
            # Generate SGD image
            sgd_weights = frame_data.get("sgd", {})
            sgd_img = cppn_manager.generate_image(
                sgd_key, 
                sgd_weights, 
                batch_request.img_size
            )
            sgd_frames.append(numpy_to_base64(sgd_img))
        
        return BatchFrameResponse(
            picbreeder_frames=picbreeder_frames,
            sgd_frames=sgd_frames
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/feature_maps/{model_key}")
async def get_feature_maps(model_key: str, img_size: int = 128):
    """Get feature maps for a specific model"""
    try:
        if model_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        features, weight_mapping = cppn_manager.generate_feature_maps(model_key, img_size)
        
        # Convert feature maps to base64 images
        feature_maps_data = []
        
        for layer_idx, layer_features in enumerate(features):
            layer_data = {
                "layer": layer_idx,
                "features": []
            }
            
            # layer_features has shape (H, W, num_neurons)
            num_neurons = layer_features.shape[-1]
            
            for neuron_idx in range(num_neurons):
                feature_map = layer_features[:, :, neuron_idx]
                feature_img = feature_map_to_base64(feature_map)
                
                # Get the corresponding weight ID for this neuron
                mapping_key = f"layer_{layer_idx}_neuron_{neuron_idx}"
                weight_id = weight_mapping.get(mapping_key, {}).get("primary_weight", 0)
                
                layer_data["features"].append({
                    "neuron": neuron_idx,
                    "weight_id": weight_id,
                    "image": feature_img,
                    "mapping_key": mapping_key
                })
            
            feature_maps_data.append(layer_data)
        
        return {
            "model_key": model_key,
            "feature_maps": feature_maps_data,
            "weight_mapping": weight_mapping
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sweep/{model_key}/{weight_id}")
async def weight_sweep(model_key: str, weight_id: int, steps: int = 20, range_val: float = 1.0):
    """Generate a sequence of images with one weight swept across a range"""
    try:
        if model_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        model = cppn_manager.models[model_key]
        if weight_id >= model["n_params"]:
            raise HTTPException(status_code=400, detail=f"Weight ID {weight_id} out of range")
        
        # Generate sweep
        weight_values = np.linspace(-range_val, range_val, steps)
        images = []
        
        for delta in weight_values:
            weight_deltas = {weight_id: delta}
            img_array = cppn_manager.generate_image(model_key, weight_deltas, 128)
            images.append(numpy_to_base64(img_array))
        
        return {
            "images": images,
            "weight_values": weight_values.tolist(),
            "weight_id": weight_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/feature_map_hires/{model_key}/{weight_id}")
async def get_feature_map_hires(model_key: str, weight_id: int, img_size: int = 500):
    """Get a high-resolution version of a specific feature map"""
    try:
        if model_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        # Generate feature maps at high resolution
        features, weight_mapping = cppn_manager.generate_feature_maps(model_key, img_size)
        
        # Find the feature map corresponding to the weight_id
        target_feature_map = None
        
        for layer_idx, layer_features in enumerate(features):
            num_neurons = layer_features.shape[-1]
            
            for neuron_idx in range(num_neurons):
                mapping_key = f"layer_{layer_idx}_neuron_{neuron_idx}"
                mapped_weight_id = weight_mapping.get(mapping_key, {}).get("primary_weight", 0)
                
                if mapped_weight_id == weight_id:
                    target_feature_map = layer_features[:, :, neuron_idx]
                    break
            
            if target_feature_map is not None:
                break
        
        if target_feature_map is None:
            # Fallback: if we can't find the exact weight mapping, 
            # try to find a reasonable feature map or generate one
            raise HTTPException(status_code=404, detail=f"Feature map for weight {weight_id} not found")
        
        # Convert to high-res base64 image
        feature_img = feature_map_to_base64(target_feature_map)
        
        return {
            "image": feature_img,
            "weight_id": weight_id,
            "model_key": model_key,
            "resolution": img_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/topology/{model_key}")
async def get_network_topology(model_key: str):
    """Get network topology data for visualization"""
    try:
        if model_key not in cppn_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        # Extract the source type and genome from model_key
        source, genome = model_key.split("_", 1)
        save_dir = f"data/{source}_{genome}"
        
        # Load the original NEAT structure if it exists (for Picbreeder)
        pbcppn_data = None
        if source == "picbreeder":
            try:
                pbcppn_data = util.load_pkl(save_dir, "pbcppn")
            except:
                pass
        
        # Get architecture and parameters
        model = cppn_manager.models[model_key]
        arch_str = model["arch"]
        
        # Parse architecture
        n_layers_str, activation_neurons_str = arch_str.split(";")
        n_layers = int(n_layers_str)
        
        activations = []
        d_hidden = []
        for layer_spec in activation_neurons_str.split(","):
            activation, count = layer_spec.split(":")
            activations.append(activation.strip())
            d_hidden.append(int(count))
        
        layer_size = sum(d_hidden)
        input_size = 4  # x, y, d, bias
        output_size = 3  # h, s, v
        
        # Build topology data
        topology = {
            "model_key": model_key,
            "source": source,
            "genome": genome,
            "n_layers": n_layers,
            "layer_size": layer_size,
            "activations": activations,
            "d_hidden": d_hidden,
            "nodes": [],
            "links": [],
            "original_neat": None
        }
        
        # Add input nodes
        input_names = ["x", "y", "d", "bias"]
        for i, name in enumerate(input_names):
            topology["nodes"].append({
                "id": f"input_{i}",
                "label": name,
                "type": "input",
                "activation": "identity",
                "layer": -1,
                "x": 0,
                "y": i * 50
            })
        
        # Add hidden layer nodes (sample only to avoid performance issues)
        max_layers_to_show = min(5, n_layers)  # Limit to first 5 layers
        for layer_idx in range(max_layers_to_show):
            y_offset = 0
            neuron_idx_in_layer = 0
            for act_idx, (activation, count) in enumerate(zip(activations, d_hidden)):
                # Only show first few neurons of each activation type
                max_neurons_per_type = min(5, count)
                for neuron_in_group in range(max_neurons_per_type):
                    topology["nodes"].append({
                        "id": f"hidden_{layer_idx}_{neuron_idx_in_layer}",
                        "label": f"L{layer_idx}N{neuron_idx_in_layer}" if count <= 5 else f"L{layer_idx}{activation[:3]}",
                        "type": "hidden",
                        "activation": activation,
                        "layer": layer_idx,
                        "x": (layer_idx + 1) * 200,
                        "y": y_offset + neuron_in_group * 30
                    })
                    neuron_idx_in_layer += 1
                # Add ellipsis node if there are more neurons
                if count > max_neurons_per_type:
                    topology["nodes"].append({
                        "id": f"hidden_{layer_idx}_{neuron_idx_in_layer}_more",
                        "label": f"...+{count - max_neurons_per_type}",
                        "type": "hidden",
                        "activation": "ellipsis",
                        "layer": layer_idx,
                        "x": (layer_idx + 1) * 200,
                        "y": y_offset + max_neurons_per_type * 30
                    })
                    neuron_idx_in_layer += 1
                y_offset += (max_neurons_per_type + (1 if count > max_neurons_per_type else 0)) * 30 + 20
        
        # Add ellipsis for remaining layers if there are more
        if n_layers > max_layers_to_show:
            topology["nodes"].append({
                "id": f"layers_more",
                "label": f"...+{n_layers - max_layers_to_show} layers",
                "type": "hidden",
                "activation": "ellipsis",
                "layer": max_layers_to_show,
                "x": (max_layers_to_show + 1) * 200,
                "y": 100
            })
        
        # Add output nodes
        output_names = ["h", "s", "v"]
        for i, name in enumerate(output_names):
            topology["nodes"].append({
                "id": f"output_{i}",
                "label": name,
                "type": "output",
                "activation": "identity",
                "layer": n_layers,
                "x": (n_layers + 1) * 200,
                "y": i * 50
            })
        
        # Create a list of actual node IDs for reference
        existing_node_ids = set(node["id"] for node in topology["nodes"])
        
        # Add connections (only between existing nodes)
        # Input to first hidden layer (sample subset)
        for input_idx in range(input_size):
            # Connect to first few actual neurons in layer 0
            layer_0_nodes = [node["id"] for node in topology["nodes"] 
                           if node["type"] == "hidden" and node["layer"] == 0 and "more" not in node["id"]]
            for target_id in layer_0_nodes[:6]:  # Connect to first 6 actual nodes
                topology["links"].append({
                    "id": f"link_input_{input_idx}_to_{target_id}",
                    "source": f"input_{input_idx}",
                    "target": target_id,
                    "weight": 1.0,
                    "type": "input_to_hidden"
                })
        
        # Hidden to hidden layers (simplified - only show some connections between adjacent layers)
        if max_layers_to_show > 1:
            for layer_idx in range(min(2, max_layers_to_show - 1)):  # Only first 2 layer transitions
                source_nodes = [node["id"] for node in topology["nodes"] 
                              if node["type"] == "hidden" and node["layer"] == layer_idx and "more" not in node["id"]]
                target_nodes = [node["id"] for node in topology["nodes"] 
                              if node["type"] == "hidden" and node["layer"] == layer_idx + 1 and "more" not in node["id"]]
                
                # Connect first few nodes from each layer
                for i, source_id in enumerate(source_nodes[:4]):
                    for j, target_id in enumerate(target_nodes[:4]):
                        if i % 2 == 0 and j % 2 == 0:  # Sparse sampling
                            topology["links"].append({
                                "id": f"link_{source_id}_to_{target_id}",
                                "source": source_id,
                                "target": target_id,
                                "weight": 1.0,
                                "type": "hidden_to_hidden"
                            })
        
        # Last shown hidden layer to output (sample connections)
        last_layer_idx = max_layers_to_show - 1
        last_layer_nodes = [node["id"] for node in topology["nodes"] 
                           if node["type"] == "hidden" and node["layer"] == last_layer_idx and "more" not in node["id"]]
        
        for output_idx in range(output_size):
            # Connect to first few neurons from last layer
            for source_id in last_layer_nodes[:4]:
                topology["links"].append({
                    "id": f"link_{source_id}_to_output_{output_idx}",
                    "source": source_id,
                    "target": f"output_{output_idx}",
                    "weight": 1.0,
                    "type": "hidden_to_output"
                })
        
        # If we have original NEAT data (Picbreeder), include it
        if pbcppn_data:
            neat_topology = {
                "nodes": [],
                "links": []
            }
            
            # Get special nodes mapping for proper input/output identification
            special_nodes = pbcppn_data.get('special_nodes', {})
            input_node_ids = {str(special_nodes.get(name)) for name in ['x', 'y', 'd', 'bias'] if special_nodes.get(name) is not None}
            output_node_ids = {str(special_nodes.get(name)) for name in ['h', 's', 'v'] if special_nodes.get(name) is not None}
            
            # Process NEAT nodes
            if 'nodes' in pbcppn_data:
                for node in pbcppn_data['nodes']:
                    node_id = str(node.get('id', ''))
                    
                    # Determine node type based on special_nodes mapping
                    if node_id in input_node_ids:
                        node_type = "input"
                    elif node_id in output_node_ids:
                        node_type = "output"
                    else:
                        node_type = "hidden"
                    
                    neat_topology["nodes"].append({
                        "id": node_id,
                        "label": node.get('label', ''),
                        "activation": node.get('activation', 'identity'),
                        "type": node_type
                    })
            
            # Process NEAT connections
            if 'links' in pbcppn_data:
                for link in pbcppn_data['links']:
                    neat_topology["links"].append({
                        "id": str(link.get('id', '')),
                        "source": str(link.get('source', '')),
                        "target": str(link.get('target', '')),
                        "weight": float(link.get('weight', 0.0))
                    })
            
            topology["original_neat"] = neat_topology
        
        return topology
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting FER CPPN Explorer...")
    print("Visit http://localhost:8000 to use the app")
    uvicorn.run(app, host="0.0.0.0", port=8000) 