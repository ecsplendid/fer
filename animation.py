#!/usr/bin/env python3
"""
Animation script for FER paper - Apple CPPN Parameter Sweeps
Shows how individual weights control different aspects of the apple image.
"""

import os
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time

# Add the src directory to path to import modules
sys.path.append('src')

try:
    from cppn import CPPN, FlattenCPPNParameters
    import util
except ImportError:
    print("Error: Could not import required modules. Make sure you're in the fer project directory and have the required dependencies.")
    print("You may need to run: pip install jax matplotlib numpy einops tqdm pandas")
    sys.exit(1)

class AppleWeightSweepAnimator:
    def __init__(self, source="picbreeder", img_size=256):
        """
        Initialize the animator for apple weight sweeps.
        
        Args:
            source: "picbreeder" or "sgd" - which model to use
            img_size: Size of generated images
        """
        self.source = source
        self.img_size = img_size
        self.load_model()
        self.setup_weight_data()
        
    def load_model(self):
        """Load the apple CPPN model"""
        save_dir = f"data/{self.source}_apple"
        
        if not os.path.exists(save_dir):
            print(f"Error: Model directory {save_dir} not found.")
            print("Please make sure you have the pre-trained models in the data directory.")
            sys.exit(1)
            
        try:
            self.arch = util.load_pkl(save_dir, "arch")
            self.params = util.load_pkl(save_dir, "params")
            self.cppn = FlattenCPPNParameters(CPPN(self.arch))
            print(f"Loaded {self.source} apple model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def setup_weight_data(self):
        """Setup weight sweep data for apple model"""
        if self.source == "picbreeder":
            self.weight_ids = [42178, 4140, 34459, 17131]
            self.descriptions = ["Controls Stem Angle", "Controls Apple Size", "Cleans Background", "Removes Stem"]
            self.weight_sweep_controls = [(-1., 2.), None, None, None]
        else:  # sgd
            self.weight_ids = [37753, 135, 37721, 37809]
            self.descriptions = ["Weight 37753", "≈ Controls Apple Size", "≈ Cleans Background", "Weight 37809"]
            self.weight_sweep_controls = [None, None, None, None]
    
    def sweep_weight(self, weight_id, center_weight=None, r=1, n_steps=50):
        """
        Sweep a single weight across a range of values.
        
        Args:
            weight_id: Index of weight to sweep
            center_weight: Center the sweep around this value (None for original)
            r: Range to sweep (-r to +r)
            n_steps: Number of steps in the sweep
        """
        try:
            import jax.numpy as jnp
            from einops import repeat
            import jax
        except ImportError:
            print("Error: JAX not available. Please install jax.")
            sys.exit(1)
            
        weight_values = jnp.linspace(-r, r, n_steps)
        if center_weight is not None:
            weight_values = weight_values + center_weight
        else:
            weight_values = weight_values + self.params[weight_id]
        
        # Create parameter arrays for each weight value
        params_sweep = repeat(self.params, "p -> n p", n=n_steps).at[:, weight_id].set(weight_values)
        
        # Generate images for each parameter setting
        imgs = jax.vmap(partial(self.cppn.generate_image, img_size=self.img_size))(params_sweep)
        return np.array(imgs), weight_values
    
    def create_single_weight_animation(self, weight_idx, duration=3.0):
        """Create animation for a single weight sweep"""
        weight_id = self.weight_ids[weight_idx]
        description = self.descriptions[weight_idx]
        controls = self.weight_sweep_controls[weight_idx]
        
        print(f"Generating animation for: {description}")
        
        if controls is None:
            imgs, weight_values = self.sweep_weight(weight_id, r=1, n_steps=60)
        else:
            c, r = controls
            imgs, weight_values = self.sweep_weight(weight_id, center_weight=c, r=r, n_steps=60)
        
        # Create the animation
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create image display
        im = ax.imshow(imgs[0], extent=[0.1, 0.9, 0.3, 0.9], aspect='equal')
        
        # Add title and weight info
        title = ax.text(0.5, 0.95, f"{self.source.capitalize()} Apple: {description}", 
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       transform=ax.transAxes)
        
        weight_text = ax.text(0.5, 0.25, f"Weight ID: {weight_id}", 
                             ha='center', va='center', fontsize=12,
                             transform=ax.transAxes)
        
        value_text = ax.text(0.5, 0.2, "", ha='center', va='center', fontsize=12,
                            transform=ax.transAxes)
        
        # Progress bar
        progress_bg = Rectangle((0.1, 0.1), 0.8, 0.05, facecolor='lightgray', 
                               transform=ax.transAxes)
        ax.add_patch(progress_bg)
        progress_bar = Rectangle((0.1, 0.1), 0, 0.05, facecolor='darkblue',
                                transform=ax.transAxes)
        ax.add_patch(progress_bar)
        
        def animate(frame):
            # Calculate which direction we're going (forward or backward)
            cycle_length = len(imgs)
            total_frames = cycle_length * 2  # Go forward then backward
            
            if frame < cycle_length:
                img_idx = frame
                direction = "→"
            else:
                img_idx = cycle_length * 2 - frame - 1
                direction = "←"
            
            # Update image
            im.set_array(imgs[img_idx])
            
            # Update weight value text
            weight_val = weight_values[img_idx]
            original_val = self.params[weight_id]
            delta = weight_val - original_val
            value_text.set_text(f"Δw = {delta:.3f} {direction}")
            
            # Update progress bar
            progress = frame / (total_frames - 1)
            progress_bar.set_width(0.8 * progress)
            
            return [im, value_text, progress_bar]
        
        # Create animation
        total_frames = len(imgs) * 2  # Forward and backward
        interval = (duration * 1000) / total_frames  # Convert to milliseconds
        
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=interval, blit=True, repeat=True)
        
        plt.tight_layout()
        return fig, anim
    
    def create_multi_weight_animation(self, duration_per_weight=4.0):
        """Create animation showing all weight sweeps in sequence"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        axes = [ax1, ax2, ax3, ax4]
        
        # Pre-generate all sweep data
        all_sweeps = []
        for i, (weight_id, description, controls) in enumerate(zip(
            self.weight_ids, self.descriptions, self.weight_sweep_controls)):
            
            print(f"Pre-generating sweep {i+1}/4: {description}")
            if controls is None:
                imgs, weight_values = self.sweep_weight(weight_id, r=1, n_steps=30)
            else:
                c, r = controls
                imgs, weight_values = self.sweep_weight(weight_id, center_weight=c, r=r, n_steps=30)
            all_sweeps.append((imgs, weight_values, description, weight_id))
        
        # Setup subplots
        ims = []
        texts = []
        for i, ax in enumerate(axes):
            ax.axis('off')
            imgs, _, description, weight_id = all_sweeps[i]
            im = ax.imshow(imgs[0])
            ax.set_title(f"{description}\nWeight ID: {weight_id}", fontsize=10)
            ims.append(im)
        
        # Main title
        fig.suptitle(f"{self.source.capitalize()} Apple CPPN - Weight Sweeps", 
                    fontsize=14, fontweight='bold')
        
        def animate(frame):
            updates = []
            for i, (im, (imgs, weight_values, _, _)) in enumerate(zip(ims, all_sweeps)):
                # Cycle through the sweep for each weight
                cycle_length = len(imgs)
                total_cycle = cycle_length * 2  # Forward and backward
                
                # Stagger the animations slightly
                offset_frame = (frame + i * 10) % total_cycle
                
                if offset_frame < cycle_length:
                    img_idx = offset_frame
                else:
                    img_idx = cycle_length * 2 - offset_frame - 1
                
                im.set_array(imgs[img_idx])
                updates.append(im)
            
            return updates
        
        # Create animation
        total_frames = max(len(sweep[0]) for sweep in all_sweeps) * 2
        interval = (duration_per_weight * 1000) / total_frames
        
        anim = animation.FuncAnimation(fig, animate, frames=total_frames * 3,  # Make it longer
                                     interval=interval, blit=True, repeat=True)
        
        plt.tight_layout()
        return fig, anim

def main():
    """Main function to run the animation"""
    print("Apple CPPN Parameter Sweep Animation")
    print("=====================================")
    
    # Ask user which model to use
    while True:
        choice = input("Which model to animate? (p)icbreeder or (s)gd [p]: ").lower().strip()
        if choice in ['', 'p', 'picbreeder']:
            source = "picbreeder"
            break
        elif choice in ['s', 'sgd']:
            source = "sgd"
            break
        else:
            print("Please enter 'p' for picbreeder or 's' for sgd")
    
    # Create animator
    animator = AppleWeightSweepAnimator(source=source)
    
    # Ask what type of animation
    while True:
        anim_type = input("Animation type? (s)ingle weight, (m)ulti weight [m]: ").lower().strip()
        if anim_type in ['', 'm', 'multi']:
            print("Creating multi-weight animation...")
            fig, anim = animator.create_multi_weight_animation()
            break
        elif anim_type in ['s', 'single']:
            # Ask which weight
            print("Available weights:")
            for i, desc in enumerate(animator.descriptions):
                print(f"  {i}: {desc}")
            
            while True:
                try:
                    weight_idx = int(input(f"Which weight (0-{len(animator.descriptions)-1}) [0]: ") or "0")
                    if 0 <= weight_idx < len(animator.descriptions):
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(animator.descriptions)-1}")
                except ValueError:
                    print("Please enter a valid number")
            
            print(f"Creating single weight animation for: {animator.descriptions[weight_idx]}")
            fig, anim = animator.create_single_weight_animation(weight_idx)
            break
        else:
            print("Please enter 's' for single or 'm' for multi")
    
    print("\nAnimation created! Close the window to exit.")
    print("Tip: You can record your screen while this is running to create a video.")
    
    plt.show()

if __name__ == "__main__":
    main() 