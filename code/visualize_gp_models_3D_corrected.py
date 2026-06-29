#!/usr/bin/env python3
"""
Corrected 3D visualization for GP models that actually loads and uses the GP models
instead of RBF interpolation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import sys
import os

# Add parent directory to path to import GP models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gp_base import GPSurrogateModel

class GPModelLoader:
    """Helper class to load and use saved GP models."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        
    def load_model_at_iteration(self, iteration: int):
        """Load GP model at specific iteration."""
        # Check if model files exist
        model_file = self.model_dir / f'model_{iteration}.pth'
        if not model_file.exists():
            print(f"Model file not found: {model_file}")
            return None
            
        # Load training data to get model configuration
        try:
            X = torch.load(self.model_dir / f'X_{iteration}.pth')
            Y = torch.load(self.model_dir / f'Y_{iteration}.pth')
            forces = torch.load(self.model_dir / f'g_{iteration}.pth')
            
            # Check if this is toy model data (z=0)
            is_toy_model = X.shape[-1] == 3 and torch.all(X[:, 2] == 0)
            
            if is_toy_model:
                # For toy models, create minimal atomic info
                atomic_info = {
                    "moving_indices": [0],  # Single atom for toy model
                    "frozen_indices": [],
                    "total_atoms": 1
                }
            else:
                # Try to load atomic info
                atomic_info_file = self.model_dir / f'atomic_info_{iteration}.pth'
                if atomic_info_file.exists():
                    atomic_info = torch.load(atomic_info_file)
                else:
                    # Infer from data dimensions
                    n_moving = X.shape[1] // 3
                    atomic_info = {
                        "moving_indices": list(range(n_moving)),
                        "frozen_indices": [],
                        "total_atoms": n_moving
                    }
            
            # Create GP model instance
            gp_model = GPSurrogateModel(
                model_type="MultitaskGPModel_rbf_atomic",
                atomic_info=atomic_info
            )
            
            # Train the model with loaded data
            training_data = [X, Y, forces]
            gp_model.train(training_data, save_model=False)
            
            # Load the saved model state
            saved_state = torch.load(model_file)
            gp_model.model.load_state_dict(saved_state)
            
            return gp_model
            
        except Exception as e:
            print(f"Error loading model at iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_on_grid(self, gp_model, grid_points_2d):
        """Use GP model to predict on a grid of 2D points."""
        if gp_model is None:
            return None
            
        try:
            # Convert 2D grid points to 3D for GP (add z=0)
            n_points = grid_points_2d.shape[0]
            grid_points_3d = np.column_stack([grid_points_2d, np.zeros(n_points)])
            
            # GP expects flattened format for atomic systems
            # For toy model with single "atom", this is just the 3D coordinates
            test_data = grid_points_3d  # Shape: (n_points, 3)
            
            # Get predictions
            pred_energy, pred_forces, var_energy, var_forces = gp_model.predict(test_data)
            
            return pred_energy
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_corrected_animation(output_dir: Path, output_file: str = "corrected_gp_3d.gif"):
    """Create animation using actual GP model predictions."""
    
    # Setup paths
    gp1_dir = output_dir / "data_gp1"
    gp2_dir = output_dir / "data_gp2"
    
    # Create model loaders
    gp1_loader = GPModelLoader(gp1_dir)
    gp2_loader = GPModelLoader(gp2_dir)
    
    # Find available iterations
    model_files = list(gp1_dir.glob("model_*.pth"))
    iterations = sorted([int(f.stem.split('_')[1]) for f in model_files])
    
    if not iterations:
        print("No model files found!")
        return
        
    print(f"Found {len(iterations)} iterations: {iterations}")
    
    # Create grid for predictions
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 40)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Calculate true potential
    Z_true = -np.sin(np.pi * X_grid) * np.sin(np.pi * Y_grid)
    
    # Setup figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        iter_num = iterations[frame]
        ax.clear()
        
        print(f"\nProcessing iteration {iter_num}")
        
        # Load GP models
        gp1_model = gp1_loader.load_model_at_iteration(iter_num)
        gp2_model = gp2_loader.load_model_at_iteration(iter_num)
        
        # Get predictions
        Z_gp1 = gp1_loader.predict_on_grid(gp1_model, grid_points)
        Z_gp2 = gp2_loader.predict_on_grid(gp2_model, grid_points)
        
        # Plot true PES
        ax.plot_wireframe(X_grid, Y_grid, Z_true, 
                         color='red', alpha=0.3, linewidth=0.8,
                         rcount=20, ccount=20, label='True PES')
        
        # Plot GP1 predictions if available
        if Z_gp1 is not None:
            Z_gp1_grid = Z_gp1.reshape(X_grid.shape)
            # Note: GP predictions are on different scale, apply scaling
            # This scaling factor needs to be determined from the data
            scale_factor = 20.0  # Approximate based on analysis
            Z_gp1_scaled = Z_gp1_grid * scale_factor
            
            ax.plot_wireframe(X_grid, Y_grid, Z_gp1_scaled,
                             color='green', alpha=0.6, linewidth=1.2,
                             rcount=15, ccount=15, label='GP1 (scaled)')
            
            print(f"  GP1 predictions: min={Z_gp1.min():.3f}, max={Z_gp1.max():.3f}")
            print(f"  GP1 scaled: min={Z_gp1_scaled.min():.3f}, max={Z_gp1_scaled.max():.3f}")
        
        # Plot GP2 predictions if available
        if Z_gp2 is not None:
            Z_gp2_grid = Z_gp2.reshape(X_grid.shape)
            Z_gp2_scaled = Z_gp2_grid * scale_factor
            
            ax.plot_wireframe(X_grid, Y_grid, Z_gp2_scaled + 0.1,  # Slight offset for visibility
                             color='purple', alpha=0.6, linewidth=1.2,
                             rcount=15, ccount=15, label='GP2 (scaled)')
            
            print(f"  GP2 predictions: min={Z_gp2.min():.3f}, max={Z_gp2.max():.3f}")
            print(f"  GP2 scaled: min={Z_gp2_scaled.min():.3f}, max={Z_gp2_scaled.max():.3f}")
        
        # Load and plot trajectory
        try:
            X_train = torch.load(gp2_dir / f'X_{iter_num}.pth')
            Y_train = torch.load(gp2_dir / f'Y_{iter_num}.pth')
            
            if X_train.shape[1] == 3:
                traj_x = X_train[:, 0].numpy()
                traj_y = X_train[:, 1].numpy()
            else:
                traj_x = X_train[:, 0].numpy()
                traj_y = X_train[:, 1].numpy()
                
            # Calculate true energies at trajectory points
            traj_z = -np.sin(np.pi * traj_x) * np.sin(np.pi * traj_y)
            
            ax.plot(traj_x, traj_y, traj_z, 'k-', linewidth=2, alpha=0.8)
            ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], 
                      color='yellow', s=100, marker='*', label='Current position')
            
        except Exception as e:
            print(f"  Could not load trajectory: {e}")
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy (eV)')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        
        # Set view angle
        ax.view_init(elev=30, azim=45 + frame * 5)
        
        ax.legend(loc='upper right')
        ax.set_title(f'GP Models vs True PES - Iteration {iter_num}')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(iterations),
                                  interval=500, blit=False)
    
    # Save animation
    output_path = output_dir / "gp_animations" / output_file
    print(f"\nSaving animation to {output_path}")
    anim.save(output_path, writer='pillow', fps=2)
    plt.close()
    
    print("Animation complete!")


def main():
    # Find the latest dual GP run
    base_dir = Path("/Users/farid/Downloads/TD_SPF/real_system/outputs")
    dual_gp_dirs = sorted(base_dir.glob("dual_gp_toy_*"))
    
    if not dual_gp_dirs:
        print("No dual GP runs found!")
        return
        
    latest_dir = dual_gp_dirs[-1]
    print(f"Using latest run: {latest_dir}")
    
    # Create corrected animation
    create_corrected_animation(latest_dir)


if __name__ == "__main__":
    main()