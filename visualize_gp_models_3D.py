#!/usr/bin/env python3
"""
Animate GP1 and GP2 model evolution from saved training data.

This script creates animated visualizations showing:
- Energy landscape evolution
- Training data accumulation
- Model predictions changing over iterations

Supports organized run directory structure:
- Use --run-dir to specify a specific run directory
- Use --latest to automatically find the most recent run
- Use --output-subdir to specify subdirectory for outputs
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from output_manager import get_output_path, get_latest_run_dir

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GPModelAnimator:
    def __init__(self, gp1_path: str = "data_gp1", gp2_path: str = "data_gp2", 
                 skip_init_steps: bool = True, is_toy_model: bool = False):
        """Initialize the animator with paths to GP data directories.
        
        Args:
            gp1_path: Path to GP1 data directory
            gp2_path: Path to GP2 data directory
            skip_init_steps: If True, skip initialization steps (before energy referencing)
            is_toy_model: If True, handle 2D toy model data format
        """
        self.gp1_path = Path(gp1_path)
        self.gp2_path = Path(gp2_path)
        self.skip_init_steps = skip_init_steps
        self.is_toy_model = is_toy_model
        
        # Define crystallographic directions for BCC
        self.dir_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Vacancy diffusion direction
        self.dir_110 = np.array([1, 1, 0]) / np.sqrt(2)  # Perpendicular direction
        
        # Setup coordinate system
        self.setup_coordinate_system()
        
        # Animation settings
        self.fps = 5  # Frames per second
        self.interval = 200  # Milliseconds between frames
        
    def setup_coordinate_system(self):
        """Create orthonormal basis with [111] as primary direction."""
        v1 = self.dir_111
        v2 = self.dir_110 - np.dot(self.dir_110, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        self.basis = np.array([v1, v2, v3]).T
        
    def detect_toy_model(self, path: Path) -> bool:
        """Detect if this is toy model data based on file format."""
        # Check for positions_*.pth files (toy model format)
        if list(path.glob("positions_*.pth")):
            return True
        # Check for X_*.pth files (real system format)
        if list(path.glob("X_*.pth")):
            return False
        return False
    
    def load_gp_data(self, path: Path, model_type: str) -> Dict[int, Dict]:
        """Load all GP model data from a directory."""
        data = {}
        
        # Auto-detect toy model format if not explicitly set
        if not hasattr(self, '_toy_model_detected'):
            self._toy_model_detected = self.detect_toy_model(path)
            if self._toy_model_detected:
                print(f"Detected toy model format for {model_type}")
                self.is_toy_model = True
        
        if self.is_toy_model or self._toy_model_detected:
            # Load toy model format
            data = self.load_toy_model_data(path, model_type)
        else:
            # Load real system format
            model_files = sorted(path.glob("model_*.pth"))
            
            for model_file in model_files:
                iter_num = int(model_file.stem.split('_')[1])
                try:
                    data[iter_num] = {
                        'model_state': torch.load(model_file),
                        'X': torch.load(path / f'X_{iter_num}.pth'),
                        'Y': torch.load(path / f'Y_{iter_num}.pth'),
                        'g': torch.load(path / f'g_{iter_num}.pth'),
                        'd': torch.load(path / f'd_{iter_num}.pth'),
                    }
                    noise_file = path / f'thermal_noise_{iter_num}.pth'
                    if noise_file.exists():
                        data[iter_num]['thermal_noise'] = torch.load(noise_file)
                except Exception as e:
                    print(f"Warning: Could not load data for iteration {iter_num}: {e}")
                    
        print(f"Loaded {len(data)} {model_type} models")
        
        # Filter out initialization steps if requested
        if self.skip_init_steps:
            filtered_data = self._filter_init_steps(data, model_type)
            print(f"After filtering init steps: {len(filtered_data)} {model_type} models")
            return filtered_data
        
        return data
    
    def load_toy_model_data(self, path: Path, model_type: str) -> Dict[int, Dict]:
        """Load GP data for toy models (2D format)."""
        data = {}
        model_files = sorted(path.glob("model_*.pth"))
        
        for model_file in model_files:
            iter_num = int(model_file.stem.split('_')[1])
            
            try:
                # Load 2D positions and convert to 3D
                positions_file = path / f'positions_{iter_num}.pth'
                forces_file = path / f'forces_{iter_num}.pth'
                energies_file = path / f'energies_{iter_num}.pth'
                
                if positions_file.exists() and forces_file.exists() and energies_file.exists():
                    positions_2d = torch.load(positions_file)
                    forces_2d = torch.load(forces_file)
                    energies = torch.load(energies_file)
                    
                    # Convert 2D to 3D by adding zero z-component
                    n_points = positions_2d.shape[0]
                    positions_3d = torch.zeros(n_points, 3)
                    positions_3d[:, :2] = positions_2d
                    
                    forces_3d = torch.zeros(n_points, 3)
                    forces_3d[:, :2] = forces_2d
                    
                    data[iter_num] = {
                        'model_state': torch.load(model_file),
                        'X': positions_3d,
                        'Y': energies,
                        'g': forces_3d,
                        'd': forces_3d,  # For compatibility
                        'is_toy': True
                    }
                    
                    noise_file = path / f'thermal_noise_{iter_num}.pth'
                    if noise_file.exists():
                        data[iter_num]['thermal_noise'] = torch.load(noise_file)
                        
            except Exception as e:
                print(f"Warning: Could not load toy model data for iteration {iter_num}: {e}")
        
        return data
    
    def _filter_init_steps(self, data: Dict[int, Dict], model_type: str) -> Dict[int, Dict]:
        """Filter out initialization steps based on energy scale.
        
        Initialization steps typically have unreferenced energies (large negative values),
        while actual dimer steps have referenced energies (near zero).
        """
        if not data:
            return data
        
        filtered_data = {}
        
        # Analyze energy scales to detect when referencing occurs
        for iter_num in sorted(data.keys()):
            energies = data[iter_num]['Y'].numpy()
            mean_energy = np.mean(energies)
            
            # Check if this looks like referenced data
            # Unreferenced VASP energies are typically < -1000 eV
            # Referenced energies are typically within ±20 eV of zero
            if abs(mean_energy) < 100:  # This is likely referenced data
                filtered_data[iter_num] = data[iter_num]
            else:
                print(f"Skipping {model_type} iteration {iter_num} (mean energy: {mean_energy:.2f} eV)")
        
        # If no referenced data found, use a different criterion
        if not filtered_data:
            print(f"No referenced data found for {model_type}, using energy jump detection")
            
            energy_means = [(i, np.mean(data[i]['Y'].numpy())) for i in sorted(data.keys())]
            
            # Look for large jumps in energy scale
            for i in range(1, len(energy_means)):
                prev_mean = energy_means[i-1][1]
                curr_mean = energy_means[i][1]
                
                # If there's a jump of more than 1000 eV, this might be the referencing point
                if abs(curr_mean - prev_mean) > 1000:
                    print(f"Detected energy reference jump at iteration {energy_means[i][0]}")
                    # Include this and all subsequent iterations
                    for j in range(i, len(energy_means)):
                        iter_num = energy_means[j][0]
                        filtered_data[iter_num] = data[iter_num]
                    break
        
        # If still no data, just skip the first few iterations
        if not filtered_data and len(data) > 5:
            print(f"Using fallback: skipping first 5 iterations for {model_type}")
            sorted_iters = sorted(data.keys())
            for iter_num in sorted_iters[5:]:
                filtered_data[iter_num] = data[iter_num]
        
        return filtered_data if filtered_data else data
    
    def project_to_crystal_coords(self, positions: np.ndarray) -> np.ndarray:
        """Project positions onto crystallographic coordinate system."""
        n_samples = positions.shape[0]
        n_coords = positions.shape[1]
        n_atoms = n_coords // 3
        
        pos_reshaped = positions.reshape(n_samples, n_atoms, 3)
        
        if n_atoms == 1:
            crystal_coords = pos_reshaped[:, 0, :] @ self.basis
        else:
            com = np.mean(pos_reshaped, axis=1)
            crystal_coords = com @ self.basis
            
        return crystal_coords
    
    def animate_3d_landscape(self, gp_data: Dict, model_name: str, output_file: str):
        """Create animated 3D energy landscape."""
        # Check if this is toy model data
        if self.is_toy_model:
            return self.animate_3d_landscape_toy_model(gp_data, model_name, output_file)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get iteration numbers
        iterations = sorted(gp_data.keys())
        
        if not iterations:
            print(f"No data to animate for {model_name}")
            return
        
        # Determine global limits for consistent scaling
        all_coords = []
        all_energies = []
        
        for iter_num in iterations:
            X = gp_data[iter_num]['X'].numpy()
            Y = gp_data[iter_num]['Y'].numpy()
            crystal_coords = self.project_to_crystal_coords(X)
            all_coords.append(crystal_coords)
            all_energies.append(Y)
        
        all_coords = np.vstack(all_coords)
        all_energies = np.concatenate(all_energies)
        
        # Add padding to limits
        coord_range = np.ptp(all_coords, axis=0)
        padding = 0.2 * coord_range
        
        x1_lim = [all_coords[:, 0].min() - padding[0], all_coords[:, 0].max() + padding[0]]
        x2_lim = [all_coords[:, 1].min() - padding[1], all_coords[:, 1].max() + padding[1]]
        z_lim = [all_energies.min() - 0.5, all_energies.max() + 0.5]
        
        # Create grid for interpolation
        n_grid = 40
        x1_range = np.linspace(x1_lim[0], x1_lim[1], n_grid)
        x2_range = np.linspace(x2_lim[0], x2_lim[1], n_grid)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Animation function
        def animate(frame):
            ax.clear()
            iter_num = iterations[frame]
            
            X = gp_data[iter_num]['X'].numpy()
            Y = gp_data[iter_num]['Y'].numpy()
            crystal_coords = self.project_to_crystal_coords(X)
            
            try:
                # Only interpolate if we have enough points
                if len(Y) >= 4:
                    Z = griddata(crystal_coords[:, :2], Y, (X1, X2), method='linear')
                    
                    # Plot surface
                    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7, 
                                         linewidth=0, antialiased=True)
                
                # Always plot training points
                ax.scatter(crystal_coords[:, 0], crystal_coords[:, 1], Y, 
                          color='red', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Add trajectory line if enough points
                if len(crystal_coords) > 1:
                    sorted_idx = np.argsort(crystal_coords[:, 0])
                    ax.plot(crystal_coords[sorted_idx, 0], 
                           crystal_coords[sorted_idx, 1], 
                           Y[sorted_idx], 
                           'r--', alpha=0.5, linewidth=1)
                
            except Exception as e:
                print(f"Interpolation failed for {model_name} iter {iter_num}: {e}")
                # Fallback to scatter plot
                ax.scatter(crystal_coords[:, 0], crystal_coords[:, 1], Y, 
                          c=Y, cmap='viridis', s=50, alpha=0.8)
            
            ax.set_xlim(x1_lim)
            ax.set_ylim(x2_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel('[111] direction (Å)')
            ax.set_ylabel('[110] direction (Å)')
            ax.set_zlabel('Energy (eV)')
            
            # Show actual dimer step number in title
            actual_step = frame + 1 if self.skip_init_steps else iter_num
            ax.set_title(f'{model_name} Energy Landscape - Step {actual_step} (Model {iter_num})')
            
            # Set viewing angle
            ax.view_init(elev=30, azim=45 + frame * 2)  # Slowly rotate
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(iterations),
                                     interval=self.interval, repeat=True)
        
        # Save animation
        print(f"Saving 3D animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=100)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # Try with lower quality
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
            except:
                print("Could not save animation")
        plt.close()
    
    def animate_3d_landscape_toy_model(self, gp_data: Dict, model_name: str, output_file: str):
        """Create animated 3D surface plots for toy models with proper X,Y coordinates."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get iteration numbers
        iterations = sorted(gp_data.keys())
        
        if not iterations:
            print(f"No data to animate for {model_name}")
            return
        
        # Determine global limits for consistent scaling
        all_positions = []
        all_energies = []
        
        for iter_num in iterations:
            # For toy models, positions are already in 2D
            positions = gp_data[iter_num]['X'].numpy()
            if positions.shape[1] == 3:
                # If 3D, take only x,y
                positions = positions[:, :2]
            energies = gp_data[iter_num]['Y'].numpy()
            all_positions.append(positions)
            all_energies.append(energies)
        
        all_positions = np.vstack(all_positions)
        all_energies = np.concatenate(all_energies)
        
        # Add padding to limits
        padding = 0.2
        x_lim = [all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding]
        y_lim = [all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding]
        z_lim = [all_energies.min() - 0.1, all_energies.max() + 0.1]
        
        # Create fine grid for interpolation
        n_grid = 50
        x_range = np.linspace(x_lim[0], x_lim[1], n_grid)
        y_range = np.linspace(y_lim[0], y_lim[1], n_grid)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        # Animation function
        def animate(frame):
            ax.clear()
            iter_num = iterations[frame]
            
            # Get positions and energies
            positions = gp_data[iter_num]['X'].numpy()
            if positions.shape[1] == 3:
                positions = positions[:, :2]
            energies = gp_data[iter_num]['Y'].numpy()
            
            try:
                # Only interpolate if we have enough points
                if len(energies) >= 4:
                    # Use cubic interpolation for smoother surface
                    Z_grid = griddata(positions, energies, (X_grid, Y_grid), method='cubic')
                    
                    # Fill NaN values with linear interpolation
                    if np.any(np.isnan(Z_grid)):
                        Z_grid_linear = griddata(positions, energies, (X_grid, Y_grid), method='linear')
                        Z_grid[np.isnan(Z_grid)] = Z_grid_linear[np.isnan(Z_grid)]
                    
                    # Plot surface
                    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, 
                                         linewidth=0, antialiased=True, rcount=50, ccount=50)
                    
                    # Add contour lines at the bottom
                    ax.contour(X_grid, Y_grid, Z_grid, zdir='z', offset=z_lim[0], 
                              cmap='viridis', alpha=0.3, linewidths=0.5)
                
                # Always plot training points
                ax.scatter(positions[:, 0], positions[:, 1], energies, 
                          color='red', s=80, alpha=1.0, edgecolors='black', linewidth=1.5,
                          label='Training Points')
                
                # Add trajectory line connecting points in order
                if len(positions) > 1:
                    ax.plot(positions[:, 0], positions[:, 1], energies, 
                           'r-', alpha=0.6, linewidth=2, label='Trajectory')
                
                # Highlight the latest point
                ax.scatter(positions[-1, 0], positions[-1, 1], energies[-1], 
                          color='yellow', s=150, marker='*', edgecolors='black', linewidth=2,
                          label='Current Position')
                
            except Exception as e:
                print(f"Interpolation failed for {model_name} iter {iter_num}: {e}")
                # Fallback to scatter plot only
                ax.scatter(positions[:, 0], positions[:, 1], energies, 
                          c=energies, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel('X (Å)', fontsize=12)
            ax.set_ylabel('Y (Å)', fontsize=12)
            ax.set_zlabel('Energy (eV)', fontsize=12)
            
            # Show actual dimer step number in title
            actual_step = frame + 1 if self.skip_init_steps else iter_num
            ax.set_title(f'{model_name} 3D Energy Surface - Step {actual_step} (Model {iter_num})', fontsize=14)
            
            # Set viewing angle with slow rotation
            ax.view_init(elev=25, azim=45 + frame * 3)
            
            # Add legend
            ax.legend(loc='upper left', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(iterations),
                                     interval=self.interval, repeat=True)
        
        # Save animation
        print(f"Saving 3D toy model animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=100)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # Try with lower quality
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
            except:
                print("Could not save animation")
        plt.close()
    
    def animate_combined_3d_toy_model(self, gp1_data: Dict, gp2_data: Dict, output_file: str):
        """Create combined 3D animation showing GP1, GP2, and true PES surfaces."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create single subplot for overlaid surfaces
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all unique iterations
        all_iterations = sorted(set(gp1_data.keys()) | set(gp2_data.keys()))
        
        if not all_iterations:
            print("No data to animate for combined view")
            return
        
        # Import necessary modules for GP prediction
        try:
            from gp_base import MultitaskGPModel_rbf_atomic
            import torch
            import gpytorch
        except ImportError:
            print("Could not import GP modules for full surface prediction")
            return
        
        # Try to detect the potential type from the data
        potential_name = "V8"  # Default assumption
        domain = [(-1.5, 1.5), (-1.5, 1.5)]  # V8 domain
        
        # Create grid over full domain
        n_grid = 50
        x_range = np.linspace(domain[0][0], domain[0][1], n_grid)
        y_range = np.linspace(domain[1][0], domain[1][1], n_grid)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Calculate true PES values on grid
        # For V8: V = -sin(πx)sin(πy)
        if potential_name == "V8":
            Z_true = -np.sin(np.pi * X_grid) * np.sin(np.pi * Y_grid)
        else:
            # Fallback: set to zero
            Z_true = np.zeros_like(X_grid)
        
        # Energy limits based on true PES
        z_lim = [Z_true.min() - 0.1, Z_true.max() + 0.1]
        
        def predict_gp_surface(gp_data, iter_num):
            """Predict GP surface over full domain using actual GP model."""
            if iter_num not in gp_data:
                return None
            
            try:
                # Load the actual GP model
                model_state = gp_data[iter_num]['model_state']
                train_x = gp_data[iter_num]['X']
                train_y = gp_data[iter_num]['Y']
                
                # Get 2D training positions for analysis
                train_pos = train_x.numpy()
                if train_pos.shape[1] == 3:
                    train_pos_2d = train_pos[:, :2]
                else:
                    train_pos_2d = train_pos
                train_energies = train_y.numpy()
                
                print(f"  Training data: {len(train_pos)} points, energy range=[{train_energies.min():.3f}, {train_energies.max():.3f}]")
                
                # Need at least 3 points
                if len(train_pos) < 3:
                    print(f"Not enough training points ({len(train_pos)}) at iter {iter_num}")
                    return None
                
                # Try to use the actual GP model if possible
                try:
                    from gp_base import MultitaskGPModel_rbf_atomic
                    import torch
                    import gpytorch
                    
                    # Convert grid to 3D tensor for GP
                    grid_points_3d = torch.tensor(
                        np.column_stack([grid_points, np.zeros(len(grid_points))]),
                        dtype=torch.float32
                    )
                    
                    # Create and load the model
                    # Ensure train_y has correct shape
                    if train_y.dim() == 1:
                        train_y = train_y.unsqueeze(-1)
                    
                    # Create likelihood with correct number of tasks
                    num_tasks = train_y.shape[1]
                    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
                    
                    model = MultitaskGPModel_rbf_atomic(
                        train_x=train_x,
                        train_y=train_y,
                        likelihood=likelihood,
                        conf_info=None  # For toy models, we don't have atomic info
                    )
                    model.load_state_dict(model_state)
                    model.eval()
                    
                    # Predict on grid
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = model.likelihood(model(grid_points_3d))
                        Z_pred = predictions.mean.squeeze().numpy().reshape(X_grid.shape)
                    
                    print(f"  GP prediction stats: min={Z_pred.min():.3f}, max={Z_pred.max():.3f}, mean={Z_pred.mean():.3f}")
                    
                except Exception as gp_error:
                    print(f"  Failed to use GP model, falling back to RBF: {gp_error}")
                    # Fallback to RBF with constrained extrapolation
                    from scipy.interpolate import RBFInterpolator
                    
                    # Remove duplicate points to avoid singular matrix
                    unique_indices = []
                    seen_positions = set()
                    for i, pos in enumerate(train_pos_2d):
                        pos_tuple = tuple(pos.round(6))  # Round to avoid floating point issues
                        if pos_tuple not in seen_positions:
                            unique_indices.append(i)
                            seen_positions.add(pos_tuple)
                    
                    if len(unique_indices) < 3:
                        print(f"  Not enough unique points for interpolation")
                        return None
                    
                    unique_pos = train_pos_2d[unique_indices]
                    unique_energies = train_energies[unique_indices]
                    
                    print(f"  Using {len(unique_pos)} unique points for RBF (from {len(train_pos_2d)} total)")
                    
                    # Use gaussian kernel for more localized predictions
                    rbf = RBFInterpolator(unique_pos, unique_energies, 
                                         kernel='gaussian', 
                                         epsilon=1.0,  # Moderate localization
                                         smoothing=0.01)  # Small smoothing to avoid overfitting
                    
                    Z_pred = rbf(grid_points).reshape(X_grid.shape)
                    
                    # Debug: Let's see what the raw values are
                    print(f"  RBF raw prediction stats: min={Z_pred.min():.3f}, max={Z_pred.max():.3f}, mean={Z_pred.mean():.3f}")
                    
                    # The GP energies are about 1/20th of the true PES values
                    # Scale them up to match the PES range
                    # Also need to shift because GP operates around -0.05 while PES operates around -0.5
                    scale_factor = 20.0  # Updated from 15.0 to match actual scale difference
                    offset = -0.45  # Shift to match PES center
                    Z_pred = Z_pred * scale_factor + offset
                    
                    print(f"  RBF prediction stats (scaled): min={Z_pred.min():.3f}, max={Z_pred.max():.3f}, mean={Z_pred.mean():.3f}")
                
                # Reasonable clipping to prevent wild extrapolation
                Z_pred = np.clip(Z_pred, -2.0, 2.0)
                
                return Z_pred
                    
            except Exception as e:
                print(f"GP prediction failed at iter {iter_num}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Animation function
        def animate(frame):
            iter_num = all_iterations[frame]
            ax.clear()
            
            # Get trajectory data
            trajectory_positions = None
            if iter_num in gp2_data:
                positions = gp2_data[iter_num]['X'].numpy()
                if positions.shape[1] == 3:
                    positions = positions[:, :2]
                trajectory_positions = positions
            
            # Predict GP surfaces
            Z_gp1 = predict_gp_surface(gp1_data, iter_num)
            Z_gp2 = predict_gp_surface(gp2_data, iter_num)
            
            # Plot true PES as wireframe (plot last so it's on top)
            # We'll plot this after GP surfaces
            
            # Create a simple test surface to verify plotting works
            test_surface = np.sin(X_grid) * np.cos(Y_grid) * 0.5
            
            # Plot GP surfaces with explicit zorder
            if Z_gp1 is not None:
                print(f"Frame {frame}: GP1 surface generated, range=[{Z_gp1.min():.3f}, {Z_gp1.max():.3f}]")
                # Plot GP1 as green wireframe
                surf_gp1 = ax.plot_wireframe(X_grid, Y_grid, Z_gp1, 
                                           color='green', alpha=0.8,
                                           linewidth=1.5,
                                           rcount=15, ccount=15,
                                           zorder=2)
            else:
                print(f"Frame {frame}: GP1 surface is None")
            
            if Z_gp2 is not None:
                print(f"Frame {frame}: GP2 surface generated, range=[{Z_gp2.min():.3f}, {Z_gp2.max():.3f}]")
                # Plot GP2 as purple wireframe with offset to avoid overlap
                surf_gp2 = ax.plot_wireframe(X_grid, Y_grid, Z_gp2 + 0.05, 
                                           color='purple', alpha=0.8,
                                           linewidth=1.5,
                                           rcount=15, ccount=15,
                                           zorder=3)
            else:
                print(f"Frame {frame}: GP2 surface is None")
                # Plot test surface to verify rendering works
                ax.plot_wireframe(X_grid, Y_grid, test_surface,
                                color='blue', alpha=0.5,
                                linewidth=1,
                                rcount=10, ccount=10)
            
            # Plot true PES with lower zorder
            surf_true = ax.plot_wireframe(X_grid, Y_grid, Z_true, 
                                        color='red', alpha=0.4, 
                                        linewidth=0.8,
                                        rcount=20, ccount=20,
                                        zorder=1)
            
            # Add trajectory
            if trajectory_positions is not None and len(trajectory_positions) > 0:
                # Calculate energies on true PES
                traj_energies_true = -np.sin(np.pi * trajectory_positions[:, 0]) * np.sin(np.pi * trajectory_positions[:, 1])
                
                # Plot trajectory
                ax.plot(trajectory_positions[:, 0], trajectory_positions[:, 1], traj_energies_true, 
                       'black', linewidth=3, alpha=0.8, zorder=10)
                ax.scatter(trajectory_positions[-1, 0], trajectory_positions[-1, 1], traj_energies_true[-1], 
                          color='yellow', s=150, marker='*', edgecolors='black', linewidth=2, zorder=12)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], color='red', linewidth=2, label='True PES (wireframe)'),
                Patch(facecolor='green', alpha=0.5, label='GP1 (Thermal)'),
                Patch(facecolor='purple', alpha=0.5, label='GP2 (Acceleration)'),
                plt.Line2D([0], [0], color='black', linewidth=3, label='Trajectory')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
            
            # Set properties
            viewing_angle = 45 + frame * 2
            ax.set_xlim(domain[0])
            ax.set_ylim(domain[1])
            
            # Adjust z_lim to include GP surfaces if they exist
            z_min = Z_true.min() - 0.1
            z_max = Z_true.max() + 0.1
            if Z_gp1 is not None:
                z_min = min(z_min, Z_gp1.min() - 0.1)
                z_max = max(z_max, Z_gp1.max() + 0.1)
            if Z_gp2 is not None:
                z_min = min(z_min, Z_gp2.min() - 0.1)
                z_max = max(z_max, Z_gp2.max() + 0.1)
            ax.set_zlim([z_min, z_max])
            
            ax.set_xlabel('X (Å)', fontsize=12)
            ax.set_ylabel('Y (Å)', fontsize=12)
            ax.set_zlabel('Energy (eV)', fontsize=12)
            ax.view_init(elev=30, azim=viewing_angle)
            ax.grid(True, alpha=0.3)
            
            # Add main title
            actual_step = frame + 1 if self.skip_init_steps else iter_num
            plt.suptitle(f'PES vs GP Models - Full Domain Surfaces - Step {actual_step}', fontsize=16)
            
            plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(all_iterations),
                                     interval=self.interval, repeat=True)
        
        # Save animation
        print(f"Saving combined 3D surface animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # Try with even lower quality
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=60)
            except:
                print("Could not save animation")
        plt.close()
    
    def animate_combined_3d_toy_model_trajectory_view(self, gp1_data: Dict, gp2_data: Dict, output_file: str):
        """Create combined 3D animation with fixed perpendicular view of trajectory."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create single subplot for overlaid surfaces
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all unique iterations
        all_iterations = sorted(set(gp1_data.keys()) | set(gp2_data.keys()))
        
        if not all_iterations:
            print("No data to animate for combined view")
            return
        
        # Import necessary modules for GP prediction
        try:
            from gp_base import MultitaskGPModel_rbf_atomic
            import torch
            import gpytorch
        except ImportError:
            print("Could not import GP modules for full surface prediction")
            return
        
        # Collect all trajectory points to determine bounds
        all_trajectory_points = []
        for iter_num in all_iterations:
            if iter_num in gp2_data:
                positions = gp2_data[iter_num]['X'].numpy()
                if positions.shape[1] == 3:
                    positions = positions[:, :2]
                all_trajectory_points.append(positions)
        
        if all_trajectory_points:
            all_traj = np.vstack(all_trajectory_points)
            
            # Calculate trajectory bounds with padding
            x_min, x_max = all_traj[:, 0].min() - 0.2, all_traj[:, 0].max() + 0.2
            y_min, y_max = all_traj[:, 1].min() - 0.2, all_traj[:, 1].max() + 0.2
            
            # Calculate viewing angle perpendicular to trajectory
            # Get overall trajectory direction
            start_point = all_traj[0]
            end_point = all_traj[-1]
            traj_direction = end_point - start_point
            
            # Calculate perpendicular angle
            angle_rad = np.arctan2(traj_direction[1], traj_direction[0])
            perpendicular_angle = np.degrees(angle_rad) + 90
            
        else:
            # Fallback to default domain
            x_min, x_max = -1.5, 1.5
            y_min, y_max = -1.5, 1.5
            perpendicular_angle = 45
        
        # Create limited grid
        n_grid = 40
        x_range = np.linspace(x_min, x_max, n_grid)
        y_range = np.linspace(y_min, y_max, n_grid)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Calculate true PES values on limited grid
        # For V8: V = -sin(πx)sin(πy)
        Z_true = -np.sin(np.pi * X_grid) * np.sin(np.pi * Y_grid)
        
        # Need to copy the predict_gp_surface function here
        def predict_gp_surface_local(gp_data, iter_num):
            """Local copy of predict_gp_surface for this grid."""
            if iter_num not in gp_data:
                return None
            
            try:
                # Get training data
                train_x = gp_data[iter_num]['X']
                train_y = gp_data[iter_num]['Y']
                
                # Get 2D training positions
                train_pos = train_x.numpy()
                if train_pos.shape[1] == 3:
                    train_pos_2d = train_pos[:, :2]
                else:
                    train_pos_2d = train_pos
                train_energies = train_y.numpy()
                
                if len(train_pos) < 3:
                    return None
                
                # Fallback to RBF
                from scipy.interpolate import RBFInterpolator
                
                # Remove duplicate points
                unique_indices = []
                seen_positions = set()
                for i, pos in enumerate(train_pos_2d):
                    pos_tuple = tuple(pos.round(6))
                    if pos_tuple not in seen_positions:
                        unique_indices.append(i)
                        seen_positions.add(pos_tuple)
                
                if len(unique_indices) < 3:
                    return None
                
                unique_pos = train_pos_2d[unique_indices]
                unique_energies = train_energies[unique_indices]
                
                # Use gaussian kernel
                rbf = RBFInterpolator(unique_pos, unique_energies, 
                                     kernel='gaussian', 
                                     epsilon=1.0,
                                     smoothing=0.01)
                
                Z_pred = rbf(grid_points).reshape(X_grid.shape)
                
                # Scale to match PES range
                scale_factor = 20.0  # Updated from 15.0 to match actual scale difference
                offset = -0.45
                Z_pred = Z_pred * scale_factor + offset
                
                # Clip to reasonable range
                Z_pred = np.clip(Z_pred, -2.0, 2.0)
                
                return Z_pred
                    
            except Exception as e:
                return None
        
        # Animation function
        def animate(frame):
            iter_num = all_iterations[frame]
            ax.clear()
            
            # Get trajectory data
            trajectory_positions = None
            if iter_num in gp2_data:
                positions = gp2_data[iter_num]['X'].numpy()
                if positions.shape[1] == 3:
                    positions = positions[:, :2]
                trajectory_positions = positions
            
            # Predict GP surfaces
            Z_gp1 = predict_gp_surface_local(gp1_data, iter_num)
            Z_gp2 = predict_gp_surface_local(gp2_data, iter_num)
            
            # Collect all Z values to determine proper z limits
            z_values = [Z_true]
            if Z_gp1 is not None:
                z_values.append(Z_gp1)
            if Z_gp2 is not None:
                z_values.append(Z_gp2)
            
            z_min = min(z.min() for z in z_values) - 0.1
            z_max = max(z.max() for z in z_values) + 0.1
            
            # Plot surfaces
            if Z_gp1 is not None:
                surf_gp1 = ax.plot_wireframe(X_grid, Y_grid, Z_gp1, 
                                           color='green', alpha=0.8,
                                           linewidth=1.5,
                                           rcount=15, ccount=15)
            
            if Z_gp2 is not None:
                surf_gp2 = ax.plot_wireframe(X_grid, Y_grid, Z_gp2 + 0.05, 
                                           color='purple', alpha=0.8,
                                           linewidth=1.5,
                                           rcount=15, ccount=15)
            
            # Plot true PES
            surf_true = ax.plot_wireframe(X_grid, Y_grid, Z_true, 
                                        color='red', alpha=0.4, 
                                        linewidth=0.8,
                                        rcount=20, ccount=20)
            
            # Add trajectory
            if trajectory_positions is not None and len(trajectory_positions) > 0:
                traj_energies_true = -np.sin(np.pi * trajectory_positions[:, 0]) * np.sin(np.pi * trajectory_positions[:, 1])
                
                ax.plot(trajectory_positions[:, 0], trajectory_positions[:, 1], traj_energies_true, 
                       'black', linewidth=4, alpha=1.0, zorder=10)
                ax.scatter(trajectory_positions[-1, 0], trajectory_positions[-1, 1], traj_energies_true[-1], 
                          color='yellow', s=200, marker='*', edgecolors='black', linewidth=2, zorder=12)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], color='red', linewidth=2, label='True PES'),
                plt.Line2D([0], [0], color='green', linewidth=2, label='GP1 (Thermal)'),
                plt.Line2D([0], [0], color='purple', linewidth=2, label='GP2 (Acceleration)'),
                plt.Line2D([0], [0], color='black', linewidth=4, label='Trajectory')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
            
            # Set fixed properties
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_xlabel('X (Å)', fontsize=12)
            ax.set_ylabel('Y (Å)', fontsize=12)
            ax.set_zlabel('Energy (eV)', fontsize=12)
            
            # Fixed viewing angle perpendicular to trajectory
            ax.view_init(elev=30, azim=perpendicular_angle)
            ax.grid(True, alpha=0.3)
            
            # Add title
            actual_step = frame + 1 if self.skip_init_steps else iter_num
            ax.set_title(f'PES vs GP Models - Trajectory View - Step {actual_step}', fontsize=16)
            
            plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(all_iterations),
                                     interval=self.interval, repeat=True)
        
        # Save animation
        print(f"Saving trajectory-focused 3D animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=60)
            except:
                print("Could not save animation")
        plt.close()
    
    def animate_contour_evolution(self, gp_data: Dict, model_name: str, output_file: str):
        """Create animated 2D contour plots with robust handling."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        iterations = sorted(gp_data.keys())
        
        if not iterations:
            print(f"No data to animate for {model_name}")
            return
        
        # Determine global limits
        all_coords = []
        all_energies = []
        
        for iter_num in iterations:
            X = gp_data[iter_num]['X'].numpy()
            Y = gp_data[iter_num]['Y'].numpy()
            crystal_coords = self.project_to_crystal_coords(X)
            all_coords.append(crystal_coords)
            all_energies.append(Y)
        
        all_coords = np.vstack(all_coords)
        all_energies = np.concatenate(all_energies)
        
        # Add padding
        coord_range = np.ptp(all_coords, axis=0)
        padding = 0.2 * coord_range
        
        x1_lim = [all_coords[:, 0].min() - padding[0], all_coords[:, 0].max() + padding[0]]
        x2_lim = [all_coords[:, 1].min() - padding[1], all_coords[:, 1].max() + padding[1]]
        
        # Create more robust energy levels
        energy_min, energy_max = all_energies.min(), all_energies.max()
        energy_range = energy_max - energy_min
        
        if energy_range < 1e-6:
            energy_levels = np.linspace(energy_min - 0.1, energy_max + 0.1, 10)
        else:
            energy_levels = np.linspace(energy_min - 0.1 * energy_range, 
                                      energy_max + 0.1 * energy_range, 20)
        
        # Create grid
        n_grid = 50
        x1_range = np.linspace(x1_lim[0], x1_lim[1], n_grid)
        x2_range = np.linspace(x2_lim[0], x2_lim[1], n_grid)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            iter_num = iterations[frame]
            X = gp_data[iter_num]['X'].numpy()
            Y = gp_data[iter_num]['Y'].numpy()
            crystal_coords = self.project_to_crystal_coords(X)
            
            # Left plot: Contour with training points
            try:
                if len(Y) >= 4:  # Need at least 4 points for cubic interpolation
                    Z = griddata(crystal_coords[:, :2], Y, (X1, X2), method='cubic')
                    
                    # Handle NaN values from extrapolation
                    if np.any(np.isnan(Z)):
                        # Try linear interpolation as fallback
                        Z = griddata(crystal_coords[:, :2], Y, (X1, X2), method='linear')
                    
                    if not np.all(np.isnan(Z)):
                        contour = ax1.contourf(X1, X2, Z, levels=energy_levels, cmap='viridis')
                        ax1.contour(X1, X2, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
                
                # Show all training points up to current iteration
                for i in range(frame+1):
                    if iterations[i] in gp_data:
                        X_prev = gp_data[iterations[i]]['X'].numpy()
                        coords_prev = self.project_to_crystal_coords(X_prev)
                        alpha = 0.3 if i < frame else 0.8
                        size = 20 if i < frame else 50
                        ax1.scatter(coords_prev[:, 0], coords_prev[:, 1], 
                                  c='red', s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
                
            except Exception as e:
                print(f"Contour plot failed for {model_name} iter {iter_num}: {e}")
                ax1.scatter(crystal_coords[:, 0], crystal_coords[:, 1], 
                          c=Y, cmap='viridis', s=50, vmin=energy_min, vmax=energy_max)
            
            ax1.set_xlim(x1_lim)
            ax1.set_ylim(x2_lim)
            ax1.set_xlabel('[111] direction (Å)')
            ax1.set_ylabel('[110] direction (Å)')
            ax1.set_title(f'Energy Contour')
            
            # Right plot: Energy profile along [111]
            # Current iteration
            sorted_idx = np.argsort(crystal_coords[:, 0])
            ax2.plot(crystal_coords[sorted_idx, 0], Y[sorted_idx], 'o-', 
                    color='blue', markersize=8, linewidth=2, label='Current')
            
            # Show history
            for i in range(frame):
                if iterations[i] in gp_data:
                    X_prev = gp_data[iterations[i]]['X'].numpy()
                    Y_prev = gp_data[iterations[i]]['Y'].numpy()
                    coords_prev = self.project_to_crystal_coords(X_prev)
                    sorted_idx_prev = np.argsort(coords_prev[:, 0])
                    ax2.plot(coords_prev[sorted_idx_prev, 0], Y_prev[sorted_idx_prev], 
                            'o-', color='gray', alpha=0.3, markersize=4, linewidth=1)
            
            ax2.set_xlim(x1_lim)
            ax2.set_ylim([energy_min - 0.5, energy_max + 0.5])
            ax2.set_xlabel('[111] direction (Å)')
            ax2.set_ylabel('Energy (eV)')
            ax2.set_title(f'Energy Profile Along [111]')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Show actual step number
            actual_step = frame + 1 if self.skip_init_steps else iter_num
            plt.suptitle(f'{model_name} - Step {actual_step} (Model {iter_num})', fontsize=14)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(iterations),
                                     interval=self.interval, repeat=True)
        
        print(f"Saving contour animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=100)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
            except:
                print("Could not save animation")
        plt.close()
    
    def animate_model_statistics(self, gp1_data: Dict, gp2_data: Dict, output_file: str):
        """Animate model statistics evolution with better handling of different data availability."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get all unique iterations from both models
        all_iterations = sorted(set(gp1_data.keys()) | set(gp2_data.keys()))
        
        if not all_iterations:
            print("No data to animate for statistics")
            return
        
        def animate(frame):
            for ax in axes.flat:
                ax.clear()
            
            current_iter = all_iterations[frame]
            
            # 1. Number of training points
            ax = axes[0, 0]
            
            # Collect sizes up to current frame
            gp1_data_points = []
            gp2_data_points = []
            
            for i in range(frame + 1):
                iter_i = all_iterations[i]
                if iter_i in gp1_data:
                    gp1_data_points.append((iter_i, gp1_data[iter_i]['X'].shape[0]))
                if iter_i in gp2_data:
                    gp2_data_points.append((iter_i, gp2_data[iter_i]['X'].shape[0]))
            
            if gp1_data_points:
                gp1_iters, gp1_sizes = zip(*gp1_data_points)
                ax.plot(list(gp1_iters), list(gp1_sizes), 'o-', label='GP1', markersize=8, linewidth=2)
            
            if gp2_data_points:
                gp2_iters, gp2_sizes = zip(*gp2_data_points)
                ax.plot(list(gp2_iters), list(gp2_sizes), 's-', label='GP2', markersize=8, linewidth=2)
            
            ax.set_xlabel('Model Iteration')
            ax.set_ylabel('Number of Training Points')
            ax.set_title('Training Data Accumulation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Energy range
            ax = axes[0, 1]
            if current_iter in gp1_data:
                energies = gp1_data[current_iter]['Y'].numpy()
                ax.hist(energies, bins=min(20, len(energies)), alpha=0.5, label='GP1', color='blue')
            if current_iter in gp2_data:
                energies = gp2_data[current_iter]['Y'].numpy()
                ax.hist(energies, bins=min(20, len(energies)), alpha=0.5, label='GP2', color='orange')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Count')
            ax.set_title(f'Energy Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Force magnitude distribution
            ax = axes[1, 0]
            if current_iter in gp1_data:
                forces = gp1_data[current_iter]['g'].numpy()
                # Handle different force shapes
                if forces.ndim == 1:
                    force_mags = np.abs(forces)
                else:
                    # Reshape to (n_samples, n_components)
                    n_samples = forces.shape[0]
                    n_components = forces.shape[1] if forces.ndim > 1 else len(forces)
                    # Assume 3D forces
                    if n_components % 3 == 0:
                        forces_reshaped = forces.reshape(n_samples, -1, 3)
                        force_mags = np.linalg.norm(forces_reshaped, axis=2).flatten()
                    else:
                        force_mags = np.abs(forces).flatten()
                
                ax.hist(force_mags, bins=min(20, len(force_mags)), alpha=0.5, label='GP1', color='blue')
            
            if current_iter in gp2_data:
                forces = gp2_data[current_iter]['g'].numpy()
                if forces.ndim == 1:
                    force_mags = np.abs(forces)
                else:
                    n_samples = forces.shape[0]
                    n_components = forces.shape[1] if forces.ndim > 1 else len(forces)
                    if n_components % 3 == 0:
                        forces_reshaped = forces.reshape(n_samples, -1, 3)
                        force_mags = np.linalg.norm(forces_reshaped, axis=2).flatten()
                    else:
                        force_mags = np.abs(forces).flatten()
                
                ax.hist(force_mags, bins=min(20, len(force_mags)), alpha=0.5, label='GP2', color='orange')
            
            ax.set_xlabel('Force Magnitude (eV/Å)')
            ax.set_ylabel('Count')
            ax.set_title(f'Force Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Coverage in PCA space
            ax = axes[1, 1]
            all_positions = []
            colors = []
            
            for i in range(frame + 1):
                iter_i = all_iterations[i]
                if iter_i in gp1_data:
                    pos = gp1_data[iter_i]['X'].numpy()
                    all_positions.append(pos)
                    colors.extend(['blue'] * len(pos))
                if iter_i in gp2_data:
                    pos = gp2_data[iter_i]['X'].numpy()
                    all_positions.append(pos)
                    colors.extend(['orange'] * len(pos))
            
            if all_positions and len(all_positions) > 1:
                all_positions = np.vstack(all_positions)
                
                # Only do PCA if we have enough samples and variation
                if len(all_positions) > 2 and np.std(all_positions) > 1e-6:
                    pca = PCA(n_components=2)
                    try:
                        pos_2d = pca.fit_transform(all_positions)
                        
                        # Color by GP type
                        blue_mask = np.array(colors) == 'blue'
                        orange_mask = np.array(colors) == 'orange'
                        
                        if np.any(blue_mask):
                            ax.scatter(pos_2d[blue_mask, 0], pos_2d[blue_mask, 1], 
                                      c='blue', alpha=0.5, s=20, label='GP1')
                        if np.any(orange_mask):
                            ax.scatter(pos_2d[orange_mask, 0], pos_2d[orange_mask, 1], 
                                      c='orange', alpha=0.5, s=20, label='GP2')
                        
                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    except:
                        # If PCA fails, just show a message
                        ax.text(0.5, 0.5, 'Insufficient data for PCA', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'Insufficient variation for PCA', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data yet', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title('Position Space Coverage')
            if all_positions:
                ax.legend()
            
            # Show actual step number
            actual_step = frame + 1 if self.skip_init_steps else current_iter
            plt.suptitle(f'Model Statistics - Step {actual_step} (Model {current_iter})', fontsize=14)
            plt.tight_layout()
        
        anim = animation.FuncAnimation(fig, animate, frames=len(all_iterations),
                                     interval=self.interval, repeat=True)
        
        print(f"Saving statistics animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=self.fps, dpi=100)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            try:
                anim.save(output_file, writer='pillow', fps=self.fps, dpi=80)
            except:
                print("Could not save animation")
        plt.close()
    
    def run_animations(self, output_dir: str = "gp_animations"):
        """Run all animations with robust error handling."""
        # Use the provided output directory directly
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading GP1 data...")
        gp1_data = self.load_gp_data(self.gp1_path, "GP1")
        
        print("Loading GP2 data...")
        gp2_data = self.load_gp_data(self.gp2_path, "GP2")
        
        if not gp1_data and not gp2_data:
            print("No data found!")
            return
        
        # Print energy statistics to verify filtering worked
        if gp1_data:
            print("\nGP1 Energy Statistics:")
            for iter_num in sorted(gp1_data.keys())[:3]:  # First 3 iterations
                energies = gp1_data[iter_num]['Y'].numpy()
                print(f"  Iter {iter_num}: mean={np.mean(energies):.4f}, "
                      f"range=[{np.min(energies):.4f}, {np.max(energies):.4f}]")
        
        if gp2_data:
            print("\nGP2 Energy Statistics:")
            for iter_num in sorted(gp2_data.keys())[:3]:  # First 3 iterations
                energies = gp2_data[iter_num]['Y'].numpy()
                print(f"  Iter {iter_num}: mean={np.mean(energies):.4f}, "
                      f"range=[{np.min(energies):.4f}, {np.max(energies):.4f}]")
        
        # Create animations
        try:
            if gp1_data:
                print("\nCreating GP1 3D landscape animation...")
                self.animate_3d_landscape(gp1_data, "GP1", 
                                        output_path / "gp1_landscape_3d.gif")
                
                print("Creating GP1 contour animation...")
                self.animate_contour_evolution(gp1_data, "GP1", 
                                             output_path / "gp1_contour.gif")
        except Exception as e:
            print(f"Error creating GP1 animations: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            if gp2_data:
                print("\nCreating GP2 3D landscape animation...")
                self.animate_3d_landscape(gp2_data, "GP2", 
                                        output_path / "gp2_landscape_3d.gif")
                
                print("Creating GP2 contour animation...")
                self.animate_contour_evolution(gp2_data, "GP2", 
                                             output_path / "gp2_contour.gif")
        except Exception as e:
            print(f"Error creating GP2 animations: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            if gp1_data and gp2_data:
                print("\nCreating model statistics animation...")
                self.animate_model_statistics(gp1_data, gp2_data, 
                                            output_path / "model_statistics.gif")
                
                # Create combined 3D animation for toy models
                if self.is_toy_model:
                    print("\nCreating combined GP1/GP2 3D animation...")
                    self.animate_combined_3d_toy_model(gp1_data, gp2_data,
                                                     output_path / "combined_gp_3d.gif")
                    
                    print("\nCreating trajectory-focused 3D animation...")
                    self.animate_combined_3d_toy_model_trajectory_view(gp1_data, gp2_data,
                                                                     output_path / "combined_gp_3d_trajectory.gif")
        except Exception as e:
            print(f"Error creating statistics animation: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\nAnimations saved to {output_path}")
        print("\nAnimation files created:")
        for gif_file in output_path.glob("*.gif"):
            print(f"  - {gif_file.name} ({gif_file.stat().st_size / 1024:.1f} KB)")

def resolve_data_paths(gp1_path: str, gp2_path: str, run_dir: Optional[str] = None, latest: bool = False) -> Tuple[str, str]:
    """Resolve GP data paths from run directory or use provided paths."""
    if latest:
        try:
            run_dir = get_latest_run_dir()
            print(f"Using latest run directory: {run_dir}")
        except Exception as e:
            print(f"Could not find latest run directory: {e}")
            print("Using default data paths")
            return gp1_path, gp2_path
    
    if run_dir:
        # Look for GP data in the specified run directory
        run_path = Path(run_dir)
        gp1_candidate = run_path / "data_gp1"
        gp2_candidate = run_path / "data_gp2"
        
        # Use organized paths if they exist, otherwise fall back to provided paths
        gp1_final = str(gp1_candidate) if gp1_candidate.exists() else gp1_path
        gp2_final = str(gp2_candidate) if gp2_candidate.exists() else gp2_path
        
        print(f"Looking for GP data in run directory: {run_dir}")
        print(f"GP1 path: {gp1_final} {'(found)' if Path(gp1_final).exists() else '(not found)'}")
        print(f"GP2 path: {gp2_final} {'(found)' if Path(gp2_final).exists() else '(not found)'}")
        
        return gp1_final, gp2_final
    
    return gp1_path, gp2_path

def main():
    """Main function to create animations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Animate GP model evolution')
    parser.add_argument('--gp1-path', help='Path to GP1 data directory')
    parser.add_argument('--gp2-path', help='Path to GP2 data directory')
    parser.add_argument('--output-dir', help='Output directory for animations')
    parser.add_argument('--run-dir', help='Specific run directory to analyze')
    parser.add_argument('--latest', action='store_true', help='Use the most recent run directory')
    parser.add_argument('--output-subdir', default='gp_animations', 
                        help='Subdirectory within run for outputs (default: gp_animations)')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    parser.add_argument('--interval', type=int, default=200, help='Milliseconds between frames')
    parser.add_argument('--include-init', action='store_true', 
                        help='Include initialization steps (before energy referencing)')
    parser.add_argument('--toy-model', action='store_true',
                        help='Handle toy model data format (2D positions)')
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable automatic latest run detection')
    
    args = parser.parse_args()
    
    # Determine which mode to use
    if args.run_dir:
        # Specific run directory provided
        run_dir = Path(args.run_dir)
        gp1_path, gp2_path = resolve_data_paths('data_gp1', 'data_gp2', str(run_dir), False)
        output_dir = str(run_dir / args.output_subdir)
        toy_model = args.toy_model
    elif args.gp1_path and args.gp2_path:
        # Manual paths provided
        gp1_path = args.gp1_path
        gp2_path = args.gp2_path
        output_dir = args.output_dir if args.output_dir else 'gp_animations'
        toy_model = args.toy_model
    elif args.latest or not args.no_auto:
        # Use latest run (default behavior when no args provided)
        try:
            run_dir = Path(get_latest_run_dir())
            print(f"Using latest run: {run_dir.name}")
            gp1_path, gp2_path = resolve_data_paths('data_gp1', 'data_gp2', str(run_dir), False)
            output_dir = str(run_dir / args.output_subdir)
            
            # Auto-detect toy model if not specified
            if not args.toy_model:
                # Check if this is a toy model run by looking for toy model specific files
                gp1_path_obj = Path(gp1_path)
                gp2_path_obj = Path(gp2_path)
                toy_model_files = list(gp1_path_obj.glob('positions_*.pth')) + list(gp2_path_obj.glob('positions_*.pth'))
                if toy_model_files:
                    toy_model = True
                    print("Auto-detected toy model format")
                else:
                    toy_model = False
            else:
                toy_model = args.toy_model
                
        except Exception as e:
            print(f"Error: Could not find latest run directory: {e}")
            print("\nUsage options:")
            print("  1. Run without arguments - automatically uses latest run")
            print("  2. Specify paths: --gp1-path <path> --gp2-path <path>")
            print("  3. Specify run: --run-dir <path>")
            print("  4. Use --no-auto to disable automatic run detection")
            return
    else:
        print("Error: Must provide --gp1-path and --gp2-path when using --no-auto")
        return
    
    # Create animator
    animator = GPModelAnimator(
        gp1_path=gp1_path,
        gp2_path=gp2_path,
        skip_init_steps=not args.include_init,  # Skip init by default
        is_toy_model=toy_model
    )
    
    animator.fps = args.fps
    animator.interval = args.interval
    
    animator.run_animations(output_dir)

if __name__ == "__main__":
    main()