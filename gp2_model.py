import os
import numpy as np
from gp_base import GPSurrogateModel
import pickle
import numpy.typing as npt
from typing import List, Optional, Tuple, Any, Dict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GP2(GPSurrogateModel):
    """Gaussian Process model for saddle point search guidance.
    
    This class extends GPSurrogateModel to provide a surrogate model
    that guides the dimer method in finding saddle points. It maintains
    a training dataset of positions, energies, and forces, and updates
    the model as new points are explored.
    
    Attributes:
        training_data (List): List containing:
            - positions: Array of explored configurations
            - energies: Array of energy values
            - forces: Array of force vectors
        path (Optional[str]): Path for saving model data
        atomic_info (Optional[Dict]): Atomic information for GP2
    """
    def __init__(
        self,
        training_data: List[Any],
        path: Optional[str] = None,
        atomic_info: Optional[Dict] = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize GP2 with initial training data.
        
        Args:
            training_data: Initial [positions, energies, forces] dataset
                - positions: Full system coordinates (N_samples, 3*N_atoms)
                - energies: Energy values (N_samples,)
                - forces: Full system forces (N_samples, 3*N_atoms)
            path: Optional path for saving model data
            atomic_info: Atomic information for GP2
            use_gpu: Use GPU acceleration for training and inference
        """
        super().__init__(atomic_info=atomic_info, use_gpu=use_gpu)

        self.energy_reference = None  # Will be set by Walker when first energy is calculated

        # Validate training data dimensions
        positions, energies, forces = training_data
        n_moving = len(atomic_info['moving_indices'])
        n_atoms_total = positions.shape[1] // 3
        expected_dim = n_moving * 3
        
        # Check dimensions - should be moving atoms only
        if positions.shape[1] != expected_dim:
            raise ValueError(
                f"Position dimension mismatch. Expected {expected_dim} "
                f"(3*{n_moving} moving atoms), got {positions.shape[1]}"
            )
        
        if forces.shape[1] != expected_dim:
            raise ValueError(
                f"Force dimension mismatch. Expected {expected_dim} "
                f"(3*{n_moving} moving atoms), got {forces.shape[1]}"
            )
        
        self.training_data = [positions, energies, forces]
        self.path = path
        self.n_atoms_total = n_atoms_total
        self.n_moving = n_moving


    def update(
                self,
                datapoint: List[Any],
                thermal_noise: Optional[Tuple[float, float]] = None,
                save_curvature: bool = False
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Update model with new datapoint and make predictions.
            
            Args:
                datapoint: Either:
                    - List of [positions, energies, forces] for moving atoms only
                    - Array of positions for moving atoms only for prediction
                    
            Returns:
                Tuple of (predicted_energy, predicted_forces) for moving atoms only
            """
            if isinstance(datapoint, list) and len(datapoint) == 3:
                # Extract new data - should already be moving-atoms-only
                new_positions, new_energies, new_forces = datapoint
                
                # Verify dimensions
                expected_dim = self.n_moving * 3
                if new_positions.shape[1] != expected_dim or new_forces.shape[1] != expected_dim:
                    raise ValueError(f"New data has wrong dimensions. Expected {expected_dim} for moving atoms only")
                
                # Update training data
                self.training_data[0] = np.vstack([self.training_data[0], new_positions])
                self.training_data[1] = np.concatenate([self.training_data[1], new_energies])
                self.training_data[2] = np.vstack([self.training_data[2], new_forces])
                
                # Train with updated data
                self.train(self.training_data, thermal_noise=thermal_noise, path=self.path, model_name="GP2")
            
            # Make prediction
            if isinstance(datapoint, list):
                predict_point = datapoint[0]
            else:
                predict_point = datapoint
                
            if not isinstance(predict_point, np.ndarray):
                predict_point = np.array(predict_point)
                
            if predict_point.ndim == 1:
                predict_point = predict_point.reshape(1, -1)
            
            # Verify prediction point dimensions
            if predict_point.shape[1] != self.n_moving * 3:
                raise ValueError(
                    f"Prediction point has wrong dimension. Expected {self.n_moving * 3} "
                    f"for {self.n_moving} moving atoms, got {predict_point.shape[1]}"
                )
            
            # Make prediction - returns forces for moving atoms only
            predicted_energy, predicted_forces, variance_energy, variance_forces = self.predict(predict_point)
            
            if save_curvature:
                self.save_metadata()
                
            return predicted_energy, predicted_forces

    def get_curvature(
            self,
            tol: float = 1e-6
        ) -> Tuple[str, float, npt.NDArray[np.float64]]:
        """Calculate curvature at the latest training point.
        
        Args:
            tol: Tolerance for considering eigenvalues as zero
            
        Returns:
            Tuple containing:
                - curvature_type: "positive", "negative", or "mixed"
                - curvature_value: Sum of eigenvalues
                - eigenvalues: Array of Hessian eigenvalues
        """
        eigenvalues = np.linalg.eigvals(self.calculate_hessian(self.training_data[0][-1]))

        # Determine curvature type
        positive = np.all(eigenvalues > -tol)
        negative = np.all(eigenvalues < tol)

        # Calculate the curvature as the sum of eigenvalues
        curvature_value = np.sum(eigenvalues)

        # Classify curvature type
        if positive and not negative:
            curvature_type = "positive"
        elif negative and not positive:
            curvature_type = "negative"
        else:
            curvature_type = "mixed"

        return curvature_type, curvature_value, eigenvalues

    def save_metadata(
            self,
            save_path: str = 'gp2_curvature.pkl'
        ) -> None:
        """Save GP2 curvature metadata for analysis.
        
        Args:
            save_path: Path to save metadata
            
        Notes:
            Saves curvature information at each model update,
            allowing for analysis of curvature evolution during
            the saddle point search.
        """
        # Retrieve current curvature data
        gp2_curvature_type, gp2_curvature_value, gp2_eigenvalues = self.get_curvature()

        # Prepare new entry to be saved
        new_entry = {
            "gp2_curvature_type": gp2_curvature_type,
            "gp2_curvature_value": gp2_curvature_value,
            "gp2_eigenvalues": gp2_eigenvalues,
        }

        try:
            # Load existing data if available
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    gp2_curvature = pickle.load(f)
            else:
                gp2_curvature = []

            # Append new data and save
            gp2_curvature.append(new_entry)
            with open(save_path, 'wb') as f:
                pickle.dump(gp2_curvature, f)
                
        except Exception as e:
            print(f"Warning: Failed to save GP2 metadata: {str(e)}")