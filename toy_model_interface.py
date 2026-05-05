"""Interface to make toy models compatible with walker/run infrastructure."""

import numpy as np
from potentials_toy_models import pes_models


class ToyModelInterface:
    """Wraps toy models to provide VASP-like interface for walker classes."""
    
    def __init__(self, potential_name: str, initial_position: np.ndarray):
        """Initialize toy model interface.
        
        Args:
            potential_name: Name of the toy potential (e.g., 'Muller-Brown', 'Leps', etc.)
            initial_position: Initial position (2D for toy models)
        """
        self.potential_name = potential_name
        self.pes = pes_models(potential_name)
        self.eval_count = 0
        
        # Check dimensionality
        if len(initial_position) != 2:
            raise ValueError(f"Toy models expect 2D positions, got {len(initial_position)}D")
        
        # Map of potentials to their gradient functions
        self.grad_funcs = {
            'Muller-Brown': self.pes.mb_pot_grad,
            'Karplus': self.pes.karplus_pot_grad,
            'three_hole_pot': self.pes.three_hole_pot_grad,
            'V5': self.pes.dV5,
            'V6': self.pes.dV6,
            'V8': self.pes.dV8,
        }
        
        # Map of potentials to their energy functions
        self.energy_funcs = {
            'Muller-Brown': self.pes.mb_pot,
            'Leps': self.pes.leps_pot,
            'Leps-HO': self.pes.leps_ho_pot,
            'Karplus': self.pes.karplus_pot,
            'Wolfe-Quapp': self.pes.wq_pot,
            'three_hole_pot': self.pes.three_hole_pot,
            'V1': self.pes.V1,
            'V2': self.pes.V2,
            'V3': self.pes.V3,
            'V4': self.pes.V4,
            'V5': self.pes.V5,
            'V6': self.pes.V6,
            'V7': self.pes.V7,
            'V8': self.pes.V8,
        }
        
        if potential_name not in self.energy_funcs:
            raise ValueError(f"Unknown potential: {potential_name}")
        
        self.energy_func = self.energy_funcs[potential_name]
        
        # For potentials without explicit gradient, use finite differences
        if potential_name in self.grad_funcs:
            self.grad_func = self.grad_funcs[potential_name]
        else:
            self.grad_func = self._finite_diff_grad
    
    def scaler_y_value(self, position: np.ndarray, is_thermal: bool = False) -> float:
        """Calculate energy at given position.
        
        Args:
            position: Position vector (2D)
            is_thermal: Unused for toy models
            
        Returns:
            Energy value
        """
        self.eval_count += 1
        return self.energy_func(position)
    
    def first_derivative(self, position: np.ndarray, is_thermal: bool = False) -> np.ndarray:
        """Calculate forces at given position.
        
        Args:
            position: Position vector (2D)
            is_thermal: Unused for toy models
            
        Returns:
            Forces (negative gradient)
        """
        # Get gradient
        gradient = self.grad_func(position)
        # Return forces (negative gradient)
        return -gradient
    
    def _finite_diff_grad(self, position: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Calculate gradient using finite differences.
        
        Args:
            position: Position vector (2D)
            h: Step size for finite differences
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(position)
        
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            
            grad[i] = (self.energy_func(pos_plus) - self.energy_func(pos_minus)) / (2 * h)
        
        return grad
    
    def get_info(self) -> dict:
        """Get information about the toy model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'potential_name': self.potential_name,
            'domain': self.pes.domain,
            'energy_range': self.pes.energies,
            'eval_count': self.eval_count,
            'dimensionality': 2
        }