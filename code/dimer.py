from __future__ import annotations

import logging
import warnings
import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Dict, Any, Tuple
import traceback

logger = logging.getLogger(__name__)


class Dimer:
    """Advanced implementation of the Dimer method for saddle point searching.
    
    This implementation closely follows the reference dimer.py code while maintaining
    compatibility with the walker interface.
    """
    
    def __init__(
            self,
            x: npt.NDArray[np.float64],
            force_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            dimer_sep: float = 0.01,
            rotation_method: str = 'lbfgsext',
            translation: str = 'lbfgs',
            opt_type: str = 'double_damped_BFGS',  # For compatibility
            max_dimer_rotations: int = 10,
            T_anglerot: float = 0.01,
            T_anglerot_init: float = 0.0873,
            num_iter_initrot: Optional[int] = None,
            param_trans: Optional[npt.NDArray[np.float64]] = None,
            dimer_stopping_criteria: float = 0.01,
            **kwargs
        ) -> None:
        """Initialize the Dimer method.
        
        Args:
            x: Initial position
            force_func: Function returning forces at given position
            dimer_sep: Dimer separation distance
            rotation_method: Method for rotation ('mn', 'cg', 'lbfgs', 'lbfgsext')
            translation: Method for translation ('newton', 'cg', 'lbfgs', 'qmvv')
            opt_type: Optimization type (kept for compatibility)
            max_dimer_rotations: Maximum rotation iterations per translation
            T_anglerot: Rotation angle convergence threshold (main loop)
            T_anglerot_init: Rotation angle convergence threshold (initial rotations)
            num_iter_initrot: Number of initial rotations (None = dimension)
            param_trans: Translation parameters
            dimer_stopping_criteria: Force convergence threshold
        """
        self.x = x.copy()
        self.force_func = force_func
        self.dimer_sep = dimer_sep
        self.rotation_method = rotation_method.lower()
        self.translation_method = translation.lower()
        self.opt_type = opt_type
        self.max_dimer_rotations = max_dimer_rotations
        self.T_anglerot = T_anglerot
        self.T_anglerot_init = T_anglerot_init
        self.dimer_stopping_criteria = dimer_stopping_criteria
        
        # Set default translation parameters if not provided
        if param_trans is None:
            self.param_trans = np.array([[0.1, 0.1]])
        else:
            self.param_trans = param_trans
            
        # Set default num_iter_initrot to dimension if not specified
        self.D = x.size  # Dimension
        if num_iter_initrot is None:
            self.num_iter_initrot = self.D
        else:
            self.num_iter_initrot = num_iter_initrot
            
        # Initialize orientation vector (will be set properly later)
        self.orient = None
        
        # Flag to track if initial rotations have been performed
        self.initial_rotations_done = False
        
        # Initialize rotation info dictionary (following reference)
        self.rotinfo = {
            'F_rot_old': np.zeros((1, self.D)),
            'F_modrot_old': np.zeros((1, self.D)),
            'orient_rot_oldplane': np.zeros((1, self.D)),
            'cgiter_rot': 0,
            'num_cgiter_rot': self.D,
            'deltaR_mem': np.ndarray(shape=(0, self.D)),
            'deltaF_mem': np.ndarray(shape=(0, self.D)),
            'num_lbfgsiter_rot': self.D,
            'G1': np.ndarray(shape=(0, self.D))
        }
        
        # Initialize translation info dictionary (following reference)
        # Create wrapper to match reference potential function interface
        def potential_wrapper(R):
            """Wrapper to match reference code's potential function interface."""
            if R.ndim == 1:
                R = R.reshape(1, -1)
            forces = self.force_func(R.ravel())
            # Return (energy, gradient) where energy is dummy and gradient = -forces
            return np.array([[-1.0]]), -forces.reshape(1, -1)
            
        self.transinfo = {
            'potential': potential_wrapper,
            'F_trans_old': np.zeros((1, self.D)),
            'F_modtrans_old': np.zeros((1, self.D)),
            'V_old': np.zeros((1, self.D)),
            'zeroV': 1,
            'cgiter_trans': 0,
            'num_cgiter_trans': self.D,
            'deltaR_mem': np.ndarray(shape=(0, self.D)),
            'deltaF_mem': np.ndarray(shape=(0, self.D)),
            'num_lbfgsiter_trans': self.D
        }
        
        # Track all observations (following reference)
        self.R_all = np.ndarray(shape=(0, self.D))
        self.E_all = np.ndarray(shape=(0, 1))
        self.G_all = np.ndarray(shape=(0, self.D))
        self.E_R_acc = np.ndarray(shape=(0,))
        self.maxF_R_acc = np.ndarray(shape=(0,))
        
    def initialize_direction(self) -> npt.NDArray[np.float64]:
        """Initialize dimer orientation using a random direction."""
        if self.orient is None:
            # Random initialization
            self.orient = np.random.normal(size=(1, self.D))
            self.orient = self.orient / np.linalg.norm(self.orient)
        return self.orient
        
    def set_initial_direction(self, direction: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Set the initial dimer direction explicitly."""
        direction = direction.reshape(1, -1)
        if not np.isclose(np.linalg.norm(direction), 1.0):
            direction = direction / np.linalg.norm(direction)
        self.orient = direction
        return direction
        
    def force_rot(self, G01: npt.NDArray[np.float64], orient: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate rotational force following reference implementation."""
        F1 = -G01[1, :][np.newaxis]
        F2 = -2 * G01[0, :][np.newaxis] + G01[1, :][np.newaxis]
        F1_rot = F1 - np.dot(F1[0, :], orient[0, :]) * orient
        F2_rot = F2 - np.dot(F2[0, :], orient[0, :]) * orient
        F_rot = (F1_rot - F2_rot) / self.dimer_sep
        return F_rot
        
    def rotate_dimer(self, R, orient, G01, F_rot, potential, T_anglerot, estim_Curv=1):
        """Rotate dimer to find minimum energy orientation (following reference)."""
        F_0 = np.sqrt(np.sum(np.square(F_rot)))
        C_0 = np.dot(-G01[0, :] + G01[1, :], orient[0, :]) / self.dimer_sep
        dtheta = 0.5 * np.arctan(0.5 * F_0 / (np.abs(C_0) + 1e-10))
        
        if dtheta < T_anglerot:
            # No rotation needed
            orient_new = orient.copy()
            orient_rot_new = np.zeros((1, self.D))
            Curv = C_0 if estim_Curv else None
            G1 = np.ndarray(shape=(0, self.D))
            R1_dtheta = np.ndarray(shape=(0, self.D))
            E1_dtheta = np.ndarray(shape=(0, 1))
            G1_dtheta = np.ndarray(shape=(0, self.D))
        else:
            # Perform rotation
            orient_rot = F_rot / F_0
            orient_dtheta = np.cos(dtheta) * orient + np.sin(dtheta) * orient_rot
            orient_rot_dtheta = -np.sin(dtheta) * orient + np.cos(dtheta) * orient_rot
            
            R1_dtheta = R + self.dimer_sep * orient_dtheta
            E1_dtheta, G1_dtheta = potential(R1_dtheta)
            
            # Ensure G1_dtheta is 2D
            if G1_dtheta.ndim == 1:
                G1_dtheta = G1_dtheta.reshape(1, -1)
            
            F_rot_dtheta = self.force_rot(np.vstack((G01[0, :], G1_dtheta[0, :])), orient_dtheta)
            F_dtheta = np.dot(F_rot_dtheta[0, :], orient_rot_dtheta[0, :])
            
            a1 = (F_dtheta - F_0 * np.cos(2 * dtheta)) / (2 * np.sin(2 * dtheta))
            b1 = -0.5 * F_0
            angle_rot = 0.5 * np.arctan(b1 / (a1 + 1e-10))
            
            if angle_rot < 0:
                angle_rot = np.pi / 2 + angle_rot
                
            orient_new = np.cos(angle_rot) * orient + np.sin(angle_rot) * orient_rot
            orient_new = orient_new / np.sqrt(np.sum(np.square(orient_new)))
            orient_rot_new = -np.sin(angle_rot) * orient + np.cos(angle_rot) * orient_rot
            orient_rot_new = orient_rot_new / np.sqrt(np.sum(np.square(orient_rot_new)))
            
            if estim_Curv:
                Curv = C_0 + a1 * (np.cos(2 * angle_rot) - 1) + b1 * np.sin(2 * angle_rot)
            else:
                Curv = None
                
            G1 = np.ndarray(shape=(0, self.D))
            
        return orient_new, orient_rot_new, Curv, G1, R1_dtheta, E1_dtheta, G1_dtheta
        
    def rot_iter_lbfgsext(self, R, orient, G01, potential, T_anglerot, estim_Curv=1):
        """L-BFGS rotation with extrapolation following reference implementation."""
        F_rot_old = self.rotinfo['F_rot_old'].copy()
        deltaR_mem = self.rotinfo['deltaR_mem'].copy()
        deltaF_mem = self.rotinfo['deltaF_mem'].copy()
        num_lbfgsiter_rot = self.rotinfo['num_lbfgsiter_rot']
        
        m = deltaR_mem.shape[0]
        F_rot = self.force_rot(G01, orient)
        
        if m > 0:
            deltaF_mem = np.vstack((deltaF_mem, F_rot - F_rot_old))
            
        # L-BFGS two-loop recursion
        q = -F_rot[0, :]
        a_mem = np.zeros((m, 1))
        
        for k in range(m):
            s = deltaR_mem[m - 1 - k, :]
            y = -deltaF_mem[m - 1 - k, :]
            rho = 1 / (np.dot(y, s) + 1e-10)
            a = rho * np.dot(s, q)
            a_mem[m - 1 - k, 0] = a
            q = q - a * y
            
        if m > 0:
            s = deltaR_mem[m - 1, :]
            y = -deltaF_mem[m - 1, :]
            scaling = np.dot(s, y) / (np.dot(y, y) + 1e-10)
        else:
            scaling = 0.01
            
        r = scaling * q
        
        for k in range(1, m + 1):
            s = deltaR_mem[k - 1, :]
            y = -deltaF_mem[k - 1, :]
            rho = 1 / (np.dot(y, s) + 1e-10)
            b = rho * np.dot(y, r)
            r = r + s * (a_mem[k - 1, 0] - b)
            
        orient_rot = r[np.newaxis] - np.dot(orient[0, :], r) * orient
        orient_rot = orient_rot / (np.sqrt(np.sum(np.square(orient_rot))) + 1e-10)
        F_rot_oriented = np.dot(F_rot[0, :], orient_rot[0, :]) * orient_rot
        
        orient_new, orient_rot_new, Curv, G1, R_obs, E_obs, G_obs = self.rotate_dimer(
            R, orient, G01, F_rot_oriented, potential, T_anglerot, estim_Curv
        )
        
        if R_obs.shape[0] < 1:
            F_rot = np.zeros((1, self.D))
            deltaR_mem = np.ndarray(shape=(0, self.D))
            deltaF_mem = np.ndarray(shape=(0, self.D))
        else:
            deltaR_mem = np.vstack((deltaR_mem, orient_new - orient))
            if m >= num_lbfgsiter_rot:
                deltaR_mem = np.delete(deltaR_mem, 0, 0)
                deltaF_mem = np.delete(deltaF_mem, 0, 0)
                
        self.rotinfo['F_rot_old'] = F_rot.copy()
        self.rotinfo['deltaR_mem'] = deltaR_mem.copy()
        self.rotinfo['deltaF_mem'] = deltaF_mem.copy()
        self.rotinfo['G1'] = G1.copy()
        
        return orient_new, Curv, R_obs, E_obs, G_obs
        
    def trans_iter_lbfgs(self, R, orient, F_R, Curv, param_trans):
        """L-BFGS translation following reference implementation."""
        F_trans_old = self.transinfo['F_trans_old'].copy()
        deltaR_mem = self.transinfo['deltaR_mem'].copy()
        deltaF_mem = self.transinfo['deltaF_mem'].copy()
        num_lbfgsiter_trans = self.transinfo['num_lbfgsiter_trans']
        
        steplength_convex = param_trans[0, 0]
        max_steplength = param_trans[0, 1]
        m = deltaR_mem.shape[0]
        
        if Curv < 0:
            # Negative curvature: invert force component along dimer
            F_trans = F_R - 2 * np.dot(F_R[0, :], orient[0, :]) * orient
            
            if m > 0:
                deltaF_mem = np.vstack((deltaF_mem, F_trans - F_trans_old))
                
            # L-BFGS two-loop recursion
            q = -F_trans[0, :]
            a_mem = np.zeros((m, 1))
            
            for k in range(m):
                s = deltaR_mem[m - 1 - k, :]
                y = -deltaF_mem[m - 1 - k, :]
                rho = 1 / (np.dot(y, s) + 1e-10)
                a = rho * np.dot(s, q)
                a_mem[m - 1 - k, 0] = a
                q = q - a * y
                
            if m > 0:
                s = deltaR_mem[m - 1, :]
                y = -deltaF_mem[m - 1, :]
                scaling = np.dot(s, y) / (np.dot(y, y) + 1e-10)
            else:
                scaling = 0.01
                
            r = scaling * q
            
            for k in range(1, m + 1):
                s = deltaR_mem[k - 1, :]
                y = -deltaF_mem[k - 1, :]
                rho = 1 / (np.dot(y, s) + 1e-10)
                b = rho * np.dot(y, r)
                r = r + s * (a_mem[k - 1, 0] - b)
                
            steplength = np.sqrt(np.sum(np.square(r)))
            
            if steplength > max_steplength:
                r = max_steplength / steplength * r
                self.transinfo['F_trans_old'] = np.zeros((1, self.D))
                self.transinfo['deltaR_mem'] = np.ndarray(shape=(0, self.D))
                self.transinfo['deltaF_mem'] = np.ndarray(shape=(0, self.D))
            else:
                deltaR_mem = np.vstack((deltaR_mem, -r[np.newaxis]))
                if m >= num_lbfgsiter_trans:
                    deltaR_mem = np.delete(deltaR_mem, 0, 0)
                    deltaF_mem = np.delete(deltaF_mem, 0, 0)
                self.transinfo['F_trans_old'] = F_trans.copy()
                self.transinfo['deltaR_mem'] = deltaR_mem.copy()
                self.transinfo['deltaF_mem'] = deltaF_mem.copy()
                
            R_new = R - r[np.newaxis]
            step_size = -r[np.newaxis]
            
        else:
            # Positive curvature: climb uphill along dimer direction
            # Always move in the positive dimer direction to escape minimum
            orient_search = orient[0, :]
            R_new = R + steplength_convex * orient_search[np.newaxis]
            step_size = steplength_convex * orient_search[np.newaxis]
            
            self.transinfo['F_trans_old'] = np.zeros((1, self.D))
            self.transinfo['deltaR_mem'] = np.ndarray(shape=(0, self.D))
            self.transinfo['deltaF_mem'] = np.ndarray(shape=(0, self.D))
            
        R_obs = np.ndarray(shape=(0, self.D))
        E_obs = np.ndarray(shape=(0, 1))
        G_obs = np.ndarray(shape=(0, self.D))
        
        return R_new, R_obs, E_obs, G_obs, step_size
        
    def run(self) -> npt.NDArray[np.float64]:
        """Execute one complete dimer step (rotation + translation).
        
        This follows the reference implementation's logic more closely.
        """
        # Initialize direction if not set
        if self.orient is None:
            self.initialize_direction()
            
        # Get forces at current position
        R = self.x.reshape(1, -1)
        Fl = self.force_func(self.x)
        G_R = -Fl.reshape(1, -1)  # Convert forces to gradients
        
        # Update tracking arrays
        self.E_R_acc = np.hstack((self.E_R_acc, -1.0))  # Dummy energy
        maxF_R = np.max(np.abs(Fl))
        self.maxF_R_acc = np.hstack((self.maxF_R_acc, maxF_R))
        
        # Check convergence
        if maxF_R < self.dimer_stopping_criteria:
            logger.info(f"Converged with max force: {maxF_R:.6f}")
            return np.zeros_like(self.x)
            
        # Evaluate gradient at image 1
        R1 = R + self.dimer_sep * self.orient
        Fl_1 = self.force_func(R1.ravel())
        G1 = -Fl_1.reshape(1, -1)
        
        # Store observations
        self.R_all = np.vstack((self.R_all, R1)) if self.R_all.size > 0 else R1
        self.E_all = np.vstack((self.E_all, [[-1.0]])) if self.E_all.size > 0 else np.array([[-1.0]])
        self.G_all = np.vstack((self.G_all, G1)) if self.G_all.size > 0 else G1
        
        # Rotation phase
        for ind_iter_rot in range(1, self.max_dimer_rotations + 1):
            orient_old = self.orient.copy()
            
            # Perform rotation
            G01 = np.vstack((G_R, G1))
            if self.rotation_method == 'lbfgsext':
                orient_new, Curv, R_obs, E_obs, G_obs = self.rot_iter_lbfgsext(
                    R, self.orient, G01, self.transinfo['potential'], self.T_anglerot, 1
                )
            else:
                # Default simple rotation
                F_rot = self.force_rot(G01, self.orient)
                if np.linalg.norm(F_rot) < 0.1:
                    orient_new = self.orient.copy()
                    Curv = np.dot((-G_R[0, :] + G1[0, :]), self.orient[0, :]) / self.dimer_sep
                else:
                    orient_new = self.orient + 0.1 * F_rot / np.linalg.norm(F_rot)
                    orient_new = orient_new / np.linalg.norm(orient_new)
                    Curv = np.dot((-G_R[0, :] + G1[0, :]), orient_new[0, :]) / self.dimer_sep
                R_obs = np.ndarray(shape=(0, self.D))
                E_obs = np.ndarray(shape=(0, 1))
                G_obs = np.ndarray(shape=(0, self.D))
            
            self.orient = orient_new
            
            # Store observations
            if R_obs.shape[0] > 0:
                self.R_all = np.vstack((self.R_all, R_obs))
                self.E_all = np.vstack((self.E_all, E_obs))
                self.G_all = np.vstack((self.G_all, G_obs))
                
            # Check rotation convergence
            angle_change = np.arccos(np.clip(np.dot(orient_old[0, :], orient_new[0, :]), -1, 1))
            if R_obs.shape[0] < 1 or angle_change < self.T_anglerot:
                break
            elif self.rotinfo['G1'].shape[0] < 1:
                # Re-evaluate at new orientation
                R1 = R + self.dimer_sep * self.orient
                Fl_1 = self.force_func(R1.ravel())
                G1 = -Fl_1.reshape(1, -1)
                self.R_all = np.vstack((self.R_all, R1))
                self.E_all = np.vstack((self.E_all, [[-1.0]]))
                self.G_all = np.vstack((self.G_all, G1))
            else:
                G1 = self.rotinfo['G1'].copy()
                self.rotinfo['G1'] = np.ndarray(shape=(0, self.D))
                
        # Reset rotation memory after rotation phase (following reference)
        self.rotinfo['deltaR_mem'] = np.ndarray(shape=(0, self.D))
        self.rotinfo['deltaF_mem'] = np.ndarray(shape=(0, self.D))
        
        # Calculate curvature if not already done
        if 'Curv' not in locals() or Curv is None:
            Curv = np.dot((-G_R[0, :] + G1[0, :]), self.orient[0, :]) / self.dimer_sep
        
        # Store curvature for debugging
        self.Curv = Curv
        # print(f"Curvature estimate: {Curv:.4f} eV/Å²")
        
        # Translation phase
        F_R = -G_R  # Convert gradient to force
        
        if self.translation_method == 'lbfgs':
            R_new, R_obs, E_obs, G_obs, step_size = self.trans_iter_lbfgs(
                R, self.orient, F_R, Curv, self.param_trans
            )
        else:
            # Simple translation
            if Curv < 0:
                # Negative curvature: invert component along dimer
                F_trans = F_R - 2 * np.dot(F_R[0, :], self.orient[0, :]) * self.orient
            else:
                # Positive curvature: move along dimer direction (uphill)
                # Force for climbing is just the dimer direction itself
                F_trans = self.orient
            
            if np.linalg.norm(F_trans) > 1e-10:
                step_size = 0.1 * F_trans / np.linalg.norm(F_trans)
            else:
                step_size = np.zeros_like(F_trans)
            
            R_new = R + step_size
            R_obs = np.ndarray(shape=(0, self.D))
            E_obs = np.ndarray(shape=(0, 1))
            G_obs = np.ndarray(shape=(0, self.D))
        
        # Store translation observations
        if R_obs.shape[0] > 0:
            self.R_all = np.vstack((self.R_all, R_obs))
            self.E_all = np.vstack((self.E_all, E_obs))
            self.G_all = np.vstack((self.G_all, G_obs))
            
        # Update position
        self.x = R_new.ravel()
        
        # Store final position
        self.R_all = np.vstack((self.R_all, R_new))
        self.E_all = np.vstack((self.E_all, [[-1.0]]))
        E_R, G_R_new = self.transinfo['potential'](R_new)
        self.G_all = np.vstack((self.G_all, G_R_new))
        
        return step_size.ravel()
        
    def perform_initial_rotations(self) -> None:
        """Perform initial rotations with less strict convergence.
        
        This follows the reference implementation's approach.
        """
        if self.initial_rotations_done:
            return
            
        logger.info(f"Performing up to {self.num_iter_initrot} initial rotations")
        
        # Initialize direction if not set
        if self.orient is None:
            self.initialize_direction()
            
        # Get initial position and force
        R = self.x.reshape(1, -1)
        Fl = self.force_func(self.x)
        G_R = -Fl.reshape(1, -1)
        
        # Perform initial rotations
        for k in range(self.num_iter_initrot):
            # Evaluate at image 1
            R1 = R + self.dimer_sep * self.orient
            Fl_1 = self.force_func(R1.ravel())
            G1 = -Fl_1.reshape(1, -1)
            
            # Check if rotation needed
            G01 = np.vstack((G_R, G1))
            F_rot = self.force_rot(G01, self.orient)
            F_0 = np.sqrt(np.sum(np.square(F_rot)))
            C_0 = np.dot(-G01[0, :] + G01[1, :], self.orient[0, :]) / self.dimer_sep
            dtheta = 0.5 * np.arctan(0.5 * F_0 / (np.abs(C_0) + 1e-10))
            
            if dtheta < self.T_anglerot_init:
                logger.info(f"Initial rotations converged after {k} iterations")
                break
                
            # Perform rotation
            orient_old = self.orient.copy()
            if self.rotation_method == 'lbfgsext':
                orient_new, Curv, R_obs, E_obs, G_obs = self.rot_iter_lbfgsext(
                    R, self.orient, G01, self.transinfo['potential'], self.T_anglerot_init
                )
            else:
                # Simple rotation
                orient_new = self.orient + 0.1 * F_rot / np.linalg.norm(F_rot)
                orient_new = orient_new / np.linalg.norm(orient_new)
                
            angle_change = np.arccos(np.clip(np.dot(orient_old[0, :], orient_new[0, :]), -1, 1))
            self.orient = orient_new
            
            if angle_change < self.T_anglerot_init:
                logger.info(f"Initial rotations converged after {k+1} iterations")
                break
                
        self.initial_rotations_done = True
        logger.info("Initial rotations completed")
        
    def continue_run(self, next_x: npt.NDArray[np.float64]) -> bool:
        """Check if dimer should continue based on force convergence."""
        force = self.force_func(next_x)
        return np.linalg.norm(force) >= self.dimer_stopping_criteria
        
    def perform_hopping(self, number_of_hops: int = 10,
                       max_step_size: float = 0.2) -> npt.NDArray[np.float64]:
        """Perform multiple dimer hops from current position.
        
        Args:
            number_of_hops: Number of rotation+translation cycles to perform
            max_step_size: Maximum allowed TOTAL displacement after all hops
            
        Returns:
            Final position after all hops (constrained by max_step_size)
        """
        start_position = self.x.copy()
        
        # Perform initial rotations if not done
        if not self.initial_rotations_done:
            self.perform_initial_rotations()
        
        # Perform all hops
        for hop in range(number_of_hops):
            step = self.run()
            # Note: position is already updated in run()
        
        # Check total displacement from start
        total_displacement = self.x - start_position
        total_distance = np.linalg.norm(total_displacement)
        
        # Limit total displacement if needed
        if total_distance > max_step_size:
            # Scale back to max_step_size
            limited_displacement = total_displacement * (max_step_size / total_distance)
            self.x = start_position + limited_displacement
        
        return self.x.copy()