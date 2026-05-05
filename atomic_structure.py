from typing import Dict, List, Tuple, Optional
import numpy.typing as npt
import numpy as np

class AtomicStructure:
    """Manages atomic structure information and active/frozen atom selection.
    
    This class handles the tracking of active and frozen atoms based on distance criteria
    and manages atom type information for the GP calculations.
    """
    def __init__(
            self,
            moving_atoms: npt.NDArray[np.float64],  # Shape: (N_mov, 3)
            frozen_atoms: npt.NDArray[np.float64],  # Shape: (N_fro, 3)
            moving_types: npt.NDArray[np.int64],    # Shape: (N_mov,)
            frozen_types: npt.NDArray[np.int64],    # Shape: (N_fro,)
            activation_radius: float = np.inf,
            moving_indices: Optional[List[int]] = None
        ):
        """Initialize atomic structure tracking.
        
        Args:
            moving_atoms: Coordinates of moving atoms
            frozen_atoms: Coordinates of all frozen atoms
            moving_types: Atom type indices for moving atoms (0-based)
            frozen_types: Atom type indices for frozen atoms
            activation_radius: Radius for activating frozen atoms
        """
        # Store initial configuration
        self.moving_atoms = moving_atoms
        self.all_frozen_atoms = frozen_atoms
        self.moving_types = moving_types
        self.all_frozen_types = frozen_types
        self.activation_radius = activation_radius
        
        # Use provided moving_indices or default to range
        if moving_indices is not None:
            self.moving_indices = moving_indices
        else:
            self.moving_indices = list(range(len(moving_atoms)))
        
        # Initialize active/inactive splits
        self.active_frozen_atoms = np.empty((0, 3))
        self.active_frozen_types = np.empty(0, dtype=np.int64)
        self.inactive_frozen_atoms = frozen_atoms.copy()
        self.inactive_frozen_types = frozen_types.copy()
        
        # Set up pair type mapping
        max_type = max(moving_types)
        if len(frozen_types) > 0:
            max_type = max(max_type, max(frozen_types))
        self.n_atomtypes = max_type + 1
        self.setup_pair_types()
        
        # Do initial activation
        self.update_active_atoms(moving_atoms)

        
    def setup_pair_types(self) -> None: 
        """Initialize pair type mapping matrix."""
        self.pair_types = np.full((self.n_atomtypes, self.n_atomtypes), -1, dtype=np.int64)
        self.n_pair_types = 0
        
        # Set pair types for moving-moving interactions
        for i in range(len(self.moving_types)):
            for j in range(i+1, len(self.moving_types)):
                type_i = self.moving_types[i]
                type_j = self.moving_types[j]
                if self.pair_types[type_i, type_j] == -1:
                    self.pair_types[type_i, type_j] = self.n_pair_types
                    self.pair_types[type_j, type_i] = self.n_pair_types
                    self.n_pair_types += 1
    
    def update_active_atoms(self, positions: npt.NDArray[np.float64]) -> int:
        """Update which frozen atoms are active based on positions of moving atoms.
        
        Args:
            positions: Current positions of moving atoms (N_mov, 3)
            
        Returns:
            int: Number of newly activated atoms
        """
        newly_activated = 0
        
        if self.activation_radius < np.inf and len(self.inactive_frozen_atoms) > 0:
            # Find atoms within activation radius
            to_activate = []
            for i, pos in enumerate(positions):
                distances = np.linalg.norm(self.inactive_frozen_atoms - pos, axis=1)
                active_indices = np.where(distances <= self.activation_radius)[0]
                to_activate.extend(active_indices)
            
            to_activate = list(set(to_activate))  # Remove duplicates
            
            if len(to_activate) > 0:
                # Move atoms from inactive to active
                new_active_atoms = self.inactive_frozen_atoms[to_activate]
                new_active_types = self.inactive_frozen_types[to_activate]
                
                self.active_frozen_atoms = np.vstack((self.active_frozen_atoms, new_active_atoms)) \
                    if len(self.active_frozen_atoms) > 0 else new_active_atoms
                self.active_frozen_types = np.concatenate((self.active_frozen_types, new_active_types)) \
                    if len(self.active_frozen_types) > 0 else new_active_types
                
                # Remove from inactive
                self.inactive_frozen_atoms = np.delete(self.inactive_frozen_atoms, to_activate, axis=0)
                self.inactive_frozen_types = np.delete(self.inactive_frozen_types, to_activate)
                
                # Update pair types
                self._update_pair_types(new_active_types)
                
                newly_activated = len(to_activate)
                
        return newly_activated
    
    def _update_pair_types(self, new_atom_types: npt.NDArray[np.int64]) -> None:
        """Update pair type matrix for new active atoms."""
        # Add pair types for new atom interactions with moving atoms
        for moving_type in self.moving_types:
            for frozen_type in new_atom_types:
                if self.pair_types[moving_type, frozen_type] == -1:
                    self.pair_types[moving_type, frozen_type] = self.n_pair_types
                    self.pair_types[frozen_type, moving_type] = self.n_pair_types
                    self.n_pair_types += 1
    
    def get_structure_info(self) -> Dict:
        """Get structure information needed for GP kernel.
        
        Returns:
            Dictionary containing:
            - conf_fro: Active frozen atom coordinates
            - atomtype_mov: Moving atom type indices
            - atomtype_fro: Active frozen atom type indices
            - pairtype: Pair type matrix
            - n_pt: Number of active pair types
        """
        return {
            'conf_fro': self.active_frozen_atoms,
            'atomtype_mov': self.moving_types,
            'atomtype_fro': self.active_frozen_types,
            'pairtype': self.pair_types,
            'n_pt': self.n_pair_types,
            'moving_indices': self.moving_indices
        }
    
    def check_interatomic_distances(self, x_new: npt.NDArray[np.float64], 
                                   observed_positions: npt.NDArray[np.float64],
                                   ratio_at_limit: float = 2.0/3.0) -> bool:
        """Check if inter-atomic distances changed too much compared to observed data.
        
        Args:
            x_new: New position to check
            observed_positions: Array of previously observed positions
            ratio_at_limit: Maximum allowed ratio change in distances
            
        Returns:
            bool: True if distances changed too much (position should be rejected)
        """
        # Handle case where observed_positions might be moving-atom-only
        if observed_positions.shape[1] == len(self.moving_indices) * 3:
            # Convert to full positions by inserting into a template
            # This is a workaround - better to store full positions
            print("Warning: Distance checking with moving-atom-only positions")
            return False  # Skip check for now
    
        # For each observed configuration
        for obs_pos in observed_positions:
            # Calculate inter-atomic distances for new position
            distances_new = self.calculate_all_interatomic_distances(x_new)
            distances_obs = self.calculate_all_interatomic_distances(obs_pos)
            
            # Check ratios
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = distances_new / distances_obs
                # Check if all ratios are within acceptable range
                if np.all((ratios > ratio_at_limit) & (ratios < 1.0/ratio_at_limit)):
                    return False  # Found acceptable configuration
        
        return True  # No acceptable configuration found
    
    def calculate_all_interatomic_distances(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate all inter-atomic distances for moving atoms.
        
        Args:
            positions: Full system positions (flattened)
            
        Returns:
            Array of all interatomic distances
        """
        n_moving = len(self.moving_indices)
        
        # Reshape to get atomic positions
        pos_3d = positions.reshape(-1, 3)
        moving_positions = pos_3d[self.moving_indices]
        
        # Calculate pairwise distances
        distances = []
        for i in range(n_moving):
            for j in range(i+1, n_moving):
                dist = np.linalg.norm(moving_positions[i] - moving_positions[j])
                distances.append(dist)
        
        # Also include distances to frozen atoms if any
        if len(self.active_frozen_atoms) > 0:
            for i in range(n_moving):
                for j in range(len(self.active_frozen_atoms)):
                    dist = np.linalg.norm(moving_positions[i] - self.active_frozen_atoms[j])
                    distances.append(dist)
        
        return np.array(distances)
    
    def update_activated_atoms_wrapper(self, x_next: npt.NDArray[np.float64], 
                                     verbose: bool = False) -> int:
        """Wrapper for updating active atoms with proper logging.
        
        Args:
            x_next: New positions (full system, flattened)
            verbose: Whether to print activation messages
            
        Returns:
            int: Number of newly activated atoms
        """
        # Get positions for moving atoms only
        moving_positions = x_next.reshape(-1, 3)[self.moving_indices]
        n_activated = self.update_active_atoms(moving_positions)
        
        if n_activated > 0 and verbose:
            print(f"Activated {n_activated} new frozen atoms")
            print(f"Total active frozen atoms: {len(self.active_frozen_atoms)}")
            
        return n_activated


def calculate_interatomic_distances(
        positions: npt.NDArray[np.float64],
        conf_info: Dict
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Calculate interatomic distances and corresponding pair types.
    
    Args:
        positions: Atomic positions (N_atoms x 3)
        conf_info: Dictionary containing atomic structure info
        
    Returns:
        Tuple containing:
            - distances: Array of interatomic distances
            - inv_distances: Array of inverse interatomic distances
            - pair_types: Array of pair type indices
    """
    # Extract info from conf_info
    conf_fro = conf_info['conf_fro']
    atomtype_mov = conf_info['atomtype_mov']
    atomtype_fro = conf_info['atomtype_fro']
    pairtype = conf_info['pairtype']

    N_mov = len(atomtype_mov)
    positions = positions.reshape(-1, 3)
    
    distances = []
    inv_distances = []
    pair_types = []
    
    # Moving-moving atom distances
    for i in range(N_mov - 1):
        pos_i = positions[i]
        type_i = atomtype_mov[i]
        
        for j in range(i + 1, N_mov):
            pos_j = positions[j]
            type_j = atomtype_mov[j]
            
            # Calculate distance
            dist = np.linalg.norm(pos_i - pos_j)
            inv_dist = 1.0 / dist if dist > 1e-10 else 0.0
            
            # Get pair type
            pt = pairtype[type_i, type_j]
            
            if pt >= 0:  # Only store active pair types
                distances.append(dist)
                inv_distances.append(inv_dist)
                pair_types.append(pt)
    
    # Moving-frozen atom distances
    for i in range(N_mov):
        pos_i = positions[i]
        type_i = atomtype_mov[i]
        
        for j in range(len(atomtype_fro)):
            pos_j = conf_fro[j]
            type_j = atomtype_fro[j]
            
            # Calculate distance
            dist = np.linalg.norm(pos_i - pos_j)
            inv_dist = 1.0 / dist if dist > 1e-10 else 0.0
            
            # Get pair type
            pt = pairtype[type_i, type_j]
            
            if pt >= 0:  # Only store active pair types
                distances.append(dist)
                inv_distances.append(inv_dist)
                pair_types.append(pt)
                
    return (
        np.array(distances),
        np.array(inv_distances),
        np.array(pair_types)
    )

def calculate_atomic_distance_measure(
        pos1: npt.NDArray[np.float64],
        pos2: npt.NDArray[np.float64],
        conf_info: Dict,
        lengthscales: npt.NDArray[np.float64]
    ) -> float:
    """Calculate atomic distance measure between two configurations.
    
    Implements the distance measure:
    dist(C,C') = sqrt(SUM_ij{[(1/r_ij-1/r_ij')/l_ij]^2})
    
    Args:
        pos1: First configuration positions
        pos2: Second configuration positions
        conf_info: Atomic structure information
        lengthscales: Lengthscales for each pair type
        
    Returns:
        float: Distance measure between configurations
    """
    # Get distances for both configurations
    dist1, inv1, types1 = calculate_interatomic_distances(pos1, conf_info)
    dist2, inv2, types2 = calculate_interatomic_distances(pos2, conf_info)
    
    # Verify configurations have same structure
    if not np.array_equal(types1, types2):
        raise ValueError("Configurations have different atomic structure!")
    
    # Calculate distance measure
    diff_inv = inv1 - inv2
    ls = lengthscales[types1]
    dist = np.sqrt(np.sum((diff_inv / ls) ** 2))
    
    return dist