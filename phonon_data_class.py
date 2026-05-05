import h5py
import numpy as np
import matplotlib.pyplot as plt

class PhononData:
    def __init__(self, filename):
        """
        Load and organize phonon data from HDF5 file.
        """
        self.load_data(filename)
        
    def load_data(self, filename):
        """
        Load data from HDF5 file.
        """
        with h5py.File(filename, 'r') as f:
            # Load frequencies and q-points
            self.frequencies = f['frequencies'][:]  # Shape: (n_qpoints, n_modes)
            self.q_values = f['q_values'][:]       # Shape: (n_qpoints,)
            self.q_vector = f['q_vector'][:]       # Shape: (n_qpoints, 3)
            self.q_ticks = f['q_ticks'][:]         # Shape: (n_ticks,)
            
            # Load eigenvectors (combining real and imaginary parts)
            eigvec_re = f['eigenvectors_re'][:]    # Shape: (n_qpoints, n_modes, n_modes)
            eigvec_im = f['eigenvectors_im'][:]    # Shape: (n_qpoints, n_modes, n_modes)
            self.eigenvectors = eigvec_re + 1j * eigvec_im
            
            # Load group velocities and site projections
            self.group_velocities = f['group_velocities'][:]  # Shape: (n_qpoints, n_modes, 3)
            self.site_projections = f['site_projection_per_mode'][:]  # Shape: (n_qpoints, n_modes, n_atoms)
            
        # Store dimensions
        self.n_qpoints = self.frequencies.shape[0]
        self.n_modes = self.frequencies.shape[1]
        self.n_atoms = self.n_modes // 3
        
    def get_frequencies_at_q(self, q_idx):
        """
        Get frequencies at specific q-point index.
        """
        return self.frequencies[q_idx]
    
    def get_eigenvectors_at_q(self, q_idx):
        """
        Get eigenvectors at specific q-point index.
        """
        return self.eigenvectors[q_idx]
    
    def get_group_velocities_at_q(self, q_idx):
        """
        Get group velocities at specific q-point index.
        """
        return self.group_velocities[q_idx]
    
    def plot_dispersion(self, ylim=None, save_path=None):
        """
        Plot phonon dispersion relations.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot frequencies vs q_values
        for mode in range(self.n_modes):
            plt.plot(self.q_values, self.frequencies[:, mode], 'b-', alpha=0.5)
            
        # Set labels and title
        plt.xlabel('Wave Vector')
        plt.ylabel('Frequency (units)')
        plt.title('Phonon Dispersion Relations')
        
        # Set y-axis limits if specified
        if ylim is not None:
            plt.ylim(ylim)
            
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_numpy(self, filename):
        """
        Save key data to numpy format.
        """
        np.savez(filename,
                frequencies=self.frequencies,
                eigenvectors=self.eigenvectors,
                q_values=self.q_values,
                q_vector=self.q_vector)

def main():
    # File paths
    hdf5_file = 'outfile.dispersion_relations.hdf5'
    
    # Load data
    print("Loading phonon data...")
    phonon_data = PhononData(hdf5_file)
    
    # Print summary
    print("\nData Summary:")
    print(f"Number of q-points: {phonon_data.n_qpoints}")
    print(f"Number of modes: {phonon_data.n_modes}")
    print(f"Number of atoms: {phonon_data.n_atoms}")
    
    # Print frequency range
    print(f"\nFrequency range: {phonon_data.frequencies.min():.4f} to {phonon_data.frequencies.max():.4f}")
    
    # Plot dispersion relations
    print("\nPlotting dispersion relations...")
    phonon_data.plot_dispersion(save_path='phonon_dispersion.png')
    
    # Save data in numpy format
    print("\nSaving data in numpy format...")
    phonon_data.save_numpy('phonon_data.npz')
    
    # Example: Get data at specific q-point (e.g., Γ point, q_idx=0)
    print("\nData at Γ point:")
    frequencies_gamma = phonon_data.get_frequencies_at_q(0)
    print(f"Frequencies at Γ (first 10 modes):")
    print(frequencies_gamma[:10])

if __name__ == "__main__":
    main()