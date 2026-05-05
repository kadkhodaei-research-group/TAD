import os
from gp1_model import GP1
from thermal_sampling_utils import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ThermodynamicResidenceGP1(GP1):
    def __init__(self, current_location, temperature, mass, local_pes, use_gp_model=False, sap_training_data=None, path=None, atomic_info=None):
        super().__init__(current_location, temperature, mass, local_pes, atomic_info=atomic_info)
        self.temperature = temperature
        self.mass = mass
        self.local_pes = local_pes
        self.use_gp_model = use_gp_model
        self.gp_model = None
        self.sap_training_data = sap_training_data
        self.path = path

    def calculate_heat_capacity(self, eigenvalues_tr):
        """Calculate heat capacity Cv in units of k_B (dimensionless)."""
        Cv_total = 0.0
        
        # Use atomic units k_B
        k_B_au = 3.1668114e-6  # Hartree/K
        hbar = 1.0  # atomic units
        
        for omega in eigenvalues_tr:
            if omega > 1e-6:  # Skip acoustic modes
                # x = ℏω / (k_B * T)
                x = (hbar * omega) / (k_B_au * self.temperature)
                
                if x < 1e-6:  # High temperature limit
                    Cv_mode_dimensionless = 1.0
                elif x < 50:  # Normal calculation
                    exp_x = np.exp(x)
                    Cv_mode_dimensionless = (x**2) * exp_x / ((exp_x - 1)**2)
                else:  # Low temperature limit
                    Cv_mode_dimensionless = 0.0
                
                Cv_total += Cv_mode_dimensionless
        print(f"Total heat capacity (dimensionless): {Cv_total}")
        return Cv_total

    def calculate_thermal_noise(self, eigenvalues_tr, eigenvectors_tr, num_snapshots=50):
        # Generate snapshots (this part is correct)
        displacements = self.generate_thermal_snapshot(
            num_snapshots=num_snapshots,
            eigenvalues_tr=eigenvalues_tr,
            eigenvectors_tr=eigenvectors_tr, 
            context='thermal_noise_calculation',
            dimer_step=None
        ) - self.current_location
        
        # Calculate <u²>
        average_u_squared = np.mean(np.linalg.norm(displacements, axis=1) ** 2)
        
        # Calculate Cv (dimensionless)
        Cv_dimensionless = self.calculate_heat_capacity(eigenvalues_tr)  # ~642
        
        # Calculate thermal noise
        k_B = 8.617333262145e-5  # eV/K
        k_B_T = k_B * self.temperature  # eV
        
        # Energy fluctuations: σ_E = k_B*T * sqrt(Cv_dimensionless)
        sigma_E = k_B_T * np.sqrt(Cv_dimensionless)
        
        # Force fluctuations: σ_F = σ_E / sqrt(<u²>)
        sigma_F = sigma_E / np.sqrt(average_u_squared)
        
        print(f"\nThermal noise calculation:")
        print(f"  k_B*T = {k_B_T:.4f} eV")
        print(f"  Cv (dimensionless) = {Cv_dimensionless:.1f}")
        print(f"  <u²> = {average_u_squared:.2f} Ų")
        print(f"  σ_E = {sigma_E:.3f} eV")
        print(f"  σ_F = {sigma_F:.3f} eV/Å")
        
        return sigma_F, sigma_E