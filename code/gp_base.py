from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
import numpy.typing as npt
from gp_kernels import *
from models_utilities import *
import torch
import gpytorch
import torch
from gp_data_saver import get_gp_logger
from atomic_structure import calculate_atomic_distance_measure
torch.set_default_dtype(torch.float64)

class GPSurrogateModel(object):
    """Base class for Gaussian Process surrogate models.
    
    This class provides core functionality for training and using GP models
    to approximate potential energy surfaces and their derivatives.
    
    Attributes:
        epoch (int): Minimum number of training epochs
        model_type (str): Type of GP model to use
        model (Optional[gpytorch.models.ExactGP]): The trained GP model
        likelihood (Optional[gpytorch.likelihoods.Likelihood]): Model likelihood
        device (torch.device): Device to run computations on (CPU or GPU)
    """
    def __init__(
            self, 
            min_epoch: int = 10000,
            model_type: str = 'MultitaskGPModel',
            atomic_info: Optional[Dict] = None,
            use_gpu: bool = False
        ) -> None:
        """Initialize the GP surrogate model.
        
        Args:
            min_epoch: Minimum number of training epochs
            model_type: Type of GP model to use
            atomic_info: Dictionary containing atomic structure information
        Notes:
            Sets PyTorch to use float64 precision for numerical stability
        """

        torch.set_default_dtype(torch.float64)  # set the default dtype to float64 (very important)

        # Set device based on GPU availability and request
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                print("GPU requested but not available. Using CPU.")

        self.epoch = min_epoch
        self.model_type = model_type
        self.model = None
        self.likelihood = None
        self.atomic_info = atomic_info  # Store atomic info
        self.energy_reference = None  # For energy referencing
        self.disable_auto_referencing = False  # Flag to disable automatic energy referencing

    def set_current_iteration(self, iteration: int):
        """Set the current iteration number for tracking."""
        self.current_iteration = iteration

    def train(
        self,
        training_data: List[any],
        thermal_noise: Optional[Tuple[float, float]] = None,
        power: Optional[int] = None,
        path: Optional[str] = None,
        save_model: bool = True,
        model_name: str = "GP",
        minimum_noise: Optional[Tuple[float, float]] = None,
    ) -> 'GPSurrogateModel':
        """Train the GP model on provided data.
        
        Args:
            training_data: List containing [positions, energies, forces]
            thermal_noise: Thermal noise values (force_noise, energy_noise)
            power: Power parameter (unused)
            path: Path to save model
            save_model: Whether to save the model
            model_name: Name of the model
            minimum_noise: Minimum noise constraint from trgp1 (force_noise, energy_noise)
        """
        
        # Extract and prepare data
        x, y, g = training_data

        # Convert to numpy if needed (keep these for logging)
        x_np = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
        y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
        g_np = g.numpy() if hasattr(g, 'numpy') else np.asarray(g)
        
        # Verify dimensions match reference expectations
        n_samples = x_np.shape[0]
        n_moving = len(self.atomic_info["moving_indices"])
        expected_dim = n_moving * 3
        
        if x_np.shape[1] != expected_dim:
            raise ValueError(
                f"Position dimension mismatch. Expected {expected_dim} "
                f"(3*{n_moving} moving atoms), got {x_np.shape[1]}"
            )
        
        if g_np.shape[1] != expected_dim:
            raise ValueError(
                f"Force dimension mismatch. Expected {expected_dim} "
                f"(3*{n_moving} moving atoms), got {g_np.shape[1]}"
            )
        
        # Convert to torch tensors and move to device - data is already in moving-atom-only format
        x = torch.tensor(x_np, dtype=torch.float64).to(self.device)
        y = torch.tensor(y_np, dtype=torch.float64).reshape(-1).to(self.device)
        g = torch.tensor(g_np, dtype=torch.float64).to(self.device)
        
        if self.model_type == "MultitaskGPModel_rbf_atomic":
            # Create stacked target tensor
            stacked_y_g = torch.cat([y.unsqueeze(1), g], dim=1)

            #Initialize hyperparameters intelligently
            hp_init, energy_range, range_x = self.initialize_hyperparameters(training_data)

            # Setup likelihood and model
            likelihood = self.get_likelihood(x, stacked_y_g, thermal_noise, model_name, minimum_noise)
            
            # Update likelihood to have correct number of tasks
            num_tasks = stacked_y_g.shape[1]
            likelihood.num_tasks = num_tasks
            
            model = MultitaskGPModel_rbf_atomic(
                train_x=x,
                train_y=stacked_y_g,
                likelihood=likelihood,
                conf_info=self.atomic_info
            )

            # Set initial values in the model
            from gpytorch.priors import NormalPrior
            if hasattr(model.covar_module, 'data_covar_module'):
                base_kernel = model.covar_module.data_covar_module

                # Set initial values IMMEDIATELY after model creation
                with torch.no_grad():
                    # Don't use the constraint transform - set raw values directly
                    base_kernel.raw_magnitude.data = torch.log(torch.tensor([hp_init['magnitude']], dtype=torch.float64, device=self.device))
                    base_kernel.raw_lengthscale.data = torch.log(torch.tensor(hp_init['lengthscales'], dtype=torch.float64, device=self.device))
                
                # Magnitude prior
                mag_prior = NormalPrior(0.0, max(1.0, energy_range/3))
                base_kernel.raw_magnitude.requires_grad = True
                base_kernel.register_prior('magnitude_prior', mag_prior, 'raw_magnitude')
                
                # Lengthscale prior
                len_prior = NormalPrior(0.0, max(1.0, range_x/3))
                base_kernel.raw_lengthscale.requires_grad = True
                base_kernel.register_prior('lengthscale_prior', len_prior, 'raw_lengthscale')

                # Ensure gradients are enabled
                for param in model.parameters():
                    param.requires_grad_(True)
                
                # Set noise levels
                if thermal_noise is None:
                    # Set a base noise level
                    likelihood.noise.fill_(hp_init['energy_noise'])
                    
                    # For multitask likelihood with task-specific noises
                    if hasattr(likelihood, 'raw_task_noises'):
                        noise_values = torch.full((num_tasks,), hp_init['force_noise'], dtype=torch.float64, device=self.device)
                        noise_values[0] = hp_init['energy_noise']
                        # Transform through constraint
                        if hasattr(likelihood, 'raw_task_noises_constraint'):
                            raw_values = likelihood.raw_task_noises_constraint.inverse_transform(noise_values)
                            likelihood.raw_task_noises.data = raw_values
            
            # Add regularization through constraints on the base kernel
            if hasattr(base_kernel, 'raw_magnitude_constraint'):
                # Update the constraint bounds
                base_kernel.raw_magnitude_constraint = gpytorch.constraints.Interval(
                    0.01 * hp_init['magnitude'],
                    100 * hp_init['magnitude']
                )

            # Handle heteroscedastic noise model for GP1
            if model_name == "GP1" and hasattr(self, '_use_heteroscedastic') and self._use_heteroscedastic:
                print(f"\nSetting up position-dependent noise learning for {model_name}")
                
                # For heteroscedastic, we'll let the likelihood learn different noise levels
                # for different regions of the input space by making noise trainable
                # and using a more flexible prior
                
                if hasattr(likelihood, 'raw_task_noises'):
                    # Add a prior that allows noise to vary more
                    from gpytorch.priors import LogNormalPrior
                    
                    # Get initial noise values
                    if hasattr(self, '_heteroscedastic_init_noise'):
                        force_noise, energy_noise = self._heteroscedastic_init_noise
                        mean_noise = (force_noise + energy_noise) / 2.0
                    else:
                        mean_noise = 0.1
                    
                    # LogNormal prior allows noise to vary over orders of magnitude
                    noise_prior = LogNormalPrior(
                        loc=torch.log(torch.tensor(mean_noise)),
                        scale=1.0  # This controls how much variation is allowed
                    )
                    
                    # Register the prior
                    likelihood.raw_task_noises.requires_grad = True
                    likelihood.register_prior('noise_prior', noise_prior, lambda m: m.raw_task_noises)
                    
                    print(f"  Noise will adapt based on input position")
                    print(f"  Using LogNormal prior with mean {mean_noise:.4f}")

            # Move model and likelihood to device
            model = model.to(self.device)
            likelihood = likelihood.to(self.device)
            
            # Training
            model.train()
            likelihood.train()

            # SINGLE optimizer for ALL cases - no special handling
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            # CRITICAL: For GP1, freeze noise parameters AFTER optimizer creation
            if model_name == "GP1" and thermal_noise is not None and (not hasattr(self, 'noise_model') or self.noise_model == "fixed"):
                if hasattr(likelihood, 'raw_task_noises'):
                    likelihood.raw_task_noises.requires_grad_(False)
                    # Remove from optimizer's param groups
                    for group in optimizer.param_groups:
                        group['params'] = [p for p in group['params'] if p is not likelihood.raw_task_noises]

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Training with convergence monitoring
            loss_history = []
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            # Track loss history
            loss_window = 10  # Look at recent history

            # Training loop with error handling
            try:
                for j in range(self.epoch):
                    optimizer.zero_grad()
                    
                    output = model(model.train_inputs[0])
                    loss = -mll(output, model.train_targets)
                    
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    loss_history.append(loss_val)
                
                    # Robust Check for convergence
                    if len(loss_history) > loss_window:
                        # Calculate average loss change over window
                        recent_losses = loss_history[-loss_window:]
                        loss_gradient = abs(recent_losses[-1] - recent_losses[0]) / loss_window
                        
                        # Stop if loss is changing very slowly
                        if loss_gradient < 1e-6:  # Average change per epoch
                            # print(f"Converged: loss gradient {loss_gradient:.2e} < threshold")
                            break

                    # Also check for best loss improvement
                    if loss_val < best_loss - 1e-5:  # Looser tolerance
                        best_loss = loss_val
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= patience and j > 500:
                        break

            except Exception as e:
                pass  # Training failed, fallback
            
            # NEW_v24: Ensure model and likelihood are on correct device after training
            model = model.to(self.device)
            likelihood = likelihood.to(self.device)
            
            self.model = model
            self.likelihood = likelihood

            # Calculate and print training errors
            self._calculate_training_metrics(training_data, model_name=model_name)
            
            # Log training data to GP logger
            gp_logger = get_gp_logger()

            # Extract final hyperparameters
            final_hyperparams = {}
            if hasattr(model.covar_module, 'data_covar_module'):
                base_kernel = model.covar_module.data_covar_module
                final_hyperparams['magnitude'] = float(base_kernel.magnitude.detach().cpu().numpy())
                final_hyperparams['lengthscales'] = base_kernel.lengthscale.detach().cpu().numpy().tolist()
                
            
            # ADD THIS: Extract final noise values from likelihood
            if hasattr(likelihood, 'noise'):
                # For standard likelihood
                final_hyperparams['energy_noise'] = float(likelihood.noise.detach().cpu().numpy())
                
            if hasattr(likelihood, 'raw_task_noises'):
                # For multitask likelihood - transform raw values to actual noise values
                task_noises = likelihood.raw_task_noises_constraint.transform(likelihood.raw_task_noises)
                task_noises_np = task_noises.detach().cpu().numpy()
                
                # Assuming task 0 is energy, tasks 1+ are forces
                if len(task_noises_np) > 0:
                    final_hyperparams['energy_noise'] = float(task_noises_np[0])
                if len(task_noises_np) > 1:
                    # For forces, you might want to average them or take the first one
                    final_hyperparams['force_noise'] = float(task_noises_np[1])
                
                # Optionally save all task noises
                final_hyperparams['all_task_noises'] = task_noises_np.tolist()

            # Prepare hyperparameters dict
            logged_hyperparams = {
                'magnitude': final_hyperparams.get('magnitude', float(hp_init['magnitude']) if 'hp_init' in locals() else None),
                'lengthscales': final_hyperparams.get('lengthscales', hp_init['lengthscales'].tolist() if 'hp_init' in locals() else None),
                'energy_noise': final_hyperparams.get('energy_noise', float(hp_init['energy_noise']) if 'hp_init' in locals() else None),
                'force_noise': final_hyperparams.get('force_noise', float(hp_init['force_noise']) if 'hp_init' in locals() else None),
                'thermal_noise': thermal_noise if thermal_noise else None,
                'num_tasks': num_tasks,
                'model_type': self.model_type,
                'initial_values': {  # Keep initial values for reference
                    'magnitude': float(hp_init['magnitude']) if 'hp_init' in locals() else None,
                    'lengthscales': hp_init['lengthscales'].tolist() if 'hp_init' in locals() else None,
                    'energy_noise': float(hp_init['energy_noise']) if 'hp_init' in locals() else None,
                    'force_noise': float(hp_init['force_noise']) if 'hp_init' in locals() else None,
                },
                'final_values': final_hyperparams  # This now includes noise values
            }

            # Log with training data
            logged_hyperparams['final_values'] = final_hyperparams

            # Prepare training metrics
            training_metrics = {}
            if hasattr(self, '_last_training_metrics'):
                training_metrics = self._last_training_metrics

            # Log to GP logger
            gp_logger.log_gp_training(
                model_name=model_name,
                iteration=getattr(self, 'current_iteration', 0),
                training_data=[x_np, y_np, g_np],
                hyperparameters=logged_hyperparams,
                training_metrics=training_metrics,
                loss_history=loss_history if 'loss_history' in locals() else []
            )

            # Log model info if first time
            if not hasattr(self, '_logged_model_info'):
                gp_logger.log_model_info(
                    model_name=model_name,
                    model_type=self.model_type,
                    atomic_info=self.atomic_info if self.atomic_info else {},
                    additional_info={
                        'min_epoch': self.epoch,
                        'likelihood_type': type(likelihood).__name__
                    }
                )
                self._logged_model_info = True

            if save_model:
                self.save_model(self.model, x, y, g, stacked_y_g, thermal_noise, path)
            
            return self
        elif self.model_type == "GPModelWithDerivatives_rbf_atomic":
            # ========== NEW DERIVATIVE-BASED IMPLEMENTATION ==========
            positions = x
            energies = y
            forces = g
            n_data = positions.shape[0]
            n_dims = positions.shape[1]
            
            # Create augmented training data for derivative observations
            # Format: [f(x1), f(x2), ..., df/dx1_1, df/dx1_2, ..., df/dx2_1, ...]
            
            # Create block diagonal structure for positions
            # Each training point contributes 1 + n_dims observations
            train_x_list = []
            train_y_list = []
            
            for i in range(n_data):
                # Add the position for function value
                train_x_list.append(positions[i:i+1, :])
                train_y_list.append(energies[i:i+1])
                
                # Add the same position for each derivative observation
                for j in range(n_dims):
                    train_x_list.append(positions[i:i+1, :])
                    train_y_list.append(-forces[i:i+1, j])  # Negative because force = -gradient
            
            train_x = torch.cat(train_x_list, dim=0)
            train_y = torch.cat(train_y_list, dim=0)
            
            # Create indicator matrix for derivative observations
            # This tells the kernel which observations are derivatives and w.r.t. which dimension
            n_total = train_x.shape[0]
            derivative_indicator = torch.zeros(n_total, n_dims + 1)
            
            idx = 0
            for i in range(n_data):
                # Function value indicator
                derivative_indicator[idx, 0] = 1.0
                idx += 1
                
                # Derivative indicators
                for j in range(n_dims):
                    derivative_indicator[idx, j + 1] = 1.0
                    idx += 1
            
            # Augment train_x with derivative indicators
            train_x_augmented = torch.cat([train_x, derivative_indicator], dim=1)
            
            # CRITICAL: Determine if this is GP1 or GP2
            is_gp1 = "GP1" in self.__class__.__name__ or model_name == "GP1"
            
            # Create likelihood based on GP type
            if is_gp1 and thermal_noise:
                # GP1: Fixed noise from thermal calculations
                noise_force, noise_energy = thermal_noise
                noise_values = []
                for i in range(n_data):
                    noise_values.append(noise_energy)  # Energy noise
                    noise_values.extend([noise_force] * n_dims)  # Force noise for each dimension
                noise_values = torch.tensor(noise_values, dtype=torch.float64)
                
                self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=noise_values,
                    learn_additional_noise=False  # No additional noise to learn
                )
            else:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-1)
                )
                # Initialize with small noise
                self.likelihood.noise = 1e-4
            
            # Create model
            self.model = GPModelWithDerivatives_rbf_atomic(
                train_x_augmented, train_y, self.likelihood, self.atomic_info
            )
            
            # INITIALIZE HYPERPARAMETERS PROPERLY
            hp_init, energy_range, range_x = self.initialize_hyperparameters(training_data)

            # Override noise values in hp_init for GP1
            if is_gp1 and thermal_noise:
                noise_force, noise_energy = thermal_noise
                hp_init['energy_noise'] = noise_energy
                hp_init['force_noise'] = noise_force
            
            with torch.no_grad():
                # Set outputscale (magnitude squared)
                self.model.covar_module.outputscale = hp_init['magnitude'] ** 2
                
                # Set lengthscales
                self.model.covar_module.lengthscale = torch.tensor(
                    hp_init['lengthscales'], 
                    dtype=torch.float64
                )
                
                # Only set noise for GP2 (not GP1 with thermal noise)
                if not is_gp1 and hasattr(self.likelihood, 'noise'):
                    # Don't square the noise value - it's already a standard deviation
                    noise_value = max(hp_init['energy_noise'], 1e-4)  # Ensure minimum noise
                    self.likelihood.noise = noise_value
            
            print(f"\nInitialized {self.model_type} with:")
            print(f"  Magnitude: {hp_init['magnitude']:.4f}")
            print(f"  Lengthscales: {hp_init['lengthscales']}")
            if is_gp1 and thermal_noise:
                print(f"  Noise: FIXED thermal (E: {noise_energy:.6f}, F: {noise_force:.6f})")
            else:
                print(f"  Noise: TRAINABLE (initial: {hp_init['energy_noise']:.6f})")



            # Hyperparameter initialization
            n_pt = self.atomic_info['n_pt']
            if n_pt > 0:
                with torch.no_grad():
                    # Initialize lengthscales based on atomic distances
                    if positions.shape[0] > 1:
                        # Use atomic distance for initialization
                        conf_info_np = {
                            'conf_fro': self.atomic_info['conf_fro'].cpu().numpy() if isinstance(self.atomic_info['conf_fro'], torch.Tensor) else self.atomic_info['conf_fro'],
                            'atomtype_mov': self.atomic_info['atomtype_mov'],
                            'atomtype_fro': self.atomic_info['atomtype_fro'],
                            'pairtype': self.atomic_info['pairtype'],
                            'n_pt': self.atomic_info['n_pt']
                        }
                        
                        pos_np = positions.cpu().numpy()
                        atomic_dists = calculate_atomic_distance_measure(pos_np, pos_np, conf_info_np, np.ones(n_pt))
                        
                        # Filter out zeros and get median
                        valid_dists = atomic_dists[atomic_dists > 0]
                        if len(valid_dists) > 0:
                            median_dist = np.median(valid_dists)
                            lengthscale_value = max(median_dist / 2.0, 0.1)  # Ensure minimum value
                        else:
                            lengthscale_value = 1.0
                    else:
                        lengthscale_value = 1.0
                    
                    # Ensure lengthscale_value is positive
                    lengthscale_value = max(lengthscale_value, 0.1)
                    
                    # Set lengthscale through the covar_module's lengthscale property
                    self.model.covar_module.lengthscale = lengthscale_value * torch.ones(n_pt, dtype=torch.float64)
                    
                    # Also initialize outputscale
                    y_std = train_y.std()
                    if y_std > 0:
                        self.model.covar_module.outputscale = (0.5 * y_std) ** 2
                    else:
                        self.model.covar_module.outputscale = 1.0
            
            # Training loop
            self.model.train()
            self.likelihood.train()
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            
            loss_history = []
            print("\nStarting optimization with derivative kernel...")
            
            for i in range(self.epoch):
                optimizer.zero_grad()
                output = self.model(train_x_augmented)
                loss = -mll(output, train_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                loss_history.append(loss.item())
                
                if i < 1:
                    print(f'Iter {i} - Loss: {loss.item():.3f}')
                if i == 0:
                    break  # Early break for initial iterations
            
            # NEW_v24: Ensure model and likelihood are on correct device after training
            self.model = self.model.to(self.device)
            self.likelihood = self.likelihood.to(self.device)
            
            # Switch to eval mode
            self.model.eval()
            self.likelihood.eval()
            
            # Store augmented data for predictions
            self.train_x_augmented = train_x_augmented
            self.derivative_indicator = derivative_indicator

            # ADD THIS SECTION AFTER THE TRAINING LOOP
            # Calculate and print training errors
            self._calculate_training_metrics(training_data, model_name=model_name)

            # Log training data to GP logger
            gp_logger = get_gp_logger()

            # Extract final hyperparameters
            final_hyperparams = {}

            # Get magnitude (outputscale is magnitude squared)
            if hasattr(self.model.covar_module, 'outputscale'):
                final_hyperparams['magnitude'] = float(torch.sqrt(self.model.covar_module.outputscale).detach().cpu().numpy())

            # Get lengthscales
            if hasattr(self.model.covar_module, 'lengthscale'):
                lengthscales_tensor = self.model.covar_module.lengthscale
                if lengthscales_tensor.dim() > 1:
                    lengthscales_tensor = lengthscales_tensor.squeeze()
                final_hyperparams['lengthscales'] = lengthscales_tensor.detach().cpu().numpy().tolist()

            # Handle noise based on model type
            if is_gp1 and thermal_noise:
                noise_force, noise_energy = thermal_noise
                final_hyperparams['energy_noise'] = float(noise_energy)
                final_hyperparams['force_noise'] = float(noise_force)
                final_hyperparams['all_task_noises'] = [float(noise_energy)] + [float(noise_force)] * n_dims
            else:
                if hasattr(self.likelihood, 'noise'):
                    learned_noise_variance = self.likelihood.noise.detach().cpu().numpy()
                    learned_noise_std = float(np.sqrt(learned_noise_variance))
                    final_hyperparams['energy_noise'] = learned_noise_std
                    final_hyperparams['force_noise'] = learned_noise_std
                    final_hyperparams['all_task_noises'] = [learned_noise_std] * (n_dims + 1)

            # Prepare hyperparameters dict for logging
            logged_hyperparams = {
                'magnitude': final_hyperparams.get('magnitude'),
                'lengthscales': final_hyperparams.get('lengthscales'),
                'energy_noise': final_hyperparams.get('energy_noise'),
                'force_noise': final_hyperparams.get('force_noise'),
                'thermal_noise': thermal_noise if thermal_noise else None,
                'num_tasks': n_dims + 1,
                'model_type': self.model_type,
                'initial_values': {
                    'magnitude': float(hp_init['magnitude']),
                    'lengthscales': hp_init['lengthscales'].tolist(),
                    'energy_noise': float(hp_init['energy_noise']),
                    'force_noise': float(hp_init['force_noise']),
                },
                'final_values': final_hyperparams
            }

            # Get training metrics if available
            training_metrics = getattr(self, '_last_training_metrics', {})

            # Log to GP logger
            gp_logger.log_gp_training(
                model_name=model_name,
                iteration=getattr(self, 'current_iteration', 0),
                training_data=[x_np, y_np, g_np],
                hyperparameters=logged_hyperparams,
                training_metrics=training_metrics,
                loss_history=loss_history
            )

            # Log model info if first time
            if not hasattr(self, '_logged_model_info'):
                gp_logger.log_model_info(
                    model_name=model_name,
                    model_type=self.model_type,
                    atomic_info=self.atomic_info if self.atomic_info else {},
                    additional_info={
                        'min_epoch': self.epoch,
                        'likelihood_type': type(self.likelihood).__name__,
                        'uses_derivatives': True
                    }
                )
                self._logged_model_info = True

            # Store augmented data for predictions
            self.train_x_augmented = train_x_augmented
            self.derivative_indicator = derivative_indicator

            print(f"\nTraining completed. Final loss: {loss_history[-1]:.3f}")
            if save_model:
                # Need to prepare the stacked_y_g format for consistency
                # For derivative model, we don't have the same format but we can create a compatible one
                stacked_y_g = torch.cat([train_y.unsqueeze(1), torch.zeros(train_y.shape[0], n_dims)], dim=1)
                self.save_model(self.model, train_x, train_y, forces, stacked_y_g, thermal_noise, path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\nTraining completed. Final loss: {loss_history[-1]:.3f}")
        
        # Validate model was created
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model creation failed during training")

    # def predict(
    #         self,
    #         test_data: npt.NDArray[np.float64]
    #     ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], 
    #             npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    #     """Make predictions using the trained model.
        
    #     Following reference implementation - works only with moving atoms.
        
    #     Args:
    #         test_data: Positions of moving atoms only (N_test × 3*N_mov)
            
    #     Returns:
    #         Tuple containing:
    #             - predicted_energy: Energy predictions (N_test,)
    #             - predicted_forces: Force predictions for moving atoms (N_test, 3*N_mov)
    #             - variance_energy: Energy variance (N_test,)
    #             - variance_forces: Force variance for moving atoms (N_test, 3*N_mov)
    #     """
    #     if self.model is None or self.likelihood is None:
    #         raise RuntimeError("Model must be trained before prediction")

    #     # Verify input dimensions
    #     n_moving = len(self.atomic_info["moving_indices"])
    #     expected_dim = n_moving * 3
        
    #     if test_data.ndim == 1:
    #         test_data = test_data.reshape(1, -1)
        
    #     if test_data.shape[1] != expected_dim:
    #         raise ValueError(
    #             f"Test data dimension mismatch. Expected {expected_dim} "
    #             f"(3*{n_moving} moving atoms), got {test_data.shape[1]}"
    #         )
        
    #     # Convert to tensor - data is already in moving-atom-only format
    #     test_data = torch.tensor(test_data, dtype=torch.float64)
    #     if self.model_type == "MultitaskGPModel_rbf_atomic":
    #         with torch.no_grad():
    #             # Get predictions
    #             self.model.eval()
    #             self.likelihood.eval()
    #             predictions = self.likelihood(self.model(test_data))
                
    #             n_samples = len(test_data)
    #             n_tasks = self.likelihood.num_tasks
                
    #             # Reshape predictions
    #             mean_flat = predictions.mean
    #             var_flat = predictions.variance
                
    #             # Reshape to separate tasks
    #             mean_reshaped = mean_flat.reshape(n_samples, n_tasks)
    #             var_reshaped = var_flat.reshape(n_samples, n_tasks)
                
    #             # Extract components
    #             predicted_energy = mean_reshaped[:, 0].detach().numpy()
    #             predicted_forces = mean_reshaped[:, 1:].detach().numpy()
    #             variance_energy = var_reshaped[:, 0].detach().numpy()
    #             variance_forces = var_reshaped[:, 1:].detach().numpy()

    #         return predicted_energy, predicted_forces, variance_energy, variance_forces
    #     elif self.model_type == "GPModelWithDerivatives_rbf_atomic":
    #         # Check if model has been trained
    #         if not hasattr(self, 'train_x_augmented'):
    #             # Model hasn't been trained yet, train it with current data
    #             print("Model not trained yet, training with current data...")
    #             if hasattr(self, 'training_data') and self.training_data is not None:
    #                 self.train(
    #                     training_data=self.training_data,
    #                     thermal_noise=getattr(self, 'thermal_noise', None),
    #                     model_name="GP1" if "GP1" in self.__class__.__name__ else "GP2"
    #                 )
    #             else:
    #                 raise RuntimeError("No training data available for model training")
            
    #         # Derivative-based prediction
    #         test_x = torch.tensor(test_data, dtype=torch.float64)
    #         n_test = test_x.shape[0]
    #         n_dims = test_x.shape[1]
            
    #         # Create augmented test data
    #         test_x_list = []
            
    #         for i in range(n_test):
    #             # Function value query
    #             x_with_indicator = torch.zeros(1, test_x.shape[1] + n_dims + 1, dtype=torch.float64)
    #             x_with_indicator[0, :n_dims] = test_x[i]
    #             x_with_indicator[0, n_dims] = 1.0  # Function value indicator
    #             test_x_list.append(x_with_indicator)
                
    #             # Derivative queries
    #             for j in range(n_dims):
    #                 x_with_indicator = torch.zeros(1, test_x.shape[1] + n_dims + 1, dtype=torch.float64)
    #                 x_with_indicator[0, :n_dims] = test_x[i]
    #                 x_with_indicator[0, n_dims + 1 + j] = 1.0  # Derivative indicator
    #                 test_x_list.append(x_with_indicator)
            
    #         test_x_augmented = torch.cat(test_x_list, dim=0)
            
    #         # Make predictions - the key is to handle both eval and train modes
    #         self.model.eval()
    #         self.likelihood.eval()
            
    #         with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #             # CRITICAL FIX: Use the model's __call__ method properly
    #             # Option 1: If we're predicting on training inputs
    #             if hasattr(self, 'train_x_augmented') and torch.equal(test_x_augmented, self.train_x_augmented):
    #                 # We're predicting on training data
    #                 predictions = self.likelihood(self.model(test_x_augmented))
    #             else:
    #                 # Option 2: We're predicting on new data
    #                 # Set up the model for prediction by setting training data
    #                 if hasattr(self.model, 'set_train_data'):
    #                     # Temporarily store the current training data
    #                     train_inputs = self.model.train_inputs[0] if hasattr(self.model, 'train_inputs') else self.train_x_augmented
    #                     train_targets = self.model.train_targets if hasattr(self.model, 'train_targets') else self.model.train_targets
                        
    #                     # Make prediction
    #                     try:
    #                         predictions = self.likelihood(self.model(test_x_augmented))
    #                     except RuntimeError as e:
    #                         if "You must train on the training inputs" in str(e):
    #                             # This means we need to use a different approach
    #                             # Use the conditional distribution given training data
    #                             with gpytorch.settings.cholesky_jitter(1e-3):
    #                                 # Get the full covariance matrix
    #                                 full_output = self.model.forward(train_inputs, test_x_augmented)
    #                                 predictions = self.likelihood(full_output)
    #                         else:
    #                             raise e
    #                 else:
    #                     # Fallback: just try to predict
    #                     predictions = self.likelihood(self.model(test_x_augmented))
                
    #             mean = predictions.mean
    #             var = predictions.variance
            
    #         # Convert to numpy
    #         mean_np = mean.numpy()
    #         var_np = var.numpy()
            
    #         # Extract energies and forces
    #         pred_energies = []
    #         pred_forces = []
    #         var_energies = []
    #         var_forces = []
            
    #         idx = 0
    #         for i in range(n_test):
    #             # Energy
    #             pred_energies.append(mean_np[idx])
    #             var_energies.append(var_np[idx])
    #             idx += 1
                
    #             # Forces (negative of gradients)
    #             forces_i = []
    #             vars_i = []
    #             for j in range(n_dims):
    #                 forces_i.append(mean_np[idx])  # There must be No negative sign here! OMG!
    #                 vars_i.append(var_np[idx])
    #                 idx += 1
                
    #             pred_forces.append(forces_i)
    #             var_forces.append(vars_i)
            
    #         pred_energies = np.array(pred_energies)
    #         pred_forces = np.array(pred_forces)
    #         var_energies = np.array(var_energies)
    #         var_forces = np.array(var_forces)
            
    #         return pred_energies, pred_forces, var_energies, var_forces


    def predict(
            self,
            test_data: npt.NDArray[np.float64]
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], 
                npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Make predictions using the trained model."""
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be trained before prediction")

        # Verify input dimensions
        n_moving = len(self.atomic_info["moving_indices"])
        expected_dim = n_moving * 3
        
        if test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)
        
        if test_data.shape[1] != expected_dim:
            raise ValueError(
                f"Test data dimension mismatch. Expected {expected_dim} "
                f"(3*{n_moving} moving atoms), got {test_data.shape[1]}"
            )
        
        # Convert to tensor and move to device - data is already in moving-atom-only format
        test_data = torch.tensor(test_data, dtype=torch.float64).to(self.device)
        
        if self.model_type == "MultitaskGPModel_rbf_atomic":
            with torch.no_grad():
                # Get predictions
                self.model.eval()
                self.likelihood.eval()
                
                # IMPORTANT: For heteroscedastic noise, we need to handle it differently
                if hasattr(self, 'noise_model') and self.noise_model == "heteroscedastic" and hasattr(self.likelihood, 'noise_model'):
                    # For heteroscedastic, the likelihood needs the noise model to be evaluated first
                    self.likelihood.noise_model.eval()
                    
                predictions = self.likelihood(self.model(test_data))
                
                n_samples = len(test_data)
                n_tasks = self.likelihood.num_tasks
                
                # For Student-t likelihood, extract mean and variance differently
                if hasattr(self.likelihood, '__class__') and 'StudentT' in self.likelihood.__class__.__name__:
                    # Student-t has different variance calculation
                    mean_flat = predictions.mean
                    # For Student-t, variance includes the degrees of freedom adjustment
                    # var = scale^2 * nu / (nu - 2) for nu > 2
                    var_flat = predictions.variance
                else:
                    # Standard Gaussian case
                    mean_flat = predictions.mean
                    var_flat = predictions.variance
                
                # Reshape to separate tasks
                mean_reshaped = mean_flat.reshape(n_samples, n_tasks)
                var_reshaped = var_flat.reshape(n_samples, n_tasks)
                
                # Extract components
                predicted_energy = mean_reshaped[:, 0].detach().cpu().numpy()
                predicted_forces = mean_reshaped[:, 1:].detach().cpu().numpy()
                variance_energy = var_reshaped[:, 0].detach().cpu().numpy()
                variance_forces = var_reshaped[:, 1:].detach().cpu().numpy()

            return predicted_energy, predicted_forces, variance_energy, variance_forces
        elif self.model_type == "GPModelWithDerivatives_rbf_atomic":
            # Check if model has been trained
            if not hasattr(self, 'train_x_augmented'):
                # Model hasn't been trained yet, train it with current data
                print("Model not trained yet, training with current data...")
                if hasattr(self, 'training_data') and self.training_data is not None:
                    self.train(
                        training_data=self.training_data,
                        thermal_noise=getattr(self, 'thermal_noise', None),
                        model_name="GP1" if "GP1" in self.__class__.__name__ else "GP2"
                    )
                else:
                    raise RuntimeError("No training data available for model training")
            
            # Derivative-based prediction
            test_x = torch.tensor(test_data, dtype=torch.float64)
            n_test = test_x.shape[0]
            n_dims = test_x.shape[1]
            
            # Create augmented test data
            test_x_list = []
            
            for i in range(n_test):
                # Function value query
                x_with_indicator = torch.zeros(1, test_x.shape[1] + n_dims + 1, dtype=torch.float64)
                x_with_indicator[0, :n_dims] = test_x[i]
                x_with_indicator[0, n_dims] = 1.0  # Function value indicator
                test_x_list.append(x_with_indicator)
                
                # Derivative queries
                for j in range(n_dims):
                    x_with_indicator = torch.zeros(1, test_x.shape[1] + n_dims + 1, dtype=torch.float64)
                    x_with_indicator[0, :n_dims] = test_x[i]
                    x_with_indicator[0, n_dims + 1 + j] = 1.0  # Derivative indicator
                    test_x_list.append(x_with_indicator)
            
            test_x_augmented = torch.cat(test_x_list, dim=0)
            
            # Make predictions - the key is to handle both eval and train modes
            self.model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # CRITICAL FIX: Use the model's __call__ method properly
                # Option 1: If we're predicting on training inputs
                if hasattr(self, 'train_x_augmented') and torch.equal(test_x_augmented, self.train_x_augmented.to(test_x_augmented.device)):
                    # We're predicting on training data
                    predictions = self.likelihood(self.model(test_x_augmented))
                else:
                    # Option 2: We're predicting on new data
                    # Set up the model for prediction by setting training data
                    if hasattr(self.model, 'set_train_data'):
                        # Temporarily store the current training data
                        train_inputs = self.model.train_inputs[0] if hasattr(self.model, 'train_inputs') else self.train_x_augmented
                        train_targets = self.model.train_targets if hasattr(self.model, 'train_targets') else self.model.train_targets
                        
                        # Make prediction
                        try:
                            predictions = self.likelihood(self.model(test_x_augmented))
                        except RuntimeError as e:
                            if "You must train on the training inputs" in str(e):
                                # This means we need to use a different approach
                                # Use the conditional distribution given training data
                                with gpytorch.settings.cholesky_jitter(1e-3):
                                    # Get the full covariance matrix
                                    full_output = self.model.forward(train_inputs, test_x_augmented)
                                    predictions = self.likelihood(full_output)
                            else:
                                raise e
                    else:
                        # Fallback: just try to predict
                        predictions = self.likelihood(self.model(test_x_augmented))
                
                mean = predictions.mean
                var = predictions.variance
            
            # Convert to numpy
            mean_np = mean.numpy()
            var_np = var.numpy()
            
            # Extract energies and forces
            pred_energies = []
            pred_forces = []
            var_energies = []
            var_forces = []
            
            idx = 0
            for i in range(n_test):
                # Energy
                pred_energies.append(mean_np[idx])
                var_energies.append(var_np[idx])
                idx += 1
                
                # Forces (negative of gradients)
                forces_i = []
                vars_i = []
                for j in range(n_dims):
                    forces_i.append(mean_np[idx])  # There must be No negative sign here! OMG!
                    vars_i.append(var_np[idx])
                    idx += 1
                
                pred_forces.append(forces_i)
                var_forces.append(vars_i)
            
            pred_energies = np.array(pred_energies)
            pred_forces = np.array(pred_forces)
            var_energies = np.array(var_energies)
            var_forces = np.array(var_forces)
            
            return pred_energies, pred_forces, var_energies, var_forces
        
        elif self.model_type == "MultitaskGPModel":
            # Handle the new MultitaskGPModel (similar to MultitaskGPModel_rbf_atomic)
            with torch.no_grad():
                # Get predictions
                self.model.eval()
                self.likelihood.eval()
                
                predictions = self.likelihood(self.model(test_data))
                
                n_samples = len(test_data)
                n_tasks = self.likelihood.num_tasks
                
                # Extract mean and variance
                mean_flat = predictions.mean
                var_flat = predictions.variance
                
                # Reshape predictions (n_samples, n_tasks)
                if mean_flat.dim() == 1:
                    # If flat, reshape to (n_samples, n_tasks)
                    mean = mean_flat.view(n_samples, n_tasks)
                    var = var_flat.view(n_samples, n_tasks)
                else:
                    mean = mean_flat
                    var = var_flat
                
                # Extract energy (first task) and forces (remaining tasks)
                pred_energies = mean[:, 0].cpu().numpy()
                pred_forces = mean[:, 1:].cpu().numpy()
                var_energies = var[:, 0].cpu().numpy()
                var_forces = var[:, 1:].cpu().numpy()
                
                return pred_energies, pred_forces, var_energies, var_forces
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


    def get_predictions(self, position: npt.NDArray[np.float64]) -> Tuple[float, float]:
        """Get predictions WITHOUT retraining - unified method for GP1 and GP2.
        
        Args:
            position: Position to predict at
            
        Returns:
            Tuple of (energy, force_norm)
        """
        try:
            pred_e, pred_f, _, _ = self.predict(position.reshape(1, -1))
            pred_e = float(pred_e[0]) if hasattr(pred_e, '__len__') else float(pred_e)
            
            # Check if this looks like an unreferenced energy (unless disabled)
            if (hasattr(self, 'energy_reference') and self.energy_reference is not None 
                and not getattr(self, 'disable_auto_referencing', False)):
                if abs(pred_e) > 100:  # Likely unreferenced
                    pred_e = pred_e - self.energy_reference
            
            pred_fn = float(np.linalg.norm(pred_f.flatten()))
            return pred_e, pred_fn
        except:
            return float('nan'), float('nan')

    def validate_accuracy(self, split_ratio: float = 0.8, model_name: str = "GP") -> Dict[str, float]:
        """Validate GP accuracy on training data with proper force handling.
        
        Args:
            split_ratio: Ratio of data to use for training (rest for validation)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing validation metrics
        """
        if not hasattr(self, 'training_data') or len(self.training_data[0]) < 5:
            return {}  # Need at least 5 points for validation
        
        # Import the logger
        gp_logger = get_gp_logger()
        
        # Get training data
        positions = self.training_data[0]
        energies = self.training_data[1]
        forces = self.training_data[2]

        # Check if all energies are properly referenced
        if not hasattr(self, 'energy_reference') or self.energy_reference is None:
            print("WARNING: No energy reference set. Skipping validation.")
            return {}
            
        # Check energy scale
        if abs(np.mean(energies)) > 10:  # Referenced energies should be near zero
            print(f"WARNING: Energy scale looks wrong (mean={np.mean(energies):.4f}). Skipping validation.")
            return {}
        
        # Check if we have mixed referenced/unreferenced energies
        energy_min = np.min(energies)
        energy_max = np.max(energies)
        energy_range = energy_max - energy_min
        
        print(f"\nValidation - Training data energy statistics:")
        print(f"  Mean: {np.mean(energies):.4f}")
        print(f"  Range: [{energy_min:.4f}, {energy_max:.4f}]")
        print(f"  Reference: {self.energy_reference if hasattr(self, 'energy_reference') else 'None'}")
        
        # If we have a huge range, likely mixed referenced/unreferenced data (unless disabled)
        if (energy_range > 100 and hasattr(self, 'energy_reference') and self.energy_reference is not None
            and not getattr(self, 'disable_auto_referencing', False)):
            print("  Detected mixed referenced/unreferenced energies in training data")
            # Apply reference to any unreferenced energies (those with large positive values)
            unreferenced_mask = energies > 100  # Energies above 100 eV are likely unreferenced
            if np.any(unreferenced_mask):
                print(f"  Applying reference to {np.sum(unreferenced_mask)} unreferenced energies")
                energies[unreferenced_mask] = energies[unreferenced_mask] - self.energy_reference
                print(f"  New range: [{np.min(energies):.4f}, {np.max(energies):.4f}]")
        
        # Skip validation if energies still look wrong
        if abs(np.mean(energies)) > 50:
            print("WARNING: Energy scale still looks wrong. Skipping validation.")
            return {}

        # Random split
        n_samples = len(positions)
        indices = np.random.permutation(n_samples)
        n_train = int(split_ratio * n_samples)
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # Create training data for temporary model
        train_data = [
            positions[train_idx],
            energies[train_idx],
            forces[train_idx]
        ]
        
        # Create and train temporary GP for validation
        from gp2_model import GP2
        temp_gp = GP2(
            training_data=[
                train_data[0][0:1],  # Use first point for initialization
                train_data[1][0:1],
                train_data[2][0:1]
            ],
            atomic_info=self.atomic_info
        )
        temp_gp.model_type = self.model_type

        # IMPORTANT: Copy the energy reference to the temporary model
        if hasattr(self, 'energy_reference'):
            temp_gp.energy_reference = self.energy_reference
        
        # Train the temporary model with the rest of the training data
        for i in range(1, len(train_idx)):
            update_data = [
                train_data[0][i:i+1],
                train_data[1][i:i+1],
                train_data[2][i:i+1]
            ]
            temp_gp.update(update_data, thermal_noise=getattr(self, 'thermal_noise', None))
        
        # Test on held-out data
        test_errors_e = []
        test_errors_f = []
        detailed_results = []
        
        for idx in test_idx:
            # Get prediction - this returns forces for moving atoms only
            pred_e, pred_f, var_e, var_f = temp_gp.predict(positions[idx:idx+1])
            
            # Get true values
            true_e = energies[idx]
            true_f_full = forces[idx]  # Full system forces
            
            # Extract forces for moving atoms from true forces
            moving_indices = self.atomic_info["moving_indices"]
            true_f_moving = []
            for atom_idx in moving_indices:
                true_f_moving.extend(true_f_full[atom_idx*3:(atom_idx+1)*3])
            true_f_moving = np.array(true_f_moving)
            
            # Calculate errors
            error_e = abs(pred_e[0] - true_e)
            test_errors_e.append(error_e)
            
            # Force error - compare moving atom forces
            if len(pred_f.flatten()) == len(true_f_moving):
                force_diff = pred_f.flatten() - true_f_moving
                error_f_mae = np.mean(np.abs(force_diff))
                error_f_rmse = np.sqrt(np.mean(force_diff**2))
                error_f_max = np.max(np.abs(force_diff))
                test_errors_f.append(error_f_mae)
                
                # Store detailed results for logging
                detailed_results.append({
                    'sample_idx': int(idx),
                    'energy_error': float(error_e),
                    'force_mae': float(error_f_mae),
                    'force_rmse': float(error_f_rmse),
                    'force_max': float(error_f_max),
                    'pred_force_magnitude': float(np.linalg.norm(pred_f)),
                    'true_force_magnitude': float(np.linalg.norm(true_f_moving))
                })
            else:
                print(f"Warning: Force dimension mismatch in validation")
                print(f"  Predicted: {len(pred_f.flatten())}, True: {len(true_f_moving)}")
        
        # Calculate overall metrics
        mean_test_error_e = np.mean(test_errors_e) if test_errors_e else 0.0
        mean_test_error_f = np.mean(test_errors_f) if test_errors_f else 0.0
        
        # Prepare validation data for logging
        validation_data = {
            'n_train': n_train,
            'n_test': len(test_idx),
            'energy_mae': float(mean_test_error_e),
            'energy_rmse': float(np.sqrt(np.mean(np.array(test_errors_e)**2))) if test_errors_e else 0.0,
            'energy_max': float(np.max(test_errors_e)) if test_errors_e else 0.0,
            'force_mae': float(mean_test_error_f),
            'force_rmse': float(np.sqrt(np.mean(np.array(test_errors_f)**2))) if test_errors_f else 0.0,
            'force_max': float(np.max(test_errors_f)) if test_errors_f else 0.0,
            'detailed_results': detailed_results,
            'n_moving_atoms': len(self.atomic_info["moving_indices"]),
            'force_errors_computed': len(test_errors_f)
        }
        
        # Log to GP logger
        gp_logger.log_gp_validation(model_name, getattr(self, 'current_iteration', 0), validation_data)
        
        # Print results
        print(f"\n=== {model_name} Cross-Validation (on {len(test_idx)} test points) ===")
        print(f"Energy MAE: {mean_test_error_e:.6f} eV")
        print(f"Energy RMSE: {validation_data['energy_rmse']:.6f} eV")
        print(f"Energy Max: {validation_data['energy_max']:.6f} eV")
        print(f"Force MAE:  {mean_test_error_f:.6f} eV/Å")
        print(f"Force RMSE: {validation_data['force_rmse']:.6f} eV/Å")
        print(f"Force Max:  {validation_data['force_max']:.6f} eV/Å")
        print(f"Force errors computed: {len(test_errors_f)}/{len(test_idx)}")
        print("=" * 50 + "\n")
        
        if mean_test_error_e > 0.1:  # Threshold in eV
            print("WARNING: GP validation error is high!")
            
        # Additional diagnostics if force errors are missing
        if len(test_errors_f) == 0:
            print("WARNING: No force errors were computed!")
            print("This might indicate a problem with force predictions or data format.")
            print(f"Number of moving atoms: {len(self.atomic_info['moving_indices'])}")
            print(f"Expected force dimension: {len(self.atomic_info['moving_indices']) * 3}")
            
        return validation_data

    # def get_likelihood(self, x: torch.Tensor, y: torch.Tensor, noise: Optional[Tuple[float, float]] = None, model_name: str = "GP"):
    #     """
    #     Create an appropriate multitask likelihood for the GP model.
    #     For GP1: Fixed noise when thermal noise is provided
    #     For GP2: Always trainable noise
    #     """
    #     num_tasks = y.size(1)
    #     dtype = y.dtype
    #     device = y.device
        
    #     # Use higher noise floor to avoid numerical issues
    #     likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    #         num_tasks=num_tasks,
    #         noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    #     )
        
    #     # Determine model type from class name
    #     is_gp1 = "GP1" in self.__class__.__name__
    #     print(f"Creating likelihood for {model_name} - is_gp1: {is_gp1}, num_tasks: {num_tasks}")

    #     if noise is not None:
    #         force_noise_val = abs(float(noise[0]))
    #         energy_noise_val = abs(float(noise[1]))
            
    #         # Create a tensor for the desired noise variance for each task
    #         noise_values_per_task = torch.full((num_tasks,), force_noise_val, dtype=dtype, device=device)
    #         noise_values_per_task[0] = energy_noise_val
            
    #         if hasattr(likelihood, 'raw_task_noises') and hasattr(likelihood, 'raw_task_noises_constraint'):
    #             raw_values = likelihood.raw_task_noises_constraint.inverse_transform(noise_values_per_task)
                
    #             # Set the values
    #             with torch.no_grad():
    #                 if likelihood.raw_task_noises.shape == raw_values.shape:
    #                     likelihood.raw_task_noises.data.copy_(raw_values)
    #                 elif likelihood.raw_task_noises.dim() == raw_values.dim() + 1 and likelihood.raw_task_noises.shape[0] == 1:
    #                     likelihood.raw_task_noises.data.copy_(raw_values.unsqueeze(0))
                
    #             # CRITICAL: Set requires_grad based on model type
    #             if is_gp1:
    #                 # GP1: Fixed noise (not trainable)
    #                 likelihood.raw_task_noises.requires_grad_(False)
    #             else:
    #                 # GP2: Trainable noise
    #                 likelihood.raw_task_noises.requires_grad_(True)
        
    #     return likelihood

    def get_likelihood(self, x: torch.Tensor, y: torch.Tensor, noise: Optional[Tuple[float, float]] = None, model_name: str = "GP", minimum_noise: Optional[Tuple[float, float]] = None):
        """
        Create an appropriate multitask likelihood for the GP model.
        
        Args:
            x: Input tensor
            y: Target tensor
            noise: Noise values (force_noise, energy_noise)
            model_name: Name of the model
            minimum_noise: Minimum noise constraint from trgp1 (force_noise, energy_noise)
        """
        num_tasks = y.size(1)
        dtype = y.dtype
        device = y.device
        
        # Determine if this is GP1
        is_gp1 = "GP1" in self.__class__.__name__ or model_name == "GP1"
        
        if is_gp1:
            # GP1 cases
            if hasattr(self, 'noise_model') and self.noise_model != "fixed":
                # Special noise models for GP1
                if self.noise_model == "heteroscedastic":
                    print(f"\n{model_name}: Using HETEROSCEDASTIC noise model")
                    
                    # In GPyTorch, heteroscedastic noise is handled differently
                    # We use a standard MultitaskGaussianLikelihood but will modify
                    # the noise model in the training phase
                    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=num_tasks,
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                        has_global_noise=False,  # Each task can have different noise
                        has_task_noise=True
                    )
                    
                    # Store thermal noise for initialization if provided
                    if noise is not None:
                        self._heteroscedastic_init_noise = noise
                        # Initialize with thermal noise
                        force_noise_val = abs(float(noise[0]))
                        energy_noise_val = abs(float(noise[1]))
                        
                        # Apply minimum noise constraint if provided
                        if minimum_noise is not None:
                            min_force_noise = abs(float(minimum_noise[0]))
                            min_energy_noise = abs(float(minimum_noise[1]))
                            force_noise_val = max(force_noise_val, min_force_noise)
                            energy_noise_val = max(energy_noise_val, min_energy_noise)
                            self._heteroscedastic_min_noise = minimum_noise
                            print(f"  Applied minimum noise constraint: F={min_force_noise:.6f}, E={min_energy_noise:.6f}")
                        
                        noise_values_per_task = torch.full((num_tasks,), force_noise_val, dtype=dtype, device=device)
                        noise_values_per_task[0] = energy_noise_val
                        
                        if hasattr(likelihood, 'raw_task_noises') and hasattr(likelihood, 'raw_task_noises_constraint'):
                            raw_values = likelihood.raw_task_noises_constraint.inverse_transform(noise_values_per_task)
                            with torch.no_grad():
                                if likelihood.raw_task_noises.shape == raw_values.shape:
                                    likelihood.raw_task_noises.data.copy_(raw_values)
                                elif likelihood.raw_task_noises.dim() == raw_values.dim() + 1:
                                    likelihood.raw_task_noises.data.copy_(raw_values.unsqueeze(0))
                        
                        # Make noise trainable for heteroscedastic
                        if hasattr(likelihood, 'raw_task_noises'):
                            likelihood.raw_task_noises.requires_grad_(True)
                        
                        print(f"  Will learn position-dependent noise")
                        print(f"  Initial: E={energy_noise_val:.6f}, F={force_noise_val:.6f}")
                    
                    # Mark that we need special handling in training
                    self._use_heteroscedastic = True
                    
                elif self.noise_model == "student_t":
                    print(f"\n{model_name}: Using STUDENT-T likelihood (df={self.student_t_df})")
                    
                    # Check if adaptive df is enabled
                    effective_df = self.student_t_df
                    
                    if hasattr(self, 'use_adaptive_df') and self.use_adaptive_df:
                        # Determine effective df based on iteration
                        if hasattr(self, 'current_iteration'):
                            try:
                                # Extract iteration number
                                iter_str = str(self.current_iteration)
                                if 'eval_' in iter_str:
                                    iter_num = int(iter_str.split('_')[-1])
                                else:
                                    iter_num = 0
                                
                                # Define adaptive strategy
                                if hasattr(self, 'adaptive_df_schedule'):
                                    # Use custom schedule if provided
                                    schedule = self.adaptive_df_schedule
                                else:
                                    # Default schedule: transition from initial df to target df
                                    start_iter = getattr(self, 'adaptive_df_start_iter', 15)
                                    end_iter = getattr(self, 'adaptive_df_end_iter', 25)
                                    target_df = getattr(self, 'adaptive_df_target', 1.0)
                                    
                                    if iter_num <= start_iter:
                                        effective_df = self.student_t_df
                                    elif iter_num >= end_iter:
                                        effective_df = target_df
                                    else:
                                        # Linear interpolation
                                        progress = (iter_num - start_iter) / (end_iter - start_iter)
                                        effective_df = self.student_t_df + progress * (target_df - self.student_t_df)
                                
                                if effective_df != self.student_t_df:
                                    print(f"  Adaptive df: {effective_df:.2f} (iteration {iter_num})")
                            except Exception as e:
                                print(f"  Warning: Could not determine iteration for adaptive df: {e}")
                                effective_df = self.student_t_df
                    
                    # StudentTLikelihood is single-task only, so we need a different approach
                    # Option 1: Use a single StudentTLikelihood and handle multitask manually
                    # Option 2: Use robust Gaussian with adjusted parameters
                    
                    # For now, let's use a robust approach with Gaussian likelihood
                    # but with modified noise handling to simulate heavy tails
                    print(f"  Note: Using robust Gaussian likelihood to simulate Student-t behavior")
                    
                    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=num_tasks,
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)  # Larger floor for robustness
                    )
                    
                    # Initialize with modified noise for robustness
                    if noise is not None:
                        force_noise_val = abs(float(noise[0]))
                        energy_noise_val = abs(float(noise[1]))
                        
                        # Apply minimum noise constraint if provided
                        if minimum_noise is not None:
                            min_force_noise = abs(float(minimum_noise[0]))
                            min_energy_noise = abs(float(minimum_noise[1]))
                            force_noise_val = max(force_noise_val, min_force_noise)
                            energy_noise_val = max(energy_noise_val, min_energy_noise)
                            print(f"  Applied minimum noise constraint: F={min_force_noise:.6f}, E={min_energy_noise:.6f}")
                        
                        # Inflate noise to account for heavy tails
                        # Lower df means heavier tails, so more inflation
                        inflation_factor = max(1.0, 4.0 / effective_df)
                        
                        force_noise_val *= inflation_factor
                        energy_noise_val *= inflation_factor
                        
                        noise_values_per_task = torch.full((num_tasks,), force_noise_val, dtype=dtype, device=device)
                        noise_values_per_task[0] = energy_noise_val
                        
                        if hasattr(likelihood, 'raw_task_noises') and hasattr(likelihood, 'raw_task_noises_constraint'):
                            raw_values = likelihood.raw_task_noises_constraint.inverse_transform(noise_values_per_task)
                            with torch.no_grad():
                                if likelihood.raw_task_noises.shape == raw_values.shape:
                                    likelihood.raw_task_noises.data.copy_(raw_values)
                                elif likelihood.raw_task_noises.dim() == raw_values.dim() + 1 and likelihood.raw_task_noises.shape[0] == 1:
                                    likelihood.raw_task_noises.data.copy_(raw_values.unsqueeze(0))
                            
                            # Make it trainable to adapt to outliers
                            likelihood.raw_task_noises.requires_grad_(True)
                        
                    
                    # Store flag to indicate robust mode
                    self._using_robust_gaussian = True
                
                else:
                    raise ValueError(f"Unknown noise model: {self.noise_model}")
                    
            else:
                # Standard GP1 with fixed thermal noise (original behavior)
                
                # Use higher noise floor to avoid numerical issues
                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks,
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
                )
                
                if noise is not None:
                    force_noise_val = abs(float(noise[0]))
                    energy_noise_val = abs(float(noise[1]))
                    
                    # Create a tensor for the desired noise variance for each task
                    noise_values_per_task = torch.full((num_tasks,), force_noise_val, dtype=dtype, device=device)
                    noise_values_per_task[0] = energy_noise_val
                    
                    if hasattr(likelihood, 'raw_task_noises') and hasattr(likelihood, 'raw_task_noises_constraint'):
                        raw_values = likelihood.raw_task_noises_constraint.inverse_transform(noise_values_per_task)
                        
                        # Set the values
                        with torch.no_grad():
                            if likelihood.raw_task_noises.shape == raw_values.shape:
                                likelihood.raw_task_noises.data.copy_(raw_values)
                            elif likelihood.raw_task_noises.dim() == raw_values.dim() + 1 and likelihood.raw_task_noises.shape[0] == 1:
                                likelihood.raw_task_noises.data.copy_(raw_values.unsqueeze(0))
                        
                        # CRITICAL: Fixed noise for standard GP1
                        likelihood.raw_task_noises.requires_grad_(False)
                else:
                    pass  # No thermal noise provided

        else:
            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_tasks,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            )
            
            # Initialize with small values for GP2
            with torch.no_grad():
                if hasattr(likelihood, 'raw_task_noises'):
                    # Different noise for each task
                    init_noise = torch.full((num_tasks,), 1e-4, dtype=dtype, device=device)
                    init_noise[0] = 1e-4  # Energy noise
                    raw_values = likelihood.raw_task_noises_constraint.inverse_transform(init_noise)
                    
                    if likelihood.raw_task_noises.shape == raw_values.shape:
                        likelihood.raw_task_noises.data.copy_(raw_values)
                    elif likelihood.raw_task_noises.dim() == raw_values.dim() + 1:
                        likelihood.raw_task_noises.data.copy_(raw_values.unsqueeze(0))
                    
                    # Ensure it's trainable for GP2
                    likelihood.raw_task_noises.requires_grad_(True)
                else:
                    # Single noise parameter
                    likelihood.noise = 1e-4
            
        return likelihood

    def prepare_training_data(
            self,
            training_data: List[npt.NDArray[np.float64]]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for GP training.
        
        Args:
            training_data: List of [positions, energies, forces]
            
        Returns:
            Tuple containing:
                - x: Position tensor
                - y: Energy tensor
                - g: Force tensor
                - stacked_y_g: Combined tensor of energies and forces
        """
        x, y, g = training_data
        g = g.reshape(np.shape(x))
        x = torch.tensor(x)
        y = torch.tensor(y.ravel())
        g = torch.tensor(g)
        stacked_y_g = torch.stack(
            [y, torch.tensor(g[:, 0]), torch.tensor(g[:, 1])], -1
        ).squeeze(1)
        
        return x, y, g, stacked_y_g

    def save_model(
        self,
        model: gpytorch.models.ExactGP,
        x: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        stacked_y_g: torch.Tensor,
        thermal_noise: Optional[Tuple[float, float]],
        path: Optional[str]
        ) -> None:
        """Save the trained model and its training data.
        
        Args:
            model: Trained GP model
            x: Position tensor
            y: Energy tensor
            g: Force tensor
            stacked_y_g: Combined tensor of energies and forces
            thermal_noise: Optional tuple of (force_noise, energy_noise)
            path: Directory to save model files
            
        Notes:
            - Files are saved with incrementing numbers (model_1.pth, model_2.pth, etc.)
            - Each save includes the model state, training data, and noise parameters
        """
        if path is None:
            return

        check = check_model_files(model, x, y, g, stacked_y_g, path, thermal_noise, 
                                 self.atomic_info if hasattr(self, 'atomic_info') else None)
        if check is False:
            # First model was already saved by check_model_files
            model_no = 1
        else:
            # check is a sorted list of existing model files
            model_no = int(check[-1].split('_')[1].split('.')[0]) + 1
            # Save the model for subsequent calls
            torch.save(model.state_dict(), path + '/model_' + str(model_no) + '.pth')
            torch.save(x, path + '/X_' + str(model_no) + '.pth')
            torch.save(y, path + '/Y_' + str(model_no) + '.pth')
            torch.save(g, path + '/g_' + str(model_no) + '.pth')
            torch.save(stacked_y_g,  path + '/d_' + str(model_no) + '.pth')
            if thermal_noise is not None:
                torch.save(torch.tensor(thermal_noise), path + '/thermal_noise_' + str(model_no) + '.pth')
            # ADD THIS LINE to save atomic info
            if hasattr(self, 'atomic_info') and self.atomic_info is not None:
                torch.save(self.atomic_info, path + '/atomic_info_' + str(model_no) + '.pth')
        
        # Also save in toy model format if this is 2D data (z-component is all zeros)
        if x.shape[-1] == 3 and torch.all(x[:, 2] == 0):
            # This is toy model data - save in 2D format
            positions_2d = x[:, :2]
            forces_2d = g[:, :2] if g.shape[-1] == 3 else g
            torch.save(positions_2d, path + '/positions_' + str(model_no) + '.pth')
            torch.save(forces_2d, path + '/forces_' + str(model_no) + '.pth')
            torch.save(y, path + '/energies_' + str(model_no) + '.pth')


    def _calculate_training_metrics(self, training_data, model_name="GP"):
        """Calculate and print training metrics."""
        x, y, g = training_data
        
        # Ensure all tensors are on the same device as the model
        if hasattr(x, 'to'):
            x = x.to(self.device)
        if hasattr(y, 'to'):
            y = y.to(self.device)
        if hasattr(g, 'to'):
            g = g.to(self.device)
        
        try:
            # Check if this is a derivative-based model
            if self.model_type == "GPModelWithDerivatives_rbf_atomic":
                # Make predictions on training data
                with torch.no_grad():
                    self.model.eval()
                    self.likelihood.eval()
                    # Convert to appropriate format for prediction
                    x_np = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
                    y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
                    g_np = g.numpy() if hasattr(g, 'numpy') else np.asarray(g)
                    # For derivative models, we need to use the augmented training inputs
                    if hasattr(self, 'train_x_augmented'):
                        # Use the stored augmented training data
                        predictions = self.likelihood(self.model(self.train_x_augmented))
                        pred_mean = predictions.mean
                        
                        # Extract predictions in the correct order
                        n_data = x_np.shape[0]
                        n_dims = x_np.shape[1]
                        
                        pred_energies = []
                        pred_forces_list = []
                        
                        idx = 0
                        for i in range(n_data):
                            # Energy prediction
                            pred_energies.append(pred_mean[idx].item())
                            idx += 1
                            
                            # Force predictions (remember they're negative gradients)
                            forces_i = []
                            for j in range(n_dims):
                                forces_i.append(-pred_mean[idx].item())
                                idx += 1
                            pred_forces_list.append(forces_i)
                        
                        pred_energy = np.array(pred_energies)
                        pred_forces = np.array(pred_forces_list)
                        
                        # True values
                        true_energy = y_np
                        true_forces = g_np
                        
                    else:
                        print(f"WARNING: No augmented training data available for {model_name}")
                        return
            else:
                # Make predictions on training data
                with torch.no_grad():
                    self.model.eval()
                    self.likelihood.eval()
                    
                    # Convert to appropriate format for prediction
                    x_np = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
                    
                    # Data is already in moving-atom-only format - no extraction needed
                    x_test = torch.tensor(x_np, dtype=torch.float64)
                    
                    # Check for NaN in input
                    if torch.isnan(x_test).any():
                        print(f"WARNING: NaN detected in test input data for {model_name}")
                        print(f"Number of NaN values: {torch.isnan(x_test).sum().item()}")
                        return
                    
                    # Try to get predictions with error handling
                    try:
                        # First check if the model can evaluate without likelihood
                        model_output = self.model(x_test)
                        
                        # Check for NaN in model output
                        if torch.isnan(model_output.mean).any() or torch.isnan(model_output.covariance_matrix).any():
                            print(f"WARNING: NaN detected in model output for {model_name}")
                            print("Skipping training metrics due to numerical issues")
                            return
                        
                        # Now try with likelihood
                        predictions = self.likelihood(model_output)
                        pred_mean = predictions.mean
                        
                        # Check for NaN in predictions
                        if torch.isnan(pred_mean).any():
                            return

                    except Exception as e:
                        return
                    
                    # Separate energy and force predictions
                    pred_energy = pred_mean[:, 0].numpy()
                    pred_forces = pred_mean[:, 1:].numpy()
                    
                    # Get true values
                    true_energy = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
                    true_forces = g.numpy() if hasattr(g, 'numpy') else np.asarray(g)
                    
                    # Forces are already for moving atoms only - no extraction needed
                    
                    # Check for NaN in results before calculating metrics
                    if np.isnan(pred_energy).any() or np.isnan(pred_forces).any():
                        return
                    
                    # Calculate errors
                    energy_mae = np.mean(np.abs(pred_energy - true_energy))
                    energy_rmse = np.sqrt(np.mean((pred_energy - true_energy)**2))
                    energy_max_error = np.max(np.abs(pred_energy - true_energy))
                    
                    force_mae = np.mean(np.abs(pred_forces - true_forces))
                    force_rmse = np.sqrt(np.mean((pred_forces - true_forces)**2))
                    force_max_error = np.max(np.abs(pred_forces - true_forces))
                    

                    self._last_training_metrics = {
                        'energy_mae': float(energy_mae),
                        'energy_rmse': float(energy_rmse),
                        'energy_max': float(energy_max_error),
                        'force_mae': float(force_mae),
                        'force_rmse': float(force_rmse),
                        'force_max': float(force_max_error)
                    }
                
        except Exception as e:
            print(f"WARNING: Could not calculate training metrics for {model_name}: {e}")
            print("This typically happens when the model is not properly trained")
        
        finally:
            # Always set model back to training mode
            if self.model is not None:
                self.model.train()
            if self.likelihood is not None:
                self.likelihood.train()


    def calculate_hessian(self, location: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate Hessian using automatic differentiation with proper gradient tracking."""
        
        # Get dimensions
        n_full = len(location.flatten())
        moving_idx = np.asarray(self.atomic_info["moving_indices"])
        n_moving = len(moving_idx)
        
        # For very large systems, use finite differences instead
        if n_full > 750:  # ~250 atoms
            print(f"System too large for autodiff ({n_full} DOF), using finite differences...")
            return self._calculate_hessian_finite_diff(location)
        
        # Convert to torch tensor
        x_full = torch.tensor(location.flatten(), dtype=torch.float64, requires_grad=True)
        
        try:
            # Forward pass to get energy
            with torch.enable_grad():
                # Extract moving atom coordinates
                x_moving = []
                for idx in moving_idx:
                    x_moving.extend([x_full[idx*3], x_full[idx*3+1], x_full[idx*3+2]])
                x_moving = torch.stack(x_moving)
                
                # Reshape for model input
                x_moving_batch = x_moving.unsqueeze(0)
                
                # Get model prediction
                self.model.eval()
                self.likelihood.eval()
                predictions = self.likelihood(self.model(x_moving_batch))
                energy = predictions.mean[0, 0]  # First element is energy
                
                # Compute gradient with respect to full coordinates
                grad = torch.autograd.grad(energy, x_full, create_graph=True, retain_graph=True)[0]
                
                # Compute Hessian
                hessian = torch.zeros(n_full, n_full, dtype=torch.float64)
                
                # Calculate each row of the Hessian
                for i in range(n_full):
                    if grad[i].requires_grad:
                        grad2 = torch.autograd.grad(
                            grad[i], 
                            x_full, 
                            retain_graph=(i < n_full - 1),  # Only retain for non-last iteration
                            create_graph=False
                        )[0]
                        hessian[i] = grad2
                    else:
                        # This coordinate doesn't affect the energy (frozen atom)
                        hessian[i, i] = 0.0
            
            # Clip extreme values to prevent numerical issues
            hessian = np.clip(hessian, -1e6, 1e6)
            
            # Ensure symmetry
            hessian = 0.5 * (hessian + hessian.T)
            
            return hessian.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Autodiff failed: {e}. Falling back to finite differences.")
            return self._calculate_hessian_finite_diff(location)

    def _calculate_hessian_finite_diff(self, location: npt.NDArray[np.float64], epsilon: float = 1e-4) -> npt.NDArray[np.float64]:
        """Fallback finite difference Hessian calculation."""
        n = len(location.flatten())
        hessian = np.zeros((n, n))
        
        # Get base energy
        f_0 = float(self.predict(location.reshape(1, -1))[0][0])
        
        # Only calculate diagonal elements for large systems
        if n > 100:
            print(f"Large system - only calculating diagonal Hessian elements...")
            for i in range(n):
                loc_plus = location.flatten().copy()
                loc_plus[i] += epsilon
                f_plus = float(self.predict(loc_plus.reshape(1, -1))[0][0])
                
                loc_minus = location.flatten().copy()
                loc_minus[i] -= epsilon
                f_minus = float(self.predict(loc_minus.reshape(1, -1))[0][0])
                
                hessian[i, i] = (f_plus - 2 * f_0 + f_minus) / (epsilon ** 2)
            
            return hessian
        
        # Full Hessian for smaller systems
        # First pass: diagonal elements
        f_plus = np.zeros(n)
        f_minus = np.zeros(n)
        
        for i in range(n):
            loc_plus = location.flatten().copy()
            loc_plus[i] += epsilon
            f_plus[i] = float(self.predict(loc_plus.reshape(1, -1))[0][0])
            
            loc_minus = location.flatten().copy()
            loc_minus[i] -= epsilon
            f_minus[i] = float(self.predict(loc_minus.reshape(1, -1))[0][0])
            
            hessian[i, i] = (f_plus[i] - 2 * f_0 + f_minus[i]) / (epsilon ** 2)
        
        # Second pass: off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                loc_pp = location.flatten().copy()
                loc_pp[i] += epsilon
                loc_pp[j] += epsilon
                f_pp = float(self.predict(loc_pp.reshape(1, -1))[0][0])
                
                hessian[i, j] = (f_pp - f_plus[i] - f_plus[j] + f_0) / (epsilon ** 2)
                hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def initialize_hyperparameters(self, training_data):
        """Initialize hyperparameters based on data statistics - following reference approach."""
        from scipy.stats import norm
        
        x, y, g = training_data
        
        # Convert to numpy for statistics
        x_np = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
        y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
        g_np = g.numpy() if hasattr(g, 'numpy') else np.asarray(g)
        
        # 1. Energy scale analysis (like reference code)
        mean_y = np.mean(y_np)
        range_y = np.max(y_np) - np.min(y_np)
        
        # Handle edge case where all energies are the same
        if range_y < 1e-6:
            range_y = max(1.0, abs(mean_y))
        
        # 2. Spatial scale analysis using atomic distance measure
        # This is equivalent to their range_x calculation
        n_samples = x_np.shape[0]
        if n_samples > 1 and hasattr(self, 'atomic_info') and self.atomic_info is not None:
            # Calculate maximum pairwise distance
            distances = []
            for i in range(min(n_samples, 20)):  # Sample more points
                for j in range(i+1, min(n_samples, 20)):
                    try:
                        # Calculate atomic distance with unit lengthscales
                        dist = calculate_atomic_distance_measure(
                            x_np[i], x_np[j], 
                            self.atomic_info, 
                            np.ones(self.atomic_info['n_pt'])
                        )
                        if not np.isnan(dist) and not np.isinf(dist):
                            distances.append(dist)
                    except:
                        pass
            
            if distances:
                range_x = np.max(distances)
            else:
                # Fallback to Euclidean distance
                dists = []
                for i in range(min(n_samples, 10)):
                    for j in range(i+1, min(n_samples, 10)):
                        dists.append(np.linalg.norm(x_np[i] - x_np[j]))
                range_x = np.max(dists) if dists else 1.0
        else:
            range_x = 1.0
        
        # 3. Initialize hyperparameters following reference approach
        # Using norm.ppf(0.75, 0, sigma) which gives 0.674*sigma
        # This is the 75th percentile of a normal distribution
        
        # Magnitude - following their approach exactly
        initial_magnitude = norm.ppf(0.75, 0, range_y/3)
        if initial_magnitude < 0.1:  # Ensure reasonable minimum
            initial_magnitude = 0.1
        
        # Add regularization for small data ranges
        if range_y < 0.1:  # Very small energy range
            initial_magnitude = max(0.5, range_y * 2)  # More conservative
        
        # Lengthscales - following their approach
        if hasattr(self, 'atomic_info') and self.atomic_info is not None:
            n_pt = self.atomic_info['n_pt']
            initial_lengthscales = norm.ppf(0.75, 0, range_x/3) * np.ones(n_pt)
        else:
            initial_lengthscales = np.array([norm.ppf(0.75, 0, range_x/3)])
        
        # Ensure minimum lengthscale
        initial_lengthscales = np.maximum(initial_lengthscales, 0.05)
        
        # 4. Noise initialization
        # The reference code uses fixed small noise values
        # They constrain likelihood variance to 1e-8
        energy_noise = 1e-8  # Small but not too small
        force_noise = 1e-8   # Slightly larger for forces
        
        # print(f"\nRange-based hyperparameter initialization:")
        # print(f"  Data statistics:")
        # print(f"    Energy mean: {mean_y:.4f}")
        # print(f"    Energy range: {range_y:.4f}")
        # print(f"    Spatial range: {range_x:.4f}")
        # print(f"  Initial hyperparameters:")
        # print(f"    Magnitude: {initial_magnitude:.4f}")
        # print(f"    Lengthscales: {initial_lengthscales}")
        # print(f"    Energy noise: {energy_noise:.6f}")
        # print(f"    Force noise: {force_noise:.6f}")
        
        return {
            'magnitude': initial_magnitude,
            'lengthscales': initial_lengthscales,
            'energy_noise': energy_noise,
            'force_noise': force_noise
        }, range_y, range_x


class MultitaskGPModel(gpytorch.models.ExactGP):
    """Multitask GP Model with standard RBF kernel for toy models.
    
    This model is specifically designed for toy models where we need
    better control over length scales. It uses a regular RBF kernel
    instead of the atomic version.
    """
    
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=1):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        
        # Use standard RBF kernel with better length scale control
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.shape[-1],  # ARD for each dimension
                lengthscale_constraint=gpytorch.constraints.Interval(0.01, 10.0)  # Reasonable bounds
            ),
            num_tasks=num_tasks,
            rank=rank
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train_multitask_gp_toy_model(
    self,
    training_data: List[any],
    thermal_noise: Optional[Tuple[float, float]] = None,
    model_name: str = "GP",
    initial_lengthscale: Optional[float] = None,
    verbose: bool = False,
    path: Optional[str] = None
) -> 'GPSurrogateModel':
    """Train MultitaskGPModel specifically for toy models.
    
    This method provides better hyperparameter initialization for toy models
    with highly nonlinear potentials like V8.
    
    Args:
        training_data: List containing [positions, energies, forces]
        thermal_noise: Thermal noise values (not used for toy models)
        model_name: Name of the model
        initial_lengthscale: Initial length scale (if None, will be estimated)
        verbose: Whether to print training information
    """
    # Extract training data
    train_positions = training_data[0]
    train_energies = training_data[1]
    train_forces = training_data[2]
    
    n_train = len(train_positions)
    n_dims = train_positions.shape[1]
    
    # Prepare data for multitask learning and move to device
    train_x = torch.tensor(train_positions, dtype=torch.float64).to(self.device)
    
    # Stack energies and forces
    train_y_energy = torch.tensor(train_energies, dtype=torch.float64).unsqueeze(-1).to(self.device)
    train_y_forces = torch.tensor(train_forces, dtype=torch.float64).to(self.device)
    train_y = torch.cat([train_y_energy, train_y_forces], dim=-1)
    
    # Initialize likelihood with small fixed noise
    num_tasks = 1 + n_dims  # 1 for energy + n_dims for force components
    self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-4)
    )
    
    # Initialize model and move to device
    self.model = MultitaskGPModel(train_x, train_y, self.likelihood, num_tasks).to(self.device)
    self.likelihood = self.likelihood.to(self.device)
    
    # Estimate appropriate length scale if not provided
    if initial_lengthscale is None:
        # Calculate pairwise distances to estimate correlation length
        if n_train > 1:
            dists = []
            for i in range(min(n_train, 50)):  # Sample to avoid O(n^2)
                for j in range(i+1, min(n_train, 50)):
                    dist = np.linalg.norm(train_positions[i] - train_positions[j])
                    if dist > 0:
                        dists.append(dist)
            if dists:
                # Use median distance as initial length scale
                initial_lengthscale = np.median(dists) * 0.5
                # Ensure it's within bounds [0.01, 10.0]
                initial_lengthscale = max(0.01, min(initial_lengthscale, 10.0))
            else:
                initial_lengthscale = 0.3  # Default for toy models
        else:
            initial_lengthscale = 0.3
    
    # Set initial hyperparameters
    # For MultitaskKernel, the base kernel is accessed via data_covar_module
    self.model.covar_module.data_covar_module.lengthscale = initial_lengthscale
    
    # Calculate appropriate kernel magnitude
    energy_std = np.std(train_energies) if len(train_energies) > 1 else 1.0
    force_magnitudes = np.linalg.norm(train_forces, axis=1)
    force_std = np.std(force_magnitudes) if len(force_magnitudes) > 1 else 1.0
    
    # Use geometric mean of energy and force scales
    initial_magnitude = np.sqrt(energy_std * force_std)
    if initial_magnitude < 0.1:
        initial_magnitude = 0.1
    
    # Note: We can't directly set outputscale for MultitaskKernel
    # It will be learned during optimization
    
    # Set fixed small noise (respecting constraints)
    self.likelihood.noise = 1e-4  # Small but within typical constraints
    
    if verbose:
        print(f"\nGP2 Toy Model Training:")
        print(f"  Training points: {n_train}")
        print(f"  Initial lengthscale: {initial_lengthscale:.4f}")
        print(f"  Initial magnitude estimate: {initial_magnitude:.4f}")
        print(f"  Fixed noise: 1e-6")
    
    # Training mode
    self.model.train()
    self.likelihood.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
    
    # Loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 50
    
    for i in range(self.epoch):
        optimizer.zero_grad()
        output = self.model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            break

    # NEW_v24: Always ensure model and likelihood are on correct device after training
    self.model = self.model.to(self.device)
    self.likelihood = self.likelihood.to(self.device)

    # Save the model if path is provided
    if path is not None:
        # Create stacked_y_g for saving
        stacked_y_g = torch.cat([train_y.unsqueeze(1), train_g], dim=1)
        self.save_model(self.model, train_x, train_y[:, 0], train_g, stacked_y_g, thermal_noise, path)
    
    return self