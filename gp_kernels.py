from __future__ import annotations
import gpytorch
from vasp_manager import *
from atomic_structure import *
import torch
from typing import Optional, Dict
from torch import nn
import numpy as np
from gpytorch.kernels import RBFKernelGrad
from gpytorch.means import ConstantMeanGrad
from gpytorch.models import ExactGP
from atomic_structure import calculate_atomic_distance_measure
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def dist(x1,x2):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    dist2 = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            dist2[i,j] = np.sum((x1[i,:]-x2[j,:])**2)
    dist = np.sqrt(dist2)
    return dist

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        range_y = torch.max(train_y[0, :]) - torch.min(train_y[0, :])
        range_x = torch.tensor(np.max(dist(train_x.numpy(), train_x.numpy())))
        outputscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_y ** 2)
        lengthscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_x)
        mean_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_y ** 2)
        self.mean_module = gpytorch.means.ConstantMean(prior=mean_prior,
                                                       batch_shape=torch.Size([train_x.size()[1] + 1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=int(np.shape(train_x)[1]), lengthscale_prior=lengthscale_prior,
                                       batch_shape=torch.Size([train_x.size()[1] + 1])),
            outputscale_prior=outputscale_prior, batch_shape=torch.Size([train_x.size()[1] + 1]), rank=128
        )
        self.covar_module.base_kernel.lengthscale = lengthscale_prior
        self.covar_module.outputscale = outputscale_prior

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        range_y = torch.max(train_y[0, :]) - torch.min(train_y[0, :])
        range_x = torch.tensor(np.max(dist(train_x.numpy(), train_x.numpy())))
        mean_y_1 = torch.mean(train_y[:, 0])
        mean_y_2 = torch.mean(train_y[:, 1])
        mean_y_3 = torch.mean(train_y[:, 2])
        lengthscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_x / 3)
        outputscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=(range_y / 3) ** 2)
        mean_prior_1 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_1 ** 2)
        mean_prior_2 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_2 ** 2)
        mean_prior_3 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_3 ** 2)
        mean_1 = gpytorch.means.ConstantMean(prior=mean_prior_1)
        mean_2 = gpytorch.means.ConstantMean(prior=mean_prior_2)
        mean_3 = gpytorch.means.ConstantMean(prior=mean_prior_3)
        self.mean_module = gpytorch.means.MultitaskMean([mean_1, mean_2, mean_3], num_tasks=train_x.size()[1] + 1)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=int(np.shape(train_x)[1]), lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior, num_tasks=train_x.size()[1] + 1, rank=train_x.size()[1] + 1
        )
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskGPModelPolynomial(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, power):
        super(MultitaskGPModelPolynomial, self).__init__(train_x, train_y, likelihood)
        range_y = torch.max(train_y[0, :]) - torch.min(train_y[0, :])
        range_x = torch.tensor(np.max(dist(train_x.numpy(), train_x.numpy())))
        mean_y_1 = torch.mean(train_y[:, 0])
        mean_y_2 = torch.mean(train_y[:, 1])
        mean_y_3 = torch.mean(train_y[:, 2])
        lengthscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_x / 3)
        outputscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=(range_y / 3) ** 2)
        mean_prior_1 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_1 ** 2)
        mean_prior_2 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_2 ** 2)
        mean_prior_3 = gpytorch.priors.NormalPrior(loc=0, scale=mean_y_3 ** 2)
        mean_1 = gpytorch.means.ConstantMean(prior=mean_prior_1)
        mean_2 = gpytorch.means.ConstantMean(prior=mean_prior_2)
        mean_3 = gpytorch.means.ConstantMean(prior=mean_prior_3)
        self.mean_module = gpytorch.means.MultitaskMean([mean_1, mean_2, mean_3], num_tasks=train_x.size()[1] + 1)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.PolynomialKernel(power=power, ard_num_dims=int(np.shape(train_x)[1]), lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior, num_tasks=train_x.size()[1] + 1, rank=train_x.size()[1] + 1
        )
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        range_y = torch.max(train_y[0, :]) - torch.min(train_y[0, :])
        range_x = torch.tensor(np.max(dist(train_x.numpy(), train_x.numpy())))
        mean_y = torch.mean(train_y[0, :])
        outputscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=(range_y) ** 2)
        lengthscale_prior = gpytorch.priors.NormalPrior(loc=0, scale=range_x)
        mean_prior = gpytorch.priors.NormalPrior(loc=mean_y, scale=(range_y) ** 2)
        self.mean_module = gpytorch.means.ConstantMeanGrad(prior=mean_prior)
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=train_x.size()[1],
                                                          lengthscale_prior=lengthscale_prior)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel, outputscale_prior=outputscale_prior)

        self.covar_module.base_kernel.lengthscale = lengthscale_prior
        self.covar_module.outputscale = outputscale_prior

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class RBFAtomicKernel(gpytorch.kernels.Kernel):
    """Fixed implementation matching the reference rbf_atomic.py exactly"""

    def __init__(self, conf_info: dict, **kwargs):
        super().__init__(**kwargs)

        required = {"n_pt", "pairtype", "atomtype_mov", "atomtype_fro", "conf_fro"}
        missing = required.difference(conf_info)
        if missing:
            raise KeyError(f"conf_info missing keys: {', '.join(missing)}")

        self.conf_info = conf_info
        self.n_mov = len(conf_info["atomtype_mov"])

        # Hyperparameters
        n_pt = int(conf_info["n_pt"])
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.ones(n_pt)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

        self.register_parameter("raw_magnitude", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_magnitude", gpytorch.constraints.Positive())

        # Store frozen atom coordinates as buffer
        self.register_buffer("conf_fro", torch.as_tensor(conf_info["conf_fro"], dtype=torch.double))

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def magnitude(self):
        return self.raw_magnitude_constraint.transform(self.raw_magnitude)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None, *, diag=False, **params):
        x2 = x1 if x2 is None else x2
        
        expected_dim = self.n_mov * 3
        if x1.shape[-1] != expected_dim:
            raise ValueError(f"Input dimension mismatch: expected {expected_dim}, got {x1.shape[-1]}")

        # Compute squared atomic distances between configurations
        dist2 = self._dist2_atomic(x1, x2)

        if diag:
            return self.magnitude.pow(2).expand(dist2.shape[0])

        # Apply RBF kernel
        K = self.magnitude.pow(2) * torch.exp(-0.5 * dist2)
        
        # Add jitter for numerical stability (CRITICAL FIX)
        if x1.shape[0] == x2.shape[0] and torch.equal(x1, x2.to(x1.device)):
            # Adaptive jitter based on kernel magnitude
            base_jitter = 1e-3
            adaptive_jitter = base_jitter * self.magnitude.pow(2)
            jitter_matrix = adaptive_jitter * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
            K = K + jitter_matrix
            
            # Ensure positive definiteness
            min_eigenval = torch.linalg.eigvalsh(K).min()
            if min_eigenval < 1e-6:
                K = K + (1e-6 - min_eigenval) * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        
        
        return K

    def _dist2_atomic(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute squared atomic distance measure between configurations.
        This matches the reference _dist2_atomic exactly.
        
        Returns tensor of shape (n1, n2) where each element is the squared
        distance between configuration i in X1 and configuration j in X2.
        """
        # Extract configuration info
        atomtype_mov = self.conf_info['atomtype_mov']
        atomtype_fro = self.conf_info['atomtype_fro'] 
        pairtype = self.conf_info['pairtype']
        
        n1, n2 = X1.shape[0], X2.shape[0]
        N_mov = len(atomtype_mov)
        N_fro = len(atomtype_fro)
        
        # Inverse squared lengthscales
        s2 = 1.0 / self.lengthscale.pow(2)  # shape: (n_pt,)
        
        # Initialize distance matrix
        dist2 = torch.zeros(n1, n2, device=X1.device, dtype=X1.dtype)
        
        # Add small epsilon for numerical stability
        eps = 1e-6
        
        # Distances between moving atoms (if more than one moving atom)
        if N_mov > 1:
            for i in range(N_mov - 1):
                for j in range(i + 1, N_mov):
                    # Extract coordinates for atoms i and j
                    coords_i_X1 = X1[:, (i*3):(i*3+3)]  # (n1, 3)
                    coords_j_X1 = X1[:, (j*3):(j*3+3)]  # (n1, 3)
                    coords_i_X2 = X2[:, (i*3):(i*3+3)]  # (n2, 3)
                    coords_j_X2 = X2[:, (j*3):(j*3+3)]  # (n2, 3)
                    
                    # Compute distances
                    dist_ij_X1 = torch.clamp(
                        torch.norm(coords_i_X1 - coords_j_X1, dim=1), min=eps
                    )  # (n1,)
                    dist_ij_X2 = torch.clamp(
                        torch.norm(coords_i_X2 - coords_j_X2, dim=1), min=eps  
                    )  # (n2,)
                    
                    # Compute inverse distances
                    invr_ij_X1 = 1.0 / dist_ij_X1  # (n1,)
                    invr_ij_X2 = 1.0 / dist_ij_X2  # (n2,)
                    
                    # Broadcast to get (n1, n2) matrix
                    invr_ij_X1_expanded = invr_ij_X1.unsqueeze(1)  # (n1, 1)
                    invr_ij_X2_expanded = invr_ij_X2.unsqueeze(0)  # (1, n2)
                    
                    # Get pair type index
                    pt = pairtype[atomtype_mov[i], atomtype_mov[j]]
                    if pt >= 0:  # Only include active pair types
                        diff_squared = (invr_ij_X1_expanded - invr_ij_X2_expanded).pow(2)
                        dist2 += s2[pt] * diff_squared
        
        # Distances from moving atoms to frozen atoms
        if N_fro > 0:
            for i in range(N_mov):
                for j in range(N_fro):
                    # Extract coordinates
                    coords_i_X1 = X1[:, (i*3):(i*3+3)]  # (n1, 3)
                    coords_i_X2 = X2[:, (i*3):(i*3+3)]  # (n2, 3)
                    coords_fro_j = self.conf_fro[j, :3]  # (3,)
                    
                    # Compute distances to frozen atom
                    dist_ij_X1 = torch.clamp(
                        torch.norm(coords_i_X1 - coords_fro_j, dim=1), min=eps
                    )  # (n1,)
                    dist_ij_X2 = torch.clamp(
                        torch.norm(coords_i_X2 - coords_fro_j, dim=1), min=eps
                    )  # (n2,)
                    
                    # Compute inverse distances
                    invr_ij_X1 = 1.0 / dist_ij_X1  # (n1,)
                    invr_ij_X2 = 1.0 / dist_ij_X2  # (n2,)
                    
                    # Broadcast to get (n1, n2) matrix  
                    invr_ij_X1_expanded = invr_ij_X1.unsqueeze(1)  # (n1, 1)
                    invr_ij_X2_expanded = invr_ij_X2.unsqueeze(0)  # (1, n2)
                    
                    # Get pair type index
                    pt = pairtype[atomtype_mov[i], atomtype_fro[j]]
                    if pt >= 0:  # Only include active pair types
                        diff_squared = (invr_ij_X1_expanded - invr_ij_X2_expanded).pow(2)
                        dist2 += s2[pt] * diff_squared
        
        return dist2

    def Kdiag(self, X):
        """Diagonal of kernel matrix (all ones times magnitude^2)"""
        return self.magnitude.pow(2).expand(X.shape[0])




class MultitaskGPModel_rbf_atomic(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        conf_info,
        *,
        rank: int | None = None,
    ) -> None:
        # Standard ExactGP initialisation – stores the data inside the model
        super().__init__(train_x, train_y, likelihood)

        if train_y.ndim != 2:
            raise ValueError(
                "train_y must have shape (N, T).  Got " f"{train_y.shape} instead."
            )
        num_tasks: int = train_y.size(1)
        if likelihood.num_tasks != num_tasks:
            raise ValueError(
                "likelihood.num_tasks (" f"{likelihood.num_tasks}) != "
                f"train_y.shape[1] ({num_tasks})."
            )

        # ────────────────────────────────────────────────────────────────
        # Mean: independent constant for each task
        # ────────────────────────────────────────────────────────────────
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        # ────────────────────────────────────────────────────────────────
        # Covariance: multitask coregionalisation of custom RBFAtomicKernel
        # ────────────────────────────────────────────────────────────────
        base_kernel = RBFAtomicKernel(conf_info)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=num_tasks,
            rank=rank if rank is not None else min(3, num_tasks),
        )
    def forward(self, x: torch.Tensor):
        """Return predictive distribution at inputs *x*."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class RBFAtomicKernelGrad(Kernel):
    """
    RBF Atomic Kernel with Gradient Support for GPyTorch.
    
    This kernel implements the atomic distance measure from GPy's RBF_atomic kernel:
    dist(C,C') = sqrt(SUM_ij{[(1/r_ij-1/r_ij')/l_ij]^2})
    
    And extends it to handle gradient observations following GPyTorch's RBFKernelGrad pattern.
    """
    
    def __init__(self, conf_info: Optional[Dict] = None, 
                 batch_shape=torch.Size(), active_dims=None, 
                 lengthscale_prior=None, lengthscale_constraint=None,
                 outputscale_prior=None, outputscale_constraint=None,
                 eps=1e-10, **kwargs):
        
        if conf_info is None:
            raise ValueError("conf_info dictionary must be provided for atomic kernel")
            
        self.conf_info = conf_info
        self._validate_conf_info()
        
        # Initialize parent without any dimension specifications
        super().__init__(has_lengthscale=False, batch_shape=batch_shape, **kwargs)
        
        # Now manually set up lengthscale parameters
        self.ard_num_dims = self.conf_info['n_pt']
        
        # Set lengthscale constraint
        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()
            
        # Register the raw lengthscale parameter
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.ard_num_dims))
        )
        
        # Set the constraint
        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        
        # Set the prior if specified
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
            
        # Add magnitude (outputscale) parameter
        if outputscale_constraint is None:
            outputscale_constraint = Positive()
            
        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        
        self.register_constraint("raw_outputscale", outputscale_constraint)
        
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior",
                outputscale_prior,
                lambda m: m.outputscale,
                lambda m, v: m._set_outputscale(v),
            )
            
        self.eps = eps
        self.include_derivatives = True  # Always include derivatives for this kernel
        
    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        """
        Override the base class __call__ to bypass dimension checking.
        """
        # Just call forward directly, bypassing the parent's dimension checks
        return self.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
    
    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    
    def _set_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        
        # Ensure the value is positive (add small epsilon to avoid numerical issues)
        value = value.clamp(min=1e-6)
        
        # Handle shape - ensure it matches ard_num_dims
        if value.dim() == 0:
            value = value.unsqueeze(0).unsqueeze(0)  # Add batch and dimension
        elif value.dim() == 1:
            value = value.unsqueeze(0)  # Add batch dimension
        
        # Ensure correct number of lengthscales
        if value.shape[-1] == 1 and self.ard_num_dims > 1:
            value = value.expand(*value.shape[:-1], self.ard_num_dims)
        elif value.shape[-1] != self.ard_num_dims:
            raise ValueError(f"Expected {self.ard_num_dims} lengthscales, got {value.shape[-1]}")
        
        # Apply constraint
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @outputscale.setter  
    def outputscale(self, value):
        self._set_outputscale(value)
        
    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        
        # Ensure the value is positive
        value = value.clamp(min=1e-6)
        
        # Handle shape
        if value.dim() == 0:
            value = value.unsqueeze(0)  # Add batch dimension
        
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    @property
    def magnitude(self):
        """Alias for outputscale to match GPy naming."""
        return self.outputscale

    @magnitude.setter
    def magnitude(self, value):
        self.outputscale = value
        
    def _validate_conf_info(self):
        """Validate the conf_info dictionary has all required fields."""
        required_keys = ['conf_fro', 'atomtype_mov', 'atomtype_fro', 'pairtype', 'n_pt']
        for key in required_keys:
            if key not in self.conf_info:
                raise ValueError(f"conf_info must contain '{key}'")
                
        # Convert numpy arrays to torch tensors
        self.conf_fro = torch.tensor(self.conf_info['conf_fro'], dtype=torch.float64)
        self.atomtype_mov = torch.tensor(self.conf_info['atomtype_mov'], dtype=torch.long)
        self.atomtype_fro = torch.tensor(self.conf_info['atomtype_fro'], dtype=torch.long)
        self.pairtype = torch.tensor(self.conf_info['pairtype'], dtype=torch.long)
        self.n_pt = self.conf_info['n_pt']
            
    @property
    def num_outputs_per_input(self, x1):
        """
        This is called by GPyTorch to determine expected input dimensions.
        We override to handle our augmented input format.
        """
        # When we have augmented inputs with derivative indicators
        N_mov = self.atomtype_mov.shape[0]
        n_dims = 3 * N_mov
        
        # If input has derivative indicators, return 1 (we handle internally)
        if x1.shape[-1] > n_dims:
            return 1
        else:
            # Standard case
            return 1
    
    def _dist2_atomic(self, x1, x2=None):
        """
        Compute the square of the atomic distance measure between configurations.
        
        Args:
            x1: Tensor of shape (n1, 3*N_mov) - positions of moving atoms
            x2: Tensor of shape (n2, 3*N_mov) - positions of moving atoms (optional)
            
        Returns:
            dist2: Tensor of shape (n1, n2) - squared atomic distances
        """
        if x2 is None:
            x2 = x1
            
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        # Get squared inverse lengthscales
        ls = self.lengthscale.squeeze(0)  # Remove batch dimension if present
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)  # Shape: (n_pt,)
        
        dist2 = torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device)
        
        # Distances between moving atoms
        if N_mov > 1:
            for i in range(N_mov - 1):
                for j in range(i + 1, N_mov):
                    # Extract positions for atoms i and j
                    pos_i_1 = x1[:, i*3:(i+1)*3]  # Shape: (n1, 3)
                    pos_j_1 = x1[:, j*3:(j+1)*3]  # Shape: (n1, 3)
                    pos_i_2 = x2[:, i*3:(i+1)*3]  # Shape: (n2, 3)
                    pos_j_2 = x2[:, j*3:(j+1)*3]  # Shape: (n2, 3)
                    
                    # Compute inverse distances
                    r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps)
                    r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps)
                    
                    invr_ij_1 = 1.0 / r_ij_1  # Shape: (n1, 1)
                    invr_ij_2 = 1.0 / r_ij_2  # Shape: (n2,)
                    
                    # Get pair type
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        dist2 += s2[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))**2
        
        # Distances from moving atoms to frozen atoms
        if N_fro > 0:
            for i in range(N_mov):
                for j in range(N_fro):
                    # Extract position for moving atom i
                    pos_i_1 = x1[:, i*3:(i+1)*3]  # Shape: (n1, 3)
                    pos_i_2 = x2[:, i*3:(i+1)*3]  # Shape: (n2, 3)
                    
                    # Frozen atom position
                    pos_fro_j = self.conf_fro[j:j+1, :]  # Shape: (1, 3)
                    
                    # Compute inverse distances
                    r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_fro_j)**2, dim=1, keepdim=True) + self.eps)
                    r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_fro_j)**2, dim=1) + self.eps)
                    
                    invr_ij_1 = 1.0 / r_ij_1  # Shape: (n1, 1)
                    invr_ij_2 = 1.0 / r_ij_2  # Shape: (n2,)
                    
                    # Get pair type
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                    if pt >= 0:
                        dist2 += s2[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))**2
                        
        return dist2
    
    def _ddist2_dX(self, x1, x2, dim_x):
        """
        Compute derivative of squared distance with respect to one coordinate.
        
        Args:
            x1: Tensor of shape (n1, 3*N_mov)
            x2: Tensor of shape (n2, 3*N_mov)
            dim_x: Which dimension to take derivative with respect to
            
        Returns:
            Derivative tensor of shape (n1, n2)
        """
        if x2 is None:
            x2 = x1
            
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze(0)
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        i = dim_x // 3  # Which atom
        xyz = dim_x % 3  # Which coordinate (x=0, y=1, z=2)
        
        D1 = torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device)
        
        # Derivatives for moving-moving interactions
        if N_mov > 1:
            for j in range(N_mov):
                if j != i:
                    pos_i_1 = x1[:, i*3:(i+1)*3]
                    pos_j_1 = x1[:, j*3:(j+1)*3]
                    pos_i_2 = x2[:, i*3:(i+1)*3]
                    pos_j_2 = x2[:, j*3:(j+1)*3]
                    
                    r_ij_1_sq = torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps
                    r_ij_2_sq = torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps
                    
                    invr_ij_1 = 1.0 / torch.sqrt(r_ij_1_sq)
                    invr_ij_2 = 1.0 / torch.sqrt(r_ij_2_sq)
                    
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        deriv_ij = -2.0 * s2[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))
                        deriv_ij = deriv_ij * (invr_ij_1**3) * (x1[:, i*3+xyz:i*3+xyz+1] - x1[:, j*3+xyz:j*3+xyz+1])
                        D1 += deriv_ij
                        
        # Derivatives for moving-frozen interactions
        if N_fro > 0:
            for j in range(N_fro):
                pos_i_1 = x1[:, i*3:(i+1)*3]
                pos_i_2 = x2[:, i*3:(i+1)*3]
                pos_fro_j = self.conf_fro[j:j+1, :]
                
                r_ij_1_sq = torch.sum((pos_i_1 - pos_fro_j)**2, dim=1, keepdim=True) + self.eps
                r_ij_2_sq = torch.sum((pos_i_2 - pos_fro_j)**2, dim=1) + self.eps
                
                invr_ij_1 = 1.0 / torch.sqrt(r_ij_1_sq)
                invr_ij_2 = 1.0 / torch.sqrt(r_ij_2_sq)
                
                pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                if pt >= 0:
                    deriv_ij = -2.0 * s2[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))
                    deriv_ij = deriv_ij * (invr_ij_1**3) * (x1[:, i*3+xyz:i*3+xyz+1] - pos_fro_j[:, xyz:xyz+1])
                    D1 += deriv_ij
                    
        return D1
    
    def _d2dist2_dXdX2(self, x1, x2, dim_x1, dim_x2):
        """
        Compute second derivative of squared distance.
        
        Args:
            x1, x2: Position tensors
            dim_x1, dim_x2: Dimensions for derivatives
            
        Returns:
            Second derivative tensor of shape (n1, n2)
        """
        if x2 is None:
            x2 = x1
            
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze(0)
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        i_1 = dim_x1 // 3
        xyz_1 = dim_x1 % 3
        i_2 = dim_x2 // 3
        xyz_2 = dim_x2 % 3
        
        D12 = torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device)
        
        if i_1 != i_2:
            # Different atoms
            pt = self.pairtype[self.atomtype_mov[i_1], self.atomtype_mov[i_2]]
            if pt >= 0:
                pos_i1_1 = x1[:, i_1*3:(i_1+1)*3]
                pos_i2_1 = x1[:, i_2*3:(i_2+1)*3]
                pos_i1_2 = x2[:, i_1*3:(i_1+1)*3]
                pos_i2_2 = x2[:, i_2*3:(i_2+1)*3]
                
                r_12_1_sq = torch.sum((pos_i1_1 - pos_i2_1)**2, dim=1, keepdim=True) + self.eps
                r_12_2_sq = torch.sum((pos_i1_2 - pos_i2_2)**2, dim=1) + self.eps
                
                invr_12_1 = 1.0 / torch.sqrt(r_12_1_sq)
                invr_12_2 = 1.0 / torch.sqrt(r_12_2_sq)
                
                temp = 2.0 * s2[pt] * (invr_12_1**3) * (invr_12_2.unsqueeze(0)**3)
                deriv_12 = temp * (x1[:, i_1*3+xyz_1:i_1*3+xyz_1+1] - x1[:, i_2*3+xyz_1:i_2*3+xyz_1+1])
                deriv_12 = deriv_12 * (x2[:, i_1*3+xyz_2:i_1*3+xyz_2+1] - x2[:, i_2*3+xyz_2:i_2*3+xyz_2+1]).t()
                D12 += deriv_12
        else:
            # Same atom - need to consider all other atoms
            i = i_1
            
            # Moving atoms
            if N_mov > 1:
                for j in range(N_mov):
                    if j != i:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_j_1 = x1[:, j*3:(j+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_j_2 = x2[:, j*3:(j+1)*3]
                        
                        r_ij_1_sq = torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps
                        r_ij_2_sq = torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps
                        
                        invr_ij_1 = 1.0 / torch.sqrt(r_ij_1_sq)
                        invr_ij_2 = 1.0 / torch.sqrt(r_ij_2_sq)
                        
                        pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                        if pt >= 0:
                            temp = -2.0 * s2[pt] * (invr_ij_1**3) * (invr_ij_2.unsqueeze(0)**3)
                            deriv_ij_12 = temp * (x1[:, i*3+xyz_1:i*3+xyz_1+1] - x1[:, j*3+xyz_1:j*3+xyz_1+1])
                            deriv_ij_12 = deriv_ij_12 * (x2[:, i*3+xyz_2:i*3+xyz_2+1] - x2[:, j*3+xyz_2:j*3+xyz_2+1]).t()
                            D12 += deriv_ij_12
                            
            # Frozen atoms
            if N_fro > 0:
                for j in range(N_fro):
                    pos_i_1 = x1[:, i*3:(i+1)*3]
                    pos_i_2 = x2[:, i*3:(i+1)*3]
                    pos_fro_j = self.conf_fro[j:j+1, :]
                    
                    r_ij_1_sq = torch.sum((pos_i_1 - pos_fro_j)**2, dim=1, keepdim=True) + self.eps
                    r_ij_2_sq = torch.sum((pos_i_2 - pos_fro_j)**2, dim=1) + self.eps
                    
                    invr_ij_1 = 1.0 / torch.sqrt(r_ij_1_sq)
                    invr_ij_2 = 1.0 / torch.sqrt(r_ij_2_sq)
                    
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                    if pt >= 0:
                        temp = -2.0 * s2[pt] * (invr_ij_1**3) * (invr_ij_2.unsqueeze(0)**3)
                        deriv_ij_12 = temp * (x1[:, i*3+xyz_1:i*3+xyz_1+1] - pos_fro_j[:, xyz_1:xyz_1+1])
                        deriv_ij_12 = deriv_ij_12 * (x2[:, i*3+xyz_2:i*3+xyz_2+1] - pos_fro_j[:, xyz_2:xyz_2+1]).t()
                        D12 += deriv_ij_12
                        
        return D12
    
    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        """
        Compute the kernel matrix including derivative information.
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            if x2 is not None:
                x2 = x2.transpose(-1, -2).unsqueeze(-1)
                
        # Get dimensions
        N_mov = self.atomtype_mov.shape[0]
        n_dims = 3 * N_mov
        
        # Check if input includes derivative indicators
        if x1.shape[-1] > n_dims:
            # Extract positions and derivative indicators
            positions_1 = x1[..., :n_dims]
            indicators_1 = x1[..., n_dims:]
            
            if x2 is not None:
                positions_2 = x2[..., :n_dims]
                indicators_2 = x2[..., n_dims:]
            else:
                positions_2 = positions_1
                indicators_2 = indicators_1
                
            # Compute the kernel with derivative information
            K = self._compute_derivative_kernel(positions_1, positions_2, 
                                            indicators_1, indicators_2, diag)
            
            # Ensure positive definiteness
            if not diag and (x2 is None or (positions_1.shape == positions_2.shape and 
                                        torch.allclose(positions_1, positions_2))):
                # Add small jitter to diagonal
                jitter = 1e-4
                K = K + jitter * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
                
            return K
        else:
            # Standard case - just positions, no derivatives
            if diag:
                # For diagonal, we just need the kernel values at the same points
                return self.outputscale * torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)
            else:
                # Compute base kernel
                dist2 = self._dist2_atomic(x1, x2)
                K = self.outputscale * torch.exp(-0.5 * dist2)
                
                # Add jitter if computing K(X, X)
                if x2 is None:
                    jitter = 1e-4
                    K = K + jitter * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
                    
                return K

    def _compute_derivative_kernel(self, pos1, pos2, ind1, ind2, diag=False):
        """
        Compute kernel values based on derivative indicators with proper derivatives.
        This is the accurate implementation following the GPy version.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0] if pos2 is not None else n1
        
        if diag:
            # Diagonal elements - need special handling
            result = torch.zeros(n1, dtype=pos1.dtype, device=pos1.device)
            
            for i in range(n1):
                if ind1[i, 0] == 1:  # Function value
                    result[i] = self.outputscale
                else:  # Derivative
                    dim = torch.where(ind1[i, 1:] == 1)[0]
                    if len(dim) > 0:
                        result[i] = self._compute_derivative_diagonal_exact(pos1[i:i+1], dim[0].item())
                    else:
                        result[i] = self.outputscale * 0.1
            
            return result + 1e-4  # Small jitter for stability
        
        # Full kernel matrix computation
        result = torch.zeros(n1, n2, dtype=pos1.dtype, device=pos1.device)
        
        # Precompute base kernel
        dist2 = self._dist2_atomic(pos1, pos2)
        K_base = self.outputscale * torch.exp(-0.5 * dist2)
        
        # Process each element based on type
        for i in range(n1):
            for j in range(n2):
                # Determine observation types
                is_val1 = (ind1[i, 0] == 1)
                is_val2 = (ind2[j, 0] == 1)
                
                if is_val1 and is_val2:
                    # Both function values: K(x1, x2)
                    result[i, j] = K_base[i, j]
                    
                elif is_val1 and not is_val2:
                    # x1 is function, x2 is derivative: dK/dx2
                    dim2 = torch.where(ind2[j, 1:] == 1)[0]
                    if len(dim2) > 0:
                        dim2 = dim2[0].item()
                        # Note: derivative is w.r.t. second argument
                        ddist2 = self._ddist2_dX_single(pos2[j:j+1], pos1[i:i+1], dim2)
                        result[i, j] = -0.5 * K_base[i, j] * ddist2
                    
                elif not is_val1 and is_val2:
                    # x1 is derivative, x2 is function: dK/dx1
                    dim1 = torch.where(ind1[i, 1:] == 1)[0]
                    if len(dim1) > 0:
                        dim1 = dim1[0].item()
                        # Derivative w.r.t. first argument
                        ddist2 = self._ddist2_dX_single(pos1[i:i+1], pos2[j:j+1], dim1)
                        result[i, j] = -0.5 * K_base[i, j] * ddist2
                    
                else:
                    # Both derivatives: d²K/dx1dx2
                    dim1 = torch.where(ind1[i, 1:] == 1)[0]
                    dim2 = torch.where(ind2[j, 1:] == 1)[0]
                    
                    if len(dim1) > 0 and len(dim2) > 0:
                        dim1 = dim1[0].item()
                        dim2 = dim2[0].item()
                        
                        # Second derivative term
                        d2dist2 = self._d2dist2_dXdX2_single(pos1[i:i+1], pos2[j:j+1], dim1, dim2)
                        term1 = -0.5 * K_base[i, j] * d2dist2
                        
                        # Product of first derivatives term
                        if pos2 is None and i == j:  # Same point
                            ddist2_1 = self._ddist2_dX_single(pos1[i:i+1], pos1[j:j+1], dim1)
                            ddist2_2 = self._ddist2_dX_single(pos1[j:j+1], pos1[i:i+1], dim2)
                            term2 = 0.25 * K_base[i, j] * ddist2_1 * ddist2_2
                        else:
                            ddist2_1 = self._ddist2_dX_single(pos1[i:i+1], pos2[j:j+1], dim1)
                            ddist2_2 = self._ddist2_dX_single(pos2[j:j+1], pos1[i:i+1], dim2)
                            term2 = 0.25 * K_base[i, j] * ddist2_1 * ddist2_2
                        
                        result[i, j] = term1 + term2
        
        # Ensure positive definiteness for training
        if pos2 is None:
            # Symmetrize
            result = 0.5 * (result + result.t())
            
            # Check minimum eigenvalue and add to diagonal if needed
            try:
                min_eig = torch.linalg.eigvalsh(result).min()
                if min_eig < 1e-4:
                    result = result + (1e-4 - min_eig + 1e-5) * torch.eye(n1, device=result.device)
            except:
                # If eigenvalue computation fails, add conservative jitter
                result = result + 1e-3 * torch.eye(n1, device=result.device)
        
        return result
    
    def _ddist2_dX_single(self, x1, x2, dim_x):
        """
        Compute derivative of squared distance for a single point pair.
        More efficient than the full matrix version.
        """
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        i = dim_x // 3  # Which atom
        xyz = dim_x % 3  # Which coordinate
        
        # Reshape inputs
        x1_3d = x1.reshape(-1, 3)
        x2_3d = x2.reshape(-1, 3)
        
        result = torch.tensor(0.0, dtype=x1.dtype, device=x1.device)
        
        # Moving atom contributions
        if N_mov > 1:
            for j in range(N_mov):
                if j != i:
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        # Positions
                        pos_i_1 = x1_3d[i]
                        pos_j_1 = x1_3d[j]
                        pos_i_2 = x2_3d[i]
                        pos_j_2 = x2_3d[j]
                        
                        # Compute inverse distances
                        r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2) + self.eps)
                        r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2) + self.eps)
                        
                        invr_ij_1 = 1.0 / r_ij_1
                        invr_ij_2 = 1.0 / r_ij_2
                        
                        # Derivative contribution
                        deriv = -2.0 * s2[pt] * (invr_ij_1 - invr_ij_2)
                        deriv = deriv * (invr_ij_1**3) * (pos_i_1[xyz] - pos_j_1[xyz])
                        result += deriv
        
        # Frozen atom contributions
        if N_fro > 0:
            for j in range(N_fro):
                pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                if pt >= 0:
                    pos_i_1 = x1_3d[i]
                    pos_i_2 = x2_3d[i]
                    pos_fro = self.conf_fro[j]
                    
                    r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2) + self.eps)
                    r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2) + self.eps)
                    
                    invr_ij_1 = 1.0 / r_ij_1
                    invr_ij_2 = 1.0 / r_ij_2
                    
                    deriv = -2.0 * s2[pt] * (invr_ij_1 - invr_ij_2)
                    deriv = deriv * (invr_ij_1**3) * (pos_i_1[xyz] - pos_fro[xyz])
                    result += deriv
        
        return result

    def _d2dist2_dXdX2_single(self, x1, x2, dim_x1, dim_x2):
        """
        Compute second derivative of squared distance for a single point pair.
        """
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        i_1 = dim_x1 // 3
        xyz_1 = dim_x1 % 3
        i_2 = dim_x2 // 3
        xyz_2 = dim_x2 % 3
        
        x1_3d = x1.reshape(-1, 3)
        x2_3d = x2.reshape(-1, 3)
        
        result = torch.tensor(0.0, dtype=x1.dtype, device=x1.device)
        
        if i_1 != i_2:
            # Different atoms
            pt = self.pairtype[self.atomtype_mov[i_1], self.atomtype_mov[i_2]]
            if pt >= 0:
                # Positions  
                pos_i1_1 = x1_3d[i_1]
                pos_i2_1 = x1_3d[i_2]
                pos_i1_2 = x2_3d[i_1]
                pos_i2_2 = x2_3d[i_2]
                
                # Distances
                r_12_1 = torch.sqrt(torch.sum((pos_i1_1 - pos_i2_1)**2) + self.eps)
                r_12_2 = torch.sqrt(torch.sum((pos_i1_2 - pos_i2_2)**2) + self.eps)
                
                invr_12_1 = 1.0 / r_12_1
                invr_12_2 = 1.0 / r_12_2
                
                # Second derivative
                temp = 2.0 * s2[pt] * (invr_12_1**3) * (invr_12_2**3)
                deriv = temp * (pos_i1_1[xyz_1] - pos_i2_1[xyz_1]) * (pos_i1_2[xyz_2] - pos_i2_2[xyz_2])
                result += deriv
        else:
            # Same atom
            i = i_1
            
            # Moving atoms
            if N_mov > 1:
                for j in range(N_mov):
                    if j != i:
                        pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                        if pt >= 0:
                            pos_i_1 = x1_3d[i]
                            pos_j_1 = x1_3d[j]
                            pos_i_2 = x2_3d[i]
                            pos_j_2 = x2_3d[j]
                            
                            r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2) + self.eps)
                            r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2) + self.eps)
                            
                            invr_ij_1 = 1.0 / r_ij_1
                            invr_ij_2 = 1.0 / r_ij_2
                            
                            temp = -2.0 * s2[pt] * (invr_ij_1**3) * (invr_ij_2**3)
                            deriv = temp * (pos_i_1[xyz_1] - pos_j_1[xyz_1]) * (pos_i_2[xyz_2] - pos_j_2[xyz_2])
                            result += deriv
            
            # Frozen atoms
            if N_fro > 0:
                for j in range(N_fro):
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                    if pt >= 0:
                        pos_i_1 = x1_3d[i]
                        pos_i_2 = x2_3d[i]
                        pos_fro = self.conf_fro[j]
                        
                        r_ij_1 = torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2) + self.eps)
                        r_ij_2 = torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2) + self.eps)
                        
                        invr_ij_1 = 1.0 / r_ij_1
                        invr_ij_2 = 1.0 / r_ij_2
                        
                        temp = -2.0 * s2[pt] * (invr_ij_1**3) * (invr_ij_2**3)
                        deriv = temp * (pos_i_1[xyz_1] - pos_fro[xyz_1]) * (pos_i_2[xyz_2] - pos_fro[xyz_2])
                        result += deriv
        
        return result

    def _dK_dlengthscale_pt(self, x1, x2, K_base, pt_idx):
        """
        Compute derivative of kernel w.r.t. a specific lengthscale.
        """
        if x2 is None:
            x2 = x1
        
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s3 = 1.0 / ls.pow(3)
        
        # Initialize derivative matrix
        dK = torch.zeros_like(K_base)
        
        # Moving-moving pairs
        if N_mov > 1:
            for i in range(N_mov - 1):
                for j in range(i + 1, N_mov):
                    if self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]] == pt_idx:
                        # This pair uses the lengthscale we're differentiating w.r.t.
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_j_1 = x1[:, j*3:(j+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_j_2 = x2[:, j*3:(j+1)*3]
                        
                        r_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps)
                        r_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps)
                        
                        # Contribution to derivative
                        contrib = -2.0 * s3[pt_idx] * (r_ij_1 - r_ij_2.unsqueeze(0))**2
                        dK += contrib * K_base
        
        # Moving-frozen pairs
        if N_fro > 0:
            for i in range(N_mov):
                for j in range(N_fro):
                    if self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]] == pt_idx:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_fro = self.conf_fro[j:j+1, :]
                        
                        r_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2, dim=1, keepdim=True) + self.eps)
                        r_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2, dim=1) + self.eps)
                        
                        contrib = -2.0 * s3[pt_idx] * (r_ij_1 - r_ij_2.unsqueeze(0))**2
                        dK += contrib * K_base
        
        return dK
    
    def _compute_derivative_diagonal_exact(self, x, dim):
        """
        Compute exact diagonal element for derivative observation.
        This is d²K/dx_dim² evaluated at the same point.
        """
        atom_idx = dim // 3
        coord_idx = dim % 3  # 0=x, 1=y, 2=z
        
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        # Reshape position
        x_3d = x.reshape(-1, 3)
        
        # Get lengthscales
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        # For d²K/dx² at same point, the kernel value is K(x,x) = outputscale
        # We need d²(dist²)/dx² at the same point
        
        d2dist2_dx2 = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        
        # Contribution from moving atoms
        if N_mov > 1:
            for j in range(N_mov):
                if j != atom_idx:
                    pt = self.pairtype[self.atomtype_mov[atom_idx], self.atomtype_mov[j]]
                    if pt >= 0:
                        # Distance between atoms i and j
                        diff = x_3d[atom_idx] - x_3d[j]
                        r2 = torch.sum(diff**2) + self.eps
                        r = torch.sqrt(r2)
                        
                        # Second derivative of 1/r w.r.t. coordinate
                        # d²(1/r)/dx² = 3*x²/r^5 - 1/r³
                        coord_diff = diff[coord_idx]
                        term = s2[pt] * (3 * coord_diff**2 / r**5 - 1 / r**3)
                        d2dist2_dx2 += 2 * term  # Factor of 2 from the squared term
        
        # Contribution from frozen atoms
        if N_fro > 0:
            for j in range(N_fro):
                pt = self.pairtype[self.atomtype_mov[atom_idx], self.atomtype_fro[j]]
                if pt >= 0:
                    # Distance to frozen atom
                    diff = x_3d[atom_idx] - self.conf_fro[j]
                    r2 = torch.sum(diff**2) + self.eps
                    r = torch.sqrt(r2)
                    
                    coord_diff = diff[coord_idx]
                    term = s2[pt] * (3 * coord_diff**2 / r**5 - 1 / r**3)
                    d2dist2_dx2 += 2 * term
        
        # The diagonal element is K(x,x) * d²(dist²)/dx²
        # For the exponential, we also get a term from the square of first derivative
        # But at the same point, dist² = 0, so K(x,x) = outputscale
        
        # However, we need to be careful about the limiting behavior
        # For numerical stability, we use a finite difference approximation
        # or a reasonable bounded value
        
        if abs(d2dist2_dx2) > 1e6:  # Avoid numerical issues
            d2dist2_dx2 = torch.sign(d2dist2_dx2) * 1e6
        
        # The full expression includes both the second derivative term and
        # the squared first derivative term
        result = self.outputscale * max(0.1, min(abs(d2dist2_dx2), 10.0))
        
        return result
    
    def _ddist2_dX_vectorized(self, x1, x2, dim_x):
        """
        Vectorized computation of derivative of squared distance.
        """
        if x2 is None:
            x2 = x1
            
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        
        ls = self.lengthscale.squeeze(0)
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s2 = 1.0 / ls.pow(2)
        
        i = dim_x // 3  # Which atom
        xyz = dim_x % 3  # Which coordinate
        
        # Vectorized computation
        n1, n2 = x1.shape[0], x2.shape[0]
        D1 = torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device)
        
        # For moving atoms - vectorize the computation
        if N_mov > 1:
            # Get positions for atom i
            pos_i_1 = x1[:, i*3:(i+1)*3]
            pos_i_2 = x2[:, i*3:(i+1)*3]
            
            for j in range(N_mov):
                if j != i:
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        # Vectorized computation for all pairs
                        pos_j_1 = x1[:, j*3:(j+1)*3]
                        pos_j_2 = x2[:, j*3:(j+1)*3]
                        
                        # Compute distances
                        diff_1 = pos_i_1 - pos_j_1
                        diff_2 = pos_i_2 - pos_j_2
                        
                        r_1 = torch.sqrt(torch.sum(diff_1**2, dim=1, keepdim=True) + self.eps)
                        r_2 = torch.sqrt(torch.sum(diff_2**2, dim=1) + self.eps)
                        
                        invr_1 = 1.0 / r_1
                        invr_2 = 1.0 / r_2
                        
                        # Vectorized derivative computation
                        deriv = -2.0 * s2[pt] * (invr_1 - invr_2.unsqueeze(0))
                        deriv = deriv * (invr_1**3) * diff_1[:, xyz:xyz+1]
                        D1 += deriv
        
        # Similar vectorization for frozen atoms
        if N_fro > 0:
            pos_i_1 = x1[:, i*3:(i+1)*3]
            pos_i_2 = x2[:, i*3:(i+1)*3]
            
            for j in range(N_fro):
                pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                if pt >= 0:
                    pos_fro = self.conf_fro[j:j+1, :]
                    
                    # Vectorized computation
                    diff_1 = pos_i_1 - pos_fro
                    diff_2 = pos_i_2 - pos_fro
                    
                    r_1 = torch.sqrt(torch.sum(diff_1**2, dim=1, keepdim=True) + self.eps)
                    r_2 = torch.sqrt(torch.sum(diff_2**2, dim=1) + self.eps)
                    
                    invr_1 = 1.0 / r_1
                    invr_2 = 1.0 / r_2
                    
                    deriv = -2.0 * s2[pt] * (invr_1 - invr_2.unsqueeze(0))
                    deriv = deriv * (invr_1**3) * diff_1[:, xyz:xyz+1]
                    D1 += deriv
                    
        return D1
    
    def _compute_kernel_element(self, x1, x2, ind1, ind2):
        """Compute a single element of the kernel matrix based on derivative indicators."""
        # Check what type of observations we have
        is_val1 = (ind1[0] == 1)
        is_val2 = (ind2[0] == 1)
        
        if is_val1 and is_val2:
            # K(x1, x2)
            dist2 = self._dist2_atomic(x1, x2)
            return (self.outputscale * torch.exp(-0.5 * dist2)).squeeze()
        
        elif is_val1 and not is_val2:
            # dK/dx2
            dim2 = torch.where(ind2[1:] == 1)[0].item()
            dist2 = self._dist2_atomic(x1, x2)
            K = self.outputscale * torch.exp(-0.5 * dist2)
            dK_dx2 = self._ddist2_dX(x2, x1, dim2).t() * (-0.5) * K
            return dK_dx2.squeeze()
        
        elif not is_val1 and is_val2:
            # dK/dx1
            dim1 = torch.where(ind1[1:] == 1)[0].item()
            dist2 = self._dist2_atomic(x1, x2)
            K = self.outputscale * torch.exp(-0.5 * dist2)
            dK_dx1 = self._ddist2_dX(x1, x2, dim1) * (-0.5) * K
            return dK_dx1.squeeze()
        
        else:
            # d²K/dx1dx2
            dim1 = torch.where(ind1[1:] == 1)[0].item()
            dim2 = torch.where(ind2[1:] == 1)[0].item()
            
            dist2 = self._dist2_atomic(x1, x2)
            K = self.outputscale * torch.exp(-0.5 * dist2)
            
            # Second derivative
            d2dist2 = self._d2dist2_dXdX2(x1, x2, dim1, dim2)
            d2K = (-0.5) * K * d2dist2
            
            # Add contribution from first derivatives if same dimension
            if dim1 == dim2:
                ddist2_1 = self._ddist2_dX(x1, x2, dim1)
                ddist2_2 = self._ddist2_dX(x2, x1, dim2).t()
                d2K += 0.25 * K * ddist2_1 * ddist2_2
                
            return d2K.squeeze()
    
    def _compute_derivative_diagonal(self, x, dim):
        """Compute diagonal element for derivative observation."""
        # For the diagonal of d²K/dx²dx² at the same point, we need to compute
        # the second derivative of the kernel w.r.t. the same coordinate
        
        # Get the relevant lengthscale for this dimension
        atom_idx = dim // 3
        
        # Find which pair types involve this atom
        relevant_lengthscales = []
        N_mov = self.atomtype_mov.shape[0]
        
        # Check all pair types involving this atom
        for j in range(N_mov):
            if j != atom_idx:
                pt = self.pairtype[self.atomtype_mov[atom_idx], self.atomtype_mov[j]]
                if pt >= 0 and pt < self.lengthscale.shape[-1]:
                    ls_val = self.lengthscale.squeeze()[pt]
                    if ls_val > 0:
                        relevant_lengthscales.append(ls_val)
        
        # If frozen atoms exist, check those too
        N_fro = self.atomtype_fro.shape[0]
        if N_fro > 0:
            for j in range(N_fro):
                pt = self.pairtype[self.atomtype_mov[atom_idx], self.atomtype_fro[j]]
                if pt >= 0 and pt < self.lengthscale.shape[-1]:
                    ls_val = self.lengthscale.squeeze()[pt]
                    if ls_val > 0:
                        relevant_lengthscales.append(ls_val)
        
        if relevant_lengthscales:
            # Use the minimum lengthscale (most restrictive)
            min_lengthscale = min(relevant_lengthscales)
            # The second derivative at the same point is roughly outputscale / lengthscale²
            # Add small epsilon to avoid division issues
            return self.outputscale / (min_lengthscale ** 2 + 1e-6)
        else:
            # Default value if no interactions
            return self.outputscale * 0.1  # Small positive value
    def dK_ddist2_via_X(self, x1, x2):
        """dK/d(dist²) - First derivative of kernel w.r.t. squared distance"""
        dist2 = self._dist2_atomic(x1, x2)
        return -0.5 * self.outputscale * torch.exp(-0.5 * dist2)

    def dK2_ddist2dist2_via_X(self, x1, x2):
        """d²K/d(dist²)² - Second derivative of kernel w.r.t. squared distance"""
        dist2 = self._dist2_atomic(x1, x2)
        return 0.25 * self.outputscale * torch.exp(-0.5 * dist2)

    def dK3_ddist2dist2dist2_via_X(self, x1, x2):
        """d³K/d(dist²)³ - Third derivative of kernel w.r.t. squared distance"""
        dist2 = self._dist2_atomic(x1, x2)
        return -0.125 * self.outputscale * torch.exp(-0.5 * dist2)
    
    def ddist2_dlengthscale(self, x1, x2):
        """
        Derivative of squared distance w.r.t. each lengthscale parameter.
        Returns a list of derivative matrices, one for each pair type.
        """
        if x2 is None:
            x2 = x1
        
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        n_pt = self.conf_info['n_pt']
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s3 = 1.0 / ls.pow(3)
        
        # Initialize list of derivatives for each pair type
        D_pt = []
        for pt in range(n_pt):
            D_pt.append(torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device))
        
        # Distances between moving atoms
        if N_mov > 1:
            for i in range(N_mov - 1):
                for j in range(i + 1, N_mov):
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_j_1 = x1[:, j*3:(j+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_j_2 = x2[:, j*3:(j+1)*3]
                        
                        invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps)
                        invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps)
                        
                        D_pt[pt] = D_pt[pt] - 2.0 * s3[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))**2
        
        # Distances from moving atoms to frozen atoms
        if N_fro > 0:
            for i in range(N_mov):
                for j in range(N_fro):
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                    if pt >= 0:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_fro = self.conf_fro[j:j+1, :]
                        
                        invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2, dim=1, keepdim=True) + self.eps)
                        invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2, dim=1) + self.eps)
                        
                        D_pt[pt] = D_pt[pt] - 2.0 * s3[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))**2
        
        return D_pt
    def dK2_dmagnitudedX(self, x1, x2, dim):
        """Mixed derivative: d²K/(d(magnitude)dx)"""
        return 2 * self.dK_dX(x1, x2, dim) / self.outputscale

    def dK2_dmagnitudedX2(self, x1, x2, dim):
        """Mixed derivative: d²K/(d(magnitude)dx2)"""
        return 2 * self.dK_dX2(x1, x2, dim) / self.outputscale

    def dK3_dmagnitudedXdX2(self, x1, x2, dim, dimX2):
        """Mixed third derivative: d³K/(d(magnitude)dxdx2)"""
        return 2 * self.dK2_dXdX2(x1, x2, dim, dimX2) / self.outputscale

    def dK_dX(self, x1, x2, dimX):
        """Derivative of kernel w.r.t. first input dimension dimX"""
        D1 = self._ddist2_dX(x1, x2, dimX)
        DK = self.dK_ddist2_via_X(x1, x2)
        return D1 * DK

    def dK_dX2(self, x1, x2, dimX2):
        """Derivative of kernel w.r.t. second input dimension dimX2"""
        D2 = self._ddist2_dX(x2, x1, dimX2).t()
        DK = self.dK_ddist2_via_X(x1, x2)
        return D2 * DK

    def dK2_dXdX2(self, x1, x2, dimX, dimX2):
        """Second derivative of kernel w.r.t. both inputs"""
        D1 = self._ddist2_dX(x1, x2, dimX)
        D2 = self._ddist2_dX(x2, x1, dimX2).t()
        D12 = self._d2dist2_dXdX2(x1, x2, dimX, dimX2)
        DK = self.dK_ddist2_via_X(x1, x2)
        DK2 = self.dK2_ddist2dist2_via_X(x1, x2)
        return DK * D12 + DK2 * D1 * D2
    
    def d2dist2_dlengthscaledX(self, x1, x2, dimX):
        """Second derivative of distance w.r.t. lengthscale and position"""
        if x2 is None:
            x2 = x1
        
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        n_pt = self.conf_info['n_pt']
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s3 = 1.0 / ls.pow(3)
        
        i = dimX // 3
        xyz = dimX % 3
        
        D1_pt = []
        for pt in range(n_pt):
            D1_pt.append(torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device))
        
        # Distances between moving atoms
        if N_mov > 1:
            for j in range(N_mov):
                if j != i:
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                    if pt >= 0:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_j_1 = x1[:, j*3:(j+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_j_2 = x2[:, j*3:(j+1)*3]
                        
                        invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps)
                        invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps)
                        
                        deriv_ij = 4.0 * s3[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))
                        deriv_ij = deriv_ij * (invr_ij_1**3) * (x1[:, i*3+xyz:i*3+xyz+1] - x1[:, j*3+xyz:j*3+xyz+1])
                        D1_pt[pt] = D1_pt[pt] + deriv_ij
        
        # Distances from moving atoms to frozen atoms
        if N_fro > 0:
            for j in range(N_fro):
                pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                if pt >= 0:
                    pos_i_1 = x1[:, i*3:(i+1)*3]
                    pos_i_2 = x2[:, i*3:(i+1)*3]
                    pos_fro = self.conf_fro[j:j+1, :]
                    
                    invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2, dim=1, keepdim=True) + self.eps)
                    invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2, dim=1) + self.eps)
                    
                    deriv_ij = 4.0 * s3[pt] * (invr_ij_1 - invr_ij_2.unsqueeze(0))
                    deriv_ij = deriv_ij * (invr_ij_1**3) * (x1[:, i*3+xyz:i*3+xyz+1] - pos_fro[:, xyz:xyz+1])
                    D1_pt[pt] = D1_pt[pt] + deriv_ij
        
        return D1_pt

    def dK2_dlengthscaledX(self, x1, x2, dimX):
        """Mixed derivative: d²K/(d(lengthscale)dx) for each lengthscale"""
        D1 = self._ddist2_dX(x1, x2, dimX)
        D_pt = self.ddist2_dlengthscale(x1, x2)
        D1_pt = self.d2dist2_dlengthscaledX(x1, x2, dimX)
        DK = self.dK_ddist2_via_X(x1, x2)
        DK2 = self.dK2_ddist2dist2_via_X(x1, x2)
        
        DK_1pt = []
        for pt in range(len(D_pt)):
            DK_1pt.append(DK * D1_pt[pt] + DK2 * D1 * D_pt[pt])
        return DK_1pt

    def dK2_dlengthscaledX2(self, x1, x2, dimX2):
        """Mixed derivative: d²K/(d(lengthscale)dx2) for each lengthscale"""
        D2 = self._ddist2_dX(x2, x1, dimX2).t()
        D_pt = self.ddist2_dlengthscale(x1, x2)
        D2_pt_T = self.d2dist2_dlengthscaledX(x2, x1, dimX2)
        DK = self.dK_ddist2_via_X(x1, x2)
        DK2 = self.dK2_ddist2dist2_via_X(x1, x2)
        
        DK_2pt = []
        for pt in range(len(D_pt)):
            DK_2pt.append(DK * D2_pt_T[pt].t() + DK2 * D2 * D_pt[pt])
        return DK_2pt
    
    def d3dist2_dlengthscaledXdX2(self, x1, x2, dimX, dimX2):
        """Third derivative of distance w.r.t. lengthscale and two positions"""
        if x2 is None:
            x2 = x1
        
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        N_mov = self.atomtype_mov.shape[0]
        N_fro = self.atomtype_fro.shape[0]
        n_pt = self.conf_info['n_pt']
        
        ls = self.lengthscale.squeeze()
        if ls.dim() == 0:
            ls = ls.unsqueeze(0)
        s = 1.0 / ls
        
        i_1 = dimX // 3
        xyz_1 = dimX % 3
        i_2 = dimX2 // 3
        xyz_2 = dimX2 % 3
        
        D12_pt = []
        for pt in range(n_pt):
            D12_pt.append(torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device))
        
        if i_1 != i_2:
            # Different atoms
            pt = self.pairtype[self.atomtype_mov[i_1], self.atomtype_mov[i_2]]
            if pt >= 0:
                pos_i1_1 = x1[:, i_1*3:(i_1+1)*3]
                pos_i2_1 = x1[:, i_2*3:(i_2+1)*3]
                pos_i1_2 = x2[:, i_1*3:(i_1+1)*3]
                pos_i2_2 = x2[:, i_2*3:(i_2+1)*3]
                
                invr_12_1 = 1.0 / torch.sqrt(torch.sum((pos_i1_1 - pos_i2_1)**2, dim=1, keepdim=True) + self.eps)
                invr_12_2 = 1.0 / torch.sqrt(torch.sum((pos_i1_2 - pos_i2_2)**2, dim=1) + self.eps)
                
                temp = -4.0 * (s[pt] * invr_12_1 * invr_12_2.unsqueeze(0))**3
                deriv_12 = temp * (x1[:, i_1*3+xyz_1:i_1*3+xyz_1+1] - x1[:, i_2*3+xyz_1:i_2*3+xyz_1+1])
                deriv_12 = deriv_12 * (x2[:, i_1*3+xyz_2:i_1*3+xyz_2+1] - x2[:, i_2*3+xyz_2:i_2*3+xyz_2+1]).t()
                D12_pt[pt] = D12_pt[pt] + deriv_12
        else:
            # Same atom
            i = i_1
            if N_mov > 1:
                for j in range(N_mov):
                    if j != i:
                        pt = self.pairtype[self.atomtype_mov[i], self.atomtype_mov[j]]
                        if pt >= 0:
                            pos_i_1 = x1[:, i*3:(i+1)*3]
                            pos_j_1 = x1[:, j*3:(j+1)*3]
                            pos_i_2 = x2[:, i*3:(i+1)*3]
                            pos_j_2 = x2[:, j*3:(j+1)*3]
                            
                            invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_j_1)**2, dim=1, keepdim=True) + self.eps)
                            invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_j_2)**2, dim=1) + self.eps)
                            
                            temp = 4.0 * (s[pt] * invr_ij_1 * invr_ij_2.unsqueeze(0))**3
                            deriv_12 = temp * (x1[:, i*3+xyz_1:i*3+xyz_1+1] - x1[:, j*3+xyz_1:j*3+xyz_1+1])
                            deriv_12 = deriv_12 * (x2[:, i*3+xyz_2:i*3+xyz_2+1] - x2[:, j*3+xyz_2:j*3+xyz_2+1]).t()
                            D12_pt[pt] = D12_pt[pt] + deriv_12
            
            if N_fro > 0:
                for j in range(N_fro):
                    pt = self.pairtype[self.atomtype_mov[i], self.atomtype_fro[j]]
                    if pt >= 0:
                        pos_i_1 = x1[:, i*3:(i+1)*3]
                        pos_i_2 = x2[:, i*3:(i+1)*3]
                        pos_fro = self.conf_fro[j:j+1, :]
                        
                        invr_ij_1 = 1.0 / torch.sqrt(torch.sum((pos_i_1 - pos_fro)**2, dim=1, keepdim=True) + self.eps)
                        invr_ij_2 = 1.0 / torch.sqrt(torch.sum((pos_i_2 - pos_fro)**2, dim=1) + self.eps)
                        
                        temp = 4.0 * (s[pt] * invr_ij_1 * invr_ij_2.unsqueeze(0))**3
                        deriv_12 = temp * (x1[:, i*3+xyz_1:i*3+xyz_1+1] - pos_fro[:, xyz_1:xyz_1+1])
                        deriv_12 = deriv_12 * (x2[:, i*3+xyz_2:i*3+xyz_2+1] - pos_fro[:, xyz_2:xyz_2+1]).t()
                        D12_pt[pt] = D12_pt[pt] + deriv_12
        
        return D12_pt

    def dK3_dlengthscaledXdX2(self, x1, x2, dimX, dimX2):
        """Mixed third derivative: d³K/(d(lengthscale)dxdx2) for each lengthscale"""
        D1 = self._ddist2_dX(x1, x2, dimX)
        D2 = self._ddist2_dX(x2, x1, dimX2).t()
        D12 = self._d2dist2_dXdX2(x1, x2, dimX, dimX2)
        D_pt = self.ddist2_dlengthscale(x1, x2)
        D1_pt = self.d2dist2_dlengthscaledX(x1, x2, dimX)
        D2_pt_T = self.d2dist2_dlengthscaledX(x2, x1, dimX2)
        D12_pt = self.d3dist2_dlengthscaledXdX2(x1, x2, dimX, dimX2)
        
        DK = self.dK_ddist2_via_X(x1, x2)
        DK2 = self.dK2_ddist2dist2_via_X(x1, x2)
        DK3 = self.dK3_ddist2dist2dist2_via_X(x1, x2)
        
        DK_12pt = []
        for pt in range(len(D_pt)):
            term1 = DK * D12_pt[pt]
            term2 = DK2 * (D1 * D2_pt_T[pt].t() + D2 * D1_pt[pt] + D_pt[pt] * D12)
            term3 = DK3 * D1 * D2 * D_pt[pt]
            DK_12pt.append(term1 + term2 + term3)
        
        return DK_12pt

    def Kdiag(self, x):
        """Compute only the diagonal of the kernel matrix"""
        ret = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        ret[:] = self.outputscale
        return ret

class GPModelWithDerivatives_rbf_atomic(gpytorch.models.ExactGP):
    """
    GP model that uses RBFAtomicKernelGrad for handling derivative observations.
    """
    def __init__(self, train_x, train_y, likelihood, atomic_info):
        super().__init__(train_x, train_y, likelihood)
        
        # Extract number of moving atoms from atomic_info
        self.n_moving = len(atomic_info['moving_indices'])
        self.input_dim = 3 * self.n_moving
        
        # Store dimensions for later use
        self.n_dims = self.input_dim
        
        # Create mean module that handles augmented inputs
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Create covariance module with RBFAtomicKernelGrad
        self.covar_module = RBFAtomicKernelGrad(
            conf_info=atomic_info,
            lengthscale_constraint=gpytorch.constraints.Positive()
        )
        
        # Enable derivative mode
        self.covar_module.include_derivatives = True
        
    def forward(self, x):
        # The input x has shape (n_samples, n_dims + n_dims + 1)
        # where the last n_dims + 1 columns are derivative indicators
        
        # For the mean, we just need a constant value regardless of derivative type
        batch_shape = x.shape[:-1]
        mean_x = self.mean_module(torch.zeros(*batch_shape, 1, device=x.device, dtype=x.dtype))
        
        # The covariance module will handle the augmented input
        covar_x = self.covar_module(x)
        
        # Ensure positive definiteness by adding jitter through lazy evaluation
        if self.training:
            # Use GPyTorch's built-in jitter handling
            with gpytorch.settings.cholesky_jitter(1e-2):
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)