#!/bin/bash

# Environment installation script for TD_SPF in Docker container
# This script installs all required packages for TD_SPF with GPU support

set -e

echo "=========================================="
echo "Installing TD_SPF Environment"
echo "=========================================="

# Ensure we're using Anaconda from the container
export PATH=/opt/anaconda3/bin:$PATH

# Create conda environment
echo "Creating conda environment: td_spf_unified"
conda create -n td_spf_unified python=3.10 -y

# Activate environment
source /opt/anaconda3/bin/activate td_spf_unified

echo ""
echo "Installing core scientific packages..."
conda install -y numpy scipy matplotlib pandas scikit-learn -c conda-forge

echo ""
echo "Installing materials science packages..."
pip install ase==3.22.1
pip install pymatgen==2024.5.1
pip install spglib

echo ""
echo "Installing PyTorch with CUDA support..."
# First uninstall any existing PyTorch to avoid conflicts
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
# Install PyTorch with CUDA 11.8 (compatible with most recent GPUs)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing JAX with GPU support..."
# Install JAX with CUDA support
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "Installing additional ML packages..."
pip install gpytorch==1.11
pip install botorch==0.9.2
pip install scikit-optimize

echo ""
echo "Installing optimization and numerical packages..."
pip install cvxpy
pip install qpsolvers
pip install osqp

echo ""
echo "Installing utility packages..."
pip install tqdm
pip install pyyaml
pip install joblib
pip install h5py
pip install seaborn
pip install plotly

echo ""
echo "Installing development tools..."
pip install ipython
pip install jupyter
pip install black
pip install flake8
pip install pytest

echo ""
echo "Verifying GPU access..."
python -c "
import torch
import jax

print('PyTorch GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('PyTorch GPU device:', torch.cuda.get_device_name(0))
    print('PyTorch CUDA version:', torch.version.cuda)

print('')
try:
    from jax.lib import xla_bridge
    print('JAX backend:', xla_bridge.get_backend().platform)
    import jax.numpy as jnp
    x = jnp.ones(10)
    print('JAX GPU test successful')
except Exception as e:
    print('JAX GPU test failed:', e)
"

echo ""
echo "Creating environment activation script..."
cat > /opt/anaconda3/envs/td_spf_unified/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash

# Set environment variables for TD_SPF
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# CUDA settings for optimal performance
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2

# JAX settings
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "TD_SPF environment variables set"
EOF

chmod +x /opt/anaconda3/envs/td_spf_unified/etc/conda/activate.d/env_vars.sh

echo ""
echo "=========================================="
echo "✓ Environment Installation Complete!"
echo "=========================================="
echo ""
echo "Environment name: td_spf_unified"
echo "Python version: 3.10"
echo "PyTorch: 2.1.0 with CUDA 11.8"
echo "JAX: Latest with CUDA support"
echo ""
echo "To activate this environment:"
echo "  conda activate td_spf_unified"
echo ""
echo "To test GPU support:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo "=========================================="