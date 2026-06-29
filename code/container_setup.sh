#!/bin/bash
set -e

echo '=========================================='
echo 'Setting up TD_SPF environment in container'
echo '=========================================='

# Update system packages
echo 'Updating system packages...'
apt-get update -qq
apt-get install -y -qq wget bzip2 git vim nano curl build-essential

# Install Anaconda if not already installed
echo ''
echo 'Installing Anaconda...'
if [ ! -d /opt/anaconda3 ]; then
    cd /tmp
    wget -q https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh
    bash anaconda.sh -b -p /opt/anaconda3
    rm anaconda.sh
    echo 'Anaconda installed.'
else
    echo 'Anaconda already installed.'
fi

# Add Anaconda to PATH
export PATH=/opt/anaconda3/bin:$PATH
echo 'export PATH=/opt/anaconda3/bin:$PATH' >> ~/.bashrc

# Initialize conda
/opt/anaconda3/bin/conda init bash
source ~/.bashrc

# Navigate to TD_SPF directory
cd /workspace/TD_SPF

# Check for install_env.sh and run with GPU flag
if [ -f install_env.sh ]; then
    echo ''
    echo 'Running install_env.sh with GPU support...'
    bash install_env.sh --gpu
elif [ -f scripts/install_env.sh ]; then
    echo ''
    echo 'Running scripts/install_env.sh with GPU support...'
    bash scripts/install_env.sh --gpu
else
    echo ''
    echo 'WARNING: install_env.sh not found!'
    echo 'Creating a basic td_spf_unified environment with GPU support...'
    
    # Create basic environment if install_env.sh is missing
    /opt/anaconda3/bin/conda create -n td_spf_unified python=3.11 -y
    source /opt/anaconda3/bin/activate td_spf_unified
    
    # Install packages with GPU support
    pip install "numpy<2.0"
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    pip install --upgrade "jax[cuda12]"
    pip install scipy matplotlib pandas
    pip install ase==3.22.1 pymatgen
    pip install gpytorch==1.11 botorch
    
    echo 'Basic GPU-enabled environment created.'
fi

# Create activation script
cat > /activate_td_spf.sh << 'EOF'
#!/bin/bash
export PATH=/opt/anaconda3/bin:$PATH
source /opt/anaconda3/bin/activate td_spf_unified
cd /workspace/TD_SPF
echo 'TD_SPF environment activated'
echo 'Working directory: /workspace/TD_SPF'
EOF

chmod +x /activate_td_spf.sh

echo ''
echo '=========================================='
echo 'Setup complete!'
echo '=========================================='
echo ''
echo 'Environment: td_spf_unified'
echo 'TD_SPF location: /workspace/TD_SPF'
echo ''
echo 'To activate the environment, run:'
echo '  source /activate_td_spf.sh'
echo ''