#!/bin/bash

# This script forces permission fixes from inside the container
# Run this INSIDE the Docker container

echo "=========================================="
echo "Forcing Permission Fixes (Inside Container)"
echo "=========================================="

# Set the most permissive umask
umask 000
echo "Set umask to 000 (all files created will be world-writable)"

# Try to make everything writable
echo "Attempting to fix permissions..."

# Method 1: Change permissions directly
chmod -R 777 /workspace/TD_SPF/real_system/outputs 2>/dev/null && echo "✓ Fixed outputs directory" || echo "✗ Cannot fix outputs directly"
chmod -R 777 /workspace/TD_SPF/real_system/scripts 2>/dev/null && echo "✓ Fixed scripts directory" || echo "✗ Cannot fix scripts directly"

# Method 2: Create ACLs if available
if command -v setfacl &> /dev/null; then
    echo "Setting ACLs..."
    setfacl -R -m u::rwx,g::rwx,o::rwx /workspace/TD_SPF/real_system/outputs 2>/dev/null || true
    setfacl -d -R -m u::rwx,g::rwx,o::rwx /workspace/TD_SPF/real_system/outputs 2>/dev/null || true
fi

# Method 3: Pre-create all possible output directories
echo "Pre-creating output directories..."
cd /workspace/TD_SPF/real_system
for tr in 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0; do
    for snap in 25 50 100; do
        for rep in 1 2 3 4; do
            dir="outputs/run_TRSnap_TR${tr}_snap${snap}_rep${rep}"
            mkdir -p "$dir" 2>/dev/null
            chmod 777 "$dir" 2>/dev/null
            mkdir -p "$dir/checkpoints" 2>/dev/null
            chmod 777 "$dir/checkpoints" 2>/dev/null
        done
    done
done
echo "✓ Pre-created all output directories"

# Method 4: Create wrapper script that bypasses permissions
cat > /workspace/run_bypassing_permissions.sh << 'EOF'
#!/bin/bash

# This script bypasses NFS permission issues by copying everything to /tmp first

echo "Running with permission bypass..."

# Set permissive umask
umask 000

# Activate environment
source /activate_td_spf.sh

# Copy the entire workspace to /tmp (which is always writable)
echo "Copying workspace to /tmp (this may take a moment)..."
rm -rf /tmp/td_spf_workspace 2>/dev/null
cp -r /workspace/TD_SPF /tmp/td_spf_workspace

# Change to the temp directory
cd /tmp/td_spf_workspace/real_system/scripts

# Make everything executable
chmod +x *.sh *.py 2>/dev/null

# Run the study
echo "Starting runs in /tmp..."
bash run_tr_snapshot_study.sh

# Copy results back
echo "Copying results back to NFS..."
cp -r /tmp/td_spf_workspace/real_system/outputs/* /workspace/TD_SPF/real_system/outputs/ 2>/dev/null || {
    echo "Cannot copy back to NFS. Results are in /tmp/td_spf_workspace/real_system/outputs/"
}
EOF

chmod +x /workspace/run_bypassing_permissions.sh

# Method 5: Python wrapper that intercepts file operations
cat > /workspace/permission_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper that intercepts file operations to handle permission issues
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Create a temp directory for outputs
temp_base = tempfile.mkdtemp(prefix="td_spf_", dir="/tmp")
print(f"Using temporary directory: {temp_base}")

# Map NFS paths to temp paths
nfs_outputs = "/workspace/TD_SPF/real_system/outputs"
temp_outputs = f"{temp_base}/outputs"
os.makedirs(temp_outputs, exist_ok=True)

# Monkey-patch os.makedirs and open to redirect to temp
original_makedirs = os.makedirs
original_open = open

def makedirs_redirect(path, *args, **kwargs):
    if str(path).startswith(nfs_outputs):
        new_path = str(path).replace(nfs_outputs, temp_outputs)
        return original_makedirs(new_path, *args, **kwargs)
    return original_makedirs(path, *args, **kwargs)

def open_redirect(path, *args, **kwargs):
    if str(path).startswith(nfs_outputs):
        new_path = str(path).replace(nfs_outputs, temp_outputs)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        return original_open(new_path, *args, **kwargs)
    return original_open(path, *args, **kwargs)

os.makedirs = makedirs_redirect
__builtins__.open = open_redirect

# Now run the actual script
script_path = sys.argv[1] if len(sys.argv) > 1 else "run_tr_snapshot_study.sh"
if script_path.endswith('.sh'):
    os.system(f"bash {script_path}")
else:
    exec(open(script_path).read())

# Try to sync back
print(f"\nSyncing results from {temp_outputs} to {nfs_outputs}...")
os.system(f"cp -r {temp_outputs}/* {nfs_outputs}/ 2>/dev/null || echo 'Sync failed - results in {temp_outputs}'")
EOF

chmod +x /workspace/permission_wrapper.py

echo ""
echo "=========================================="
echo "Permission Fix Options Available:"
echo "=========================================="
echo ""
echo "Option 1: Run with umask 000 (simplest):"
echo "  umask 000"
echo "  cd /workspace/TD_SPF/real_system/scripts"
echo "  bash run_tr_snapshot_study.sh"
echo ""
echo "Option 2: Run with bypass script (copies to /tmp):"
echo "  /workspace/run_bypassing_permissions.sh"
echo ""
echo "Option 3: Run with Python wrapper (redirects writes):"
echo "  cd /workspace/TD_SPF/real_system/scripts"
echo "  python /workspace/permission_wrapper.py run_tr_snapshot_study.sh"
echo ""
echo "All output directories have been pre-created with 777 permissions."
echo "=========================================="
