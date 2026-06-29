#!/usr/bin/env python
"""Centralized output directory management for TD-SPF calculations."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class OutputManager:
    """Manages output directories and paths for TD-SPF calculations."""
    
    _instance = None
    _output_base = None
    _run_dir = None
    _input_base = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OutputManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the output manager."""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # Handle both old structure (scripts in root) and new structure (scripts in scripts/)
            current_dir = os.getcwd()
            if os.path.basename(current_dir) == 'scripts':
                # We're in the scripts directory, go up to parent
                self.code_dir = os.path.dirname(current_dir)
            else:
                # We're in the root directory
                self.code_dir = current_dir
    
    @classmethod
    def setup(cls, base_dir: Optional[str] = None, run_name: Optional[str] = None, 
              input_dir: Optional[str] = None):
        """Set up the output directory structure.
        
        Args:
            base_dir: Base directory for all outputs (default: ./outputs)
            run_name: Name for this run (default: timestamp)
            input_dir: Directory containing input files (default: ./inputs)
        """
        instance = cls()
        
        # Set input directory
        if input_dir is None:
            input_dir = os.path.join(instance.code_dir, "inputs")
        cls._input_base = os.path.abspath(input_dir)
        
        # Check if input directory exists
        if not os.path.exists(cls._input_base):
            raise FileNotFoundError(
                f"Input directory not found: {cls._input_base}\n"
                f"Please create the 'inputs' directory and place your input files there:\n"
                f"  - POSCAR\n"
                f"  - FORCE_CONSTANTS\n"
                f"  - INCAR, KPOINTS, POTCAR\n"
                f"  - Any other input files"
            )
        
        # Set base output directory
        if base_dir is None:
            base_dir = os.path.join(instance.code_dir, "outputs")
        cls._output_base = os.path.abspath(base_dir)
        
        # Create run-specific directory
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        cls._run_dir = os.path.join(cls._output_base, run_name)
        os.makedirs(cls._run_dir, exist_ok=True)
        
        # Create standard subdirectories
        subdirs = [
            'vasp_runs',
            'checkpoints', 
            'results',
            'data_gp1',
            'data_gp2',
            'data_trgp1',
            'logs',
            'plots',
            'dimer_runs'
        ]
        
        for subdir in subdirs:
            os.makedirs(cls.get_path(subdir), exist_ok=True)
        
        # Create a symlink to latest run for convenience
        latest_link = os.path.join(cls._output_base, "latest")
        if os.path.lexists(latest_link):  # lexists returns True for broken symlinks too
            os.remove(latest_link)
        os.symlink(cls._run_dir, latest_link)
        
        # Check for required input files and warn if missing
        cls._check_input_files()
        
        print(f"\nOutput directory initialized:")
        print(f"  Code:   {instance.code_dir}")
        print(f"  Inputs: {cls._input_base}")
        print(f"  Output: {cls._output_base}")
        print(f"  Run:    {cls._run_dir}")
        print(f"  Link:   {latest_link} -> {run_name}")
        
        return instance
    
    @classmethod
    def get_path(cls, *paths: str, create: bool = False) -> str:
        """Get a path relative to the run directory.
        
        Args:
            *paths: Path components to join
            create: If True, create the directory if it doesn't exist
            
        Returns:
            Absolute path to the requested location
        """
        if cls._run_dir is None:
            raise RuntimeError("OutputManager not initialized. Call OutputManager.setup() first.")
        
        full_path = os.path.join(cls._run_dir, *paths)
        
        if create:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        return full_path
    
    @classmethod
    def _check_input_files(cls):
        """Check for common input files and warn if missing."""
        common_files = {
            'POSCAR': 'Initial structure file',
            'INCAR': 'VASP input parameters',
            'KPOINTS': 'K-point mesh',
            'POTCAR': 'Pseudopotentials',
            'FORCE_CONSTANTS': 'Phonon force constants (for thermal calculations)',
            'infile.ssposcar': 'TDEP supercell structure (optional)',
            'infile.ucposcar': 'TDEP primitive cell structure (optional)'
        }
        
        print("\nChecking input files:")
        missing = []
        for filename, description in common_files.items():
            filepath = os.path.join(cls._input_base, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}: {description}")
            else:
                print(f"  ✗ {filename}: {description} (NOT FOUND)")
                missing.append(filename)
        
        if missing:
            print(f"\nWarning: {len(missing)} input files not found in {cls._input_base}")
            print("Some calculations may fail if required files are missing.")
    
    @classmethod
    def get_input_path(cls, *paths: str) -> str:
        """Get a path relative to the input directory.
        
        Args:
            *paths: Path components to join
            
        Returns:
            Absolute path to the requested location in input directory
        """
        if cls._input_base is None:
            # Fallback to code directory if not initialized
            instance = cls()
            return os.path.join(instance.code_dir, "inputs", *paths)
        
        return os.path.join(cls._input_base, *paths)
    
    @classmethod
    def cleanup_run(cls):
        """Clean up the current run directory."""
        if cls._run_dir and os.path.exists(cls._run_dir):
            shutil.rmtree(cls._run_dir)
            print(f"Cleaned up run directory: {cls._run_dir}")
    
    @classmethod
    def get_run_info(cls) -> dict:
        """Get information about the current run."""
        return {
            'code_dir': cls().code_dir,
            'input_dir': cls._input_base,
            'output_base': cls._output_base,
            'run_dir': cls._run_dir,
            'run_name': os.path.basename(cls._run_dir) if cls._run_dir else None
        }
    
    @classmethod
    def save_run_metadata(cls, metadata: dict):
        """Save metadata about the current run."""
        import json
        import numpy as np
        
        # Custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        metadata_file = cls.get_path('run_metadata.json')
        
        # Add run info
        metadata['run_info'] = cls.get_run_info()
        metadata['timestamp'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)


# Convenience functions
def setup_output_dir(base_dir: Optional[str] = None, run_name: Optional[str] = None,
                     input_dir: Optional[str] = None) -> OutputManager:
    """Set up output directory structure."""
    return OutputManager.setup(base_dir, run_name, input_dir)


def get_output_path(*paths: str, create: bool = False) -> str:
    """Get a path relative to the current run directory."""
    return OutputManager.get_path(*paths, create=create)


def get_input_path(*paths: str) -> str:
    """Get a path relative to the input directory."""
    return OutputManager.get_input_path(*paths)


def cleanup_outputs():
    """Clean up the current run directory."""
    OutputManager.cleanup_run()


def get_latest_run_dir() -> str:
    """Get the path to the most recent run directory."""
    # Handle both old structure (scripts in root) and new structure (scripts in scripts/)
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'scripts':
        # We're in the scripts directory, go up to parent
        base_dir = os.path.dirname(current_dir)
    else:
        # We're in the root directory
        base_dir = current_dir
    
    outputs_dir = os.path.join(base_dir, "outputs")
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError("No outputs directory found")
    
    # First check if there's a 'latest' symlink
    latest_link = os.path.join(outputs_dir, "latest")
    if os.path.exists(latest_link) and os.path.islink(latest_link):
        return os.path.realpath(latest_link)
    
    # Otherwise find all run directories
    run_dirs = []
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        # Accept directories that start with "run_" or contain GP runs
        if os.path.isdir(item_path) and (item.startswith("run_") or 
                                         "gp" in item.lower() or 
                                         item != "latest"):
            run_dirs.append(item_path)
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found in outputs")
    
    # Sort by modification time and return the most recent
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return run_dirs[0]
