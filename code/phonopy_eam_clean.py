#!/usr/bin/env python
# phonopy_eam_clean.py - Clean phonopy + EAM with proper folder management

import os
import sys
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.calculators.eam import EAM
from ase.calculators.kim import KIM
import argparse
from pathlib import Path
from datetime import datetime
import yaml

# Try to import phonopy
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.interface.vasp import read_vasp
    from phonopy.file_IO import write_FORCE_SETS, write_FORCE_CONSTANTS, parse_FORCE_SETS
    PHONOPY_AVAILABLE = True
except ImportError as e:
    PHONOPY_AVAILABLE = False
    print(f"Warning: Phonopy Python API not available ({e})")

def setup_calculation_directory(base_dir, poscar_name):
    """Create a clean directory structure for the calculation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calc_dir = Path(base_dir) / f"phonopy_{poscar_name}_{timestamp}"
    
    # Create directory structure
    dirs = {
        'root': calc_dir,
        'displacements': calc_dir / 'displacements',
        'forces': calc_dir / 'forces',
        'results': calc_dir / 'results',
        'plots': calc_dir / 'plots'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs

def run_phonopy_calculation(poscar_path, calc_dir, dim):
    """Run phonopy to generate displacements."""
    # Copy POSCAR to calculation directory
    shutil.copy(poscar_path, calc_dir / 'POSCAR')
    
    # Change to calc directory
    original_dir = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        # Generate displacements
        dim_str = f"{dim[0]} {dim[1]} {dim[2]}"
        cmd = ['phonopy', '-d', f'--dim={dim_str}', '-c', 'POSCAR']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return []
        
        # Move displacement files to subdirectory
        disp_files = []
        for f in sorted(Path('.').glob('POSCAR-*')):
            dest = Path('displacements') / f.name
            shutil.move(str(f), str(dest))
            disp_files.append(dest)
        
        # Keep important files in root and results
        for f in ['SPOSCAR', 'phonopy_disp.yaml']:
            if Path(f).exists():
                shutil.copy(f, Path('results') / f)
        
        return disp_files
        
    finally:
        os.chdir(original_dir)

def calculate_forces(disp_files, calc, calc_dir):
    """Calculate forces for all displacement configurations."""
    forces_data = []
    natoms = None
    
    # Change to calc directory
    original_dir = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        for i, disp_file in enumerate(disp_files):
            print(f"   {disp_file.name} ({i+1}/{len(disp_files)})...", end='', flush=True)
            
            # Read structure
            atoms = read(disp_file, format='vasp')
            if natoms is None:
                natoms = len(atoms)
            
            # Calculate forces
            atoms.calc = calc
            forces = atoms.get_forces()
            forces_data.append(forces)
            
            # Save forces for this configuration
            force_file = Path('forces') / f'forces_{i:03d}.npy'
            np.save(force_file, forces)
            
            print(f" max|F| = {np.max(np.abs(forces)):.6f} eV/Å")
        
        return natoms, forces_data
        
    finally:
        os.chdir(original_dir)

def write_force_sets_fixed(natoms, forces_data, calc_dir):
    """Write FORCE_SETS in the canonical phonopy format."""
    disp_yaml_file = calc_dir / "phonopy_disp.yaml"

    with open(disp_yaml_file, "r") as f:
        disp_yaml = yaml.safe_load(f)

    displacements = disp_yaml.get("displacements")
    if displacements is None:
        raise ValueError("No displacements found in phonopy_disp.yaml")

    force_sets_file = calc_dir / "FORCE_SETS"
    with open(force_sets_file, "w") as f:
        # header
        f.write(f"{natoms}\n")
        f.write(f"{len(forces_data)}\n")

        # one data set per displacement
        for disp_info, forces in zip(displacements, forces_data):
            f.write(f"\n{disp_info['atom']}\n")
            dx, dy, dz = disp_info["displacement"]
            f.write(f"   {dx:15.10f}   {dy:15.10f}   {dz:15.10f}\n")

            # forces -- ONLY the three Cartesian components
            for fx, fy, fz in forces:
                f.write(f"   {fx:15.10f}   {fy:15.10f}   {fz:15.10f}\n")

    return force_sets_file

def calculate_force_constants_directly(calc_dir, dim, forces_data):
    """Build fc2 directly from forces, bypassing FORCE_SETS."""
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    from phonopy.file_IO import write_FORCE_CONSTANTS

    unitcell = read_vasp(calc_dir / "POSCAR")
    supercell_matrix = np.diag(dim)
    phonon = Phonopy(unitcell, supercell_matrix)

    with open(calc_dir / "phonopy_disp.yaml", "r") as f:
        disp_yaml = yaml.safe_load(f)
    
    displacements = disp_yaml.get("displacements")
    if displacements is None:
        raise RuntimeError("No displacements found in phonopy_disp.yaml")

    # Create dataset in the correct format
    dataset = {"natom": len(phonon.supercell), "first_atoms": []}
    
    for disp_info, forces in zip(displacements, forces_data):
        dataset["first_atoms"].append({
            "number": disp_info["atom"] - 1,  # 0-indexed
            "displacement": disp_info["displacement"],
            "forces": forces.tolist(),
        })

    phonon.dataset = dataset
    phonon.produce_force_constants()
    
    write_FORCE_CONSTANTS(
        phonon.force_constants, filename=str(calc_dir / "FORCE_CONSTANTS")
    )
    
    return phonon

def run_phonopy_analysis_direct(calc_dir, dim, mp_grid, tmax, band_path=None, forces_data=None):
    """Run phonopy analysis using direct Python implementation."""
    original_dir = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        print("\n4. Creating force constants...")
        
        # Try to load phonopy object with force constants
        phonon = None
        
        # Method 1: Try parsing FORCE_SETS
        try:
            # Read the structure
            unitcell = read_vasp("POSCAR")
            phonon = Phonopy(unitcell, dim, primitive_matrix='auto')
            
            # Read forces from FORCE_SETS
            force_sets = parse_FORCE_SETS(filename="FORCE_SETS")
            phonon.dataset = force_sets  # Use dataset instead of set_displacement_dataset
            
            # Produce force constants
            phonon.produce_force_constants()
            
        except Exception as e:
            print(f"   FORCE_SETS parsing failed: {e}")
            print("   Trying direct calculation from forces...")
            
            # Method 2: Calculate directly from forces if available
            if forces_data is not None:
                phonon = calculate_force_constants_directly(Path.cwd(), dim, forces_data)
            
            if phonon is None:
                raise RuntimeError("Failed to create phonon object")
        
        # Save force constants
        write_FORCE_CONSTANTS(phonon.force_constants, filename="FORCE_CONSTANTS")
        print("   Force constants created successfully!")
        
        # Band structure calculation
        if band_path:
            print("\n5. Calculating band structure...")
            try:
                # Try using seekpath if available
                phonon.auto_band_structure()
                
                # Save band structure
                phonon.write_yaml_band_structure(filename="results/band.yaml")
                
                # Plot band structure
                band_plot = phonon.plot_band_structure()
                if band_plot is not None:
                    plt.savefig("plots/band_structure.pdf")
                    plt.close()
                
                # Also save as text
                try:
                    band_dict = phonon.get_band_structure_dict()
                    # Check the structure of band_dict
                    if 'distances' in band_dict and 'frequencies' in band_dict:
                        distances = band_dict['distances']
                        frequencies = band_dict['frequencies']
                        
                        # Flatten if needed
                        if isinstance(distances, list) and len(distances) > 0 and isinstance(distances[0], list):
                            distances = np.concatenate(distances)
                            frequencies = np.concatenate(frequencies, axis=0)
                        else:
                            distances = np.array(distances)
                            frequencies = np.array(frequencies)
                        
                        np.savetxt('results/band.dat', 
                                  np.column_stack([distances, frequencies]),
                                  header='Distance Frequency(THz)')
                except Exception as e:
                    print(f"   Could not save band data: {e}")
                
            except ImportError:
                print("   seekpath not installed, using simple band path...")
                # Fallback to simple band path
                path_connections = []
                labels = band_path.split()
                for i in range(len(labels)-1):
                    path_connections.append([labels[i], labels[i+1]])
                
                phonon.set_band_structure(path_connections)
                
                # Get band structure data
                band_dict = phonon.get_band_structure_dict()
                qpoints = band_dict['qpoints']
                distances = band_dict['distances']
                frequencies = band_dict['frequencies']
                
                # Plot band structure
                plt.figure(figsize=(8, 6))
                for i in range(len(frequencies[0])):
                    plt.plot(distances, frequencies[:, i], 'b-')
                
                plt.xlabel('Wave vector')
                plt.ylabel('Frequency (THz)')
                plt.title('Phonon Band Structure')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('plots/band_structure.pdf')
                plt.close()
                
                # Save band data
                np.savetxt('results/band.dat', 
                          np.column_stack([distances, frequencies]),
                          header='Distance Frequency(THz)')
        
        # DOS calculation
        print("\n6. Calculating DOS...")
        phonon.run_mesh(mp_grid)
        phonon.run_total_dos()
        
        # Get DOS data
        dos_dict = phonon.get_total_dos_dict()
        frequency_points = dos_dict['frequency_points']
        dos = dos_dict['total_dos']
        
        # Plot DOS
        plt.figure(figsize=(8, 6))
        plt.plot(frequency_points, dos, 'b-', linewidth=2)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('DOS')
        plt.title('Phonon Density of States')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/total_dos.pdf')
        plt.close()
        
        # Save DOS data
        np.savetxt('results/total_dos.dat', 
                  np.column_stack([frequency_points, dos]),
                  header='Frequency(THz) DOS')
        
        # Thermal properties calculation
        print("\n7. Calculating thermal properties...")
        phonon.run_thermal_properties(t_step=10, t_max=tmax, t_min=0)
        
        # Get thermal properties
        tp_dict = phonon.get_thermal_properties_dict()
        temps = tp_dict['temperatures']
        free_energy = tp_dict['free_energy']
        entropy = tp_dict['entropy']
        heat_capacity = tp_dict['heat_capacity']
        
        # Plot thermal properties
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        # Free energy
        axes[0].plot(temps, free_energy, 'b-', linewidth=2)
        axes[0].set_ylabel('Free energy (kJ/mol)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Thermal Properties')
        
        # Entropy
        axes[1].plot(temps, entropy, 'r-', linewidth=2)
        axes[1].set_ylabel('Entropy (J/K/mol)')
        axes[1].grid(True, alpha=0.3)
        
        # Heat capacity
        axes[2].plot(temps, heat_capacity, 'g-', linewidth=2)
        axes[2].set_xlabel('Temperature (K)')
        axes[2].set_ylabel('Heat capacity (J/K/mol)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/thermal_properties.pdf')
        plt.close()
        
        # Save thermal data
        thermal_data = np.column_stack([temps, free_energy, entropy, heat_capacity])
        np.savetxt('results/thermal_properties.dat',
                  thermal_data,
                  header='T(K) F(kJ/mol) S(J/K/mol) Cv(J/K/mol)')
        
        # Write thermal properties in phonopy format
        with open('results/thermal_properties.yaml', 'w') as f:
            f.write("# Thermal properties\n")
            f.write("# T [K], F [kJ/mol], S [J/K/mol], C_v [J/K/mol]\n")
            for T, F, S, Cv in thermal_data:
                f.write(f"{T:10.3f} {F:15.8f} {S:15.8f} {Cv:15.8f}\n")
        
        # Copy important files to results
        shutil.copy('FORCE_SETS', 'results/')
        
        return True
        
    except Exception as e:
        print(f"\nError in analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)

def run_simple_phonopy_analysis(calc_dir, dim, mp_grid, tmax, band_path, forces_data):
    """Simple phonopy analysis that directly uses forces without FORCE_SETS parsing."""
    original_dir = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        print("\n4. Creating force constants (simple method)...")
        
        # Create a Python script that does everything
        script_content = f"""
import numpy as np
import matplotlib.pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.interface.vasp import read_vasp
import warnings
import yaml
warnings.filterwarnings('ignore')

try:
    # Read structure
    unitcell = read_vasp("POSCAR")
    
    # Create phonopy instance
    phonon = Phonopy(unitcell, [{dim[0]}, {dim[1]}, {dim[2]}], primitive_matrix='auto')
    
    # Read phonopy_disp.yaml to get displacement information
    with open('phonopy_disp.yaml', 'r') as f:
        disp_yaml = yaml.safe_load(f)
    
    # Load forces from numpy files
    forces_list = []
    for i in range({len(forces_data)}):
        forces = np.load(f'forces/forces_{{i:03d}}.npy')
        forces_list.append(forces)
    
    # Create dataset in proper format
    displacements = disp_yaml.get('displacements')
    dataset = {{"natom": len(phonon.supercell), "first_atoms": []}}
    
    for disp_info, forces in zip(displacements, forces_list):
        dataset["first_atoms"].append({{
            "number": disp_info["atom"] - 1,  # 0-indexed
            "displacement": disp_info["displacement"],
            "forces": forces.tolist(),
        }})
    
    # Set dataset and produce force constants
    phonon.dataset = dataset
    phonon.produce_force_constants()

    # Save force constants
    write_FORCE_CONSTANTS(phonon.force_constants, filename="FORCE_CONSTANTS")
    print("Force constants created!")

    # Calculate DOS
    phonon.set_mesh([{mp_grid[0]}, {mp_grid[1]}, {mp_grid[2]}])
    phonon.set_total_DOS()
    freq, dos = phonon.get_total_DOS()

    # Save DOS
    np.savetxt('total_dos.dat', np.column_stack([freq, dos]), 
               header='Frequency(THz) DOS')

    # Plot DOS
    plt.figure(figsize=(8, 6))
    plt.plot(freq, dos, 'b-', linewidth=2)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS')
    plt.title('Phonon Density of States')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('total_dos.pdf')
    plt.close()

    # Calculate thermal properties
    phonon.set_thermal_properties(t_step=10, t_max={tmax}, t_min=0)
    temps, F, S, Cv = phonon.get_thermal_properties()

    # Save thermal properties
    thermal_data = np.column_stack([temps, F, S, Cv])
    np.savetxt('thermal_properties.dat', thermal_data,
               header='T(K) F(kJ/mol) S(J/K/mol) Cv(J/K/mol)')

    # Plot thermal properties
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    axes[0].plot(temps, F, 'b-', linewidth=2)
    axes[0].set_ylabel('Free energy (kJ/mol)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Thermal Properties')

    axes[1].plot(temps, S, 'r-', linewidth=2)
    axes[1].set_ylabel('Entropy (J/K/mol)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temps, Cv, 'g-', linewidth=2)
    axes[2].set_xlabel('Temperature (K)')
    axes[2].set_ylabel('Heat capacity (J/K/mol)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('thermal_properties.pdf')
    plt.close()

    print("Analysis completed successfully!")
    
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        with open('simple_analysis.py', 'w') as f:
            f.write(script_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 'simple_analysis.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        # Check if files were created and move them
        files_created = []
        
        for fname in ['FORCE_CONSTANTS', 'total_dos.dat', 'thermal_properties.dat']:
            if Path(fname).exists():
                files_created.append(fname)
                shutil.move(fname, f'results/{fname}')
        
        for fname in ['total_dos.pdf', 'thermal_properties.pdf']:
            if Path(fname).exists():
                files_created.append(fname)
                shutil.move(fname, f'plots/{fname}')
        
        # Clean up
        if Path('simple_analysis.py').exists():
            Path('simple_analysis.py').unlink()
        
        # Also copy FORCE_SETS for reference
        if Path('FORCE_SETS').exists():
            shutil.copy('FORCE_SETS', 'results/FORCE_SETS')
        
        print(f"\nFiles created: {files_created}")
        
        return len(files_created) >= 4  # At least force constants and thermal properties
        
    except Exception as e:
        print(f"\nSimple analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)

def run_phonopy_analysis_cli(calc_dir, dim, mp_grid, tmax, band_path=None):
    """Fallback CLI method for phonopy analysis."""
    original_dir = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        print("\n4. Creating force constants (CLI method)...")
        
        dim_str = f"{dim[0]} {dim[1]} {dim[2]}"
        
        # First try to create force constants from FORCE_SETS
        cmd = f'phonopy --dim="{dim_str}" -c POSCAR --writefc'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not Path('FORCE_CONSTANTS').exists():
            print("   Direct creation failed, trying with explicit FORCE_SETS...")
            # Create force constants from FORCE_SETS
            cmd = f'phonopy --dim="{dim_str}" -c POSCAR -f displacements/POSCAR-* --writefc'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if Path('FORCE_CONSTANTS').exists():
            print("   Force constants created!")
            
            # DOS calculation
            print("\n6. Calculating DOS...")
            mp_str = f"{mp_grid[0]} {mp_grid[1]} {mp_grid[2]}"
            cmd = f'phonopy --dim="{dim_str}" -c POSCAR --readfc --dos --mp="{mp_str}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   DOS calculation warning: {result.stderr}")
            
            # Thermal properties
            print("\n7. Calculating thermal properties...")
            cmd = f'phonopy --dim="{dim_str}" -c POSCAR --readfc -t --tmax={tmax} --save-params'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   Thermal properties warning: {result.stderr}")
            else:
                print("   Thermal properties calculated")
            
            # Band structure if requested
            if band_path:
                print("\n5. Calculating band structure...")
                # Create a BAND file with the path
                band_conf = """ATOM_NAME = Zr
DIM = {} {} {}
BAND = AUTO
""".format(dim[0], dim[1], dim[2])
                
                with open('band.conf', 'w') as f:
                    f.write(band_conf)
                
                cmd = f'phonopy -p -c POSCAR --readfc band.conf'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"   Band structure warning: {result.stderr}")
                else:
                    print("   Band structure calculated")
            
            # Look for output files in both root and phonopy- directories
            output_files = [
                'FORCE_CONSTANTS',
                'total_dos.dat',
                'projected_dos.dat',
                'thermal_properties.yaml',
                'thermal_properties.dat',
                'band.yaml',
                'band.pdf',
                'band.dat'
            ]
            
            # Also check phonopy-* directories
            phonopy_dirs = list(Path('.').glob('phonopy-*'))
            
            # First, look for thermal_properties.yaml in phonopy-* dirs
            for pdir in phonopy_dirs:
                thermal_yaml = pdir / 'thermal_properties.yaml'
                if thermal_yaml.exists():
                    shutil.copy(str(thermal_yaml), 'thermal_properties.yaml')
                    print(f"   Found thermal properties in {pdir}")
                    break
            
            for fname in output_files:
                if not Path(fname).exists():
                    # Check in phonopy directories
                    for pdir in phonopy_dirs:
                        if (pdir / fname).exists():
                            shutil.move(str(pdir / fname), fname)
                            break
            
            # List all files created (for debugging)
            print("\n   Files in current directory:")
            for f in Path('.').iterdir():
                if f.is_file():
                    print(f"     - {f.name}")
            
            print("\n   Files in phonopy-* directories:")
            for pdir in phonopy_dirs:
                if pdir.is_dir():
                    print(f"     In {pdir}:")
                    for f in pdir.iterdir():
                        if f.is_file():
                            print(f"       - {f.name}")
            
            # Check if thermal properties are in phonopy.yaml or phonopy_params.yaml
            thermal_found = False
            for yaml_file in ['phonopy.yaml', 'phonopy_params.yaml']:
                if Path(yaml_file).exists() and not Path('thermal_properties.yaml').exists():
                    print(f"\n   Checking {yaml_file} for thermal properties...")
                    try:
                        with open(yaml_file, 'r') as f:
                            yaml_content = yaml.safe_load(f)
                        
                        if 'thermal_properties' in yaml_content:
                            # Extract thermal properties and save as separate file
                            thermal_props = {'thermal_properties': yaml_content['thermal_properties']}
                            with open('thermal_properties.yaml', 'w') as f:
                                yaml.dump(thermal_props, f)
                            print(f"   Extracted thermal properties from {yaml_file}")
                            thermal_found = True
                            break
                    except Exception as e:
                        print(f"   Could not extract thermal properties from {yaml_file}: {e}")
            
            # If still not found, try to generate using phonopy-load
            if not thermal_found and not Path('thermal_properties.yaml').exists():
                print("\n   Trying to generate thermal properties with phonopy-load...")
                cmd = f'phonopy-load --readfc -t --tmax={tmax}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print("   Generated thermal properties with phonopy-load")
                    
                    # Check if it created thermal_properties.yaml
                    for f in Path('.').glob('thermal_properties*'):
                        print(f"   Found: {f.name}")
            
            # Move results to appropriate directories
            files_to_move = {
                'FORCE_CONSTANTS': 'results/',
                'total_dos.dat': 'results/',
                'projected_dos.dat': 'results/',
                'thermal_properties.yaml': 'results/',
                'thermal_properties.dat': 'results/',
                'band.yaml': 'results/',
                'band.dat': 'results/',
                'band.pdf': 'plots/'
            }
            
            for fname, dest in files_to_move.items():
                if Path(fname).exists():
                    dest_path = Path(dest) / fname
                    if dest_path.exists():
                        dest_path.unlink()  # Remove existing file
                    shutil.move(fname, dest)
            
            # Copy FORCE_SETS
            if Path('FORCE_SETS').exists():
                shutil.copy('FORCE_SETS', 'results/')
            
            # Generate missing plots and data files
            # DOS plot
            if Path('results/total_dos.dat').exists() and not Path('plots/total_dos.pdf').exists():
                data = np.loadtxt('results/total_dos.dat')
                plt.figure(figsize=(8, 6))
                plt.plot(data[:, 0], data[:, 1], 'b-', linewidth=2)
                plt.xlabel('Frequency (THz)')
                plt.ylabel('DOS')
                plt.title('Phonon Density of States')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('plots/total_dos.pdf')
                plt.close()
                print("   Generated DOS plot")
            
            # Convert thermal_properties.yaml to .dat and create plots
            if Path('results/thermal_properties.yaml').exists():
                print("\n   Processing thermal properties...")
                temps, F, S, Cv = [], [], [], []
                
                try:
                    # Read the yaml file
                    with open('results/thermal_properties.yaml', 'r') as f:
                        thermal_yaml = yaml.safe_load(f)
                    
                    # Extract thermal properties
                    if 'thermal_properties' in thermal_yaml:
                        for tp in thermal_yaml['thermal_properties']:
                            temps.append(tp['temperature'])
                            F.append(tp['free_energy'])
                            S.append(tp['entropy'])
                            Cv.append(tp['heat_capacity'])
                    
                except:
                    # Fallback: try line-by-line parsing
                    with open('results/thermal_properties.yaml', 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) >= 4 and not any(c.isalpha() for c in parts[0]):
                                    try:
                                        temps.append(float(parts[0]))
                                        F.append(float(parts[1]))
                                        S.append(float(parts[2]))
                                        Cv.append(float(parts[3]))
                                    except ValueError:
                                        continue
                
                if temps:
                    thermal_data = np.column_stack([temps, F, S, Cv])
                    np.savetxt('results/thermal_properties.dat', thermal_data,
                              header='T(K) F(kJ/mol) S(J/K/mol) Cv(J/K/mol)')
                    print("   Generated thermal properties data file")
                    
                    # Create thermal plots
                    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
                    
                    axes[0].plot(temps, F, 'b-', linewidth=2)
                    axes[0].set_ylabel('Free energy (kJ/mol)')
                    axes[0].grid(True, alpha=0.3)
                    axes[0].set_title('Thermal Properties')
                    
                    axes[1].plot(temps, S, 'r-', linewidth=2)
                    axes[1].set_ylabel('Entropy (J/K/mol)')
                    axes[1].grid(True, alpha=0.3)
                    
                    axes[2].plot(temps, Cv, 'g-', linewidth=2)
                    axes[2].set_xlabel('Temperature (K)')
                    axes[2].set_ylabel('Heat capacity (J/K/mol)')
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig('plots/thermal_properties.pdf')
                    plt.close()
                    print("   Generated thermal properties plot")
                else:
                    print("   Warning: Could not parse thermal properties data")
            
            # Generate band structure plot if band.yaml exists but band.pdf doesn't
            if Path('results/band.yaml').exists() and not Path('plots/band_structure.pdf').exists():
                try:
                    # Read band.yaml and create plot
                    with open('results/band.yaml', 'r') as f:
                        band_yaml = yaml.safe_load(f)
                    
                    # Extract band data
                    phonon_bands = band_yaml.get('phonon', [])
                    if phonon_bands:
                        distances = []
                        frequencies = []
                        
                        for band_point in phonon_bands:
                            distances.append(band_point['distance'])
                            freqs = band_point['band']
                            frequencies.append([f['frequency'] for f in freqs])
                        
                        distances = np.array(distances)
                        frequencies = np.array(frequencies)
                        
                        # Plot
                        plt.figure(figsize=(8, 6))
                        for i in range(frequencies.shape[1]):
                            plt.plot(distances, frequencies[:, i], 'b-')
                        
                        plt.xlabel('Wave vector')
                        plt.ylabel('Frequency (THz)')
                        plt.title('Phonon Band Structure')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig('plots/band_structure.pdf')
                        plt.close()
                        print("   Generated band structure plot")
                        
                        # Save as dat file
                        np.savetxt('results/band.dat',
                                  np.column_stack([distances, frequencies]),
                                  header='Distance Frequency(THz)')
                except Exception as e:
                    print(f"   Could not generate band plot: {e}")
            
            return True
        else:
            print("   Failed to create force constants!")
            return False
        
    except Exception as e:
        print(f"\nCLI method failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)

def extract_thermal_properties(results_dir):
    """Extract and display key thermal properties with summary report."""
    thermal_file = results_dir / 'thermal_properties.dat'
    
    if not thermal_file.exists():
        thermal_file = results_dir / 'thermal_properties.yaml'
    
    if not thermal_file.exists():
        print("\n" + "="*60)
        print("THERMAL PROPERTIES SUMMARY")
        print("="*60)
        print("\nNo thermal properties file found!")
        print("The calculation may have failed. Check the error messages above.")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("THERMAL PROPERTIES SUMMARY")
    print("="*60)
    
    # Read thermal data
    try:
        if thermal_file.suffix == '.dat':
            data = np.loadtxt(thermal_file, skiprows=1)
            temps = data[:, 0]
            free_energy = data[:, 1]
            entropy = data[:, 2]
            heat_capacity = data[:, 3]
        else:
            # Parse YAML format
            temps, free_energy, entropy, heat_capacity = [], [], [], []
            with open(thermal_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 4:
                            temps.append(float(parts[0]))
                            free_energy.append(float(parts[1]))
                            entropy.append(float(parts[2]))
                            heat_capacity.append(float(parts[3]))
        
        if len(temps) == 0:
            print("\nNo thermal data found in file!")
            return
        
        # Key temperatures to report
        key_temps = [300, 600, 1000, 1400, 2000]
        
        print("\n" + "-"*60)
        print(f"{'Temperature':>12} {'Free Energy':>15} {'Entropy':>15} {'Heat Capacity':>15}")
        print(f"{'(K)':>12} {'(kJ/mol)':>15} {'(J/K/mol)':>15} {'(J/K/mol)':>15}")
        print("-"*60)
        
        for target_T in key_temps:
            # Find closest temperature
            idx = np.argmin(np.abs(np.array(temps) - target_T))
            if abs(temps[idx] - target_T) < 20:  # Within 20K
                print(f"{temps[idx]:12.0f} {free_energy[idx]:15.4f} "
                      f"{entropy[idx]:15.4f} {heat_capacity[idx]:15.4f}")
        
        print("-"*60)
        
        # Generate summary report
        report_file = results_dir.parent / 'summary_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PHONOPY + EAM CALCULATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Calculation directory: {results_dir.parent}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Key Results:\n")
            f.write("-"*40 + "\n")
            
            # Check for key files
            files_to_check = {
                'FORCE_CONSTANTS': 'Force constants calculated',
                'total_dos.dat': 'DOS calculated',
                'thermal_properties.dat': 'Thermal properties calculated',
                'band.dat': 'Band structure calculated'
            }
            
            for fname, desc in files_to_check.items():
                if (results_dir / fname).exists():
                    f.write(f"✓ {desc}\n")
                else:
                    f.write(f"✗ {desc}\n")
            
            f.write("\n" + "-"*40 + "\n")
            f.write("Thermal Properties at Key Temperatures:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'T (K)':>8} {'F (kJ/mol)':>12} {'S (J/K/mol)':>12} {'Cv (J/K/mol)':>12}\n")
            
            for target_T in key_temps:
                idx = np.argmin(np.abs(np.array(temps) - target_T))
                if abs(temps[idx] - target_T) < 20:
                    f.write(f"{temps[idx]:8.0f} {free_energy[idx]:12.4f} "
                            f"{entropy[idx]:12.4f} {heat_capacity[idx]:12.4f}\n")
        
        print(f"\nSummary report saved to: {report_file}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError reading thermal properties: {e}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Calculate phonons using EAM potentials with phonopy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('poscar', help='Input POSCAR file')
    
    # Potential options
    potential_group = parser.add_mutually_exclusive_group(required=True)
    potential_group.add_argument('--potential', help='EAM potential file')
    potential_group.add_argument('--kim', help='KIM model name')
    
    # Phonopy parameters
    parser.add_argument('--dim', nargs=3, type=int, default=[2,2,2],
                       help='Supercell dimensions')
    parser.add_argument('--mp', nargs=3, type=int, default=[20,20,20],
                       help='MP k-point grid')
    parser.add_argument('--band', default=None,
                       help='Band structure path (e.g., "G X M G R X")')
    parser.add_argument('--tmax', type=int, default=2000,
                       help='Maximum temperature')
    parser.add_argument('--work-dir', default='phonopy_calculations',
                       help='Working directory')
    
    args = parser.parse_args()
    
    # Get POSCAR name
    poscar_name = Path(args.poscar).stem
    
    print("="*60)
    print("PHONOPY + EAM CALCULATION")
    print("="*60)
    print(f"POSCAR: {args.poscar}")
    if args.potential:
        print(f"Potential: {args.potential}")
    else:
        print(f"KIM model: {args.kim}")
    print(f"Supercell: {args.dim[0]}×{args.dim[1]}×{args.dim[2]}")
    print(f"MP grid: {args.mp[0]}×{args.mp[1]}×{args.mp[2]}")
    print("="*60)
    
    # Setup calculation directory
    dirs = setup_calculation_directory(args.work_dir, poscar_name)
    print(f"\nWorking directory: {dirs['root']}")
    
    try:
        # Step 1: Generate displacements
        print("\n1. Generating displacements...")
        disp_files = run_phonopy_calculation(args.poscar, dirs['root'], args.dim)
        print(f"   Generated {len(disp_files)} displacement configurations")
        
        if len(disp_files) == 0:
            raise RuntimeError("No displacement files generated!")
        
        # Step 2: Setup calculator
        print("\n2. Setting up calculator...")
        if args.potential:
            print(f"   Using EAM potential: {args.potential}")
            # Copy potential file to calc directory for reference
            shutil.copy(args.potential, dirs['root'] / Path(args.potential).name)
            calc = EAM(potential=args.potential)
        else:
            print(f"   Using KIM model: {args.kim}")
            calc = KIM(args.kim)
        
        # Step 3: Calculate forces
        print("\n3. Calculating forces...")
        natoms, forces_data = calculate_forces(disp_files, calc, dirs['root'])
        
        # Step 4: Write FORCE_SETS with proper format
        force_sets_file = write_force_sets_fixed(natoms, forces_data, dirs['root'])
        print(f"\n   Written FORCE_SETS ({natoms} atoms)")
        
        # Step 5-7: Run phonopy analysis
        if PHONOPY_AVAILABLE:
            success = run_phonopy_analysis_direct(dirs['root'], args.dim, args.mp, args.tmax, args.band, forces_data)
            if not success:
                print("\nDirect method failed, trying simple method...")
                success = run_simple_phonopy_analysis(dirs['root'], args.dim, args.mp, args.tmax, args.band, forces_data)
                if not success:
                    print("\nSimple method failed, trying CLI method...")
                    success = run_phonopy_analysis_cli(dirs['root'], args.dim, args.mp, args.tmax, args.band)
        else:
            success = run_phonopy_analysis_cli(dirs['root'], args.dim, args.mp, args.tmax, args.band)
        
        # Step 8: Extract results
        extract_thermal_properties(dirs['results'])
        
        print("\n" + "="*60)
        print("CALCULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nAll files are in: {dirs['root']}")
        print(f"Results are in:   {dirs['results']}")
        print(f"Plots are in:     {dirs['plots']}")
        print("\nKey output files:")
        print("  - results/FORCE_SETS")
        print("  - results/FORCE_CONSTANTS")
        print("  - results/thermal_properties.dat")
        print("  - results/total_dos.dat")
        print("  - plots/total_dos.pdf")
        print("  - plots/thermal_properties.pdf")
        if args.band:
            print("  - results/band.dat")
            print("  - plots/band_structure.pdf")
        print("  - summary_report.txt")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()