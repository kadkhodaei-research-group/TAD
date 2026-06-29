import logging
import numpy as np
from typing import Dict, List, Any, Optional
import os

class VerboseLogger:
    """Handles verbose logging and table output for Walker class.
    
    This class encapsulates all the verbose output formatting logic
    that was previously in the Walker class, following the single
    responsibility principle.
    """
    
    def __init__(self, temperature: float, logger: logging.Logger = None):
        """Initialize the verbose logger.
        
        Args:
            temperature: System temperature for display
            logger: Optional logger instance
        """
        self.temperature = temperature
        self.verbose_history = []
        self.logger = logger or logging.getLogger(__name__)
        self.gp2_history = []  # Add this for GP2-only mode

        
    def set_history(self, history: List[Dict]):
        """Set the verbose history (used for restoration).
        
        Args:
            history: List of historical row data
        """
        self.verbose_history = history
        print(f"Restored {len(history)} previous steps to verbose history")
        
    def add_row(self, step_idx: Any, gp1_e: float, gp1_fn: float, gp1_curv: float,
                gp2_e: float, gp2_fn: float, gp2_curv: float, energy_diff: float):
        """Add a row to the verbose history.
        
        Args:
            step_idx: Step identifier (int or string like 'Init-0')
            gp1_e: GP1 energy prediction
            gp1_fn: GP1 force norm
            gp1_curv: GP1 curvature
            gp2_e: GP2 energy prediction
            gp2_fn: GP2 force norm
            gp2_curv: GP2 curvature
            energy_diff: Absolute difference between GP1 and GP2 energies
        """
        row_data = {
            'step': step_idx,
            'gp1_e': gp1_e,
            'gp1_fn': gp1_fn,
            'gp1_curv': gp1_curv,
            'gp2_e': gp2_e,
            'gp2_fn': gp2_fn,
            'gp2_curv': gp2_curv,
            'energy_diff': energy_diff
        }
        self.verbose_history.append(row_data)
        
    def print_table(self, dimer_dirs: int = 0, thermal_batches: int = 0):
        """Print the full verbose table with all rows.
        
        Args:
            dimer_dirs: Number of dimer directories
            thermal_batches: Number of thermal batches
        """
        # Print header
        print("\n" + "="*90)
        print(f"{'TEMPERATURE-DEPENDENT SADDLE POINT FINDER (TD-SPF)':^90}")
        print(f"{'Temperature: ' + str(self.temperature) + ' K':^90}")
        print("="*90)
        
        print(f"{'Step':^7} | {'GP1 E':>9} {'|F|':>8} {'λ_min':>9} | "
              f"{'GP2 E':>9} {'|F|':>8} {'λ_min':>9} | {'ΔE(GP1-GP2)':>10}")
        print(f"{'':^7} | {'(eV)':>9} {'(eV/Å)':>8} {'':>9} | "
              f"{'(eV)':>9} {'(eV/Å)':>8} {'':>9} | {'(eV)':>10}")
        print("-"*90)

        # Debug: Check how many rows we have
        if len(self.verbose_history) == 0:
            print("WARNING: No verbose history entries found!")
        
        # Print all historical rows
        for row in self.verbose_history:
            print(f"{row['step']:^7} | "
                  f"{self._format_energy(row['gp1_e'])} "
                  f"{self._format_force(row['gp1_fn'])} "
                  f"{self._format_curvature(row['gp1_curv'])} | "
                  f"{self._format_energy(row['gp2_e'])} "
                  f"{self._format_force(row['gp2_fn'])} "
                  f"{self._format_curvature(row['gp2_curv'])} | "
                  f"{self._format_diff(row['energy_diff'])}")
            
        # Summary
        print(f"\n[Total Dimer Directories: {dimer_dirs} | Total Thermal Batches: {thermal_batches}]")
        
        # Log to file as well
        self._log_table(dimer_dirs, thermal_batches)
        
    def _log_table(self, dimer_dirs: int, thermal_batches: int):
        """Log the table to the logger."""
        table_lines = []
        
        # Header
        table_lines.append("\n" + "="*90)
        table_lines.append(f"{'TEMPERATURE-DEPENDENT SADDLE POINT FINDER (TD-SPF)':^90}")
        table_lines.append(f"{'Temperature: ' + str(self.temperature) + ' K':^90}")
        table_lines.append("="*90)
        table_lines.append(f"{'Step':^7} | {'GP1 E':>9} {'|F|':>8} {'λ_min':>9} | "
                          f"{'GP2 E':>9} {'|F|':>8} {'λ_min':>9} | {'ΔE(GP1-GP2)':>10}")
        table_lines.append(f"{'':^7} | {'(eV)':>9} {'(eV/Å)':>8} {'':>9} | "
                          f"{'(eV)':>9} {'(eV/Å)':>8} {'':>9} | {'(eV)':>10}")
        table_lines.append("-"*90)
        
        # Data rows
        for row in self.verbose_history:
            table_lines.append(f"{row['step']:^7} | "
                              f"{self._format_energy(row['gp1_e'])} "
                              f"{self._format_force(row['gp1_fn'])} "
                              f"{self._format_curvature(row['gp1_curv'])} | "
                              f"{self._format_energy(row['gp2_e'])} "
                              f"{self._format_force(row['gp2_fn'])} "
                              f"{self._format_curvature(row['gp2_curv'])} | "
                              f"{self._format_diff(row['energy_diff'])}")
        
        # Summary
        table_lines.append(f"\n[Total Dimer Directories: {dimer_dirs} | Total Thermal Batches: {thermal_batches}]")
        
        # Log the entire table
        table_str = '\n'.join(table_lines)
        self.logger.info(table_str)
        
    def print_convergence_message(self, converged: bool, max_steps_reached: bool = False):
        """Print convergence or completion message.
        
        Args:
            converged: Whether the optimization converged
            max_steps_reached: Whether maximum steps were reached
        """
        print("-"*130)
        if converged:
            print(f"{'CONVERGED':^130}")
        elif max_steps_reached:
            print(f"{'MAXIMUM STEPS REACHED':^130}")
        print("="*130)
        
    def log_message(self, message: str):
        """Log and print a message.
        
        Args:
            message: Message to log and print
        """
        print(message)
        self.logger.info(message)
        
    # Formatting helper methods
    def _format_curvature(self, curv: float) -> str:
        """Format curvature values with fixed width."""
        if np.isnan(curv):
            return "      NaN"
        elif abs(curv) > 1e6:
            return f"{curv:9.2e}"
        elif abs(curv) > 1000:
            return f"{curv:9.1f}"
        elif abs(curv) < 0.001 and curv != 0:
            return f"{curv:9.2e}"
        else:
            return f"{curv:9.4f}"
    
    def _format_energy(self, e: float) -> str:
        """Format energy values with fixed width."""
        if np.isnan(e):
            return "      nan"
        elif abs(e) > 1e6:
            return f"{e:9.2e}"
        else:
            return f"{e:9.4f}"
    
    def _format_force(self, f: float) -> str:
        """Format force values with fixed width."""
        if np.isnan(f):
            return "     nan"
        elif f > 1e4:
            return f"{f:8.2e}"
        elif f < 0.0001:
            return f"{f:8.2e}"
        else:
            return f"{f:8.4f}"
    
    def _format_diff(self, d: float) -> str:
        """Format energy difference values."""
        if np.isnan(d):
            return "       nan"
        elif d > 1e4:
            return f"{d:10.2e}"
        elif d < 0.0001:
            return f"{d:10.2e}"
        else:
            return f"{d:10.4f}"
    
    def add_gp2_row(self, step, gp2_energy, gp2_force_norm, gp2_curvature):
        """Add a row for GP2-only walker."""
        if not hasattr(self, 'gp2_history'):
            self.gp2_history = []
        
        self.gp2_history.append({
            'step': step,
            'gp2_energy': gp2_energy,
            'gp2_force_norm': gp2_force_norm,
            'gp2_curvature': gp2_curvature,
        })

    def print_gp2_table(self, show_all=False):
        """Print verbose table for GP2-only walker."""
        if not hasattr(self, 'gp2_history') or not self.gp2_history:
            return
        
        # Clear screen/previous output for cleaner display (optional)
        # print("\033[2J\033[H")  # Uncomment to clear screen
        
        print("\n" + "="*70)
        print(f"{'Step':^8} | {'GP2 Energy (eV)':^15} | {'GP2 |Force| (eV/Å)':^18} | {'GP2 Min λ':^12}")
        print("-"*70)
        
        # Determine which rows to show
        if show_all or len(self.gp2_history) <= 20:
            rows_to_show = self.gp2_history
        else:
            # Show first 5 and last 15 rows
            rows_to_show = self.gp2_history[:5] + [{'step': '...', 'gp2_energy': '...', 'gp2_force_norm': '...', 'gp2_curvature': '...'}] + self.gp2_history[-15:]
        
        for row in rows_to_show:
            if row['step'] == '...':
                print(f"{'...':^8} | {'...':^15} | {'...':^18} | {'...':^12}")
            else:
                print(f"{row['step']:^8d} | "
                    f"{row['gp2_energy']:^15.6f} | "
                    f"{row['gp2_force_norm']:^18.6f} | "
                    f"{row['gp2_curvature']:^12.6f}")
        
        print("="*70)
        
        # Print current statistics
        if self.gp2_history:
            latest = self.gp2_history[-1]
            min_force = min(row['gp2_force_norm'] for row in self.gp2_history)
            print(f"\nCurrent step: {latest['step']}")
            print(f"Minimum |Force| achieved: {min_force:.6f} eV/Å")