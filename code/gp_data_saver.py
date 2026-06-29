import pickle
import numpy as np
import torch
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from output_manager import get_output_path

class GPDataLogger:
    """Logger for GP training and validation data."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize the GP data logger.
        
        Args:
            save_dir: Directory to save diagnostic data (default: uses output manager)
        """
        if save_dir is None:
            self.save_dir = get_output_path('gp_diagnostics')
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data storage
        self.gp1_data = {
            'training_history': [],
            'validation_history': [],
            'hyperparameters': [],
            'predictions': [],
            'training_data': [],
            'model_info': {}
        }
        
        self.gp2_data = {
            'training_history': [],
            'validation_history': [],
            'hyperparameters': [],
            'predictions': [],
            'training_data': [],
            'cross_validation': [],
            'model_info': {}
        }
        
        self.iteration_counter = 0
        
    def log_gp_training(self, 
                       model_name: str,
                       iteration: int,
                       training_data: List[np.ndarray],
                       hyperparameters: Dict[str, Any],
                       training_metrics: Dict[str, float],
                       loss_history: List[float],
                       model_state: Optional[Dict] = None):
        """Log GP training data.
        
        Args:
            model_name: 'GP1' or 'GP2'
            iteration: Current iteration number
            training_data: [positions, energies, forces]
            hyperparameters: Initial hyperparameters
            training_metrics: Training error metrics
            loss_history: Loss values during training
            model_state: Optional model state dict
        """
        data_dict = self.gp1_data if model_name == 'GP1' else self.gp2_data
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'n_training_points': len(training_data[0]),
            'hyperparameters': hyperparameters.copy(),
            'training_metrics': training_metrics.copy(),
            'loss_history': loss_history.copy() if loss_history else [],
            'final_loss': loss_history[-1] if loss_history else None,
            'n_epochs': len(loss_history) if loss_history else 0
        }
        
        # Store training data snapshot
        training_snapshot = {
            'positions_shape': training_data[0].shape,
            'energies_shape': training_data[1].shape,
            'forces_shape': training_data[2].shape,
            'energy_stats': {
                'mean': np.mean(training_data[1]),
                'std': np.std(training_data[1]),
                'min': np.min(training_data[1]),
                'max': np.max(training_data[1])
            },
            'force_stats': {
                'mean_magnitude': np.mean(np.linalg.norm(training_data[2], axis=1)),
                'std_magnitude': np.std(np.linalg.norm(training_data[2], axis=1)),
                'max_magnitude': np.max(np.linalg.norm(training_data[2], axis=1))
            }
        }
        entry['training_data_stats'] = training_snapshot
        
        data_dict['training_history'].append(entry)
        
        # Save immediately
        self._save_current_data()
        
    def log_gp_validation(self,
                         model_name: str,
                         iteration: int,
                         validation_data: Dict[str, Any]):
        """Log GP validation/cross-validation results.
        
        Args:
            model_name: 'GP1' or 'GP2'
            iteration: Current iteration number
            validation_data: Dictionary containing validation results
        """
        data_dict = self.gp1_data if model_name == 'GP1' else self.gp2_data
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **validation_data
        }
        
        if model_name == 'GP2' and 'cross_validation' in validation_data:
            data_dict['cross_validation'].append(entry)
        else:
            data_dict['validation_history'].append(entry)
            
        # Save immediately
        self._save_current_data()
        
    def log_gp_prediction(self,
                         model_name: str,
                         iteration: int,
                         position: np.ndarray,
                         prediction: Dict[str, Any],
                         true_values: Optional[Dict[str, Any]] = None):
        """Log a GP prediction for analysis.
        
        Args:
            model_name: 'GP1' or 'GP2'
            iteration: Current iteration number
            position: Input position
            prediction: Dictionary with predicted values
            true_values: Optional true values for comparison
        """
        data_dict = self.gp1_data if model_name == 'GP1' else self.gp2_data
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'position_shape': position.shape,
            'prediction': prediction.copy(),
            'true_values': true_values.copy() if true_values else None
        }
        
        if true_values:
            # Calculate errors
            errors = {}
            if 'energy' in prediction and 'energy' in true_values:
                errors['energy_error'] = abs(prediction['energy'] - true_values['energy'])
            if 'forces' in prediction and 'forces' in true_values:
                errors['force_error'] = np.linalg.norm(
                    np.array(prediction['forces']) - np.array(true_values['forces'])
                )
            entry['errors'] = errors
            
        data_dict['predictions'].append(entry)
        
        # Keep only last 100 predictions to avoid huge files
        if len(data_dict['predictions']) > 100:
            data_dict['predictions'] = data_dict['predictions'][-100:]
            
        # Save immediately
        self._save_current_data()
        
    def log_model_info(self,
                      model_name: str,
                      model_type: str,
                      atomic_info: Dict[str, Any],
                      additional_info: Optional[Dict] = None):
        """Log model configuration information.
        
        Args:
            model_name: 'GP1' or 'GP2'
            model_type: Type of GP model being used
            atomic_info: Atomic structure information
            additional_info: Any additional model information
        """
        data_dict = self.gp1_data if model_name == 'GP1' else self.gp2_data
        
        data_dict['model_info'] = {
            'model_type': model_type,
            'atomic_info': atomic_info.copy(),
            'timestamp': datetime.now().isoformat(),
            **(additional_info or {})
        }
        
        # Save immediately
        self._save_current_data()
        
    def _save_current_data(self):
        """Save current data to pickle files."""
        # Save GP1 data
        gp1_file = os.path.join(self.save_dir, 'gp1_diagnostics.pkl')
        with open(gp1_file, 'wb') as f:
            pickle.dump(self.gp1_data, f)
            
        # Save GP2 data
        gp2_file = os.path.join(self.save_dir, 'gp2_diagnostics.pkl')
        with open(gp2_file, 'wb') as f:
            pickle.dump(self.gp2_data, f)
            
        # Save a summary file
        summary = {
            'last_updated': datetime.now().isoformat(),
            'gp1_training_count': len(self.gp1_data['training_history']),
            'gp2_training_count': len(self.gp2_data['training_history']),
            'gp2_cv_count': len(self.gp2_data['cross_validation'])
        }
        summary_file = os.path.join(self.save_dir, 'gp_diagnostics_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
            
    def create_diagnostic_report(self, output_file: Optional[str] = None):
        """Create a text report of GP diagnostics.
        
        Args:
            output_file: Optional path to save the report
        """
        if output_file is None:
            output_file = os.path.join(self.save_dir, 'gp_diagnostic_report.txt')
            
        with open(output_file, 'w') as f:
            f.write("GP DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # GP1 Summary
            f.write("GP1 SUMMARY\n")
            f.write("-" * 40 + "\n")
            if self.gp1_data['training_history']:
                last_gp1 = self.gp1_data['training_history'][-1]
                f.write(f"Total training sessions: {len(self.gp1_data['training_history'])}\n")
                f.write(f"Last training:\n")
                f.write(f"  - Iteration: {last_gp1['iteration']}\n")
                f.write(f"  - Training points: {last_gp1['n_training_points']}\n")
                f.write(f"  - Final loss: {last_gp1['final_loss']:.6f}\n")
                if 'training_metrics' in last_gp1:
                    f.write(f"  - Energy MAE: {last_gp1['training_metrics'].get('energy_mae', 'N/A')}\n")
                    f.write(f"  - Force MAE: {last_gp1['training_metrics'].get('force_mae', 'N/A')}\n")
            f.write("\n")
            
            # GP2 Summary
            f.write("GP2 SUMMARY\n")
            f.write("-" * 40 + "\n")
            if self.gp2_data['training_history']:
                last_gp2 = self.gp2_data['training_history'][-1]
                f.write(f"Total training sessions: {len(self.gp2_data['training_history'])}\n")
                f.write(f"Last training:\n")
                f.write(f"  - Iteration: {last_gp2['iteration']}\n")
                f.write(f"  - Training points: {last_gp2['n_training_points']}\n")
                f.write(f"  - Final loss: {last_gp2['final_loss']:.6f}\n")
                if 'training_metrics' in last_gp2:
                    f.write(f"  - Energy MAE: {last_gp2['training_metrics'].get('energy_mae', 'N/A')}\n")
                    f.write(f"  - Force MAE: {last_gp2['training_metrics'].get('force_mae', 'N/A')}\n")
            
            # Cross-validation results
            if self.gp2_data['cross_validation']:
                f.write(f"\nCross-validation results: {len(self.gp2_data['cross_validation'])}\n")
                last_cv = self.gp2_data['cross_validation'][-1]
                f.write(f"Last CV:\n")
                f.write(f"  - Energy MAE: {last_cv.get('energy_mae', 'N/A')}\n")
                f.write(f"  - Force MAE: {last_cv.get('force_mae', 'N/A')}\n")
                
        print(f"Diagnostic report saved to: {output_file}")

    def log_model_comparison(self,
                        comparison_name: str,
                        iteration: int,
                        comparison_data: Dict[str, Any]):
        """Log comparison between different models.
        
        Args:
            comparison_name: Name of the comparison (e.g., 'DualGP')
            iteration: Current iteration number
            comparison_data: Dictionary containing comparison results
        """
        # Store in GP2 data as it's the main model
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'comparison_name': comparison_name,
            **comparison_data
        }
        
        # Add to a new comparison history list
        if 'comparison_history' not in self.gp2_data:
            self.gp2_data['comparison_history'] = []
        
        self.gp2_data['comparison_history'].append(entry)
        
        # Save immediately
        self._save_current_data()

    def log_training_progress(self,
                            model_name: str,
                            iteration: int,
                            metrics: Dict[str, Any]):
        """Log training progress metrics.
        
        Args:
            model_name: Model name
            iteration: Current iteration
            metrics: Dictionary of metrics to log
        """
        data_dict = self.gp1_data if model_name == 'GP1' else self.gp2_data
        
        # Add to training progress
        if 'training_progress' not in data_dict:
            data_dict['training_progress'] = []
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        data_dict['training_progress'].append(entry)
        
        # Save immediately
        self._save_current_data()


# Global logger instance
gp_logger = None

def get_gp_logger() -> GPDataLogger:
    """Get or create the global GP data logger."""
    global gp_logger
    if gp_logger is None:
        gp_logger = GPDataLogger()
    return gp_logger