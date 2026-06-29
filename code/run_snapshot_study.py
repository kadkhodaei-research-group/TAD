#!/usr/bin/env python3
"""
NEW_v13: Comprehensive Snapshot Study
======================================

Testing the effect of thermal snapshot count on success rate and barrier accuracy.
Using the best configurations from NEW_v12 with varying snapshot numbers.

Study Design:
- 2 best configs from NEW_v12 (Fixed TR with TR=0.4,Relax=30 and TR=0.35,Relax=40)
- 4 snapshot values: 50, 100, 200, 300
- 50 repetitions per configuration
- Total: 2 configs × 4 snapshots × 50 reps = 400 runs
- Split into 16 parts of 25 runs each (to keep under 50 runs per part)

Part allocation:
Parts 1-2:   Fixed TR=0.4, Relax=30, Snapshots=50   (50 runs)
Parts 3-4:   Fixed TR=0.4, Relax=30, Snapshots=100  (50 runs)
Parts 5-6:   Fixed TR=0.4, Relax=30, Snapshots=200  (50 runs)
Parts 7-8:   Fixed TR=0.4, Relax=30, Snapshots=300  (50 runs)
Parts 9-10:  Fixed TR=0.35, Relax=40, Snapshots=50  (50 runs)
Parts 11-12: Fixed TR=0.35, Relax=40, Snapshots=100 (50 runs)
Parts 13-14: Fixed TR=0.35, Relax=40, Snapshots=200 (50 runs)
Parts 15-16: Fixed TR=0.35, Relax=40, Snapshots=300 (50 runs)
"""

import subprocess
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import argparse
import hashlib

class SnapshotStudy:
    def __init__(self):
        """Initialize with study configurations"""
        
        # Base command template
        self.base_command = [
            "python", "run_dual_gp_improved.py",
            "--poscar-file", "../inputs/POSCAR_Zr",
            "--force-constants-file", "../inputs/FORCE_CONSTANTS_Zr_EAM",
            "--execution-mode", "eam",
            "--eam-potential-file", "potentials/Cu-Zr_4.eam.fs",
            "--moving-indices", "214",
            "--orient-atom-direction", "214:0.57,0.57,-0.57",
            "--temperature", "1400",
            "--verbose",
            "--enable-oscillation-detection",
            "--validate-parameters",
            "--parallel-eam",
            "--max-dimer-steps", "50",
            "--relaxed-saddle-criteria", "0.001",
            "--gpu",
            "--enable-trust-region"
        ]
        
        # Study configurations
        self.configs = {
            # Best config from NEW_v12: TR=0.4, Relax=30
            'config1': {
                'name': 'Best_TR0.4_R30',
                'trust_region_initial': 0.4,
                'max_inner_iterations': 30
            },
            # Second best: TR=0.35, Relax=40  
            'config2': {
                'name': 'Good_TR0.35_R40',
                'trust_region_initial': 0.35,
                'max_inner_iterations': 40
            }
        }
        
        # Snapshot values to test
        self.snapshot_values = [50, 100, 200, 300]
        
        # Part configuration (16 parts total)
        self.part_configs = {
            1:  ('config1', 50,  0,  25),   # Config1, 50 snapshots, runs 0-24
            2:  ('config1', 50,  25, 50),   # Config1, 50 snapshots, runs 25-49
            3:  ('config1', 100, 0,  25),
            4:  ('config1', 100, 25, 50),
            5:  ('config1', 200, 0,  25),
            6:  ('config1', 200, 25, 50),
            7:  ('config1', 300, 0,  25),
            8:  ('config1', 300, 25, 50),
            9:  ('config2', 50,  0,  25),   # Config2, 50 snapshots, runs 0-24
            10: ('config2', 50,  25, 50),   # Config2, 50 snapshots, runs 25-49
            11: ('config2', 100, 0,  25),
            12: ('config2', 100, 25, 50),
            13: ('config2', 200, 0,  25),
            14: ('config2', 200, 25, 50),
            15: ('config2', 300, 0,  25),
            16: ('config2', 300, 25, 50)
        }
        
        self.output_base = Path("../outputs/snapshot_study")
    
    def get_part_info(self, part_num):
        """Get configuration for a specific part"""
        if part_num not in self.part_configs:
            raise ValueError(f"Invalid part number: {part_num}. Must be 1-16.")
        
        config_key, snapshots, start_idx, end_idx = self.part_configs[part_num]
        config = self.configs[config_key]
        
        return {
            'config': config,
            'config_key': config_key,
            'snapshots': snapshots,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_runs': end_idx - start_idx
        }
    
    def run_single_config(self, config, snapshots, run_id, part_dir):
        """Run a single configuration"""
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(f"{timestamp}{run_id}".encode()).hexdigest()[:4]
        
        config_name = config['name'].replace(' ', '_')
        output_name = f"{config_name}_snap{snapshots}_rep{run_id:03d}_{timestamp}_{random_suffix}"
        output_dir = part_dir / output_name
        
        # Build command
        command = self.base_command.copy()
        command.extend([
            "--output-dir", str(output_dir),
            "--trust-region-initial", str(config['trust_region_initial']),
            "--max-inner-iterations", str(config['max_inner_iterations']),
            "--num-snapshots", str(snapshots)
        ])
        
        print(f"\n{'='*60}")
        print(f"Running: {config['name']} | Snapshots: {snapshots} | Rep: {run_id}")
        print(f"TR: {config['trust_region_initial']}, Relax: {config['max_inner_iterations']}")
        print(f"Output: {output_name}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ Completed in {elapsed/60:.1f} minutes")
                return {
                    'status': 'success',
                    'runtime': elapsed,
                    'output_dir': str(output_dir)
                }
            else:
                print(f"✗ Failed after {elapsed/60:.1f} minutes")
                error_msg = result.stderr[-500:] if result.stderr else "No error message"
                return {
                    'status': 'failed',
                    'runtime': elapsed,
                    'error': error_msg
                }
                
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_part(self, part_num):
        """Run all configurations for a specific part"""
        
        part_info = self.get_part_info(part_num)
        part_dir = self.output_base / f"part_{part_num:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"RUNNING PART {part_num}/16")
        print(f"{'='*80}")
        print(f"Configuration: {part_info['config']['name']}")
        print(f"Snapshots: {part_info['snapshots']}")
        print(f"Runs: {part_info['start_idx']} to {part_info['end_idx']-1}")
        print(f"Total runs in this part: {part_info['n_runs']}")
        print(f"Output directory: {part_dir}")
        
        results = []
        start_time = time.time()
        
        for i in range(part_info['start_idx'], part_info['end_idx']):
            run_num = i + 1
            print(f"\n[{run_num-part_info['start_idx']}/{part_info['n_runs']}] ", end='')
            
            result = self.run_single_config(
                part_info['config'],
                part_info['snapshots'],
                run_num,
                part_dir
            )
            
            result['run_id'] = run_num
            result['config'] = part_info['config']['name']
            result['snapshots'] = part_info['snapshots']
            results.append(result)
            
            # Save intermediate results
            results_file = self.output_base / f"part_{part_num:02d}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{'='*80}")
        print(f"PART {part_num} COMPLETE")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return results
    
    def show_info(self):
        """Display study information"""
        
        print("="*80)
        print("NEW_v13: COMPREHENSIVE SNAPSHOT STUDY")
        print("="*80)
        print("\nStudy Design:")
        print("-" * 40)
        print("Testing effect of thermal snapshot count on barrier accuracy")
        print("\nConfigurations:")
        print("  Config 1 (Best): TR=0.4, Relax=30")
        print("  Config 2 (Good): TR=0.35, Relax=40")
        print("\nSnapshot values: 50, 100, 200, 300")
        print("Repetitions per case: 50")
        print("Total runs: 400 (2 configs × 4 snapshots × 50 reps)")
        print("Total parts: 16 (25 runs each)")
        
        print("\n" + "="*80)
        print("PART ALLOCATION")
        print("="*80)
        
        print("\n{:<6} {:<20} {:<10} {:<15} {:<10}".format(
            "Part", "Configuration", "Snapshots", "Run Range", "Count"))
        print("-" * 70)
        
        for part in sorted(self.part_configs.keys()):
            config_key, snapshots, start, end = self.part_configs[part]
            config_name = self.configs[config_key]['name']
            print("{:<6} {:<20} {:<10} runs {:<2}-{:<2}      {:<10}".format(
                part, config_name, snapshots, start+1, end, end-start))
        
        print("\n" + "="*80)
        print("HOW TO RUN")
        print("="*80)
        
        print("\n1. Run a specific part:")
        print("   python run_snapshot_study.py --part 1")
        
        print("\n2. Run multiple parts in parallel (recommended):")
        print("   # Terminal 1")
        print("   python run_snapshot_study.py --part 1 > part1.log 2>&1 &")
        print("   # Terminal 2")  
        print("   python run_snapshot_study.py --part 2 > part2.log 2>&1 &")
        print("   # ... etc")
        
        print("\n3. Example parallel execution strategy:")
        print("   - Machine 1: Parts 1-4 (Config1, varying snapshots)")
        print("   - Machine 2: Parts 5-8 (Config1, varying snapshots)")
        print("   - Machine 3: Parts 9-12 (Config2, varying snapshots)")
        print("   - Machine 4: Parts 13-16 (Config2, varying snapshots)")
        
        print("\n" + "="*80)
        print("TIME ESTIMATES")
        print("="*80)
        print("\nAssuming ~15 minutes per run:")
        print("  Per part (25 runs): ~6.25 hours")
        print("  Per config-snapshot pair (2 parts): ~12.5 hours")
        print("  Total sequential: ~100 hours")
        print("  With 4 parallel workers: ~25 hours")
        print("  With 16 parallel workers: ~6.25 hours")
        
        print("\nNote: Higher snapshot counts may increase runtime:")
        print("  50 snapshots: ~15 min/run")
        print("  100 snapshots: ~20 min/run")
        print("  200 snapshots: ~30 min/run")
        print("  300 snapshots: ~40 min/run")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='NEW_v13 Snapshot Study')
    parser.add_argument('--part', type=int, help='Part number to run (1-16)')
    parser.add_argument('--info', action='store_true', help='Show study information')
    
    args = parser.parse_args()
    
    study = SnapshotStudy()
    
    if args.info:
        study.show_info()
    elif args.part:
        if args.part < 1 or args.part > 16:
            print(f"Error: Part must be between 1 and 16")
            sys.exit(1)
        study.run_part(args.part)
    else:
        study.show_info()
        print("\nPlease specify --part N to run a part, or --info for information")

if __name__ == "__main__":
    main()