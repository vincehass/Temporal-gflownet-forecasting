#!/usr/bin/env python
"""
Wrapper script to generate synthetic ablation data and visualize it in one step.
"""
import os
import sys
import argparse
import subprocess

def main():
    """Main function to run the generation and visualization pipeline."""
    parser = argparse.ArgumentParser(description='Generate and visualize synthetic ablation studies')
    parser.add_argument('--results_dir', type=str, default='./results/synthetic_data',
                        help='Directory to store synthetic results (default: ./results/synthetic_data)')
    parser.add_argument('--output_dir', type=str, default='./results/ablation_plots',
                        help='Directory to save plots (default: ./results/ablation_plots)')
    parser.add_argument('--experiments', type=str, nargs='*', 
                        help='List of experiments to include in visualization')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip the synthetic data generation step')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training data generation (default: 50)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing data directories')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data
    if not args.skip_generation:
        print("Step 1: Generating synthetic ablation data...")
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            generation_script = os.path.join(script_dir, 'generate_synthetic_results.py')
            
            # Build command with arguments
            cmd = [
                sys.executable,
                generation_script,
                '--output_dir', args.results_dir,
                '--num_epochs', str(args.epochs)
            ]
            
            # Add force flag if specified
            if args.force:
                cmd.append('--force')
            
            # Run the data generation script
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(f"Warnings:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating synthetic data: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error message: {e.stderr}")
            sys.exit(1)
    else:
        print("Skipping synthetic data generation...")
    
    # Step 2: Visualize the results
    print("\nStep 2: Visualizing ablation results...")
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visualization_script = os.path.join(script_dir, 'plot_synthetic_results.py')
        
        # Build the command with arguments
        cmd = [
            sys.executable,
            visualization_script,
            '--results_dir', args.results_dir,
            '--output_dir', args.output_dir
        ]
        
        # Add experiments if specified
        if args.experiments:
            cmd.extend(['--experiments'])
            cmd.extend(args.experiments)
        
        # Run the visualization script
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error visualizing results: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)
    
    print(f"\nPipeline completed successfully!")
    print(f"- Synthetic data stored in: {args.results_dir}")
    print(f"- Visualizations saved to: {args.output_dir}")
    print("\nGenerated plots:")
    try:
        for filename in os.listdir(args.output_dir):
            if filename.endswith('.png'):
                print(f"- {filename}")
    except Exception as e:
        print(f"Couldn't list plots: {e}")

if __name__ == "__main__":
    main() 