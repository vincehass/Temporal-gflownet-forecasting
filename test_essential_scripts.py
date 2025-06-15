#!/usr/bin/env python3
"""
Test Essential Scripts for Temporal GFlowNet Ablation Studies
Verifies that the essential scripts work correctly after cleanup.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Test configuration
TEST_CONFIG = {
    'python_scripts': [
        'scripts/run_single_ablation.py',
        'scripts/direct_ablation.py', 
        'scripts/plot_ablation_results.py',
        'scripts/visualize_wandb_ablations.py',
        'scripts/enhanced_ablation_viz.py',
        'scripts/compare_ablation_results.py',
        'scripts/run_ablation_visualization.py',
        'scripts/train.py',
        'scripts/evaluate.py',
        'scripts/test_functionality.py',
        'scripts/run_eeg_experiments.py',
        'scripts/quantization_analysis.py',
    ],
    'shell_scripts': [
        'scripts/run_complete_ablation_study.sh',
        'scripts/run_integrated_ablation_study.sh',
        'scripts/run_ablation.sh',
        'scripts/run_all_ablations.sh',
        'scripts/run_ablation_analysis.sh',
    ]
}

class ScriptTester:
    """Test runner for essential scripts."""
    
    def __init__(self):
        self.results = {'passed': [], 'failed': [], 'warnings': []}
        
    def test_python_import(self, script_path: str) -> Tuple[bool, str]:
        """Test if a Python script can be imported without errors."""
        try:
            # Convert script path to module name
            script_name = Path(script_path).stem
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            if spec is None:
                return False, "Could not create module spec"
            
            module = importlib.util.module_from_spec(spec)
            
            # Test import without executing
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Check for obvious syntax errors
            try:
                compile(content, script_path, 'exec')
                return True, "Syntax check passed"
            except SyntaxError as e:
                return False, f"Syntax error: {e}"
                
        except Exception as e:
            return False, f"Import error: {e}"
    
    def test_python_help(self, script_path: str) -> Tuple[bool, str]:
        """Test if a Python script shows help without errors."""
        try:
            result = subprocess.run(
                [sys.executable, script_path, '--help'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return True, "Help command successful"
            else:
                # Some scripts might not have --help but still work
                if "unrecognized arguments" in result.stderr:
                    return True, "Script runs (no --help option)"
                return False, f"Help failed: {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            return False, "Help command timed out"
        except Exception as e:
            return False, f"Help test error: {e}"
    
    def test_shell_script_syntax(self, script_path: str) -> Tuple[bool, str]:
        """Test shell script syntax."""
        try:
            result = subprocess.run(
                ['bash', '-n', script_path],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                return True, "Shell syntax check passed"
            else:
                return False, f"Shell syntax error: {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            return False, "Syntax check timed out"
        except Exception as e:
            return False, f"Shell syntax test error: {e}"
    
    def test_script_executable(self, script_path: str) -> Tuple[bool, str]:
        """Test if script is executable."""
        path = Path(script_path)
        if path.exists():
            if os.access(script_path, os.X_OK):
                return True, "Script is executable"
            else:
                return False, "Script is not executable"
        else:
            return False, "Script does not exist"
    
    def test_dependencies(self) -> Tuple[bool, str]:
        """Test if required dependencies are available."""
        required_modules = [
            'torch', 'numpy', 'yaml', 'wandb', 'matplotlib', 
            'pandas', 'scipy', 'sklearn', 'tqdm'
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            return False, f"Missing modules: {', '.join(missing)}"
        else:
            return True, "All required modules available"
    
    def test_data_module(self) -> Tuple[bool, str]:
        """Test if the data module was created correctly."""
        try:
            from src.temporal_gfn.data import TimeSeriesDataset, SyntheticTimeSeriesDataset
            from src.temporal_gfn.data import MeanScaler, StandardScaler
            from src.temporal_gfn.data import create_dataloader, create_windows
            
            # Test basic functionality
            dataset = SyntheticTimeSeriesDataset(num_samples=10, seq_length=100)
            dataloader = create_dataloader(dataset, batch_size=2)
            
            # Test one batch
            for batch in dataloader:
                context, target = batch
                if context.shape[0] > 0:  # Has samples
                    break
            
            return True, "Data module works correctly"
            
        except Exception as e:
            return False, f"Data module error: {e}"
    
    def run_all_tests(self) -> Dict[str, List]:
        """Run all tests and return results."""
        print("🧪 Testing Essential Scripts")
        print("=" * 50)
        
        # Test dependencies first
        print("\n1. Testing dependencies...")
        passed, msg = self.test_dependencies()
        if passed:
            print(f"  ✓ {msg}")
            self.results['passed'].append(('Dependencies', msg))
        else:
            print(f"  ✗ {msg}")
            self.results['failed'].append(('Dependencies', msg))
        
        # Test data module
        print("\n2. Testing data module...")
        passed, msg = self.test_data_module()
        if passed:
            print(f"  ✓ {msg}")
            self.results['passed'].append(('Data Module', msg))
        else:
            print(f"  ✗ {msg}")
            self.results['failed'].append(('Data Module', msg))
        
        # Test Python scripts
        print("\n3. Testing Python scripts...")
        for script in TEST_CONFIG['python_scripts']:
            if not Path(script).exists():
                print(f"  ⚠ {script} - File not found")
                self.results['warnings'].append((script, "File not found"))
                continue
            
            # Test import/syntax
            passed, msg = self.test_python_import(script)
            if passed:
                print(f"  ✓ {script} - {msg}")
                self.results['passed'].append((script, f"Import: {msg}"))
            else:
                print(f"  ✗ {script} - {msg}")
                self.results['failed'].append((script, f"Import: {msg}"))
                continue
            
            # Test help command
            passed, msg = self.test_python_help(script)
            if passed:
                print(f"    ✓ Help: {msg}")
            else:
                print(f"    ⚠ Help: {msg}")
                self.results['warnings'].append((script, f"Help: {msg}"))
        
        # Test shell scripts
        print("\n4. Testing shell scripts...")
        for script in TEST_CONFIG['shell_scripts']:
            if not Path(script).exists():
                print(f"  ⚠ {script} - File not found")
                self.results['warnings'].append((script, "File not found"))
                continue
            
            # Test executable
            passed, msg = self.test_script_executable(script)
            if not passed:
                print(f"  ⚠ {script} - {msg}")
                self.results['warnings'].append((script, f"Executable: {msg}"))
            
            # Test syntax
            passed, msg = self.test_shell_script_syntax(script)
            if passed:
                print(f"  ✓ {script} - {msg}")
                self.results['passed'].append((script, f"Syntax: {msg}"))
            else:
                print(f"  ✗ {script} - {msg}")
                self.results['failed'].append((script, f"Syntax: {msg}"))
        
        return self.results
    
    def run_basic_functionality_test(self) -> bool:
        """Run a basic end-to-end functionality test."""
        print("\n5. Running basic functionality test...")
        
        try:
            # Test that we can create a simple dataset and run minimal training
            from src.temporal_gfn.data import SyntheticTimeSeriesDataset, create_dataloader
            
            # Create small dataset
            dataset = SyntheticTimeSeriesDataset(num_samples=50, seq_length=100)
            dataloader = create_dataloader(dataset, batch_size=4)
            
            # Test data loading
            batch_count = 0
            for batch in dataloader:
                context, target = batch
                assert context.shape[1] == 96  # context_length
                assert target.shape[1] == 24   # prediction_horizon
                batch_count += 1
                if batch_count >= 3:  # Test a few batches
                    break
            
            print("  ✓ Basic data loading works")
            
            # Test that we can import core components
            from src.temporal_gfn.models.transformer import TemporalTransformerModel
            from src.temporal_gfn.gfn.policies import ForwardPolicy
            from src.temporal_gfn.quantization.adaptive import AdaptiveQuantization
            
            print("  ✓ Core components importable")
            return True
            
        except Exception as e:
            print(f"  ✗ Basic functionality test failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        total_tests = len(self.results['passed']) + len(self.results['failed'])
        passed_count = len(self.results['passed'])
        failed_count = len(self.results['failed'])
        warning_count = len(self.results['warnings'])
        
        print(f"Total tests: {total_tests}")
        print(f"✓ Passed: {passed_count}")
        print(f"✗ Failed: {failed_count}")
        print(f"⚠ Warnings: {warning_count}")
        
        if failed_count > 0:
            print(f"\n❌ Failed tests:")
            for name, msg in self.results['failed']:
                print(f"  - {name}: {msg}")
        
        if warning_count > 0:
            print(f"\n⚠️  Warnings:")
            for name, msg in self.results['warnings']:
                print(f"  - {name}: {msg}")
        
        # Overall status
        if failed_count == 0:
            print(f"\n🎉 All core tests passed! Ready for ablation studies.")
            return True
        else:
            print(f"\n❌ {failed_count} tests failed. Please fix before proceeding.")
            return False

def main():
    """Main test function."""
    tester = ScriptTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Run basic functionality test
    functionality_works = tester.run_basic_functionality_test()
    
    # Print summary
    success = tester.print_summary()
    
    if success and functionality_works:
        print("\n🚀 Ready to run ablation studies!")
        print("\nRecommended next steps:")
        print("1. Run a quick test: python scripts/run_single_ablation.py --help")
        print("2. Run basic ablation: ./scripts/run_ablation.sh")
        print("3. Run full study: ./scripts/run_complete_ablation_study.sh")
        return 0
    else:
        print("\n🔧 Please fix the issues above before running ablation studies.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 