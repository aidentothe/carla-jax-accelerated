#!/usr/bin/env python3
"""
CARLA JAX Diagnostic Tool

This tool diagnoses common issues with the CARLA JAX setup
and provides solutions.
"""

import sys
import os
import subprocess
import platform
import json
from pathlib import Path
import importlib.util

class DiagnosticTool:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.results = {}
    
    def print_header(self, text):
        print(f"\n{'='*60}")
        print(f"🔍 {text}")
        print('='*60)
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.print_header("Python Version Check")
        
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {self.python_version}")
        
        if self.python_version in ["3.7", "3.8", "3.9", "3.10"]:
            self.successes.append(f"✅ Python {self.python_version} is compatible")
            self.results['python_compatible'] = True
        else:
            self.issues.append(f"❌ Python {self.python_version} has limited CARLA support")
            self.results['python_compatible'] = False
            print("\nRecommended: Use Python 3.8 for best compatibility")
    
    def check_conda_environment(self):
        """Check conda environment setup."""
        self.print_header("Conda Environment Check")
        
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        if conda_env:
            print(f"Current conda environment: {conda_env}")
            
            if conda_env == "carla-jax":
                self.successes.append("✅ Using recommended 'carla-jax' environment")
                self.results['conda_correct'] = True
            else:
                self.warnings.append(f"⚠️  Using '{conda_env}' instead of 'carla-jax'")
                self.results['conda_correct'] = False
        else:
            print("Not using conda environment")
            self.results['conda_available'] = False
            
            # Check if conda is available
            try:
                result = subprocess.run(['conda', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.warnings.append("⚠️  Conda available but not activated")
                    print(f"Conda version: {result.stdout.strip()}")
            except:
                print("Conda not found in PATH")
    
    def check_carla_installation(self):
        """Check CARLA package installation."""
        self.print_header("CARLA Package Check")
        
        try:
            import carla
            try:
                version = carla.__version__
            except AttributeError:
                version = "unknown"
            self.successes.append(f"✅ CARLA {version} installed")
            self.results['carla_installed'] = True
            print(f"CARLA version: {version}")
            print(f"CARLA module location: {carla.__file__}")
        except ImportError as e:
            self.issues.append("❌ CARLA package not installed")
            self.results['carla_installed'] = False
            print(f"Import error: {e}")
            
            # Check for egg files
            self.check_carla_eggs()
    
    def check_carla_eggs(self):
        """Check for CARLA egg files."""
        print("\nSearching for CARLA egg files...")
        
        search_paths = [
            Path.home() / "carla_simulator",
            Path("/opt/carla-simulator"),
            Path.cwd().parent,
        ]
        
        egg_found = False
        for path in search_paths:
            if path.exists():
                eggs = list(path.rglob("carla-*.egg"))
                if eggs:
                    print(f"\nFound {len(eggs)} egg file(s) in {path}:")
                    for egg in eggs[:3]:  # Show first 3
                        print(f"  - {egg.name}")
                    egg_found = True
        
        if not egg_found:
            print("No CARLA egg files found")
    
    def check_jax_installation(self):
        """Check JAX installation."""
        self.print_header("JAX Package Check")
        
        try:
            import jax
            version = jax.__version__
            self.successes.append(f"✅ JAX {version} installed")
            self.results['jax_installed'] = True
            
            print(f"JAX version: {version}")
            print(f"JAX devices: {jax.devices()}")
            print(f"JAX backend: {jax.default_backend()}")
            
            # Test basic operation
            import jax.numpy as jnp
            x = jnp.array([1, 2, 3])
            result = jnp.sum(x)
            print(f"JAX test: sum([1,2,3]) = {result}")
            
        except ImportError as e:
            self.issues.append("❌ JAX package not installed")
            self.results['jax_installed'] = False
            print(f"Import error: {e}")
    
    def check_carla_server(self):
        """Check CARLA server availability."""
        self.print_header("CARLA Server Check")
        
        # Check if server is running
        try:
            result = subprocess.run(['nc', '-z', 'localhost', '2000'], 
                                  capture_output=True)
            if result.returncode == 0:
                self.successes.append("✅ CARLA server is running on port 2000")
                self.results['carla_server_running'] = True
                print("CARLA server is running")
            else:
                print("CARLA server is not running")
                self.results['carla_server_running'] = False
        except:
            print("Could not check CARLA server status")
            self.results['carla_server_running'] = False
        
        # Check for CARLA installation
        carla_paths = [
            Path.home() / "carla_simulator" / "CARLA_0.9.15",
            Path.home() / "carla_simulator" / "CARLA_0.9.14",
            Path.home() / "carla_simulator" / "CARLA_0.9.13",
            Path("/opt/carla-simulator"),
        ]
        
        carla_found = False
        for path in carla_paths:
            if (path / "CarlaUE4.sh").exists():
                print(f"\nCARLA server found at: {path}")
                carla_found = True
                self.results['carla_server_path'] = str(path)
                break
        
        if not carla_found:
            self.warnings.append("⚠️  CARLA server installation not found")
            print("\nCARLA server not found in common locations")
    
    def check_environment_variables(self):
        """Check environment variables."""
        self.print_header("Environment Variables")
        
        important_vars = {
            'CARLA_JAX_HOME': 'CARLA JAX installation directory',
            'CARLA_ROOT': 'CARLA server installation directory',
            'PYTHONPATH': 'Python module search path',
            'CONDA_DEFAULT_ENV': 'Current conda environment',
        }
        
        for var, description in important_vars.items():
            value = os.environ.get(var)
            if value:
                print(f"{var}: {value}")
                if var == 'CARLA_JAX_HOME':
                    if Path(value).exists():
                        self.successes.append(f"✅ {var} is set correctly")
                    else:
                        self.issues.append(f"❌ {var} points to non-existent directory")
            else:
                print(f"{var}: Not set")
    
    def check_directory_structure(self):
        """Check CARLA JAX directory structure."""
        self.print_header("Directory Structure Check")
        
        carla_jax_home = os.environ.get('CARLA_JAX_HOME', os.getcwd())
        base_path = Path(carla_jax_home)
        
        required_files = [
            'carla_jax_cli.py',
            'carla-jax-universal',
            'install_system_wide.sh',
            'PythonAPI/examples_jax/generate_traffic_jax.py',
        ]
        
        print(f"Checking directory: {base_path}")
        
        missing_files = []
        for file_path in required_files:
            full_path = base_path / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - MISSING")
                missing_files.append(file_path)
        
        if missing_files:
            self.issues.append(f"❌ Missing {len(missing_files)} required files")
        else:
            self.successes.append("✅ All required files present")
    
    def generate_solutions(self):
        """Generate solutions for identified issues."""
        self.print_header("Recommended Solutions")
        
        solutions = []
        
        # Python version issue
        if not self.results.get('python_compatible', True):
            solutions.append("""
1. Create conda environment with Python 3.8:
   conda create -n carla-jax python=3.8
   conda activate carla-jax
   pip install jax jaxlib carla==0.9.13
""")
        
        # CARLA not installed
        if not self.results.get('carla_installed', True):
            solutions.append("""
2. Install CARLA package:
   python install_carla_package.py
   
   Or manually:
   pip install carla==0.9.13
""")
        
        # JAX not installed
        if not self.results.get('jax_installed', True):
            solutions.append("""
3. Install JAX:
   pip install jax[cpu] jaxlib numpy
""")
        
        # CARLA server not found
        if 'carla_server_path' not in self.results:
            solutions.append("""
4. Install CARLA server:
   ./install_carla.sh
   
   Or download manually:
   wget https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz
   tar -xzf CARLA_0.9.15.tar.gz
""")
        
        for solution in solutions:
            print(solution)
        
        if not solutions:
            print("No critical issues found! System appears ready.")
    
    def save_diagnostic_report(self):
        """Save diagnostic report to file."""
        report = {
            'timestamp': str(Path.cwd()),
            'python_version': self.python_version,
            'platform': platform.platform(),
            'issues': self.issues,
            'warnings': self.warnings,
            'successes': self.successes,
            'results': self.results,
        }
        
        report_file = Path.home() / '.config' / 'carla-jax' / 'diagnostic_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDiagnostic report saved to: {report_file}")
    
    def run_diagnostics(self):
        """Run all diagnostic checks."""
        print("🏥 CARLA JAX Diagnostic Tool")
        print("=" * 60)
        
        # Run all checks
        self.check_python_version()
        self.check_conda_environment()
        self.check_carla_installation()
        self.check_jax_installation()
        self.check_carla_server()
        self.check_environment_variables()
        self.check_directory_structure()
        
        # Summary
        self.print_header("Diagnostic Summary")
        
        print(f"\n✅ Successes ({len(self.successes)}):")
        for success in self.successes:
            print(f"   {success}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
        
        if self.issues:
            print(f"\n❌ Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")
        
        # Generate solutions
        if self.issues:
            self.generate_solutions()
        
        # Save report
        self.save_diagnostic_report()
        
        # Return status
        return len(self.issues) == 0

def main():
    tool = DiagnosticTool()
    success = tool.run_diagnostics()
    
    print("\n" + "="*60)
    if success:
        print("✅ System is ready to use!")
        print("Run: carla-jax")
    else:
        print("❌ Issues found. Please follow the recommendations above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())