#!/usr/bin/env python3
"""
CARLA Package Installation Helper

This script handles CARLA installation across different Python versions
and provides fallback methods.
"""

import sys
import subprocess
import os
import platform
import urllib.request
import tempfile
from pathlib import Path

class CARLAInstaller:
    def __init__(self):
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        
        # CARLA version compatibility matrix
        self.compatibility = {
            "3.7": ["0.9.13", "0.9.12", "0.9.11", "0.9.10"],
            "3.8": ["0.9.15", "0.9.14", "0.9.13", "0.9.12", "0.9.11"],
            "3.9": ["0.9.15", "0.9.14", "0.9.13"],
            "3.10": ["0.9.15", "0.9.14", "0.9.13"],
            "3.11": [],  # Limited support
            "3.12": [],  # Limited support
            "3.13": [],  # No official support
        }
        
        # Direct wheel URLs for fallback
        self.wheel_urls = {
            ("3.7", "linux"): [
                "https://files.pythonhosted.org/packages/27/07/8a8c1868e1564209e4485f4e56c7bb8d4827f24fb8b72acfc7a68b768cab/carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl",
            ],
            ("3.8", "linux"): [
                "https://files.pythonhosted.org/packages/cb/4e/cfcc8a123e37fb4b9a7156ba09ab1797b42303b09c7a199d5f648088ea33/carla-0.9.13-cp38-cp38-manylinux_2_27_x86_64.whl",
            ],
            ("3.9", "linux"): [
                "https://files.pythonhosted.org/packages/21/a6/b063a296ba4e8797c2c5ac2917b8fc24ddcedd978cf8dc69c1e93c183d21/carla-0.9.13-cp39-cp39-manylinux_2_27_x86_64.whl",
            ],
            ("3.10", "linux"): [
                "https://files.pythonhosted.org/packages/40/c9/a4e6e240a3f6a87c07c63c1c52e1b73ad47d690f9e2fc456af0b69bb4074/carla-0.9.13-cp310-cp310-manylinux_2_27_x86_64.whl",
            ],
        }
    
    def check_installed(self):
        """Check if CARLA is already installed."""
        try:
            import carla
            print(f"✅ CARLA {carla.__version__} is already installed")
            return True
        except ImportError:
            return False
    
    def get_compatible_versions(self):
        """Get compatible CARLA versions for current Python."""
        return self.compatibility.get(self.python_version, [])
    
    def install_via_pip(self, version=None):
        """Try to install CARLA via pip."""
        if version:
            package = f"carla=={version}"
        else:
            package = "carla"
        
        print(f"Attempting: pip install {package}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def install_from_wheel_url(self, url):
        """Download and install wheel from URL."""
        print(f"Downloading wheel from: {url}")
        
        try:
            # Download wheel
            with tempfile.NamedTemporaryFile(suffix='.whl', delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                wheel_path = tmp.name
            
            # Install wheel
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", wheel_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clean up
            os.unlink(wheel_path)
            return True
            
        except Exception as e:
            print(f"Failed: {e}")
            return False
    
    def find_egg_files(self):
        """Find CARLA egg files in common locations."""
        search_paths = [
            Path.home() / "carla_simulator",
            Path("/opt/carla-simulator"),
            Path("/opt/carla"),
            Path.cwd().parent.parent,  # Relative to script
        ]
        
        egg_files = []
        
        for base_path in search_paths:
            if base_path.exists():
                # Search for egg files
                pattern = f"**/carla-*-py{self.python_version}*.egg"
                eggs = list(base_path.rglob(pattern))
                egg_files.extend(eggs)
        
        return egg_files
    
    def create_pth_file(self, egg_path):
        """Create .pth file to add egg to Python path."""
        import site
        site_packages = site.getsitepackages()[0]
        
        pth_file = Path(site_packages) / "carla.pth"
        
        try:
            with open(pth_file, 'w') as f:
                f.write(str(egg_path))
            print(f"✅ Created .pth file: {pth_file}")
            return True
        except Exception as e:
            print(f"Failed to create .pth file: {e}")
            return False
    
    def install(self):
        """Main installation method."""
        print(f"\n🚀 CARLA Package Installer")
        print(f"Python version: {self.python_version}")
        print(f"System: {self.system} {self.machine}")
        
        # Check if already installed
        if self.check_installed():
            return True
        
        print("\n❌ CARLA not installed. Attempting installation...")
        
        # Method 1: Try compatible versions via pip
        compatible_versions = self.get_compatible_versions()
        
        if compatible_versions:
            print(f"\n📋 Compatible versions for Python {self.python_version}: {compatible_versions}")
            
            for version in compatible_versions:
                print(f"\nTrying CARLA {version}...")
                if self.install_via_pip(version):
                    print(f"✅ Successfully installed CARLA {version}")
                    return True
        else:
            print(f"\n⚠️  No officially compatible CARLA versions for Python {self.python_version}")
        
        # Method 2: Try direct wheel download
        wheel_key = (self.python_version, self.system)
        if wheel_key in self.wheel_urls:
            print("\n📦 Trying direct wheel download...")
            
            for url in self.wheel_urls[wheel_key]:
                if self.install_from_wheel_url(url):
                    print("✅ Successfully installed from wheel")
                    return True
        
        # Method 3: Try to find and use egg files
        print("\n🔍 Searching for CARLA egg files...")
        egg_files = self.find_egg_files()
        
        if egg_files:
            print(f"Found {len(egg_files)} egg file(s)")
            
            for egg in egg_files:
                print(f"\nTrying egg: {egg.name}")
                if self.create_pth_file(egg):
                    # Test if it works
                    try:
                        import carla
                        print(f"✅ Successfully configured CARLA from egg file")
                        return True
                    except ImportError:
                        pass
        
        # Method 4: Provide manual instructions
        print("\n❌ Automatic installation failed.")
        print("\n📋 Manual installation options:")
        print("\n1. For Python 3.8 (recommended):")
        print("   conda create -n carla-jax python=3.8")
        print("   conda activate carla-jax")
        print("   pip install carla==0.9.13")
        
        print("\n2. Download CARLA and use egg file:")
        print("   wget https://github.com/carla-simulator/carla/releases/download/0.9.13/CARLA_0.9.13.tar.gz")
        print("   tar -xzf CARLA_0.9.13.tar.gz")
        print("   export PYTHONPATH=$PWD/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
        
        print("\n3. Build from source (advanced):")
        print("   https://carla.readthedocs.io/en/latest/build_linux/")
        
        return False
    
    def verify_installation(self):
        """Verify CARLA installation and show info."""
        try:
            import carla
            print(f"\n✅ CARLA import successful")
            print(f"Version: {carla.__version__}")
            
            # Try to list available classes
            classes = [c for c in dir(carla) if c[0].isupper()]
            print(f"Available classes: {len(classes)}")
            print(f"Example classes: {', '.join(classes[:5])}...")
            
            return True
            
        except ImportError as e:
            print(f"\n❌ CARLA import failed: {e}")
            return False

def main():
    installer = CARLAInstaller()
    
    # Install CARLA
    success = installer.install()
    
    # Verify installation
    if success:
        installer.verify_installation()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())