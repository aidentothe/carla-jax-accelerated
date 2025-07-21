#!/usr/bin/env python3
"""Quick test of key scenarios"""

import subprocess
import sys
import os

def run_test(name, cmd, should_pass=True):
    print(f"\n{'='*50}")
    print(f"Test: {name}")
    print(f"Command: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if should_pass and result.returncode == 0:
            print("✅ PASSED")
            return True
        elif not should_pass and result.returncode != 0:
            print("✅ PASSED (expected failure)")
            return True
        else:
            print("❌ FAILED")
            print(f"Return code: {result.returncode}")
            print(f"Stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("🧪 CARLA JAX Quick Flow Tests")
    
    tests = [
        ("CLI Status", "./carla-jax-universal --status"),
        ("No Conda Mode", "./carla-jax-universal --no-conda --status"),
        ("Help Command", "./carla-jax-universal --help"),
        ("Diagnostic Tool", "python diagnose_setup.py"),
        ("JAX Example", "python PythonAPI/examples_jax/generate_traffic_jax.py --mode jax-only --number-of-vehicles 2 --simulation-steps 3"),
        ("CARLA Import Fix", "python -c 'from PythonAPI.examples_jax.carla_import_fix import setup_carla_import; print(setup_carla_import())'"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, cmd in tests:
        if run_test(name, cmd):
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All critical flows working!")
    else:
        print("⚠️  Some issues found")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)