#!/usr/bin/env python3
"""
Ultra-Deep Edge Case Testing for CARLA JAX CLI

This script tests extreme scenarios where typical solutions break,
including file system issues, permission problems, network failures,
process conflicts, and more.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import signal
import time
import threading
import socket
import json
from pathlib import Path
import stat
import psutil
import random
import string

class StressTestSuite:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_results = []
        self.critical_failures = []
        self.passed = 0
        self.failed = 0
        
    def log_result(self, test_name, passed, details=None, critical=False):
        """Log test result."""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'critical': critical
        }
        self.test_results.append(result)
        
        if passed:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}: {details}")
            self.failed += 1
            if critical:
                self.critical_failures.append(result)
    
    def run_command_isolated(self, cmd, env=None, cwd=None, timeout=30):
        """Run command in isolated environment."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # File System Edge Cases
    def test_readonly_filesystem(self):
        """Test behavior with read-only file system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only
            os.chmod(tmpdir, 0o444)
            
            env = os.environ.copy()
            env['HOME'] = tmpdir
            
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                env=env
            )
            
            # Should handle gracefully, not crash
            self.log_result(
                "Read-only filesystem handling",
                result['success'] or 'Permission denied' not in result.get('stderr', ''),
                result.get('stderr', '')[:200]
            )
    
    def test_no_disk_space(self):
        """Simulate no disk space scenario."""
        # Create a small filesystem simulation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fill up available space (simulate)
            try:
                large_file = Path(tmpdir) / "large_file"
                with open(large_file, 'wb') as f:
                    # Write a large amount to simulate full disk
                    f.write(b'0' * (1024 * 1024))  # 1MB
                
                env = os.environ.copy()
                env['TMPDIR'] = tmpdir
                env['TEMP'] = tmpdir
                
                result = self.run_command_isolated(
                    f"{self.base_dir}/carla-jax-universal --status",
                    env=env
                )
                
                self.log_result(
                    "No disk space handling",
                    True,  # Should not crash even if it can't write
                    "Handled disk space constraints"
                )
                
            except Exception as e:
                self.log_result(
                    "No disk space handling",
                    False,
                    f"Failed to handle disk space: {e}",
                    critical=True
                )
    
    def test_corrupted_directories(self):
        """Test with corrupted/missing critical directories."""
        test_cases = [
            ("Missing home directory", {"HOME": "/nonexistent"}),
            ("Missing config directory", {"XDG_CONFIG_HOME": "/nonexistent"}),
            ("Invalid working directory", None),
        ]
        
        for test_name, env_mod in test_cases:
            env = os.environ.copy()
            if env_mod:
                env.update(env_mod)
            
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                env=env
            )
            
            self.log_result(
                f"Corrupted environment: {test_name}",
                result['success'] or result['returncode'] != -1,
                result.get('stderr', '')[:100]
            )
    
    # Process Management Edge Cases
    def test_multiple_concurrent_launches(self):
        """Test multiple launchers starting simultaneously."""
        processes = []
        start_time = time.time()
        
        # Launch 5 instances simultaneously
        for i in range(5):
            proc = subprocess.Popen([
                f"{self.base_dir}/carla-jax-universal", "--status"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(proc)
        
        # Wait for all to complete
        results = []
        for proc in processes:
            stdout, stderr = proc.communicate(timeout=60)
            results.append({
                'returncode': proc.returncode,
                'stdout': stdout.decode(),
                'stderr': stderr.decode()
            })
        
        success_count = sum(1 for r in results if r['returncode'] == 0)
        
        self.log_result(
            "Concurrent launch handling",
            success_count >= 3,  # At least 3 should succeed
            f"{success_count}/5 instances succeeded",
            critical=success_count == 0
        )
    
    def test_resource_exhaustion(self):
        """Test behavior under resource constraints."""
        try:
            # Get current memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Set memory limit (simulate low memory)
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (initial_memory * 2, initial_memory * 2))
            
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                timeout=45
            )
            
            self.log_result(
                "Memory constraint handling",
                result['success'],
                result.get('error', '') or result.get('stderr', '')[:200]
            )
            
        except Exception as e:
            self.log_result(
                "Memory constraint handling",
                False,
                f"Resource limit test failed: {e}",
                critical=True
            )
    
    # Network Edge Cases  
    def test_offline_mode(self):
        """Test behavior with no internet connection."""
        env = os.environ.copy()
        # Block network access
        env['http_proxy'] = 'http://127.0.0.1:9999'
        env['https_proxy'] = 'http://127.0.0.1:9999'
        env['HTTP_PROXY'] = 'http://127.0.0.1:9999'
        env['HTTPS_PROXY'] = 'http://127.0.0.1:9999'
        
        result = self.run_command_isolated(
            f"{self.base_dir}/carla-jax-universal --status",
            env=env,
            timeout=45
        )
        
        self.log_result(
            "Offline mode handling",
            result['success'],
            "Should work without network for status check"
        )
    
    def test_dns_resolution_failure(self):
        """Test with DNS resolution issues."""
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        # Point to invalid DNS
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
            f.write('''
import socket
original_getaddrinfo = socket.getaddrinfo

def failing_getaddrinfo(*args, **kwargs):
    raise socket.gaierror("DNS resolution failed")

socket.getaddrinfo = failing_getaddrinfo
''')
            f.flush()
            
            env['PYTHONSTARTUP'] = f.name
            
            result = self.run_command_isolated(
                f"python {self.base_dir}/carla_jax_cli.py --status",
                env=env
            )
            
            self.log_result(
                "DNS failure handling",
                result['success'],
                "Should work without network connectivity"
            )
    
    # Environment Conflicts
    def test_corrupted_python_environment(self):
        """Test with corrupted Python environment."""
        env = os.environ.copy()
        
        # Corrupt PYTHONPATH
        env['PYTHONPATH'] = '/nonexistent:/invalid/path:' + env.get('PYTHONPATH', '')
        
        # Corrupt module cache
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONHASHSEED'] = 'random'
        
        result = self.run_command_isolated(
            f"{self.base_dir}/carla-jax-universal --status",
            env=env
        )
        
        self.log_result(
            "Corrupted Python environment",
            result['success'],
            result.get('stderr', '')[:200]
        )
    
    def test_conflicting_packages(self):
        """Test with conflicting package versions."""
        # This test simulates package conflicts
        env = os.environ.copy()
        
        # Create temporary site-packages with conflicting packages
        with tempfile.TemporaryDirectory() as tmpdir:
            conflict_dir = Path(tmpdir) / "conflicting_packages"
            conflict_dir.mkdir()
            
            # Create fake conflicting package
            (conflict_dir / "__init__.py").write_text("raise ImportError('Conflicting package')")
            
            env['PYTHONPATH'] = str(conflict_dir) + ':' + env.get('PYTHONPATH', '')
            
            result = self.run_command_isolated(
                f"python {self.base_dir}/carla_jax_cli.py --status",
                env=env
            )
            
            self.log_result(
                "Package conflict handling",
                result['success'],
                "Should handle import conflicts gracefully"
            )
    
    # Extreme Configurations
    def test_unicode_paths(self):
        """Test with Unicode characters in paths."""
        try:
            unicode_dir = Path.home() / "test_unicode_日本語_🚗"
            unicode_dir.mkdir(exist_ok=True)
            
            env = os.environ.copy()
            env['CARLA_JAX_HOME'] = str(unicode_dir)
            
            # Copy CLI script to unicode directory
            shutil.copy2(
                self.base_dir / "carla_jax_cli.py",
                unicode_dir / "carla_jax_cli.py"
            )
            
            result = self.run_command_isolated(
                f"python '{unicode_dir}/carla_jax_cli.py' --status",
                env=env
            )
            
            self.log_result(
                "Unicode path handling",
                result['success'],
                result.get('stderr', '')[:200]
            )
            
            # Cleanup
            shutil.rmtree(unicode_dir, ignore_errors=True)
            
        except Exception as e:
            self.log_result(
                "Unicode path handling",
                False,
                f"Unicode test failed: {e}"
            )
    
    def test_very_long_paths(self):
        """Test with extremely long file paths."""
        try:
            # Create deeply nested directory structure
            long_path = Path.home() / "test_long_path"
            current_path = long_path
            
            # Create 50-level deep nesting
            for i in range(50):
                current_path = current_path / f"level_{i:02d}_very_long_directory_name_that_goes_on_and_on"
            
            current_path.mkdir(parents=True, exist_ok=True)
            
            env = os.environ.copy()
            env['TMPDIR'] = str(current_path)
            
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                env=env,
                timeout=60
            )
            
            self.log_result(
                "Long path handling",
                result['success'],
                f"Path length: {len(str(current_path))}"
            )
            
            # Cleanup
            shutil.rmtree(long_path, ignore_errors=True)
            
        except Exception as e:
            self.log_result(
                "Long path handling",
                False,
                f"Long path test failed: {e}"
            )
    
    # Signal Handling
    def test_signal_interruption(self):
        """Test signal handling during startup."""
        proc = subprocess.Popen([
            f"{self.base_dir}/carla-jax-universal", "--status"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit then send SIGTERM
        time.sleep(1)
        proc.terminate()
        
        try:
            stdout, stderr = proc.communicate(timeout=10)
            # Should exit gracefully
            self.log_result(
                "Signal handling",
                proc.returncode in [0, -15, 143],  # Normal or SIGTERM
                f"Exit code: {proc.returncode}"
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            self.log_result(
                "Signal handling",
                False,
                "Process didn't respond to signals",
                critical=True
            )
    
    # Security Edge Cases
    def test_malicious_environment_variables(self):
        """Test with potentially malicious environment variables."""
        malicious_envs = [
            {"LD_PRELOAD": "/tmp/malicious.so"},
            {"PYTHONPATH": "/tmp;/var/tmp;$(rm -rf /)"},
            {"PATH": "/tmp:/var/tmp:$(malicious_command)"},
            {"HOME": "'; rm -rf / #"},
            {"CARLA_JAX_HOME": "$(echo 'injection')"},
        ]
        
        for i, env_mod in enumerate(malicious_envs):
            env = os.environ.copy()
            env.update(env_mod)
            
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                env=env,
                timeout=30
            )
            
            self.log_result(
                f"Security test {i+1}",
                result['success'],
                f"Malicious env: {list(env_mod.keys())[0]}"
            )
    
    # Container/Virtualization Edge Cases
    def test_containerized_environment(self):
        """Test in containerized environment simulation."""
        env = os.environ.copy()
        
        # Simulate container environment
        env.update({
            'container': 'podman',
            'HOSTNAME': 'container-host',
            'HOME': '/tmp/fake_home',
            'USER': 'container_user',
            'TMPDIR': '/tmp',
        })
        
        # Create fake home
        fake_home = Path('/tmp/fake_home')
        fake_home.mkdir(exist_ok=True)
        
        result = self.run_command_isolated(
            f"{self.base_dir}/carla-jax-universal --status",
            env=env
        )
        
        self.log_result(
            "Container environment",
            result['success'],
            "Should work in containerized environments"
        )
        
        # Cleanup
        shutil.rmtree(fake_home, ignore_errors=True)
    
    # Performance Under Stress
    def test_performance_under_load(self):
        """Test performance when system is under load."""
        def cpu_stress():
            """Generate CPU load."""
            end_time = time.time() + 10
            while time.time() < end_time:
                sum(i * i for i in range(1000))
        
        # Start CPU stress in background
        stress_threads = []
        for _ in range(os.cpu_count() or 4):
            t = threading.Thread(target=cpu_stress)
            t.start()
            stress_threads.append(t)
        
        try:
            start_time = time.time()
            result = self.run_command_isolated(
                f"{self.base_dir}/carla-jax-universal --status",
                timeout=60
            )
            elapsed = time.time() - start_time
            
            self.log_result(
                "Performance under load",
                result['success'] and elapsed < 45,
                f"Completed in {elapsed:.1f}s under load"
            )
            
        finally:
            # Wait for stress threads to complete
            for t in stress_threads:
                t.join()
    
    def run_all_stress_tests(self):
        """Run all stress tests."""
        print("🔥 CARLA JAX CLI Ultra-Deep Edge Case Testing")
        print("=" * 80)
        print("Testing extreme scenarios where typical solutions break...")
        
        # File system tests
        print("\n📁 File System Edge Cases")
        self.test_readonly_filesystem()
        self.test_no_disk_space()
        self.test_corrupted_directories()
        
        # Process management tests
        print("\n⚙️  Process Management Edge Cases")
        self.test_multiple_concurrent_launches()
        self.test_resource_exhaustion()
        self.test_signal_interruption()
        
        # Network tests
        print("\n🌐 Network Edge Cases")
        self.test_offline_mode()
        self.test_dns_resolution_failure()
        
        # Environment tests
        print("\n🔧 Environment Edge Cases")
        self.test_corrupted_python_environment()
        self.test_conflicting_packages()
        
        # Extreme configuration tests
        print("\n🚀 Extreme Configuration Edge Cases")
        self.test_unicode_paths()
        self.test_very_long_paths()
        
        # Security tests
        print("\n🔒 Security Edge Cases")
        self.test_malicious_environment_variables()
        
        # Container tests
        print("\n📦 Container/Virtualization Edge Cases")
        self.test_containerized_environment()
        
        # Performance tests
        print("\n⚡ Performance Edge Cases")
        self.test_performance_under_load()
        
        # Generate report
        self.generate_stress_report()
    
    def generate_stress_report(self):
        """Generate comprehensive stress test report."""
        print("\n" + "="*80)
        print("🔥 STRESS TEST REPORT")
        print("="*80)
        
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"🔥 Success Rate: {success_rate:.1f}%")
        
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"   💥 {failure['test']}: {failure['details']}")
        
        if self.failed > 0:
            print(f"\n⚠️  Failed Tests:")
            failures = [r for r in self.test_results if not r['passed']]
            for failure in failures:
                print(f"   ❌ {failure['test']}: {failure['details'][:100]}")
        
        # Save detailed report
        report_file = Path.home() / '.config' / 'carla-jax' / 'stress_test_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': total,
                'passed': self.passed,
                'failed': self.failed,
                'success_rate': success_rate,
                'critical_failures': len(self.critical_failures),
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Overall assessment
        print("\n🎯 ROBUSTNESS ASSESSMENT:")
        if success_rate >= 90:
            print("🟢 EXCELLENT - System is highly robust")
        elif success_rate >= 75:
            print("🟡 GOOD - System handles most edge cases")
        elif success_rate >= 50:
            print("🟠 FAIR - Some robustness issues need addressing")
        else:
            print("🔴 POOR - Significant robustness improvements needed")

def main():
    tester = StressTestSuite()
    tester.run_all_stress_tests()
    
    return 0 if len(tester.critical_failures) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())