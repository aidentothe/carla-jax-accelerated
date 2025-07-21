#!/usr/bin/env python3
"""
CARLA JAX CLI - Interactive Command Line Interface

This CLI provides a unified interface for managing CARLA simulation,
running JAX-accelerated examples, and tracking training progress.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CARLAJAXManager:
    """Main manager for CARLA JAX system"""
    
    def __init__(self):
        # Determine base directory from environment or script location
        if os.environ.get('CARLA_JAX_HOME'):
            self.base_dir = Path(os.environ['CARLA_JAX_HOME']).absolute()
        else:
            self.base_dir = Path(__file__).parent.absolute()
        
        # Ensure we're in the correct directory
        os.chdir(self.base_dir)
        
        self.carla_process = None
        self.config_dir = Path.home() / ".config" / "carla-jax"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.json"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        self.status_file = self.config_dir / "training_status.json"
        
        # Create necessary directories
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self.load_config()
        
        # Available algorithms and scripts
        self.algorithms = {
            'PPO': {
                'name': 'Proximal Policy Optimization',
                'script': 'PythonAPI/examples/train_jax_agent.py',
                'description': 'Policy gradient method with clipped surrogate objective'
            },
            'SAC': {
                'name': 'Soft Actor-Critic',
                'script': 'algorithms/sac.py',
                'description': 'Off-policy algorithm with maximum entropy objective'
            },
            'DDPG': {
                'name': 'Deep Deterministic Policy Gradient',
                'script': 'algorithms/ddpg.py',
                'description': 'Actor-critic method for continuous control'
            },
            'A2C': {
                'name': 'Advantage Actor-Critic',
                'script': 'algorithms/a2c.py',
                'description': 'Synchronous actor-critic algorithm'
            }
        }
        
        # JAX examples
        self.jax_examples = {
            'automatic_control': {
                'name': 'JAX Vehicle Control',
                'script': 'PythonAPI/examples_jax/automatic_control_jax.py',
                'description': 'JIT-compiled vehicle control with PID optimization',
                'requires_carla': True
            },
            'traffic_simulation': {
                'name': 'JAX Traffic Simulation',
                'script': 'PythonAPI/examples_jax/generate_traffic_jax.py',
                'description': 'Vectorized multi-agent traffic simulation',
                'requires_carla': False
            },
            'sensor_fusion': {
                'name': 'JAX Sensor Fusion',
                'script': 'PythonAPI/examples_jax/sensor_synchronization_jax.py',
                'description': 'Multi-sensor data processing with JAX',
                'requires_carla': True
            },
            'vehicle_physics': {
                'name': 'JAX Vehicle Physics',
                'script': 'PythonAPI/examples_jax/vehicle_physics_jax.py',
                'description': 'Physics simulation with automatic differentiation',
                'requires_carla': False
            },
            'lidar_camera': {
                'name': 'JAX LiDAR-Camera Projection',
                'script': 'PythonAPI/examples_jax/lidar_to_camera_jax.py',
                'description': 'Geometric transformations using JAX',
                'requires_carla': True
            }
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'carla_path': f"{os.path.expanduser('~')}/carla_simulator/CARLA_0.9.15",
            'carla_port': 2000,
            'carla_timeout': 10.0,
            'preferred_renderer': 'vulkan',
            'default_quality': 'epic',
            'auto_start_carla': True,
            'log_level': 'info'
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"{Colors.WARNING}Warning: Could not load config file: {e}{Colors.ENDC}")
        
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"{Colors.FAIL}Error saving config: {e}{Colors.ENDC}")
    
    def load_training_status(self) -> Dict:
        """Load training status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_training_status(self, status: Dict):
        """Save training status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"{Colors.FAIL}Error saving training status: {e}{Colors.ENDC}")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\n{Colors.WARNING}Received interrupt signal. Cleaning up...{Colors.ENDC}")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources"""
        if self.carla_process and self.carla_process.poll() is None:
            print(f"{Colors.WARNING}Stopping CARLA server...{Colors.ENDC}")
            self.carla_process.terminate()
            try:
                self.carla_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.carla_process.kill()
    
    def is_carla_running(self) -> bool:
        """Check if CARLA server is running"""
        try:
            result = subprocess.run(['nc', '-z', 'localhost', str(self.config['carla_port'])], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def start_carla(self, quality: str = None, renderer: str = None) -> bool:
        """Start CARLA server"""
        if self.is_carla_running():
            print(f"{Colors.OKGREEN}CARLA server is already running{Colors.ENDC}")
            return True
        
        carla_path = Path(self.config['carla_path'])
        carla_executable = carla_path / "CarlaUE4.sh"
        
        if not carla_executable.exists():
            print(f"{Colors.FAIL}CARLA executable not found at: {carla_executable}{Colors.ENDC}")
            print(f"Please update the carla_path in config or run: ./install_carla.sh")
            return False
        
        print(f"{Colors.OKBLUE}Starting CARLA server...{Colors.ENDC}")
        
        # Build CARLA command
        cmd = [str(carla_executable)]
        
        # Add renderer option
        if renderer or self.config['preferred_renderer'] == 'vulkan':
            cmd.append('-vulkan')
        
        # Add quality option
        quality = quality or self.config['default_quality']
        if quality.lower() != 'epic':
            cmd.extend(['-quality-level', quality.title()])
        
        # Add port option
        cmd.extend(['-carla-rpc-port', str(self.config['carla_port'])])
        
        try:
            # Create log file
            log_file = self.logs_dir / f"carla_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                self.carla_process = subprocess.Popen(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT, 
                    cwd=carla_path,
                    preexec_fn=os.setsid
                )
            
            # Wait for server to start
            print(f"{Colors.WARNING}Waiting for CARLA server to start...{Colors.ENDC}")
            for i in range(30):  # Wait up to 30 seconds
                if self.is_carla_running():
                    print(f"{Colors.OKGREEN}CARLA server started successfully!{Colors.ENDC}")
                    return True
                time.sleep(1)
                print(".", end="", flush=True)
            
            print(f"\n{Colors.FAIL}CARLA server failed to start within 30 seconds{Colors.ENDC}")
            return False
            
        except Exception as e:
            print(f"{Colors.FAIL}Error starting CARLA: {e}{Colors.ENDC}")
            return False
    
    def stop_carla(self) -> bool:
        """Stop CARLA server"""
        if self.carla_process and self.carla_process.poll() is None:
            print(f"{Colors.WARNING}Stopping CARLA server...{Colors.ENDC}")
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.carla_process.pid), signal.SIGTERM)
                self.carla_process.wait(timeout=10)
                print(f"{Colors.OKGREEN}CARLA server stopped{Colors.ENDC}")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                os.killpg(os.getpgid(self.carla_process.pid), signal.SIGKILL)
                print(f"{Colors.WARNING}CARLA server force-killed{Colors.ENDC}")
                return True
            except Exception as e:
                print(f"{Colors.FAIL}Error stopping CARLA: {e}{Colors.ENDC}")
                return False
        else:
            # Try to kill any running CARLA processes
            try:
                subprocess.run(['pkill', '-f', 'CarlaUE4'], check=False)
                print(f"{Colors.OKGREEN}Any running CARLA processes stopped{Colors.ENDC}")
                return True
            except Exception:
                print(f"{Colors.WARNING}No CARLA processes found to stop{Colors.ENDC}")
                return True
    
    def get_training_status(self) -> Dict:
        """Get training status for all algorithms"""
        status = self.load_training_status()
        
        for alg_name in self.algorithms.keys():
            if alg_name not in status:
                status[alg_name] = {
                    'trained': False,
                    'last_training': None,
                    'episodes': 0,
                    'best_reward': None,
                    'model_path': None
                }
            
            # Check if model file exists
            model_file = self.models_dir / f"{alg_name.lower()}_model.pkl"
            if model_file.exists():
                status[alg_name]['trained'] = True
                status[alg_name]['model_path'] = str(model_file)
        
        return status
    
    def update_training_status(self, algorithm: str, episodes: int, reward: float = None):
        """Update training status for an algorithm"""
        status = self.get_training_status()
        
        status[algorithm]['trained'] = True
        status[algorithm]['last_training'] = datetime.now().isoformat()
        status[algorithm]['episodes'] = episodes
        
        if reward is not None:
            if status[algorithm]['best_reward'] is None or reward > status[algorithm]['best_reward']:
                status[algorithm]['best_reward'] = reward
        
        self.save_training_status(status)
    
    def run_script(self, script_path: str, args: List[str] = None, requires_carla: bool = False) -> bool:
        """Run a script with optional arguments"""
        if requires_carla and not self.is_carla_running():
            if self.config['auto_start_carla']:
                print(f"{Colors.WARNING}Script requires CARLA. Starting CARLA server...{Colors.ENDC}")
                if not self.start_carla():
                    return False
            else:
                print(f"{Colors.FAIL}Script requires CARLA server, but auto-start is disabled{Colors.ENDC}")
                return False
        
        full_path = self.base_dir / script_path
        if not full_path.exists():
            print(f"{Colors.FAIL}Script not found: {full_path}{Colors.ENDC}")
            return False
        
        # Use Python command from launcher if available
        python_cmd = os.environ.get('CARLA_JAX_PYTHON', sys.executable)
        
        # Build command
        cmd = [python_cmd, str(full_path)]
        if args:
            cmd.extend(args)
        
        print(f"{Colors.OKBLUE}Running: {' '.join(cmd)}{Colors.ENDC}")
        
        try:
            # Run script interactively
            result = subprocess.run(cmd, cwd=self.base_dir)
            return result.returncode == 0
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Script interrupted by user{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}Error running script: {e}{Colors.ENDC}")
            return False
    
    def show_status(self):
        """Show system status"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}CARLA JAX System Status{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        # CARLA status
        carla_status = "🟢 Running" if self.is_carla_running() else "🔴 Stopped"
        print(f"\n{Colors.BOLD}CARLA Server:{Colors.ENDC} {carla_status}")
        print(f"  Port: {self.config['carla_port']}")
        print(f"  Path: {self.config['carla_path']}")
        
        # Training status
        print(f"\n{Colors.BOLD}Training Status:{Colors.ENDC}")
        status = self.get_training_status()
        
        for alg_name, alg_info in self.algorithms.items():
            alg_status = status.get(alg_name, {})
            trained = alg_status.get('trained', False)
            episodes = alg_status.get('episodes', 0)
            last_training = alg_status.get('last_training')
            best_reward = alg_status.get('best_reward')
            
            status_icon = "✅" if trained else "❌"
            print(f"  {status_icon} {alg_name} ({alg_info['name']})")
            
            if trained:
                print(f"      Episodes: {episodes}")
                if best_reward is not None:
                    print(f"      Best Reward: {best_reward:.2f}")
                if last_training:
                    training_date = datetime.fromisoformat(last_training).strftime('%Y-%m-%d %H:%M')
                    print(f"      Last Training: {training_date}")
        
        # JAX examples
        print(f"\n{Colors.BOLD}Available JAX Examples:{Colors.ENDC}")
        for key, example in self.jax_examples.items():
            carla_req = "🔗 CARLA" if example['requires_carla'] else "🚀 Standalone"
            print(f"  • {example['name']} ({carla_req})")
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}CARLA JAX CLI - Main Menu{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
            
            # Show CARLA status
            carla_status = "🟢 Running" if self.is_carla_running() else "🔴 Stopped"
            print(f"\nCARLA Status: {carla_status}")
            
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print(f"  {Colors.OKCYAN}1.{Colors.ENDC} Show System Status")
            print(f"  {Colors.OKCYAN}2.{Colors.ENDC} CARLA Management")
            print(f"  {Colors.OKCYAN}3.{Colors.ENDC} Run JAX Examples")
            print(f"  {Colors.OKCYAN}4.{Colors.ENDC} Train RL Algorithms")
            print(f"  {Colors.OKCYAN}5.{Colors.ENDC} Configuration")
            print(f"  {Colors.OKCYAN}6.{Colors.ENDC} View Logs")
            print(f"  {Colors.OKCYAN}q.{Colors.ENDC} Quit")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
            
            if choice == '1':
                self.show_status()
            elif choice == '2':
                self.carla_menu()
            elif choice == '3':
                self.jax_examples_menu()
            elif choice == '4':
                self.training_menu()
            elif choice == '5':
                self.config_menu()
            elif choice == '6':
                self.logs_menu()
            elif choice in ['q', 'quit', 'exit']:
                print(f"{Colors.OKGREEN}Goodbye!{Colors.ENDC}")
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def carla_menu(self):
        """CARLA management menu"""
        while True:
            print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}CARLA Management{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
            
            carla_status = "🟢 Running" if self.is_carla_running() else "🔴 Stopped"
            print(f"\nCurrent Status: {carla_status}")
            
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print(f"  {Colors.OKCYAN}1.{Colors.ENDC} Start CARLA Server")
            print(f"  {Colors.OKCYAN}2.{Colors.ENDC} Stop CARLA Server")
            print(f"  {Colors.OKCYAN}3.{Colors.ENDC} Restart CARLA Server")
            print(f"  {Colors.OKCYAN}4.{Colors.ENDC} Test CARLA Connection")
            print(f"  {Colors.OKCYAN}b.{Colors.ENDC} Back to Main Menu")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
            
            if choice == '1':
                self.start_carla()
            elif choice == '2':
                self.stop_carla()
            elif choice == '3':
                self.stop_carla()
                time.sleep(2)
                self.start_carla()
            elif choice == '4':
                self.test_carla_connection()
            elif choice in ['b', 'back']:
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def test_carla_connection(self):
        """Test CARLA connection"""
        print(f"{Colors.OKBLUE}Testing CARLA connection...{Colors.ENDC}")
        
        if not self.is_carla_running():
            print(f"{Colors.FAIL}CARLA server is not running{Colors.ENDC}")
            return
        
        # Use Python command from launcher if available
        python_cmd = os.environ.get('CARLA_JAX_PYTHON', sys.executable)
        
        try:
            # Check if carla module is available first
            check_cmd = [python_cmd, "-c", "import carla"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"{Colors.FAIL}CARLA Python module not available{Colors.ENDC}")
                print(f"{Colors.WARNING}Error: {result.stderr}{Colors.ENDC}")
                return
            
            # Try to connect with Python client
            cmd = [python_cmd, "-c", f"""
import carla
import time

try:
    client = carla.Client('localhost', {self.config['carla_port']})
    client.set_timeout({self.config['carla_timeout']})
    
    world = client.get_world()
    maps = client.get_available_maps()
    
    print(f'✅ Connected to CARLA world: {{world.get_map().name}}')
    print(f'✅ Available maps: {{len(maps)}}')
    print('🎉 CARLA connection test successful!')
    
except Exception as e:
    print(f'❌ CARLA connection failed: {{e}}')
    exit(1)
"""]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"{Colors.FAIL}Connection test failed:{Colors.ENDC}")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"{Colors.FAIL}Connection test timed out{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error during connection test: {e}{Colors.ENDC}")
    
    def jax_examples_menu(self):
        """JAX examples menu"""
        while True:
            print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}JAX Examples{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
            
            carla_status = "🟢 Running" if self.is_carla_running() else "🔴 Stopped"
            print(f"\nCARLA Status: {carla_status}")
            
            print(f"\n{Colors.BOLD}Available Examples:{Colors.ENDC}")
            
            examples_list = list(self.jax_examples.items())
            for i, (key, example) in enumerate(examples_list, 1):
                carla_req = "🔗" if example['requires_carla'] else "🚀"
                available = "✅" if not example['requires_carla'] or self.is_carla_running() else "❌"
                print(f"  {Colors.OKCYAN}{i}.{Colors.ENDC} {carla_req} {example['name']} {available}")
                print(f"      {example['description']}")
            
            print(f"  {Colors.OKCYAN}b.{Colors.ENDC} Back to Main Menu")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
            
            if choice in ['b', 'back']:
                break
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(examples_list):
                    key, example = examples_list[choice_num - 1]
                    self.run_jax_example(key, example)
                else:
                    print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def run_jax_example(self, key: str, example: Dict):
        """Run a specific JAX example"""
        print(f"\n{Colors.OKBLUE}Running: {example['name']}{Colors.ENDC}")
        
        # Prepare arguments based on example type
        args = []
        
        if key == 'traffic_simulation':
            print("Traffic simulation options:")
            print("1. JAX-only mode (no CARLA needed)")
            print("2. CARLA integration mode")
            
            mode_choice = input("Choose mode (1-2): ").strip()
            
            if mode_choice == '1':
                args = ['--mode', 'jax-only', '--number-of-vehicles', '50', '--simulation-steps', '100']
            else:
                args = ['--mode', 'carla-integration', '--number-of-vehicles', '20']
        
        elif key == 'vehicle_physics':
            print("Physics demo modes:")
            print("1. Physics simulation")
            print("2. Trajectory optimization")
            print("3. Parameter identification")
            
            demo_choice = input("Choose mode (1-3): ").strip()
            demo_modes = {'1': 'physics', '2': 'optimization', '3': 'identification'}
            
            if demo_choice in demo_modes:
                args = ['--demo-mode', demo_modes[demo_choice]]
        
        elif key == 'automatic_control':
            if not example['requires_carla'] or self.is_carla_running():
                args = ['--jax-agent', '--sync']
            else:
                args = ['--demo-mode']
        
        success = self.run_script(example['script'], args, example['requires_carla'])
        
        if success:
            print(f"{Colors.OKGREEN}Example completed successfully!{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}Example failed or was interrupted{Colors.ENDC}")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    
    def training_menu(self):
        """Training menu"""
        while True:
            print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}RL Algorithm Training{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
            
            status = self.get_training_status()
            
            print(f"\n{Colors.BOLD}Available Algorithms:{Colors.ENDC}")
            
            algorithms_list = list(self.algorithms.items())
            for i, (alg_name, alg_info) in enumerate(algorithms_list, 1):
                alg_status = status.get(alg_name, {})
                trained = alg_status.get('trained', False)
                episodes = alg_status.get('episodes', 0)
                
                status_icon = "✅" if trained else "❌"
                episode_info = f"({episodes} episodes)" if trained else "(not trained)"
                
                print(f"  {Colors.OKCYAN}{i}.{Colors.ENDC} {status_icon} {alg_name} - {alg_info['name']} {episode_info}")
                print(f"      {alg_info['description']}")
            
            print(f"  {Colors.OKCYAN}a.{Colors.ENDC} Train All Algorithms")
            print(f"  {Colors.OKCYAN}s.{Colors.ENDC} Show Training Status")
            print(f"  {Colors.OKCYAN}b.{Colors.ENDC} Back to Main Menu")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
            
            if choice in ['b', 'back']:
                break
            elif choice == 's':
                self.show_training_status()
            elif choice == 'a':
                self.train_all_algorithms()
            else:
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(algorithms_list):
                        alg_name, alg_info = algorithms_list[choice_num - 1]
                        self.train_algorithm(alg_name, alg_info)
                    else:
                        print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
                except ValueError:
                    print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def show_training_status(self):
        """Show detailed training status"""
        status = self.get_training_status()
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Detailed Training Status{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        for alg_name, alg_info in self.algorithms.items():
            alg_status = status.get(alg_name, {})
            print(f"\n{Colors.BOLD}{alg_name} - {alg_info['name']}{Colors.ENDC}")
            print(f"  Description: {alg_info['description']}")
            
            if alg_status.get('trained', False):
                print(f"  Status: {Colors.OKGREEN}✅ Trained{Colors.ENDC}")
                print(f"  Episodes: {alg_status.get('episodes', 'Unknown')}")
                
                if alg_status.get('best_reward') is not None:
                    print(f"  Best Reward: {alg_status['best_reward']:.2f}")
                
                if alg_status.get('last_training'):
                    training_date = datetime.fromisoformat(alg_status['last_training']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  Last Training: {training_date}")
                
                if alg_status.get('model_path'):
                    print(f"  Model Path: {alg_status['model_path']}")
            else:
                print(f"  Status: {Colors.FAIL}❌ Not Trained{Colors.ENDC}")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    
    def train_algorithm(self, alg_name: str, alg_info: Dict):
        """Train a specific algorithm"""
        print(f"\n{Colors.OKBLUE}Training {alg_name} - {alg_info['name']}{Colors.ENDC}")
        
        # Get training parameters
        episodes = input(f"Number of episodes (default: 100): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 100
        
        # Check if script exists
        script_path = self.base_dir / alg_info['script']
        if not script_path.exists():
            print(f"{Colors.WARNING}Training script not found: {script_path}{Colors.ENDC}")
            print(f"This algorithm may not be fully implemented yet.")
            input(f"{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
            return
        
        # Prepare arguments
        args = ['--episodes', str(episodes), '--algorithm', alg_name.lower()]
        
        # Add model save path
        model_path = self.models_dir / f"{alg_name.lower()}_model.pkl"
        args.extend(['--save-model', str(model_path)])
        
        print(f"{Colors.WARNING}Starting training... This may take a while.{Colors.ENDC}")
        print(f"Episodes: {episodes}")
        print(f"Model will be saved to: {model_path}")
        
        success = self.run_script(alg_info['script'], args, requires_carla=True)
        
        if success:
            print(f"{Colors.OKGREEN}Training completed successfully!{Colors.ENDC}")
            # Update training status
            self.update_training_status(alg_name, episodes)
        else:
            print(f"{Colors.FAIL}Training failed or was interrupted{Colors.ENDC}")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    
    def train_all_algorithms(self):
        """Train all algorithms sequentially"""
        print(f"\n{Colors.WARNING}This will train all algorithms sequentially.{Colors.ENDC}")
        print(f"This process may take several hours to complete.")
        
        confirm = input(f"Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return
        
        episodes = input(f"Episodes per algorithm (default: 100): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 100
        
        for alg_name, alg_info in self.algorithms.items():
            print(f"\n{Colors.OKBLUE}Training {alg_name}...{Colors.ENDC}")
            
            script_path = self.base_dir / alg_info['script']
            if not script_path.exists():
                print(f"{Colors.WARNING}Skipping {alg_name} - script not found{Colors.ENDC}")
                continue
            
            args = ['--episodes', str(episodes), '--algorithm', alg_name.lower()]
            model_path = self.models_dir / f"{alg_name.lower()}_model.pkl"
            args.extend(['--save-model', str(model_path)])
            
            success = self.run_script(alg_info['script'], args, requires_carla=True)
            
            if success:
                self.update_training_status(alg_name, episodes)
                print(f"{Colors.OKGREEN}{alg_name} training completed{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}{alg_name} training failed{Colors.ENDC}")
                break
        
        print(f"\n{Colors.OKGREEN}All algorithm training completed!{Colors.ENDC}")
        input(f"{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    
    def config_menu(self):
        """Configuration menu"""
        while True:
            print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}Configuration{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
            
            print(f"\n{Colors.BOLD}Current Settings:{Colors.ENDC}")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print(f"  {Colors.OKCYAN}1.{Colors.ENDC} Edit CARLA Path")
            print(f"  {Colors.OKCYAN}2.{Colors.ENDC} Edit CARLA Port")
            print(f"  {Colors.OKCYAN}3.{Colors.ENDC} Toggle Auto-start CARLA")
            print(f"  {Colors.OKCYAN}4.{Colors.ENDC} Edit Renderer Preference")
            print(f"  {Colors.OKCYAN}5.{Colors.ENDC} Reset to Defaults")
            print(f"  {Colors.OKCYAN}s.{Colors.ENDC} Save Configuration")
            print(f"  {Colors.OKCYAN}b.{Colors.ENDC} Back to Main Menu")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
            
            if choice == '1':
                new_path = input(f"Enter CARLA path (current: {self.config['carla_path']}): ").strip()
                if new_path:
                    self.config['carla_path'] = new_path
            elif choice == '2':
                new_port = input(f"Enter CARLA port (current: {self.config['carla_port']}): ").strip()
                if new_port.isdigit():
                    self.config['carla_port'] = int(new_port)
            elif choice == '3':
                self.config['auto_start_carla'] = not self.config['auto_start_carla']
                status = "enabled" if self.config['auto_start_carla'] else "disabled"
                print(f"Auto-start CARLA: {status}")
            elif choice == '4':
                print("Renderer options: vulkan, opengl")
                new_renderer = input(f"Enter renderer (current: {self.config['preferred_renderer']}): ").strip()
                if new_renderer in ['vulkan', 'opengl']:
                    self.config['preferred_renderer'] = new_renderer
            elif choice == '5':
                confirm = input("Reset all settings to defaults? (y/N): ").strip().lower()
                if confirm == 'y':
                    self.config = self.load_config()
                    print("Configuration reset to defaults")
            elif choice == 's':
                self.save_config()
                print(f"{Colors.OKGREEN}Configuration saved{Colors.ENDC}")
            elif choice in ['b', 'back']:
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def logs_menu(self):
        """Logs menu"""
        print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Logs{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        log_files = list(self.logs_dir.glob("*.log"))
        
        if not log_files:
            print(f"No log files found in {self.logs_dir}")
            input(f"{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Available Log Files:{Colors.ENDC}")
        for i, log_file in enumerate(log_files, 1):
            file_size = log_file.stat().st_size
            modified_time = datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {i}. {log_file.name} ({file_size} bytes, {modified_time})")
        
        print(f"  {Colors.OKCYAN}a.{Colors.ENDC} View all logs")
        print(f"  {Colors.OKCYAN}c.{Colors.ENDC} Clear all logs")
        
        choice = input(f"\n{Colors.BOLD}Enter choice (number, 'a', 'c', or Enter to go back): {Colors.ENDC}").strip().lower()
        
        if choice == 'a':
            for log_file in log_files:
                print(f"\n{Colors.BOLD}=== {log_file.name} ==={Colors.ENDC}")
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        print(content[-2000:])  # Show last 2000 characters
                except Exception as e:
                    print(f"Error reading log file: {e}")
        elif choice == 'c':
            confirm = input("Delete all log files? (y/N): ").strip().lower()
            if confirm == 'y':
                for log_file in log_files:
                    log_file.unlink()
                print(f"{Colors.OKGREEN}All log files deleted{Colors.ENDC}")
        elif choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(log_files):
                log_file = log_files[choice_num - 1]
                print(f"\n{Colors.BOLD}=== {log_file.name} ==={Colors.ENDC}")
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        print(content[-5000:])  # Show last 5000 characters
                except Exception as e:
                    print(f"Error reading log file: {e}")
        
        if choice:
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")


def check_environment():
    """Check if we're in the correct environment"""
    # Check if we're in the right directory
    if not os.path.exists('carla_jax_cli.py') and not os.environ.get('CARLA_JAX_HOME'):
        print(f"{Colors.WARNING}Warning: Not in CARLA JAX directory and CARLA_JAX_HOME not set{Colors.ENDC}")
        print(f"{Colors.WARNING}Consider running: ./install_system_wide.sh{Colors.ENDC}")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"{Colors.OKBLUE}Conda environment: {Colors.OKGREEN}{conda_env}{Colors.ENDC}")
        if conda_env != 'carla-jax':
            print(f"{Colors.WARNING}Note: Recommended conda environment is 'carla-jax'{Colors.ENDC}")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor in [7, 8, 9, 10]:
        print(f"{Colors.OKBLUE}Python version: {Colors.OKGREEN}{python_version.major}.{python_version.minor}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}Python {python_version.major}.{python_version.minor} detected. Recommended: Python 3.8{Colors.ENDC}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CARLA JAX CLI - Interactive management tool")
    parser.add_argument('--start-carla', action='store_true', help='Start CARLA server and exit')
    parser.add_argument('--stop-carla', action='store_true', help='Stop CARLA server and exit')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    parser.add_argument('--test-connection', action='store_true', help='Test CARLA connection and exit')
    parser.add_argument('--no-env-check', action='store_true', help='Skip environment check')
    
    args = parser.parse_args()
    
    # Check environment unless skipped
    if not args.no_env_check:
        check_environment()
    
    # Create manager
    manager = CARLAJAXManager()
    
    try:
        # Handle command line arguments
        if args.start_carla:
            manager.start_carla()
        elif args.stop_carla:
            manager.stop_carla()
        elif args.status:
            manager.show_status()
        elif args.test_connection:
            manager.test_carla_connection()
        else:
            # Show interactive menu
            print(f"{Colors.HEADER}{Colors.BOLD}")
            print("🚀 Welcome to CARLA JAX CLI")
            print("Interactive management for CARLA simulation and JAX-accelerated examples")
            print(f"{Colors.ENDC}")
            
            manager.show_menu()
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()