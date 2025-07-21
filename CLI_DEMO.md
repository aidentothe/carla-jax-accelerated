# CARLA JAX CLI Demo

## 🚀 Quick Start

```bash
# Launch the interactive CLI
./carla-jax
```

## 📋 CLI Interface Preview

```
============================================================
                CARLA JAX CLI - Main Menu
============================================================

CARLA Status: 🔴 Stopped

Options:
  1. Show System Status
  2. CARLA Management
  3. Run JAX Examples
  4. Train RL Algorithms
  5. Configuration
  6. View Logs
  q. Quit

Enter your choice: 
```

## 📊 System Status Display

```bash
./carla-jax --status
```

```
============================================================
                CARLA JAX System Status
============================================================

CARLA Server: 🔴 Stopped
  Port: 2000
  Path: /home/user/carla_simulator/CARLA_0.9.15

Training Status:
  ❌ PPO (Proximal Policy Optimization)
  ❌ SAC (Soft Actor-Critic)
  ❌ DDPG (Deep Deterministic Policy Gradient)
  ❌ A2C (Advantage Actor-Critic)

Available JAX Examples:
  • JAX Vehicle Control (🔗 CARLA)
  • JAX Traffic Simulation (🚀 Standalone)
  • JAX Sensor Fusion (🔗 CARLA)
  • JAX Vehicle Physics (🚀 Standalone)
  • JAX LiDAR-Camera Projection (🔗 CARLA)
```

## 🎯 JAX Examples Menu

```
==================================================
                JAX Examples
==================================================

CARLA Status: 🟢 Running

Available Examples:
  1. 🚀 JAX Vehicle Control ✅
      JIT-compiled vehicle control with PID optimization
  2. 🚀 JAX Traffic Simulation ✅
      Vectorized multi-agent traffic simulation
  3. 🔗 JAX Sensor Fusion ✅
      Multi-sensor data processing with JAX
  4. 🚀 JAX Vehicle Physics ✅
      Physics simulation with automatic differentiation
  5. 🔗 JAX LiDAR-Camera Projection ✅
      Geometric transformations using JAX
  b. Back to Main Menu

Enter your choice: 
```

## 🧠 Training Menu

```
==================================================
             RL Algorithm Training
==================================================

Available Algorithms:
  1. ❌ PPO - Proximal Policy Optimization (not trained)
      Policy gradient method with clipped surrogate objective
  2. ✅ SAC - Soft Actor-Critic (150 episodes)
      Off-policy algorithm with maximum entropy objective
  3. ❌ DDPG - Deep Deterministic Policy Gradient (not trained)
      Actor-critic method for continuous control
  4. ❌ A2C - Advantage Actor-Critic (not trained)
      Synchronous actor-critic algorithm
  a. Train All Algorithms
  s. Show Training Status
  b. Back to Main Menu

Enter your choice: 
```

## 📈 Detailed Training Status

```bash
./carla-jax --status
```

After some training:

```
============================================================
             Detailed Training Status
============================================================

PPO - Proximal Policy Optimization
  Description: Policy gradient method with clipped surrogate objective
  Status: ✅ Trained
  Episodes: 200
  Best Reward: 1247.32
  Last Training: 2024-07-21 14:30:25
  Model Path: /path/to/models/ppo_model.pkl

SAC - Soft Actor-Critic
  Description: Off-policy algorithm with maximum entropy objective
  Status: ✅ Trained
  Episodes: 150
  Best Reward: 1180.45
  Last Training: 2024-07-21 13:15:10
  Model Path: /path/to/models/sac_model.pkl

DDPG - Deep Deterministic Policy Gradient
  Description: Actor-critic method for continuous control
  Status: ❌ Not Trained

A2C - Advantage Actor-Critic
  Description: Synchronous actor-critic algorithm
  Status: ❌ Not Trained
```

## ⚙️ Configuration Management

```
==================================================
                Configuration
==================================================

Current Settings:
  carla_path: /home/user/carla_simulator/CARLA_0.9.15
  carla_port: 2000
  carla_timeout: 10.0
  preferred_renderer: vulkan
  default_quality: epic
  auto_start_carla: true
  log_level: info

Options:
  1. Edit CARLA Path
  2. Edit CARLA Port
  3. Toggle Auto-start CARLA
  4. Edit Renderer Preference
  5. Reset to Defaults
  s. Save Configuration
  b. Back to Main Menu
```

## 🔧 CARLA Management

```
==================================================
                CARLA Management
==================================================

Current Status: 🟢 Running

Options:
  1. Start CARLA Server
  2. Stop CARLA Server
  3. Restart CARLA Server
  4. Test CARLA Connection
  b. Back to Main Menu

Enter your choice: 4

Testing CARLA connection...
✅ Connected to CARLA world: Town01
✅ Available maps: 13
🎉 CARLA connection test successful!
```

## 🚀 Auto-Start Features

When you select a JAX example that requires CARLA:

```
Running: JAX Vehicle Control

Script requires CARLA. Starting CARLA server...
🚀 Starting CARLA server...
⚠️  Waiting for CARLA server to start...
.....
✅ CARLA server started successfully!
🔵 Running: python PythonAPI/examples_jax/automatic_control_jax.py --jax-agent --sync
```

## 📝 Command Line Usage

```bash
# Interactive mode (default)
./carla-jax

# Quick commands
./carla-jax --start-carla      # Start CARLA and exit
./carla-jax --stop-carla       # Stop CARLA and exit
./carla-jax --status           # Show status and exit
./carla-jax --test-connection  # Test connection and exit

# Using the full Python script
python carla_jax_cli.py --help
```

## 🎯 Key Features

### 🤖 Intelligent CARLA Management
- **Auto-detection**: Checks if CARLA is running before starting scripts
- **Auto-start**: Automatically starts CARLA when needed (configurable)
- **Smart shutdown**: Graceful process management with cleanup

### 📊 Training Progress Tracking
- **Persistent status**: Remembers which algorithms have been trained
- **Performance metrics**: Tracks episodes, best rewards, training dates
- **Model management**: Automatic model file detection and paths

### 🎮 Interactive Script Running
- **Guided parameters**: Prompts for relevant options per script
- **Real-time feedback**: Shows script output and status
- **Error handling**: Graceful failure recovery and user feedback

### ⚙️ Flexible Configuration
- **Persistent settings**: Saves configuration between sessions
- **Environment adaptation**: Detects installation paths automatically
- **User preferences**: Customizable renderer, quality, and port settings

This CLI transforms the CARLA JAX repository from a collection of scripts into a unified, user-friendly research platform! 🚀