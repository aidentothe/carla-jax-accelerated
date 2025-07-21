#!/usr/bin/env python

"""JAX-accelerated traffic generation for CARLA.

This example demonstrates how to use JAX for high-performance multi-agent
traffic simulation with vectorized operations, batch processing, and
efficient collision avoidance using JIT compilation.
"""

import argparse
import glob
import logging
import os
import random
import sys
import time
from typing import List, Dict, Tuple, Optional, NamedTuple

try:
    # First try the import fix utility
    from carla_import_fix import import_carla
    carla = import_carla()
except ImportError:
    # Fallback to old method
    try:
        sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
    
    try:
        import carla
    except ImportError:
        print("Warning: CARLA module not found. Only JAX-only mode will work.")
        carla = None

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax, random as jax_random
    import optax
    import chex
    from functools import partial
except ImportError:
    raise RuntimeError('JAX not found. Install with: pip install jax jaxlib optax chex')


# ==============================================================================
# -- JAX Agent State and Behavior Structures ---------------------------------
# ==============================================================================

class AgentState(NamedTuple):
    """JAX-compatible agent state representation."""
    position: jnp.ndarray      # [x, y, z]
    velocity: jnp.ndarray      # [vx, vy, vz]
    heading: float             # yaw angle in radians
    speed: float              # current speed
    target_speed: float       # desired speed
    lane_id: int              # current lane
    behavior_type: int        # 0=aggressive, 1=normal, 2=cautious
    collision_radius: float   # collision detection radius


class TrafficFlowState(NamedTuple):
    """Multi-agent traffic flow state."""
    agents: AgentState        # Vectorized agent states
    distances: jnp.ndarray    # Distance matrix between agents
    collisions: jnp.ndarray   # Collision detection results
    lane_occupancy: jnp.ndarray  # Lane occupancy information


# ==============================================================================
# -- JAX Traffic Behavior Models ----------------------------------------------
# ==============================================================================

@jit
def intelligent_driver_model(agent_state: AgentState,
                            lead_vehicle_distance: float,
                            lead_vehicle_speed: float,
                            dt: float) -> Tuple[float, float]:
    """JAX-compiled Intelligent Driver Model for car-following behavior."""
    
    # IDM parameters based on behavior type
    # [aggressive, normal, cautious]
    max_accel = jnp.array([2.5, 2.0, 1.5])[agent_state.behavior_type]
    comfortable_decel = jnp.array([3.0, 2.5, 2.0])[agent_state.behavior_type]
    min_gap = jnp.array([1.5, 2.0, 2.5])[agent_state.behavior_type]
    time_headway = jnp.array([1.0, 1.5, 2.0])[agent_state.behavior_type]
    speed_exponent = 4.0
    
    # Current state
    v = agent_state.speed
    v_target = agent_state.target_speed
    s = lead_vehicle_distance
    v_lead = lead_vehicle_speed
    
    # Desired minimum spacing
    s_star = min_gap + jnp.maximum(0.0, 
                                   v * time_headway + 
                                   (v * (v - v_lead)) / (2 * jnp.sqrt(max_accel * comfortable_decel)))
    
    # IDM acceleration
    free_flow_accel = max_accel * (1.0 - (v / v_target) ** speed_exponent)
    
    # Interaction term (when there's a lead vehicle)
    interaction_term = jnp.where(
        s > 0.1,  # Valid lead vehicle
        max_accel * (s_star / s) ** 2,
        0.0
    )
    
    acceleration = jnp.clip(free_flow_accel - interaction_term, 
                           -comfortable_decel, max_accel)
    
    # Update speed and position
    new_speed = jnp.clip(v + acceleration * dt, 0.0, v_target * 1.2)
    distance_traveled = v * dt + 0.5 * acceleration * dt**2
    
    return new_speed, distance_traveled


@jit
def lane_change_decision(agent_state: AgentState,
                        left_lane_gap: float,
                        right_lane_gap: float,
                        current_lane_utility: float) -> int:
    """JAX-compiled lane change decision model."""
    
    # Lane change parameters based on behavior
    gap_threshold = jnp.array([15.0, 20.0, 25.0])[agent_state.behavior_type]
    utility_threshold = jnp.array([0.1, 0.2, 0.3])[agent_state.behavior_type]
    
    # Calculate lane utilities
    left_utility = jnp.where(left_lane_gap > gap_threshold, 
                            current_lane_utility + 0.2, -1.0)
    right_utility = jnp.where(right_lane_gap > gap_threshold,
                             current_lane_utility + 0.1, -1.0)
    
    # Decision logic: -1=left, 0=stay, 1=right
    best_utility = jnp.maximum(jnp.maximum(left_utility, right_utility), 
                              current_lane_utility)
    
    lane_change = jnp.where(
        best_utility > current_lane_utility + utility_threshold,
        jnp.where(left_utility == best_utility, -1,
                 jnp.where(right_utility == best_utility, 1, 0)),
        0
    )
    
    return lane_change


@jit
def collision_avoidance_force(agent_pos: jnp.ndarray,
                             other_agents_pos: jnp.ndarray,
                             collision_radius: float) -> jnp.ndarray:
    """Compute collision avoidance forces using social force model."""
    
    # Calculate distances to all other agents
    diff_vectors = other_agents_pos - agent_pos[None, :]  # (N, 2)
    distances = jnp.linalg.norm(diff_vectors, axis=1)
    
    # Avoid division by zero
    safe_distances = jnp.where(distances < 1e-3, 1e-3, distances)
    unit_vectors = diff_vectors / safe_distances[:, None]
    
    # Exponential repulsion force
    force_magnitude = jnp.exp(-(safe_distances - collision_radius) / 2.0)
    
    # Sum forces from all nearby agents
    forces = -force_magnitude[:, None] * unit_vectors
    total_force = jnp.sum(forces, axis=0)
    
    return total_force


# ==============================================================================
# -- Vectorized Multi-Agent Operations ----------------------------------------
# ==============================================================================

@jit
def compute_agent_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise distances between all agents."""
    # positions shape: (N, 3)
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # (N, N)
    return distances


@jit
def detect_collisions(distances: jnp.ndarray, 
                     collision_radii: jnp.ndarray) -> jnp.ndarray:
    """Detect collisions between agents."""
    # Create collision threshold matrix
    threshold_matrix = collision_radii[:, None] + collision_radii[None, :]
    
    # Detect collisions (excluding self-collisions)
    collisions = (distances < threshold_matrix) & (distances > 0.0)
    
    return collisions


@jit
def update_agent_batch(agent_states: AgentState,
                      lead_distances: jnp.ndarray,
                      lead_speeds: jnp.ndarray,
                      collision_forces: jnp.ndarray,
                      dt: float) -> AgentState:
    """Vectorized update of all agents using IDM and collision avoidance."""
    
    # Vectorized IDM updates
    vectorized_idm = vmap(intelligent_driver_model, in_axes=(0, 0, 0, None))
    new_speeds, distances_traveled = vectorized_idm(
        agent_states, lead_distances, lead_speeds, dt
    )
    
    # Update positions with collision avoidance
    # Current heading vectors
    heading_vectors = jnp.stack([
        jnp.cos(agent_states.heading),
        jnp.sin(agent_states.heading),
        jnp.zeros_like(agent_states.heading)
    ], axis=1)
    
    # Nominal movement
    nominal_movement = heading_vectors * distances_traveled[:, None]
    
    # Add collision avoidance forces (scaled)
    avoidance_movement = collision_forces * 0.1 * dt
    
    # Update positions
    new_positions = agent_states.position + nominal_movement + avoidance_movement
    
    # Update velocities
    new_velocities = heading_vectors * new_speeds[:, None]
    
    return AgentState(
        position=new_positions,
        velocity=new_velocities,
        heading=agent_states.heading,
        speed=new_speeds,
        target_speed=agent_states.target_speed,
        lane_id=agent_states.lane_id,
        behavior_type=agent_states.behavior_type,
        collision_radius=agent_states.collision_radius
    )


@jit
def simulate_traffic_step(agent_states: AgentState, dt: float) -> Tuple[AgentState, TrafficFlowState]:
    """Complete traffic simulation step with all interactions."""
    
    # Compute distances between all agents
    distances = compute_agent_distances(agent_states.position)
    
    # Detect collisions
    collisions = detect_collisions(distances, agent_states.collision_radius)
    
    # Find lead vehicles for each agent (simplified: closest in front)
    # This is a simplified version - real implementation would consider lanes
    lead_distances = jnp.min(jnp.where(distances > 0, distances, 1000.0), axis=1)
    lead_indices = jnp.argmin(jnp.where(distances > 0, distances, 1000.0), axis=1)
    lead_speeds = agent_states.speed[lead_indices]
    
    # Compute collision avoidance forces for each agent
    def compute_forces_for_agent(i):
        other_positions = agent_states.position  # All positions
        return collision_avoidance_force(
            agent_states.position[i], other_positions, agent_states.collision_radius[i]
        )
    
    collision_forces = vmap(compute_forces_for_agent)(jnp.arange(len(agent_states.position)))
    
    # Update all agents
    new_agent_states = update_agent_batch(
        agent_states, lead_distances, lead_speeds, collision_forces, dt
    )
    
    # Create traffic flow state
    traffic_state = TrafficFlowState(
        agents=new_agent_states,
        distances=distances,
        collisions=collisions,
        lane_occupancy=jnp.zeros(10)  # Placeholder for lane occupancy
    )
    
    return new_agent_states, traffic_state


# ==============================================================================
# -- JAX Traffic Manager ------------------------------------------------------
# ==============================================================================

class JAXTrafficManager:
    """JAX-accelerated traffic manager for efficient multi-agent simulation."""
    
    def __init__(self, max_agents: int = 200, dt: float = 0.1):
        self.max_agents = max_agents
        self.dt = dt
        self.simulation_step = 0
        
        # Performance monitoring
        self.step_times = []
        self.collision_counts = []
        
        # Pre-compile JAX functions
        self.simulate_step = jit(simulate_traffic_step)
        self.compute_distances = jit(compute_agent_distances)
        self.detect_collisions = jit(detect_collisions)
        
        print(f"JAX Traffic Manager initialized:")
        print(f"  Max agents: {max_agents}")
        print(f"  Time step: {dt}s")
        print(f"  JAX devices: {jax.devices()}")
        
        # Initialize random state
        self.rng_key = jax_random.PRNGKey(42)
    
    def create_random_agents(self, 
                           num_agents: int,
                           spawn_area: Tuple[float, float, float, float],
                           speed_range: Tuple[float, float] = (5.0, 15.0)) -> AgentState:
        """Create random agents with JAX."""
        
        self.rng_key, *subkeys = jax_random.split(self.rng_key, 8)
        
        # Random positions in spawn area
        x_min, x_max, y_min, y_max = spawn_area
        positions = jnp.stack([
            jax_random.uniform(subkeys[0], (num_agents,), minval=x_min, maxval=x_max),
            jax_random.uniform(subkeys[1], (num_agents,), minval=y_min, maxval=y_max),
            jnp.zeros(num_agents)  # z = 0
        ], axis=1)
        
        # Random velocities and headings
        headings = jax_random.uniform(subkeys[2], (num_agents,), minval=0, maxval=2*jnp.pi)
        speeds = jax_random.uniform(subkeys[3], (num_agents,), 
                                   minval=speed_range[0], maxval=speed_range[1])
        target_speeds = jax_random.uniform(subkeys[4], (num_agents,),
                                          minval=speed_range[0], maxval=speed_range[1] * 1.2)
        
        # Random behavior types (0=aggressive, 1=normal, 2=cautious)
        behavior_types = jax_random.randint(subkeys[5], (num_agents,), 0, 3)
        
        # Lane assignments
        lane_ids = jax_random.randint(subkeys[6], (num_agents,), 0, 4)
        
        # Velocity vectors
        velocities = jnp.stack([
            speeds * jnp.cos(headings),
            speeds * jnp.sin(headings),
            jnp.zeros(num_agents)
        ], axis=1)
        
        return AgentState(
            position=positions,
            velocity=velocities,
            heading=headings,
            speed=speeds,
            target_speed=target_speeds,
            lane_id=lane_ids,
            behavior_type=behavior_types,
            collision_radius=jnp.full(num_agents, 2.0)  # 2m collision radius
        )
    
    def simulate_traffic(self, 
                        agent_states: AgentState,
                        num_steps: int = 1000) -> Tuple[List[AgentState], List[TrafficFlowState]]:
        """Simulate traffic for multiple time steps."""
        
        print(f"Starting JAX traffic simulation for {num_steps} steps...")
        
        agent_history = [agent_states]
        traffic_history = []
        
        current_agents = agent_states
        
        for step in range(num_steps):
            start_time = time.time()
            
            # Simulate one step
            new_agents, traffic_state = self.simulate_step(current_agents, self.dt)
            
            step_time = time.time() - start_time
            self.step_times.append(step_time)
            
            # Count collisions
            num_collisions = jnp.sum(traffic_state.collisions) // 2  # Avoid double counting
            self.collision_counts.append(int(num_collisions))
            
            # Store results
            agent_history.append(new_agents)
            traffic_history.append(traffic_state)
            
            current_agents = new_agents
            self.simulation_step += 1
            
            # Progress reporting
            if step % 100 == 0:
                avg_time = np.mean(self.step_times[-100:]) * 1000
                total_collisions = sum(self.collision_counts[-100:])
                print(f"Step {step}: {avg_time:.2f}ms/step, "
                      f"{total_collisions} collisions in last 100 steps")
        
        return agent_history, traffic_history
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get simulation performance statistics."""
        if not self.step_times:
            return {}
        
        step_times_ms = np.array(self.step_times) * 1000
        
        return {
            'avg_step_time_ms': np.mean(step_times_ms),
            'max_step_time_ms': np.max(step_times_ms),
            'min_step_time_ms': np.min(step_times_ms),
            'std_step_time_ms': np.std(step_times_ms),
            'simulation_fps': 1000.0 / np.mean(step_times_ms),
            'total_collisions': sum(self.collision_counts),
            'avg_collisions_per_step': np.mean(self.collision_counts) if self.collision_counts else 0
        }


# ==============================================================================
# -- CARLA Integration --------------------------------------------------------
# ==============================================================================

class JAXCARLATrafficSpawner:
    """Spawn and control CARLA vehicles using JAX-computed behaviors."""
    
    def __init__(self, world: carla.World, traffic_manager: JAXTrafficManager):
        self.world = world
        self.traffic_manager = traffic_manager
        self.vehicles = []
        self.vehicle_blueprints = []
        
        # Get vehicle blueprints
        blueprint_library = world.get_blueprint_library()
        self.vehicle_blueprints = blueprint_library.filter('vehicle.*')
        
        print(f"CARLA-JAX integration ready with {len(self.vehicle_blueprints)} vehicle types")
    
    def spawn_jax_controlled_traffic(self, 
                                   agent_states: AgentState,
                                   spawn_points: List[carla.Transform]) -> List[carla.Vehicle]:
        """Spawn CARLA vehicles at positions specified by JAX agent states."""
        
        num_agents = len(agent_states.position)
        available_spawn_points = spawn_points[:num_agents]
        
        print(f"Spawning {num_agents} JAX-controlled vehicles...")
        
        for i in range(num_agents):
            try:
                # Choose random vehicle blueprint
                blueprint = random.choice(self.vehicle_blueprints)
                
                # Use spawn point if available, otherwise use JAX position
                if i < len(available_spawn_points):
                    spawn_transform = available_spawn_points[i]
                else:
                    # Convert JAX position to CARLA transform
                    jax_pos = agent_states.position[i]
                    jax_heading = agent_states.heading[i]
                    
                    spawn_transform = carla.Transform(
                        carla.Location(x=float(jax_pos[0]), 
                                     y=float(jax_pos[1]), 
                                     z=float(jax_pos[2]) + 0.5),
                        carla.Rotation(yaw=float(np.degrees(jax_heading)))
                    )
                
                # Spawn vehicle
                vehicle = self.world.try_spawn_actor(blueprint, spawn_transform)
                
                if vehicle is not None:
                    self.vehicles.append(vehicle)
                    
                    # Set initial velocity
                    jax_vel = agent_states.velocity[i]
                    initial_velocity = carla.Vector3D(
                        x=float(jax_vel[0]),
                        y=float(jax_vel[1]),
                        z=float(jax_vel[2])
                    )
                    vehicle.set_target_velocity(initial_velocity)
                
            except Exception as e:
                print(f"Failed to spawn vehicle {i}: {e}")
        
        print(f"Successfully spawned {len(self.vehicles)} vehicles")
        return self.vehicles
    
    def update_carla_vehicles(self, agent_states: AgentState):
        """Update CARLA vehicle positions and velocities from JAX simulation."""
        
        for i, vehicle in enumerate(self.vehicles):
            if i >= len(agent_states.position):
                break
            
            try:
                # Get JAX state
                jax_pos = agent_states.position[i]
                jax_vel = agent_states.velocity[i]
                jax_heading = agent_states.heading[i]
                
                # Update vehicle transform
                new_transform = carla.Transform(
                    carla.Location(x=float(jax_pos[0]), 
                                 y=float(jax_pos[1]), 
                                 z=float(jax_pos[2]) + 0.5),
                    carla.Rotation(yaw=float(np.degrees(jax_heading)))
                )
                
                vehicle.set_transform(new_transform)
                
                # Update velocity
                new_velocity = carla.Vector3D(
                    x=float(jax_vel[0]),
                    y=float(jax_vel[1]),
                    z=float(jax_vel[2])
                )
                vehicle.set_target_velocity(new_velocity)
                
            except Exception as e:
                print(f"Failed to update vehicle {i}: {e}")
    
    def cleanup(self):
        """Destroy all spawned vehicles."""
        print(f"Cleaning up {len(self.vehicles)} vehicles...")
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        self.vehicles.clear()


# ==============================================================================
# -- Main Demonstration -------------------------------------------------------
# ==============================================================================

def main():
    """Main function demonstrating JAX-accelerated traffic generation."""
    
    argparser = argparse.ArgumentParser(description='JAX Traffic Generation Demo')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of CARLA server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port')
    argparser.add_argument('-n', '--number-of-vehicles', default=50, type=int,
                          help='Number of vehicles to spawn')
    argparser.add_argument('--sync', action='store_true', help='Synchronous mode')
    argparser.add_argument('--simulation-steps', default=1000, type=int,
                          help='Number of simulation steps')
    argparser.add_argument('--mode', default='jax-only', 
                          choices=['jax-only', 'carla-integration'],
                          help='Simulation mode')
    
    args = argparser.parse_args()
    
    print("JAX Traffic Generation Demonstration")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'jax-only':
        run_jax_only_demo(args)
    elif args.mode == 'carla-integration':
        if carla is None:
            print("\n❌ Error: CARLA module is required for integration mode.")
            print("Please install CARLA with: pip install carla")
            print("Or use: --mode jax-only")
            return
        run_carla_integration_demo(args)


def run_jax_only_demo(args):
    """Run pure JAX traffic simulation without CARLA visualization."""
    
    print("Running JAX-only traffic simulation...")
    
    # Create traffic manager
    traffic_manager = JAXTrafficManager(max_agents=args.number_of_vehicles)
    
    # Create random agents
    spawn_area = (-100.0, 100.0, -100.0, 100.0)  # x_min, x_max, y_min, y_max
    agent_states = traffic_manager.create_random_agents(
        args.number_of_vehicles, spawn_area, speed_range=(5.0, 20.0)
    )
    
    print(f"Created {args.number_of_vehicles} agents")
    print(f"Initial agent positions shape: {agent_states.position.shape}")
    
    # Run simulation
    start_time = time.time()
    agent_history, traffic_history = traffic_manager.simulate_traffic(
        agent_states, args.simulation_steps
    )
    total_time = time.time() - start_time
    
    # Performance analysis
    stats = traffic_manager.get_performance_stats()
    
    print(f"\nSimulation completed in {total_time:.2f} seconds")
    print(f"Performance Statistics:")
    print(f"  Average step time: {stats['avg_step_time_ms']:.2f}ms")
    print(f"  Simulation FPS: {stats['simulation_fps']:.1f}")
    print(f"  Total collisions: {stats['total_collisions']}")
    print(f"  Average collisions per step: {stats['avg_collisions_per_step']:.3f}")
    
    # Analyze final state
    final_agents = agent_history[-1]
    final_distances = traffic_manager.compute_distances(final_agents.position)
    min_distance = jnp.min(jnp.where(final_distances > 0, final_distances, 1000.0))
    
    print(f"Final state:")
    print(f"  Minimum distance between agents: {min_distance:.2f}m")
    print(f"  Speed range: {jnp.min(final_agents.speed):.1f} - {jnp.max(final_agents.speed):.1f} m/s")


def run_carla_integration_demo(args):
    """Run JAX simulation with CARLA visualization."""
    
    print("Running JAX traffic simulation with CARLA integration...")
    
    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Setup synchronous mode if requested
    original_settings = world.get_settings()
    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
    
    traffic_manager_carla = client.get_trafficmanager()
    traffic_manager_carla.set_synchronous_mode(args.sync)
    
    # Create JAX traffic manager
    jax_traffic_manager = JAXTrafficManager(max_agents=args.number_of_vehicles, dt=0.1)
    
    # Create CARLA-JAX spawner
    spawner = JAXCARLATrafficSpawner(world, jax_traffic_manager)
    
    try:
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        
        if len(spawn_points) < args.number_of_vehicles:
            print(f"Warning: Only {len(spawn_points)} spawn points available, "
                  f"requested {args.number_of_vehicles} vehicles")
            args.number_of_vehicles = len(spawn_points)
        
        # Create JAX agents near spawn points
        spawn_positions = []
        for i in range(args.number_of_vehicles):
            sp = spawn_points[i].location
            spawn_positions.append([sp.x, sp.y, sp.z])
        
        spawn_positions = jnp.array(spawn_positions)
        
        # Create initial agent states
        initial_agents = jax_traffic_manager.create_random_agents(
            args.number_of_vehicles, 
            (-200, 200, -200, 200),  # Large spawn area
            speed_range=(8.0, 15.0)
        )
        
        # Override positions with spawn points
        initial_agents = initial_agents._replace(position=spawn_positions)
        
        # Spawn CARLA vehicles
        vehicles = spawner.spawn_jax_controlled_traffic(initial_agents, spawn_points)
        
        print(f"Running {args.simulation_steps} simulation steps...")
        
        current_agents = initial_agents
        
        for step in range(args.simulation_steps):
            if args.sync:
                world.tick()
            
            # Run JAX simulation step
            start_time = time.time()
            new_agents, traffic_state = jax_traffic_manager.simulate_step(current_agents, 0.1)
            jax_time = time.time() - start_time
            
            # Update CARLA vehicles
            spawner.update_carla_vehicles(new_agents)
            
            current_agents = new_agents
            
            # Progress reporting
            if step % 50 == 0:
                num_collisions = jnp.sum(traffic_state.collisions) // 2
                print(f"Step {step}: JAX time {jax_time*1000:.2f}ms, "
                      f"{num_collisions} collisions detected")
            
            # Small delay for visualization
            time.sleep(0.05)
        
        print("Simulation completed!")
        
        # Final statistics
        stats = jax_traffic_manager.get_performance_stats()
        print(f"Performance Summary:")
        print(f"  Average JAX step time: {stats['avg_step_time_ms']:.2f}ms")
        print(f"  JAX simulation FPS: {stats['simulation_fps']:.1f}")
        print(f"  Total collisions: {stats['total_collisions']}")
    
    finally:
        # Cleanup
        spawner.cleanup()
        world.apply_settings(original_settings)
        print("Cleanup completed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n - Exited by user.')
    except Exception as e:
        print(f'\n - Error: {e}')
        import traceback
        traceback.print_exc()