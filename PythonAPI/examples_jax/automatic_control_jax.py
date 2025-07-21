#!/usr/bin/env python

"""JAX-accelerated automatic vehicle control for CARLA.

This example demonstrates how to use JAX for high-performance autonomous
driving with JIT compilation, vectorization, and automatic differentiation.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import re
import sys
import time
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    import optax
except ImportError:
    raise RuntimeError('cannot import JAX, make sure JAX is installed (pip install jax)')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

sys.path.append('../')
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent


# ==============================================================================
# -- JAX Vehicle Controller ---------------------------------------------------
# ==============================================================================

class JAXVehicleController:
    """JAX-accelerated vehicle controller with JIT compilation."""
    
    def __init__(self, vehicle, target_speed=30.0, dt=0.05):
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.dt = dt
        
        # PID parameters as JAX arrays
        self.pid_params = {
            'speed': jnp.array([1.0, 0.1, 0.01]),  # P, I, D
            'steer': jnp.array([1.0, 0.0, 0.1])
        }
        
        # Initialize state
        self.prev_speed_error = 0.0
        self.speed_integral = 0.0
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        
        # Compile control function
        self.control_step = jit(self._control_step)
        
    @staticmethod
    @jit
    def _control_step(vehicle_state, target_state, pid_params, prev_errors, integrals, dt):
        """JIT-compiled control step computation."""
        # Extract vehicle state
        current_speed = vehicle_state[0]
        current_heading = vehicle_state[1]
        
        # Extract target state
        target_speed = target_state[0]
        target_heading = target_state[1]
        
        # Speed control (longitudinal)
        speed_error = target_speed - current_speed
        speed_integral_new = integrals[0] + speed_error * dt
        speed_derivative = (speed_error - prev_errors[0]) / dt
        
        throttle_brake = jnp.dot(pid_params['speed'], 
                                jnp.array([speed_error, speed_integral_new, speed_derivative]))
        
        # Steering control (lateral)
        steer_error = target_heading - current_heading
        # Normalize angle to [-pi, pi]
        steer_error = jnp.arctan2(jnp.sin(steer_error), jnp.cos(steer_error))
        
        steer_integral_new = integrals[1] + steer_error * dt
        steer_derivative = (steer_error - prev_errors[1]) / dt
        
        steering = jnp.dot(pid_params['steer'],
                          jnp.array([steer_error, steer_integral_new, steer_derivative]))
        
        # Clamp outputs
        throttle = jnp.clip(throttle_brake, 0.0, 1.0)
        brake = jnp.clip(-throttle_brake, 0.0, 1.0)
        steering = jnp.clip(steering, -1.0, 1.0)
        
        # Return control and updated state
        control = jnp.array([throttle, brake, steering])
        new_prev_errors = jnp.array([speed_error, steer_error])
        new_integrals = jnp.array([speed_integral_new, steer_integral_new])
        
        return control, new_prev_errors, new_integrals
    
    def get_control(self, target_speed, target_heading):
        """Get vehicle control using JAX-accelerated computation."""
        # Get current vehicle state
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        
        transform = self.vehicle.get_transform()
        current_heading = math.radians(transform.rotation.yaw)
        
        # Prepare JAX arrays
        vehicle_state = jnp.array([current_speed, current_heading])
        target_state = jnp.array([target_speed, target_heading])
        prev_errors = jnp.array([self.prev_speed_error, self.prev_steer_error])
        integrals = jnp.array([self.speed_integral, self.steer_integral])
        
        # Compute control using JAX
        control, new_prev_errors, new_integrals = self.control_step(
            vehicle_state, target_state, self.pid_params, prev_errors, integrals, self.dt
        )
        
        # Update state
        self.prev_speed_error = float(new_prev_errors[0])
        self.prev_steer_error = float(new_prev_errors[1])
        self.speed_integral = float(new_integrals[0])
        self.steer_integral = float(new_integrals[1])
        
        # Create CARLA control
        carla_control = carla.VehicleControl()
        carla_control.throttle = float(control[0])
        carla_control.brake = float(control[1])
        carla_control.steer = float(control[2])
        
        return carla_control


# ==============================================================================
# -- JAX Multi-Agent Controller -----------------------------------------------
# ==============================================================================

class JAXMultiAgentController:
    """Vectorized multi-agent controller using JAX vmap."""
    
    def __init__(self, vehicles, target_speed=30.0, dt=0.05):
        self.vehicles = vehicles
        self.num_vehicles = len(vehicles)
        self.target_speed = target_speed
        self.dt = dt
        
        # Vectorized PID parameters
        self.pid_params = {
            'speed': jnp.tile(jnp.array([1.0, 0.1, 0.01]), (self.num_vehicles, 1)),
            'steer': jnp.tile(jnp.array([1.0, 0.0, 0.1]), (self.num_vehicles, 1))
        }
        
        # Initialize states
        self.prev_errors = jnp.zeros((self.num_vehicles, 2))
        self.integrals = jnp.zeros((self.num_vehicles, 2))
        
        # Vectorized control function
        self.batch_control = jit(vmap(JAXVehicleController._control_step, 
                                     in_axes=(0, 0, 0, 0, 0, None)))
    
    def get_batch_controls(self, target_speeds, target_headings):
        """Get controls for all vehicles using vectorized JAX computation."""
        # Collect vehicle states
        vehicle_states = []
        for vehicle in self.vehicles:
            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            transform = vehicle.get_transform()
            current_heading = math.radians(transform.rotation.yaw)
            
            vehicle_states.append([current_speed, current_heading])
        
        vehicle_states = jnp.array(vehicle_states)
        target_states = jnp.array([[speed, heading] for speed, heading in 
                                  zip(target_speeds, target_headings)])
        
        # Vectorized control computation
        controls, new_prev_errors, new_integrals = self.batch_control(
            vehicle_states, target_states, self.pid_params, 
            self.prev_errors, self.integrals, self.dt
        )
        
        # Update states
        self.prev_errors = new_prev_errors
        self.integrals = new_integrals
        
        # Convert to CARLA controls
        carla_controls = []
        for i in range(self.num_vehicles):
            control = carla.VehicleControl()
            control.throttle = float(controls[i, 0])
            control.brake = float(controls[i, 1])
            control.steer = float(controls[i, 2])
            carla_controls.append(control)
        
        return carla_controls


# ==============================================================================
# -- JAX Enhanced Agent -------------------------------------------------------
# ==============================================================================

class JAXEnhancedAgent:
    """Agent with JAX-accelerated perception and control."""
    
    def __init__(self, vehicle, target_speed=30.0):
        self.vehicle = vehicle
        self.controller = JAXVehicleController(vehicle, target_speed)
        self.target_speed = target_speed
        self.destination = None
        
        # JAX-compiled functions for trajectory processing
        self.compute_trajectory_cost = jit(self._compute_trajectory_cost)
        self.select_best_trajectory = jit(self._select_best_trajectory)
        
    @staticmethod
    @jit
    def _compute_trajectory_cost(trajectory, obstacles, target_point):
        """Compute cost for a trajectory considering obstacles and target."""
        # Distance to target cost
        final_point = trajectory[-1]
        target_cost = jnp.linalg.norm(final_point - target_point)
        
        # Obstacle avoidance cost
        obstacle_cost = 0.0
        for point in trajectory:
            distances = jnp.linalg.norm(obstacles - point[None, :], axis=1)
            min_distance = jnp.min(distances)
            obstacle_cost += jnp.where(min_distance < 3.0, 100.0 / (min_distance + 0.1), 0.0)
        
        # Smoothness cost
        if len(trajectory) > 1:
            diffs = jnp.diff(trajectory, axis=0)
            smoothness_cost = jnp.sum(jnp.linalg.norm(diffs, axis=1))
        else:
            smoothness_cost = 0.0
        
        total_cost = target_cost + obstacle_cost + 0.1 * smoothness_cost
        return total_cost
    
    @staticmethod
    @jit
    def _select_best_trajectory(trajectories, obstacles, target_point):
        """Select best trajectory from candidates."""
        costs = vmap(JAXEnhancedAgent._compute_trajectory_cost, 
                    in_axes=(0, None, None))(trajectories, obstacles, target_point)
        best_idx = jnp.argmin(costs)
        return best_idx, trajectories[best_idx]
    
    def set_destination(self, destination):
        """Set the destination for the agent."""
        self.destination = destination
    
    def run_step(self):
        """Execute one step of the agent."""
        if self.destination is None:
            return carla.VehicleControl()
        
        # Simple navigation logic (can be enhanced with path planning)
        transform = self.vehicle.get_transform()
        vehicle_location = transform.location
        
        # Calculate heading to destination
        dx = self.destination.x - vehicle_location.x
        dy = self.destination.y - vehicle_location.y
        target_heading = math.atan2(dy, dx)
        
        # Use JAX controller
        control = self.controller.get_control(self.target_speed, target_heading)
        
        return control
    
    def done(self):
        """Check if the agent has reached its destination."""
        if self.destination is None:
            return False
        
        transform = self.vehicle.get_transform()
        distance = transform.location.distance(self.destination)
        return distance < 4.0


# ==============================================================================
# -- Modified Game Loop with JAX ----------------------------------------------
# ==============================================================================

def game_loop_jax(args):
    """JAX-enhanced game loop with performance monitoring."""
    
    pygame.init()
    pygame.font.init()
    world = None
    
    # JAX performance monitoring
    jax_timings = collections.deque(maxlen=100)
    
    try:
        if args.seed:
            np.random.seed(args.seed)
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()
        
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        # Import required classes (simplified for this example)
        from examples.automatic_control import HUD, World, KeyboardControl
        
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        # Use JAX-enhanced agent
        if args.jax_agent:
            agent = JAXEnhancedAgent(world.player, 30)
        else:
            # Fall back to standard agents
            if args.agent == "Basic":
                agent = BasicAgent(world.player, 30)
            elif args.agent == "Behavior":
                agent = BehaviorAgent(world.player, behavior=args.behavior)
            else:
                agent = ConstantVelocityAgent(world.player, 30)
        
        # Set destination
        spawn_points = world.map.get_spawn_points()
        destination = np.random.choice(spawn_points).location
        agent.set_destination(destination)
        
        clock = pygame.time.Clock()
        
        print("JAX automatic control started. JAX version:", jax.__version__)
        print(f"JAX devices: {jax.devices()}")
        
        while True:
            clock.tick()
            
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            
            if controller.parse_events():
                return
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            
            # Check if agent is done
            if agent.done():
                if args.loop:
                    agent.set_destination(np.random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("Target reached, searching for another target")
                else:
                    print("Target reached, stopping simulation")
                    break
            
            # JAX-accelerated control step with timing
            start_time = time.time()
            control = agent.run_step()
            jax_time = time.time() - start_time
            jax_timings.append(jax_time)
            
            control.manual_gear_shift = False
            world.player.apply_control(control)
            
            # Performance monitoring
            if len(jax_timings) == 100:
                avg_jax_time = np.mean(jax_timings) * 1000  # ms
                world.hud.notification(f"JAX control: {avg_jax_time:.2f}ms avg", seconds=1.0)
        
    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            world.destroy()
        
        pygame.quit()


def main():
    """Main function with JAX-specific arguments."""
    
    argparser = argparse.ArgumentParser(description='CARLA JAX Automatic Control')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port')
    argparser.add_argument('--res', default='1280x720', help='Window resolution')
    argparser.add_argument('--sync', action='store_true', help='Synchronous mode')
    argparser.add_argument('--filter', default='vehicle.*', help='Actor filter')
    argparser.add_argument('--generation', default='2', help='Actor generation')
    argparser.add_argument('--agent', default='Basic', help='Agent type')
    argparser.add_argument('--behavior', default='normal', help='Behavior type')
    argparser.add_argument('--seed', type=int, help='Random seed')
    argparser.add_argument('--loop', action='store_true', help='Loop after reaching target')
    
    # JAX-specific arguments
    argparser.add_argument('--jax-agent', action='store_true', 
                          help='Use JAX-enhanced agent instead of standard agents')
    argparser.add_argument('--jax-backend', default='cpu', choices=['cpu', 'gpu'],
                          help='JAX backend to use')
    
    args = argparser.parse_args()
    
    # Configure JAX
    if args.jax_backend == 'gpu':
        if not jax.devices('gpu'):
            print("Warning: GPU requested but not available, falling back to CPU")
    
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    
    print(__doc__)
    
    try:
        game_loop_jax(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()