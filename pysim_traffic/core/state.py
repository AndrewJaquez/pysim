"""Vehicle state management using Structure of Arrays for performance."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .geometry import LaneId, VehId, VehClass, DEFAULT_VEHICLE_PARAMS
from .dynamics import DEFAULT_IDM_PARAMS
from .routing import Route


@dataclass
class SimConfig:
    """Simulation configuration parameters."""
    dt: float = 0.1           # Time step (seconds)
    seed: int = 42            # Random seed for determinism
    max_vehicles: int = 10000 # Maximum number of vehicles
    time_scale: float = 1.0   # Real-time multiplier (1.0 = real time)


class VehicleStore:
    """Structure of Arrays for efficient vehicle storage and computation."""
    
    def __init__(self, max_vehicles: int):
        self.max_vehicles = max_vehicles
        self.count = 0  # Number of active vehicles
        
        # Vehicle state arrays (SoA)
        self.active = np.zeros(max_vehicles, dtype=bool)       # Active vehicle flags
        self.lane = np.zeros(max_vehicles, dtype=np.int32)     # Lane ID
        self.position = np.zeros(max_vehicles, dtype=np.float32)  # Position along lane (m)
        self.velocity = np.zeros(max_vehicles, dtype=np.float32)  # Speed (m/s)
        self.acceleration = np.zeros(max_vehicles, dtype=np.float32)  # Accel (m/s²)
        self.vehicle_class = np.zeros(max_vehicles, dtype=np.int32)   # Car/truck
        self.length = np.zeros(max_vehicles, dtype=np.float32)   # Vehicle length (m)
        self.width = np.zeros(max_vehicles, dtype=np.float32)    # Vehicle width (m)
        self.desired_speed = np.zeros(max_vehicles, dtype=np.float32)  # Desired speed
        
        # Routing information
        self.route_index = np.zeros(max_vehicles, dtype=np.int32)  # Current route step
        self.target_sink = np.zeros(max_vehicles, dtype=np.int32)  # Destination sink ID
        
        # Timing and statistics
        self.spawn_time = np.zeros(max_vehicles, dtype=np.float32)
        self.total_delay = np.zeros(max_vehicles, dtype=np.float32)
        
        # Route storage (list per vehicle - not ideal for vectorization but needed)
        self.routes: List[Optional[Route]] = [None] * max_vehicles
        
        # Free ID management
        self._free_indices = list(range(max_vehicles))
        self._next_id = 1
    
    def add_vehicle(self, lane_id: LaneId, position: float, vehicle_class: int,
                   route: Route, target_sink_id: int, spawn_time: float) -> Optional[VehId]:
        """Add a new vehicle to the store."""
        if not self._free_indices:
            return None  # No space available
        
        # Get free index
        idx = self._free_indices.pop(0)
        veh_id = self._next_id
        self._next_id += 1
        
        # Set vehicle parameters
        self.active[idx] = True
        self.lane[idx] = lane_id
        self.position[idx] = position
        self.velocity[idx] = 0.0  # Start at rest
        self.acceleration[idx] = 0.0
        self.vehicle_class[idx] = vehicle_class
        self.route_index[idx] = 0
        self.target_sink[idx] = target_sink_id
        self.spawn_time[idx] = spawn_time
        self.total_delay[idx] = 0.0
        self.routes[idx] = route
        
        # Set physical parameters based on class
        params = DEFAULT_VEHICLE_PARAMS[vehicle_class]
        self.length[idx] = params["length"]
        self.width[idx] = params["width"]
        
        # Set desired speed (will be updated based on current lane)
        idm_params = DEFAULT_IDM_PARAMS[vehicle_class]
        self.desired_speed[idx] = idm_params.v_max
        
        self.count += 1
        return veh_id
    
    def remove_vehicle(self, veh_id: VehId) -> bool:
        """Remove vehicle from the store."""
        idx = self._find_vehicle_index(veh_id)
        if idx is None:
            return False
        
        # Mark as inactive and return index to free pool
        self.active[idx] = False
        self.routes[idx] = None
        self._free_indices.append(idx)
        self.count -= 1
        return True
    
    def get_active_mask(self) -> np.ndarray:
        """Get boolean mask for active vehicles."""
        return self.active[:self.count] if self.count > 0 else np.array([], dtype=bool)
    
    def get_active_indices(self) -> np.ndarray:
        """Get indices of all active vehicles."""
        return np.where(self.active)[0]
    
    def get_vehicles_in_lane(self, lane_id: LaneId) -> np.ndarray:
        """Get indices of vehicles in specific lane."""
        active_indices = self.get_active_indices()
        if len(active_indices) == 0:
            return np.array([], dtype=np.int32)
        
        lane_mask = self.lane[active_indices] == lane_id
        return active_indices[lane_mask]
    
    def update_desired_speeds(self, speed_limits: Dict[LaneId, float]):
        """Update desired speeds based on current lane speed limits."""
        active_indices = self.get_active_indices()
        
        for idx in active_indices:
            lane_id = self.lane[idx]
            if lane_id in speed_limits:
                speed_limit = speed_limits[lane_id]
                veh_class = self.vehicle_class[idx]
                
                # Trucks go slightly slower
                if veh_class == VehClass.TRUCK:
                    self.desired_speed[idx] = speed_limit * 0.9
                else:
                    self.desired_speed[idx] = speed_limit
    
    def get_vehicle_data(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Get vehicle data for specified indices."""
        if len(indices) == 0:
            return {key: np.array([]) for key in [
                'lane', 'position', 'velocity', 'acceleration', 
                'vehicle_class', 'length', 'desired_speed'
            ]}
        
        return {
            'lane': self.lane[indices],
            'position': self.position[indices],
            'velocity': self.velocity[indices],
            'acceleration': self.acceleration[indices],
            'vehicle_class': self.vehicle_class[indices],
            'length': self.length[indices],
            'desired_speed': self.desired_speed[indices],
            'route_index': self.route_index[indices],
            'target_sink': self.target_sink[indices]
        }
    
    def _find_vehicle_index(self, veh_id: VehId) -> Optional[int]:
        """Find internal array index for vehicle ID."""
        # For simplicity, using linear search
        # In production, would use a hash map
        active_indices = self.get_active_indices()
        # Vehicle ID mapping would need to be implemented properly
        # For now, return first active vehicle (placeholder)
        return active_indices[0] if len(active_indices) > 0 else None


class SimulationState:
    """Main simulation state management."""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.vehicles = VehicleStore(config.max_vehicles)
        self.current_time = 0.0
        self.step_count = 0
        
        # Random number generator for deterministic simulation
        self.rng = np.random.default_rng(config.seed)
        
        # Performance tracking
        self.step_times = []
        self.vehicle_counts = []
        
    def step(self) -> float:
        """Advance simulation by one time step."""
        import time
        start_time = time.perf_counter()
        
        self.current_time += self.config.dt
        self.step_count += 1
        
        # Track performance
        step_duration = time.perf_counter() - start_time
        self.step_times.append(step_duration)
        self.vehicle_counts.append(self.vehicles.count)
        
        # Keep only recent history
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-1000:]
            self.vehicle_counts = self.vehicle_counts[-1000:]
        
        return self.current_time
    
    def get_time_of_day(self) -> float:
        """Get current time of day in hours (0-24)."""
        return (self.current_time / 3600.0) % 24.0
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get simulation performance statistics."""
        if not self.step_times:
            return {"avg_step_time": 0.0, "max_step_time": 0.0, 
                   "avg_vehicles": 0.0, "max_vehicles": 0}
        
        return {
            "avg_step_time": np.mean(self.step_times),
            "max_step_time": np.max(self.step_times),
            "avg_vehicles": np.mean(self.vehicle_counts),
            "max_vehicles": np.max(self.vehicle_counts),
            "total_steps": self.step_count,
            "sim_time": self.current_time
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset simulation to initial state."""
        if seed is not None:
            self.config.seed = seed
            self.rng = np.random.default_rng(seed)
        
        # Clear all vehicles
        self.vehicles = VehicleStore(self.config.max_vehicles)
        self.current_time = 0.0
        self.step_count = 0
        self.step_times.clear()
        self.vehicle_counts.clear()
    
    def save_state(self) -> Dict[str, Any]:
        """Save complete simulation state for replay."""
        active_indices = self.vehicles.get_active_indices()
        
        state = {
            "config": {
                "dt": self.config.dt,
                "seed": self.config.seed,
                "max_vehicles": self.config.max_vehicles
            },
            "time": self.current_time,
            "step_count": self.step_count,
            "rng_state": self.rng.bit_generator.state,
            "vehicles": {
                "count": self.vehicles.count,
                "active_indices": active_indices.tolist(),
                "data": {
                    key: arr[active_indices].tolist()
                    for key, arr in {
                        "lane": self.vehicles.lane,
                        "position": self.vehicles.position,
                        "velocity": self.vehicles.velocity,
                        "acceleration": self.vehicles.acceleration,
                        "vehicle_class": self.vehicles.vehicle_class,
                        "length": self.vehicles.length,
                        "width": self.vehicles.width,
                        "desired_speed": self.vehicles.desired_speed,
                        "route_index": self.vehicles.route_index,
                        "target_sink": self.vehicles.target_sink,
                        "spawn_time": self.vehicles.spawn_time,
                        "total_delay": self.vehicles.total_delay
                    }.items()
                }
            }
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load simulation state from saved data."""
        # Update config
        config_data = state["config"]
        self.config.dt = config_data["dt"]
        self.config.seed = config_data["seed"]
        
        # Restore time and step count
        self.current_time = state["time"]
        self.step_count = state["step_count"]
        
        # Restore RNG state
        self.rng.bit_generator.state = state["rng_state"]
        
        # Recreate vehicle store
        self.vehicles = VehicleStore(config_data["max_vehicles"])
        
        # Restore vehicle data
        vehicle_data = state["vehicles"]
        active_indices = np.array(vehicle_data["active_indices"])
        
        if len(active_indices) > 0:
            data = vehicle_data["data"]
            
            # Restore arrays
            for key, values in data.items():
                if hasattr(self.vehicles, key):
                    arr = getattr(self.vehicles, key)
                    arr[active_indices] = values
            
            # Mark vehicles as active
            self.vehicles.active[active_indices] = True
            self.vehicles.count = len(active_indices)
            
            # Update free indices
            all_indices = set(range(self.config.max_vehicles))
            used_indices = set(active_indices)
            self.vehicles._free_indices = list(all_indices - used_indices)


class Simulation:
    """High-level simulation interface."""
    
    def __init__(self, config: SimConfig):
        self.state = SimulationState(config)
        self.network = None
        self.demand_manager = None
        self.router = None
        self.dynamics = None
    
    @classmethod
    def from_map(cls, map_file: str, config: SimConfig) -> "Simulation":
        """Create simulation from map file."""
        from .persist import NetworkSerializer
        
        sim = cls(config)
        sim.network = NetworkSerializer.load_network(map_file)
        return sim
    
    def step(self, ticks: int = 1) -> float:
        """Run simulation for specified number of ticks."""
        for _ in range(ticks):
            self.state.step()
        return self.state.current_time
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics."""
        # Placeholder - would be implemented with actual metrics
        return {
            "total_vehicles": self.state.vehicles.count,
            "avg_speed": 0.0,
            "total_delay": 0.0
        }
    
    def save_state(self, filename: str):
        """Save simulation state to file."""
        import orjson
        
        state = self.state.save_state()
        with open(filename, 'wb') as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))