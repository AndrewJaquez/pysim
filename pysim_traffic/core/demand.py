"""Vehicle demand management: spawners, sinks, and origin-destination flows."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .geometry import LinkId, LaneId, VehClass, DEFAULT_VEHICLE_PARAMS


class HeadwayType(Enum):
    POISSON = "poisson"
    DETERMINISTIC = "deterministic"


@dataclass
class TimeWindow:
    """Time window for demand activation."""
    start_hour: float  # 0.0 = midnight, 6.5 = 6:30 AM
    end_hour: float
    
    def is_active(self, sim_time_hours: float) -> bool:
        """Check if current time is within this window."""
        # Handle overnight windows (e.g., 22:00 to 6:00)
        if self.start_hour <= self.end_hour:
            return self.start_hour <= sim_time_hours <= self.end_hour
        else:
            return sim_time_hours >= self.start_hour or sim_time_hours <= self.end_hour


@dataclass
class DemandSource:
    """Vehicle spawning point with configurable demand patterns."""
    id: int
    link_id: LinkId
    cars_per_hour: float
    trucks_per_hour: float
    headway_type: HeadwayType = HeadwayType.POISSON
    time_windows: List[TimeWindow] = field(default_factory=list)
    od_profile: Optional[Dict[int, float]] = None  # {sink_id: probability}
    
    # Internal state
    next_spawn_time: float = 0.0
    vehicles_spawned: int = 0
    spawn_buffer: List["PendingVehicle"] = field(default_factory=list)
    
    def get_total_demand(self) -> float:
        """Total vehicles per hour."""
        return self.cars_per_hour + self.trucks_per_hour
    
    def get_truck_fraction(self) -> float:
        """Fraction of vehicles that are trucks."""
        total = self.get_total_demand()
        return self.trucks_per_hour / total if total > 0 else 0.0
    
    def is_active(self, sim_time_hours: float) -> bool:
        """Check if source is active at current time."""
        if not self.time_windows:
            return True  # Always active if no time windows specified
        
        return any(window.is_active(sim_time_hours) for window in self.time_windows)
    
    def get_arrival_rate(self, sim_time_hours: float, dt: float) -> float:
        """Get vehicle arrival rate for current time step."""
        if not self.is_active(sim_time_hours):
            return 0.0
        
        # Convert vehicles/hour to arrivals per time step
        veh_per_second = self.get_total_demand() / 3600.0
        return veh_per_second * dt


@dataclass
class DemandSink:
    """Vehicle absorption point."""
    id: int
    link_id: LinkId
    absorb_prob: float = 1.0  # Probability of absorbing arriving vehicles
    vehicles_absorbed: int = 0


@dataclass
class PendingVehicle:
    """Vehicle waiting to spawn."""
    veh_class: int
    target_sink_id: Optional[int]
    spawn_attempts: int = 0
    max_attempts: int = 2


class DemandManager:
    """Manages all demand sources and sinks in the network."""
    
    def __init__(self, dt: float, rng: np.random.Generator):
        self.dt = dt
        self.rng = rng
        self.sources: Dict[int, DemandSource] = {}
        self.sinks: Dict[int, DemandSink] = {}
        self.sim_time = 0.0
        
        # Statistics
        self.total_spawned = 0
        self.total_absorbed = 0
        self.spawn_failures = 0
    
    def add_source(self, source: DemandSource):
        """Add demand source."""
        self.sources[source.id] = source
    
    def add_sink(self, sink: DemandSink):
        """Add demand sink."""
        self.sinks[sink.id] = sink
    
    def update(self, sim_time: float):
        """Update demand for current time step."""
        self.sim_time = sim_time
        sim_time_hours = (sim_time / 3600.0) % 24.0  # Convert to hours of day
        
        for source in self.sources.values():
            self._update_source_spawning(source, sim_time_hours)
    
    def _update_source_spawning(self, source: DemandSource, sim_time_hours: float):
        """Update spawning for a single source."""
        if not source.is_active(sim_time_hours):
            return
        
        arrival_rate = source.get_arrival_rate(sim_time_hours, self.dt)
        if arrival_rate <= 0:
            return
        
        # Generate arrivals based on headway type
        if source.headway_type == HeadwayType.POISSON:
            # Poisson arrivals: binomial with p = λ*Δt
            num_arrivals = self.rng.binomial(1, min(arrival_rate, 1.0))
        else:  # DETERMINISTIC
            # Deterministic headway: track exact timing
            source.next_spawn_time -= self.dt
            num_arrivals = 1 if source.next_spawn_time <= 0 else 0
            if num_arrivals > 0:
                headway = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
                source.next_spawn_time = headway
        
        # Create pending vehicles
        for _ in range(num_arrivals):
            # Determine vehicle class
            if self.rng.random() < source.get_truck_fraction():
                veh_class = VehClass.TRUCK
            else:
                veh_class = VehClass.CAR
            
            # Select destination sink
            target_sink_id = self._select_destination_sink(source)
            
            pending_veh = PendingVehicle(
                veh_class=veh_class,
                target_sink_id=target_sink_id
            )
            source.spawn_buffer.append(pending_veh)
    
    def _select_destination_sink(self, source: DemandSource) -> Optional[int]:
        """Select destination sink based on OD profile."""
        if not source.od_profile or not self.sinks:
            # No OD profile: select random sink
            return self.rng.choice(list(self.sinks.keys())) if self.sinks else None
        
        # Use OD profile probabilities
        sink_ids = list(source.od_profile.keys())
        probabilities = list(source.od_profile.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob <= 0:
            return None
        
        probabilities = [p / total_prob for p in probabilities]
        
        try:
            selected_id = self.rng.choice(sink_ids, p=probabilities)
            return selected_id if selected_id in self.sinks else None
        except ValueError:
            return None
    
    def get_pending_spawns(self, link_id: LinkId) -> List[Tuple[int, PendingVehicle]]:
        """Get all pending vehicles for a specific link."""
        pending = []
        
        for source_id, source in self.sources.items():
            if source.link_id == link_id and source.spawn_buffer:
                for vehicle in source.spawn_buffer:
                    pending.append((source_id, vehicle))
        
        return pending
    
    def confirm_spawn(self, source_id: int, vehicle: PendingVehicle) -> bool:
        """Confirm that a vehicle was successfully spawned."""
        if source_id not in self.sources:
            return False
        
        source = self.sources[source_id]
        if vehicle in source.spawn_buffer:
            source.spawn_buffer.remove(vehicle)
            source.vehicles_spawned += 1
            self.total_spawned += 1
            return True
        
        return False
    
    def fail_spawn(self, source_id: int, vehicle: PendingVehicle) -> bool:
        """Mark spawn attempt as failed, retry or discard."""
        if source_id not in self.sources:
            return False
        
        source = self.sources[source_id]
        if vehicle not in source.spawn_buffer:
            return False
        
        vehicle.spawn_attempts += 1
        
        if vehicle.spawn_attempts >= vehicle.max_attempts:
            # Give up on this vehicle
            source.spawn_buffer.remove(vehicle)
            self.spawn_failures += 1
            return False
        
        # Keep vehicle in buffer for next attempt
        return True
    
    def absorb_vehicle(self, sink_id: int, vehicle_id: int) -> bool:
        """Absorb vehicle at destination sink."""
        if sink_id not in self.sinks:
            return False
        
        sink = self.sinks[sink_id]
        
        # Check absorption probability
        if self.rng.random() > sink.absorb_prob:
            return False
        
        sink.vehicles_absorbed += 1
        self.total_absorbed += 1
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get demand management statistics."""
        stats = {
            "total_spawned": self.total_spawned,
            "total_absorbed": self.total_absorbed,
            "spawn_failures": self.spawn_failures,
            "active_sources": 0,
            "pending_vehicles": 0
        }
        
        sim_time_hours = (self.sim_time / 3600.0) % 24.0
        
        for source in self.sources.values():
            if source.is_active(sim_time_hours):
                stats["active_sources"] += 1
            stats["pending_vehicles"] += len(source.spawn_buffer)
        
        return stats
    
    def create_od_matrix(self) -> np.ndarray:
        """Create origin-destination matrix for analysis."""
        if not self.sources or not self.sinks:
            return np.array([[]])
        
        source_ids = sorted(self.sources.keys())
        sink_ids = sorted(self.sinks.keys())
        
        matrix = np.zeros((len(source_ids), len(sink_ids)))
        
        for i, source_id in enumerate(source_ids):
            source = self.sources[source_id]
            if source.od_profile:
                for j, sink_id in enumerate(sink_ids):
                    if sink_id in source.od_profile:
                        matrix[i, j] = source.od_profile[sink_id]
        
        return matrix


def create_default_demand_pattern(network_sources: List[LinkId], 
                                network_sinks: List[LinkId]) -> DemandManager:
    """Create default demand pattern for testing."""
    dt = 0.1  # 100ms time step
    rng = np.random.default_rng(42)  # Fixed seed for deterministic testing
    
    demand_manager = DemandManager(dt, rng)
    
    # Create demand sources
    for i, link_id in enumerate(network_sources):
        # Morning and evening rush hours
        time_windows = [
            TimeWindow(6.0, 10.0),   # 6 AM - 10 AM
            TimeWindow(16.0, 19.0)   # 4 PM - 7 PM
        ]
        
        source = DemandSource(
            id=i + 1,
            link_id=link_id,
            cars_per_hour=600,  # 600 cars/hour during rush
            trucks_per_hour=60,  # 60 trucks/hour
            time_windows=time_windows
        )
        
        # Simple uniform OD: equal probability to all sinks
        if network_sinks:
            source.od_profile = {sink_id: 1.0 / len(network_sinks) 
                               for sink_id in network_sinks}
        
        demand_manager.add_source(source)
    
    # Create demand sinks
    for i, link_id in enumerate(network_sinks):
        sink = DemandSink(
            id=i + 1,
            link_id=link_id,
            absorb_prob=1.0
        )
        demand_manager.add_sink(sink)
    
    return demand_manager