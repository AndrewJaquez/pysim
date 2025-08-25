"""Vehicle dynamics: IDM car-following model and kinematics integration."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .geometry import VehClass, DEFAULT_VEHICLE_PARAMS


@dataclass
class IDMParams:
    """Intelligent Driver Model parameters."""
    a_max: float        # Maximum acceleration (m/s²)
    b: float           # Comfortable deceleration (m/s²)
    T: float           # Safe time headway (s)
    s0: float          # Minimum gap (m)
    delta: float       # Acceleration exponent (typically 4)
    v_max: float       # Maximum desired speed (m/s)


# Default IDM parameters by vehicle class
DEFAULT_IDM_PARAMS = {
    VehClass.CAR: IDMParams(
        a_max=1.2,
        b=2.0,
        T=1.2,
        s0=2.0,
        delta=4.0,
        v_max=50.0  # Will be overridden by speed limit
    ),
    VehClass.TRUCK: IDMParams(
        a_max=0.6,
        b=2.0,
        T=1.6,
        s0=3.0,
        delta=4.0,
        v_max=45.0  # Slightly lower than cars
    )
}


class IDMModel:
    """Intelligent Driver Model implementation."""
    
    @staticmethod
    def calculate_acceleration(v: float, v0: float, s: float, dv: float, 
                             params: IDMParams) -> float:
        """
        Calculate IDM acceleration.
        
        Args:
            v: Current speed (m/s)
            v0: Desired speed (m/s)  
            s: Gap to leader (m)
            dv: Speed difference to leader (v - v_leader) (m/s)
            params: IDM parameters
            
        Returns:
            Acceleration (m/s²)
        """
        if s <= 0:
            # Emergency braking if no gap
            return -params.b * 2
        
        # Free-flow acceleration term
        v_ratio = v / max(v0, 0.1)  # Avoid division by zero
        free_flow_term = 1.0 - (v_ratio ** params.delta)
        
        # Desired gap
        s_star = (params.s0 + v * params.T + 
                 (v * dv) / (2 * np.sqrt(params.a_max * params.b)))
        s_star = max(s_star, params.s0)  # Don't go below minimum gap
        
        # Interaction term
        gap_ratio = s_star / s
        interaction_term = gap_ratio * gap_ratio
        
        # Final acceleration
        acceleration = params.a_max * (free_flow_term - interaction_term)
        
        # Clamp acceleration to reasonable bounds
        return max(-params.b * 2, min(params.a_max, acceleration))
    
    @staticmethod
    def calculate_free_flow_acceleration(v: float, v0: float, 
                                       params: IDMParams) -> float:
        """Calculate acceleration in free-flow conditions (no leader)."""
        if v >= v0:
            return 0.0  # Don't accelerate above desired speed
        
        v_ratio = v / max(v0, 0.1)
        return params.a_max * (1.0 - (v_ratio ** params.delta))


class VehicleDynamics:
    """Manages vehicle physics and movement integration."""
    
    def __init__(self, dt: float):
        self.dt = dt
        self.idm = IDMModel()
    
    def update_accelerations(self, velocities: np.ndarray, 
                           desired_speeds: np.ndarray,
                           positions: np.ndarray,
                           vehicle_classes: np.ndarray,
                           lane_assignments: np.ndarray,
                           vehicle_lengths: np.ndarray) -> np.ndarray:
        """
        Update accelerations for all vehicles using IDM.
        
        Args:
            velocities: Current speeds (m/s) [N]
            desired_speeds: Desired speeds (m/s) [N]
            positions: Positions along lanes (m) [N]
            vehicle_classes: Vehicle classes [N]
            lane_assignments: Lane IDs [N]
            vehicle_lengths: Vehicle lengths (m) [N]
            
        Returns:
            Accelerations (m/s²) [N]
        """
        N = len(velocities)
        accelerations = np.zeros(N)
        
        # Group vehicles by lane for leader-follower analysis
        lane_groups = self._group_by_lane(lane_assignments, positions, 
                                         np.arange(N))
        
        for lane_id, vehicle_indices in lane_groups.items():
            if len(vehicle_indices) == 0:
                continue
            
            # Sort by position (rear to front)
            sorted_indices = vehicle_indices[np.argsort(positions[vehicle_indices])]
            
            for i, veh_idx in enumerate(sorted_indices):
                v = velocities[veh_idx]
                v0 = desired_speeds[veh_idx]
                veh_class = vehicle_classes[veh_idx]
                params = DEFAULT_IDM_PARAMS[veh_class]
                
                if i == len(sorted_indices) - 1:
                    # Lead vehicle - free flow
                    accelerations[veh_idx] = self.idm.calculate_free_flow_acceleration(
                        v, v0, params)
                else:
                    # Following vehicle - car following
                    leader_idx = sorted_indices[i + 1]
                    
                    # Calculate gap (front bumper to rear bumper)
                    gap = (positions[leader_idx] - positions[veh_idx] - 
                          vehicle_lengths[veh_idx])
                    gap = max(0.1, gap)  # Minimum small gap
                    
                    # Speed difference
                    dv = v - velocities[leader_idx]
                    
                    accelerations[veh_idx] = self.idm.calculate_acceleration(
                        v, v0, gap, dv, params)
        
        return accelerations
    
    def integrate_kinematics(self, positions: np.ndarray, 
                           velocities: np.ndarray,
                           accelerations: np.ndarray,
                           max_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate vehicle kinematics using semi-implicit Euler method.
        
        Args:
            positions: Current positions (m) [N]
            velocities: Current velocities (m/s) [N] 
            accelerations: Current accelerations (m/s²) [N]
            max_speeds: Maximum allowed speeds (m/s) [N]
            
        Returns:
            (new_positions, new_velocities)
        """
        # Semi-implicit Euler: v_new = v + a*dt, then s_new = s + v_new*dt
        new_velocities = velocities + accelerations * self.dt
        
        # Clamp velocities to [0, v_max]
        new_velocities = np.maximum(0.0, np.minimum(new_velocities, max_speeds))
        
        # Update positions using new velocities
        new_positions = positions + new_velocities * self.dt
        
        return new_positions, new_velocities
    
    def _group_by_lane(self, lane_assignments: np.ndarray, 
                      positions: np.ndarray,
                      indices: np.ndarray) -> Dict[int, np.ndarray]:
        """Group vehicle indices by lane."""
        groups = {}
        
        for i, lane_id in enumerate(lane_assignments):
            if lane_id not in groups:
                groups[lane_id] = []
            groups[lane_id].append(indices[i])
        
        # Convert to numpy arrays
        for lane_id in groups:
            groups[lane_id] = np.array(groups[lane_id])
        
        return groups
    
    def calculate_safe_spawn_gap(self, vehicle_class: int, 
                               leader_velocity: float) -> float:
        """Calculate minimum safe gap needed to spawn behind a leader."""
        params = DEFAULT_IDM_PARAMS[vehicle_class]
        
        # Use IDM desired gap formula with zero following velocity
        v_follower = 0.0  # Spawning vehicle starts at rest
        dv = v_follower - leader_velocity  # Negative since leader is faster
        
        s_star = (params.s0 + v_follower * params.T + 
                 (v_follower * dv) / (2 * np.sqrt(params.a_max * params.b)))
        
        # Add safety margin
        return max(s_star * 1.5, params.s0 + 10.0)
    
    def check_collision_risk(self, gap: float, relative_velocity: float,
                           follower_class: int) -> bool:
        """Check if gap and relative velocity indicate collision risk."""
        params = DEFAULT_IDM_PARAMS[follower_class]
        
        # Time to collision if maintaining current relative velocity
        if relative_velocity <= 0:
            return False  # No collision risk if not approaching
        
        ttc = gap / relative_velocity
        
        # Risk if time to collision is less than reaction time
        return ttc < params.T * 2  # Double reaction time for safety


class EmergencyBraking:
    """Emergency braking system for collision avoidance."""
    
    @staticmethod
    def calculate_emergency_decel(gap: float, relative_velocity: float,
                                follower_velocity: float,
                                vehicle_class: int) -> float:
        """Calculate emergency deceleration to avoid collision."""
        params = DEFAULT_IDM_PARAMS[vehicle_class]
        
        if gap <= 0:
            return params.b * 3  # Maximum emergency braking
        
        if relative_velocity <= 0:
            return 0.0  # No deceleration needed
        
        # Calculate deceleration needed to stop before collision
        # Using v² = v₀² + 2as, solve for a when v=0
        # a = -v₀²/(2s)
        required_decel = (follower_velocity ** 2) / (2 * gap)
        
        # Limit to vehicle capabilities with safety margin
        return min(required_decel, params.b * 3)


# Performance constants for tuning
DYNAMICS_CONFIG = {
    "min_gap": 0.1,           # Minimum gap to prevent division by zero (m)
    "max_emergency_decel": 8.0,  # Maximum emergency deceleration (m/s²)
    "collision_time_threshold": 3.0,  # Time threshold for collision risk (s)
    "safety_margin_factor": 1.5,     # Safety factor for spawn gaps
}