"""Core geometry classes for roads, intersections, lanes, and connections."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


NodeId = int
LinkId = int
LaneId = int
VehId = int


class LaneType:
    DRIVING = 0
    TURN_L = 1
    TURN_R = 2
    SHOULDER = 3
    BUS = 4
    TRUCK_PREF = 5


class ControlType:
    UNCTRL = 0
    STOP = 1
    YIELD = 2
    SIGNAL = 3


class VehClass:
    CAR = 0
    TRUCK = 1


@dataclass
class Node:
    """Intersection or road endpoint."""
    id: NodeId
    x: float
    y: float
    control: int = ControlType.UNCTRL
    signal_id: Optional[int] = None
    incoming: List[LaneId] = field(default_factory=list)
    outgoing: List[LaneId] = field(default_factory=list)

    def distance_to(self, other: "Node") -> float:
        """Euclidean distance to another node."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Link:
    """Road segment connecting two nodes."""
    id: LinkId
    a: NodeId  # from node
    b: NodeId  # to node
    length: float
    speed_mps: float
    lanes: List[LaneId] = field(default_factory=list)  # left→right ordering
    one_way: bool = False

    @property
    def speed_kmh(self) -> float:
        """Speed limit in km/h."""
        return self.speed_mps * 3.6

    @classmethod
    def from_speed_kmh(cls, id: LinkId, a: NodeId, b: NodeId, 
                       length: float, speed_kmh: float, **kwargs) -> "Link":
        """Create link with speed in km/h."""
        return cls(id=id, a=a, b=b, length=length, 
                  speed_mps=speed_kmh / 3.6, **kwargs)


@dataclass
class Lane:
    """Individual lane within a link."""
    id: LaneId
    link_id: LinkId
    lane_type: int = LaneType.DRIVING
    width: float = 3.5  # meters
    connections: List["LaneConn"] = field(default_factory=list)
    
    @property
    def length(self) -> float:
        """Lane length (same as parent link)."""
        # Will be filled in by network topology builder
        return self._length
    
    def set_length(self, length: float):
        """Set lane length (called by network builder)."""
        self._length = length


@dataclass
class LaneConn:
    """Legal connection from this lane to another lane."""
    to: LaneId
    allow_car: bool = True
    allow_truck: bool = True
    is_turn: bool = False
    turn_radius: Optional[float] = None  # for speed calculations


@dataclass
class Network:
    """Complete road network topology."""
    nodes: Dict[NodeId, Node] = field(default_factory=dict)
    links: Dict[LinkId, Link] = field(default_factory=dict)
    lanes: Dict[LaneId, Lane] = field(default_factory=dict)
    
    def add_node(self, node: Node) -> None:
        """Add node to network."""
        self.nodes[node.id] = node
    
    def add_link(self, link: Link) -> None:
        """Add link to network."""
        self.links[link.id] = link
        
    def add_lane(self, lane: Lane) -> None:
        """Add lane to network."""
        self.lanes[lane.id] = lane
        
    def get_link_length(self, link_id: LinkId) -> float:
        """Get length of a link."""
        link = self.links[link_id]
        node_a = self.nodes[link.a]
        node_b = self.nodes[link.b]
        return node_a.distance_to(node_b)
    
    def validate(self) -> List[str]:
        """Validate network topology and return list of errors."""
        errors = []
        
        # Check that all link endpoints exist
        for link in self.links.values():
            if link.a not in self.nodes:
                errors.append(f"Link {link.id}: from-node {link.a} not found")
            if link.b not in self.nodes:
                errors.append(f"Link {link.id}: to-node {link.b} not found")
        
        # Check that all lane links exist
        for lane in self.lanes.values():
            if lane.link_id not in self.links:
                errors.append(f"Lane {lane.id}: parent link {lane.link_id} not found")
        
        # Check that lane connections reference valid lanes
        for lane in self.lanes.values():
            for conn in lane.connections:
                if conn.to not in self.lanes:
                    errors.append(f"Lane {lane.id}: connection to lane {conn.to} not found")
        
        return errors


# Default vehicle parameters
DEFAULT_VEHICLE_PARAMS = {
    VehClass.CAR: {
        "length": 4.5,  # meters
        "width": 1.8,
        "max_accel": 1.2,  # m/s²
        "max_decel": 2.0,
        "reaction_time": 1.2,  # seconds
        "min_gap": 2.0,  # meters
    },
    VehClass.TRUCK: {
        "length": 14.0,
        "width": 2.5,
        "max_accel": 0.6,
        "max_decel": 2.0,
        "reaction_time": 1.6,
        "min_gap": 3.0,
    }
}