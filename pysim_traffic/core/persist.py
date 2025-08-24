"""Persistence layer for saving and loading network data."""

import json
import orjson
from typing import Dict, Any, List
from pathlib import Path

from .geometry import Network, Node, Link, Lane, LaneConn, LaneType, ControlType


class NetworkSerializer:
    """Handles serialization and deserialization of road networks."""
    
    @staticmethod
    def network_to_dict(network: Network) -> Dict[str, Any]:
        """Convert network to dictionary for JSON serialization."""
        return {
            "nodes": {
                str(node_id): {
                    "id": node.id,
                    "x": node.x,
                    "y": node.y,
                    "control": node.control,
                    "signal_id": node.signal_id,
                    "incoming": node.incoming,
                    "outgoing": node.outgoing
                }
                for node_id, node in network.nodes.items()
            },
            "links": {
                str(link_id): {
                    "id": link.id,
                    "a": link.a,
                    "b": link.b,
                    "length": link.length,
                    "speed_mps": link.speed_mps,
                    "speed_kmh": link.speed_kmh,
                    "lanes": link.lanes,
                    "one_way": link.one_way
                }
                for link_id, link in network.links.items()
            },
            "lanes": {
                str(lane_id): {
                    "id": lane.id,
                    "link_id": lane.link_id,
                    "lane_type": lane.lane_type,
                    "width": lane.width,
                    "connections": [
                        {
                            "to": conn.to,
                            "allow_car": conn.allow_car,
                            "allow_truck": conn.allow_truck,
                            "is_turn": conn.is_turn,
                            "turn_radius": conn.turn_radius
                        }
                        for conn in lane.connections
                    ]
                }
                for lane_id, lane in network.lanes.items()
            },
            "metadata": {
                "version": "1.0",
                "created_by": "PySimTraffic",
                "node_count": len(network.nodes),
                "link_count": len(network.links),
                "lane_count": len(network.lanes)
            }
        }
    
    @staticmethod
    def dict_to_network(data: Dict[str, Any]) -> Network:
        """Create network from dictionary loaded from JSON."""
        network = Network()
        
        # Load nodes
        for node_data in data["nodes"].values():
            node = Node(
                id=node_data["id"],
                x=node_data["x"],
                y=node_data["y"],
                control=node_data.get("control", ControlType.UNCTRL),
                signal_id=node_data.get("signal_id"),
                incoming=node_data.get("incoming", []),
                outgoing=node_data.get("outgoing", [])
            )
            network.add_node(node)
        
        # Load links
        for link_data in data["links"].values():
            link = Link(
                id=link_data["id"],
                a=link_data["a"],
                b=link_data["b"],
                length=link_data["length"],
                speed_mps=link_data["speed_mps"],
                lanes=link_data.get("lanes", []),
                one_way=link_data.get("one_way", False)
            )
            network.add_link(link)
        
        # Load lanes
        for lane_data in data["lanes"].values():
            connections = []
            for conn_data in lane_data.get("connections", []):
                conn = LaneConn(
                    to=conn_data["to"],
                    allow_car=conn_data.get("allow_car", True),
                    allow_truck=conn_data.get("allow_truck", True),
                    is_turn=conn_data.get("is_turn", False),
                    turn_radius=conn_data.get("turn_radius")
                )
                connections.append(conn)
            
            lane = Lane(
                id=lane_data["id"],
                link_id=lane_data["link_id"],
                lane_type=lane_data.get("lane_type", LaneType.DRIVING),
                width=lane_data.get("width", 3.5),
                connections=connections
            )
            
            # Set lane length from parent link
            if lane.link_id in network.links:
                lane.set_length(network.links[lane.link_id].length)
            
            network.add_lane(lane)
        
        return network
    
    @staticmethod
    def save_network(network: Network, filename: str) -> None:
        """Save network to JSON file using orjson for performance."""
        data = NetworkSerializer.network_to_dict(network)
        
        # Validate before saving
        errors = network.validate()
        if errors:
            raise ValueError(f"Network validation failed: {errors}")
        
        # Use orjson for fast serialization
        json_bytes = orjson.dumps(data, option=orjson.OPT_INDENT_2)
        
        with open(filename, 'wb') as f:
            f.write(json_bytes)
    
    @staticmethod
    def load_network(filename: str) -> Network:
        """Load network from JSON file."""
        with open(filename, 'rb') as f:
            data = orjson.loads(f.read())
        
        network = NetworkSerializer.dict_to_network(data)
        
        # Validate loaded network
        errors = network.validate()
        if errors:
            raise ValueError(f"Loaded network validation failed: {errors}")
        
        return network


def create_sample_networks():
    """Create sample network files for testing."""
    
    # Simple 4-way intersection
    network = Network()
    
    # Create intersection node
    center = Node(1, 0, 0, control=ControlType.SIGNAL)
    network.add_node(center)
    
    # Create approach nodes
    north = Node(2, 0, 100)
    south = Node(3, 0, -100)
    east = Node(4, 100, 0)
    west = Node(5, -100, 0)
    
    for node in [north, south, east, west]:
        network.add_node(node)
    
    # Create links
    links_data = [
        (1, 2, 1, 100),  # North approach
        (2, 1, 3, 100),  # North departure
        (3, 3, 1, 100),  # South approach
        (4, 1, 2, 100),  # South departure
        (5, 4, 1, 100),  # East approach
        (6, 1, 4, 100),  # East departure
        (7, 5, 1, 100),  # West approach
        (8, 1, 5, 100),  # West departure
    ]
    
    lane_id = 1
    for link_id, from_node, to_node, length in links_data:
        link = Link(
            id=link_id,
            a=from_node,
            b=to_node,
            length=length,
            speed_mps=13.89  # 50 km/h
        )
        
        # Add two lanes per link
        for i in range(2):
            lane = Lane(id=lane_id, link_id=link_id)
            lane.set_length(length)
            link.lanes.append(lane_id)
            network.add_lane(lane)
            lane_id += 1
        
        network.add_link(link)
    
    # Save sample network
    Path("pysim_traffic/scenarios").mkdir(parents=True, exist_ok=True)
    NetworkSerializer.save_network(network, "pysim_traffic/scenarios/grid_4way.json")
    
    print("Created sample network: grid_4way.json")


if __name__ == "__main__":
    create_sample_networks()