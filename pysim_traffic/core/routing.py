"""Routing system with Dijkstra's algorithm and dynamic cost calculations."""

import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .geometry import Network, Node, Link, Lane, LaneId, NodeId, LinkId


@dataclass
class RouteNode:
    """Node in the lane-end routing graph."""
    lane_id: LaneId
    node_id: NodeId  # Physical intersection node
    x: float
    y: float
    is_source: bool = False  # Entry point to network
    is_sink: bool = False    # Exit point from network


@dataclass
class RouteEdge:
    """Edge in the routing graph (lane segment or turn)."""
    from_node: LaneId
    to_node: LaneId
    base_cost: float      # length / speed_limit
    length: float
    is_turn: bool = False
    turn_radius: Optional[float] = None
    
    # Dynamic cost factors (updated during simulation)
    queue_penalty: float = 0.0      # seconds of delay due to queue
    signal_penalty: float = 0.0     # estimated red signal delay
    truck_penalty: float = 0.0      # additional cost for trucks


@dataclass
class Route:
    """Complete route from source to destination."""
    lanes: List[LaneId]
    total_cost: float
    total_length: float
    estimated_time: float


class RoutingGraph:
    """Lane-end graph for pathfinding."""
    
    def __init__(self):
        self.nodes: Dict[LaneId, RouteNode] = {}
        self.edges: Dict[Tuple[LaneId, LaneId], RouteEdge] = {}
        self.outgoing: Dict[LaneId, List[LaneId]] = {}
        self.incoming: Dict[LaneId, List[LaneId]] = {}
    
    def add_node(self, node: RouteNode):
        """Add routing node."""
        self.nodes[node.lane_id] = node
        if node.lane_id not in self.outgoing:
            self.outgoing[node.lane_id] = []
        if node.lane_id not in self.incoming:
            self.incoming[node.lane_id] = []
    
    def add_edge(self, edge: RouteEdge):
        """Add routing edge."""
        key = (edge.from_node, edge.to_node)
        self.edges[key] = edge
        
        self.outgoing[edge.from_node].append(edge.to_node)
        self.incoming[edge.to_node].append(edge.from_node)
    
    def get_edge_cost(self, from_lane: LaneId, to_lane: LaneId, 
                      is_truck: bool = False) -> float:
        """Get total cost for edge including dynamic penalties."""
        key = (from_lane, to_lane)
        if key not in self.edges:
            return float('inf')
        
        edge = self.edges[key]
        cost = edge.base_cost + edge.queue_penalty + edge.signal_penalty
        
        if is_truck:
            cost += edge.truck_penalty
        
        return cost
    
    def update_dynamic_costs(self, queue_lengths: Dict[LaneId, int],
                           signal_states: Dict[NodeId, Dict], 
                           current_time: float):
        """Update dynamic cost penalties."""
        # Update queue penalties (simple: 1 second per queued vehicle)
        for (from_lane, to_lane), edge in self.edges.items():
            queue_length = queue_lengths.get(to_lane, 0)
            edge.queue_penalty = queue_length * 1.0  # 1 sec per vehicle
        
        # Update signal penalties for turn movements
        for (from_lane, to_lane), edge in self.edges.items():
            if edge.is_turn:
                from_node = self.nodes[from_lane]
                signal_state = signal_states.get(from_node.node_id, {})
                
                # Estimate delay based on signal timing
                if 'red_time_remaining' in signal_state:
                    edge.signal_penalty = signal_state['red_time_remaining']
                else:
                    edge.signal_penalty = 0.0


class Router:
    """Main routing engine using Dijkstra's algorithm."""
    
    def __init__(self, network: Network):
        self.network = network
        self.graph = RoutingGraph()
        self.route_cache: Dict[Tuple[LaneId, LaneId], Route] = {}
        self.cache_valid_until = 0.0
        self.cache_refresh_interval = 30.0  # 30 seconds
        
        self._build_routing_graph()
    
    def _build_routing_graph(self):
        """Build lane-end routing graph from network topology."""
        # Create routing nodes for each lane end
        for lane in self.network.lanes.values():
            link = self.network.links[lane.link_id]
            
            # Lane start (at from-node)
            from_node = self.network.nodes[link.a]
            start_node = RouteNode(
                lane_id=lane.id,
                node_id=from_node.id,
                x=from_node.x,
                y=from_node.y
            )
            self.graph.add_node(start_node)
            
            # Lane end (at to-node) 
            to_node = self.network.nodes[link.b]
            end_node = RouteNode(
                lane_id=lane.id,
                node_id=to_node.id,
                x=to_node.x,
                y=to_node.y
            )
            # Use negative lane ID to distinguish lane end from lane start
            end_node.lane_id = -lane.id
            self.graph.add_node(end_node)
            
            # Lane segment edge (driving along the lane)
            segment_edge = RouteEdge(
                from_node=lane.id,
                to_node=-lane.id,
                base_cost=link.length / link.speed_mps,
                length=link.length
            )
            self.graph.add_edge(segment_edge)
        
        # Add turn connections between lanes
        for lane in self.network.lanes.values():
            lane_end_id = -lane.id
            
            for conn in lane.connections:
                target_lane = self.network.lanes[conn.to]
                target_start_id = target_lane.id
                
                # Calculate turn cost (simplified)
                turn_cost = 2.0 if conn.is_turn else 0.5  # base turn time
                if conn.turn_radius and conn.turn_radius < 20:
                    turn_cost += 1.0  # tight turn penalty
                
                turn_edge = RouteEdge(
                    from_node=lane_end_id,
                    to_node=target_start_id,
                    base_cost=turn_cost,
                    length=0.0,
                    is_turn=True,
                    turn_radius=conn.turn_radius
                )
                
                # Add truck penalty for sharp turns
                if conn.turn_radius and conn.turn_radius < 15:
                    turn_edge.truck_penalty = 3.0
                
                self.graph.add_edge(turn_edge)
    
    def find_route(self, start_lane: LaneId, end_lane: LaneId, 
                   is_truck: bool = False) -> Optional[Route]:
        """Find shortest route between lane endpoints."""
        # Check cache first
        cache_key = (start_lane, end_lane)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Use negative lane ID for destination (lane end)
        target = -end_lane
        
        # Dijkstra's algorithm
        distances = {start_lane: 0.0}
        previous = {}
        visited = set()
        pq = [(0.0, start_lane)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target:
                # Reconstruct path
                path = []
                node = current
                while node in previous:
                    path.append(node)
                    node = previous[node]
                path.append(start_lane)
                path.reverse()
                
                # Convert path to lane sequence and calculate metrics
                return self._path_to_route(path, current_dist)
            
            # Check neighbors
            for neighbor in self.graph.outgoing.get(current, []):
                if neighbor in visited:
                    continue
                
                edge_cost = self.graph.get_edge_cost(current, neighbor, is_truck)
                new_dist = current_dist + edge_cost
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return None  # No route found
    
    def _path_to_route(self, path: List[LaneId], total_cost: float) -> Route:
        """Convert node path to Route object."""
        lanes = []
        total_length = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Skip negative lane IDs (they represent lane ends)
            if from_node > 0:
                lanes.append(from_node)
            
            # Add length for lane segments
            key = (from_node, to_node)
            if key in self.graph.edges:
                total_length += self.graph.edges[key].length
        
        return Route(
            lanes=lanes,
            total_cost=total_cost,
            total_length=total_length,
            estimated_time=total_cost
        )
    
    def update_costs(self, queue_lengths: Dict[LaneId, int],
                     signal_states: Dict[NodeId, Dict], 
                     current_time: float):
        """Update dynamic routing costs and clear cache if needed."""
        self.graph.update_dynamic_costs(queue_lengths, signal_states, current_time)
        
        # Clear cache periodically
        if current_time > self.cache_valid_until:
            self.route_cache.clear()
            self.cache_valid_until = current_time + self.cache_refresh_interval
    
    def find_routes_to_sinks(self, start_lane: LaneId, sink_lanes: List[LaneId],
                           is_truck: bool = False) -> List[Tuple[LaneId, Route]]:
        """Find routes to all possible sink lanes, return sorted by cost."""
        routes = []
        
        for sink_lane in sink_lanes:
            route = self.find_route(start_lane, sink_lane, is_truck)
            if route:
                routes.append((sink_lane, route))
        
        # Sort by total cost
        routes.sort(key=lambda x: x[1].total_cost)
        return routes
    
    def get_route_statistics(self) -> Dict[str, float]:
        """Get routing performance statistics."""
        return {
            "nodes_count": len(self.graph.nodes),
            "edges_count": len(self.graph.edges),
            "cached_routes": len(self.route_cache),
            "cache_hit_rate": 0.0  # TODO: implement cache hit tracking
        }