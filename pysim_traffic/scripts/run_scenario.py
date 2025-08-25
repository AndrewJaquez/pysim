#!/usr/bin/env python3
"""
Headless simulation runner for batch execution with CSV output.

Usage:
    python scripts/run_scenario.py --scenario scenarios/grid_4way.json \\
        --minutes 30 --seed 123 --csv out/metrics.csv
"""

import argparse
import csv
import time
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pysim_traffic.core.state import Simulation, SimConfig
from pysim_traffic.core.persist import NetworkSerializer
from pysim_traffic.core.demand import create_default_demand_pattern
from pysim_traffic.core.routing import Router
from pysim_traffic.core.dynamics import VehicleDynamics


class MetricsCollector:
    """Collects and processes simulation metrics."""
    
    def __init__(self, sim: Simulation, output_interval: float = 10.0):
        self.sim = sim
        self.output_interval = output_interval
        self.last_output_time = 0.0
        self.metrics_history: List[Dict[str, Any]] = []
        
    def collect_metrics(self, current_time: float) -> Dict[str, Any]:
        """Collect metrics at current simulation time."""
        vehicles = self.sim.state.vehicles
        active_indices = vehicles.get_active_indices()
        
        if len(active_indices) == 0:
            return self._empty_metrics(current_time)
        
        # Get vehicle data
        data = vehicles.get_vehicle_data(active_indices)
        
        # Calculate basic metrics
        metrics = {
            "sim_time": current_time,
            "time_of_day": self.sim.state.get_time_of_day(),
            "total_vehicles": len(active_indices),
            "avg_speed_mps": float(np.mean(data["velocity"])) if len(data["velocity"]) > 0 else 0.0,
            "avg_speed_kmh": float(np.mean(data["velocity"]) * 3.6) if len(data["velocity"]) > 0 else 0.0,
            "max_speed_mps": float(np.max(data["velocity"])) if len(data["velocity"]) > 0 else 0.0,
            "stopped_vehicles": int(np.sum(data["velocity"] < 0.5)) if len(data["velocity"]) > 0 else 0,
            "cars": int(np.sum(data["vehicle_class"] == 0)) if len(data["vehicle_class"]) > 0 else 0,
            "trucks": int(np.sum(data["vehicle_class"] == 1)) if len(data["vehicle_class"]) > 0 else 0,
        }
        
        # Add performance metrics
        perf_stats = self.sim.state.get_performance_stats()
        metrics.update({
            "step_time_ms": perf_stats["avg_step_time"] * 1000,
            "max_step_time_ms": perf_stats["max_step_time"] * 1000,
        })
        
        # Calculate network-level metrics
        network_stats = self._calculate_network_metrics(active_indices, data)
        metrics.update(network_stats)
        
        return metrics
    
    def _empty_metrics(self, current_time: float) -> Dict[str, Any]:
        """Return metrics structure with zero values."""
        return {
            "sim_time": current_time,
            "time_of_day": self.sim.state.get_time_of_day(),
            "total_vehicles": 0,
            "avg_speed_mps": 0.0,
            "avg_speed_kmh": 0.0,
            "max_speed_mps": 0.0,
            "stopped_vehicles": 0,
            "cars": 0,
            "trucks": 0,
            "step_time_ms": 0.0,
            "max_step_time_ms": 0.0,
            "total_delay": 0.0,
            "avg_delay_per_veh": 0.0,
            "throughput_veh_per_hour": 0.0
        }
    
    def _calculate_network_metrics(self, active_indices: np.ndarray, 
                                  data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate network-wide performance metrics."""
        
        vehicles = self.sim.state.vehicles
        current_time = self.sim.state.current_time
        
        # Calculate delays
        if len(active_indices) > 0:
            travel_times = current_time - vehicles.spawn_time[active_indices]
            # Estimate free-flow travel times (simplified)
            free_flow_times = np.maximum(travel_times * 0.5, 60.0)  # Minimum 1 minute
            delays = np.maximum(0, travel_times - free_flow_times)
            
            total_delay = float(np.sum(delays))
            avg_delay = float(np.mean(delays))
        else:
            total_delay = 0.0
            avg_delay = 0.0
        
        # Calculate throughput (vehicles completing trips per hour)
        # This would require tracking completed trips
        throughput = 0.0  # Placeholder
        
        return {
            "total_delay": total_delay,
            "avg_delay_per_veh": avg_delay,
            "throughput_veh_per_hour": throughput
        }
    
    def should_output(self, current_time: float) -> bool:
        """Check if it's time to output metrics."""
        return current_time - self.last_output_time >= self.output_interval
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics for later output."""
        self.metrics_history.append(metrics.copy())
        self.last_output_time = metrics["sim_time"]


class HeadlessSimRunner:
    """Main headless simulation runner."""
    
    def __init__(self, scenario_file: str, config: SimConfig):
        self.scenario_file = scenario_file
        self.config = config
        self.sim = None
        self.metrics_collector = None
        
    def setup_simulation(self):
        """Initialize simulation components."""
        print(f"Loading scenario: {self.scenario_file}")
        
        # Load network
        self.sim = Simulation.from_map(self.scenario_file, self.config)
        
        # Setup routing
        self.sim.router = Router(self.sim.network)
        
        # Setup vehicle dynamics
        self.sim.dynamics = VehicleDynamics(self.config.dt)
        
        # Setup demand (create default pattern for now)
        # In a full implementation, this would be loaded from scenario file
        source_links = list(self.sim.network.links.keys())[:2]  # First 2 links as sources
        sink_links = list(self.sim.network.links.keys())[-2:]   # Last 2 links as sinks
        
        if source_links and sink_links:
            self.sim.demand_manager = create_default_demand_pattern(
                source_links, sink_links)
        
        # Setup metrics collector
        self.metrics_collector = MetricsCollector(self.sim)
        
        print(f"Simulation initialized: {len(self.sim.network.nodes)} nodes, "
              f"{len(self.sim.network.links)} links, {len(self.sim.network.lanes)} lanes")
    
    def run_simulation(self, duration_minutes: float, output_interval: float = 10.0):
        """Run simulation for specified duration."""
        duration_seconds = duration_minutes * 60.0
        total_steps = int(duration_seconds / self.config.dt)
        
        print(f"Running simulation for {duration_minutes:.1f} minutes "
              f"({total_steps} steps, dt={self.config.dt}s)")
        
        # Setup metrics collector
        self.metrics_collector.output_interval = output_interval
        
        start_wall_time = time.time()
        
        for step in range(total_steps):
            # Step simulation
            current_time = self.sim.step()
            
            # Collect metrics
            if self.metrics_collector.should_output(current_time):
                metrics = self.metrics_collector.collect_metrics(current_time)
                self.metrics_collector.record_metrics(metrics)
                
                # Progress update
                progress = (step + 1) / total_steps * 100
                if step % (total_steps // 10) == 0:  # Update every 10%
                    elapsed_wall_time = time.time() - start_wall_time
                    print(f"Progress: {progress:5.1f}% | "
                          f"Sim time: {current_time:6.1f}s | "
                          f"Vehicles: {metrics['total_vehicles']:4d} | "
                          f"Avg speed: {metrics['avg_speed_kmh']:5.1f} km/h | "
                          f"Wall time: {elapsed_wall_time:6.1f}s")
        
        # Final metrics
        final_metrics = self.metrics_collector.collect_metrics(self.sim.state.current_time)
        self.metrics_collector.record_metrics(final_metrics)
        
        elapsed_wall_time = time.time() - start_wall_time
        sim_speed_factor = duration_seconds / elapsed_wall_time
        
        print(f"\nSimulation completed!")
        print(f"Wall time: {elapsed_wall_time:.2f} seconds")
        print(f"Simulation speed: {sim_speed_factor:.1f}x real-time")
        print(f"Final vehicles: {final_metrics['total_vehicles']}")
        print(f"Average speed: {final_metrics['avg_speed_kmh']:.1f} km/h")
    
    def save_metrics(self, output_file: str):
        """Save collected metrics to CSV file."""
        if not self.metrics_collector or not self.metrics_collector.metrics_history:
            print("No metrics to save")
            return
        
        print(f"Saving metrics to: {output_file}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Get all metric keys
        all_keys = set()
        for metrics in self.metrics_collector.metrics_history:
            all_keys.update(metrics.keys())
        
        # Sort keys for consistent column ordering
        fieldnames = sorted(all_keys)
        
        # Write CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in self.metrics_collector.metrics_history:
                # Fill missing keys with None
                row = {key: metrics.get(key, None) for key in fieldnames}
                writer.writerow(row)
        
        print(f"Saved {len(self.metrics_collector.metrics_history)} metric records")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run traffic simulation scenario")
    
    parser.add_argument("--scenario", required=True,
                       help="Path to scenario JSON file")
    parser.add_argument("--minutes", type=float, default=30.0,
                       help="Simulation duration in minutes (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Time step in seconds (default: 0.1)")
    parser.add_argument("--csv", 
                       help="Output CSV file for metrics")
    parser.add_argument("--output-interval", type=float, default=10.0,
                       help="Metrics output interval in seconds (default: 10)")
    parser.add_argument("--max-vehicles", type=int, default=10000,
                       help="Maximum number of vehicles (default: 10000)")
    
    args = parser.parse_args()
    
    # Validate scenario file
    if not Path(args.scenario).exists():
        print(f"Error: Scenario file not found: {args.scenario}")
        sys.exit(1)
    
    try:
        # Create simulation config
        config = SimConfig(
            dt=args.dt,
            seed=args.seed,
            max_vehicles=args.max_vehicles
        )
        
        # Setup and run simulation
        runner = HeadlessSimRunner(args.scenario, config)
        runner.setup_simulation()
        runner.run_simulation(args.minutes, args.output_interval)
        
        # Save metrics if requested
        if args.csv:
            runner.save_metrics(args.csv)
        
        print("Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()