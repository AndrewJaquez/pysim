#!/usr/bin/env python3
"""Main entry point for PySimTraffic road builder and simulation."""

import sys
import argparse
from pysim_traffic.ui.pygame_app import RoadBuilder


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PySimTraffic - Traffic Simulation")
    parser.add_argument("--width", type=int, default=1200, 
                       help="Window width (default: 1200)")
    parser.add_argument("--height", type=int, default=800,
                       help="Window height (default: 800)")
    
    args = parser.parse_args()
    
    try:
        app = RoadBuilder(args.width, args.height)
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()