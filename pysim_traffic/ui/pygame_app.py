"""Main pygame application for road building and simulation visualization."""

import pygame
import sys
import math
from typing import Optional, List, Tuple, Set
from enum import Enum
from tkinter import filedialog
import tkinter as tk

from ..core.geometry import Network, Node, Link, Lane, LaneType, ControlType
from ..core.persist import NetworkSerializer


class Tool(Enum):
    DRAW_ROAD = "draw_road"
    LANE_TOOL = "lane_tool"
    SPEED_TOOL = "speed_tool"
    CONTROL_TOOL = "control_tool"
    ERASER = "eraser"
    SELECT = "select"


class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    
    # Road colors
    ROAD_FILL = (64, 64, 64)
    LANE_DIVIDER = (255, 255, 255)
    CENTERLINE = (255, 255, 0)


class Camera:
    """Simple 2D camera with pan and zoom."""
    
    def __init__(self, width: int, height: int):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.width = width
        self.height = height
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - self.width // 2) / self.zoom + self.x
        world_y = (screen_y - self.height // 2) / self.zoom + self.y
        return world_x, world_y
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((world_x - self.x) * self.zoom + self.width // 2)
        screen_y = int((world_y - self.y) * self.zoom + self.height // 2)
        return screen_x, screen_y
    
    def pan(self, dx: float, dy: float):
        """Pan camera by screen pixels."""
        self.x -= dx / self.zoom
        self.y -= dy / self.zoom
    
    def zoom_at(self, screen_x: int, screen_y: int, zoom_factor: float):
        """Zoom at specific screen point."""
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))  # Clamp zoom
        
        # Adjust position to keep same world point under cursor
        new_world_x, new_world_y = self.screen_to_world(screen_x, screen_y)
        self.x += new_world_x - world_x
        self.y += new_world_y - world_y


class RoadBuilder:
    """Main application for building road networks."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PySimTraffic - Road Builder")
        
        self.clock = pygame.time.Clock()
        self.camera = Camera(width, height)
        self.network = Network()
        
        # Current tool and state
        self.current_tool = Tool.DRAW_ROAD
        self.is_drawing = False
        self.draw_start: Optional[Tuple[float, float]] = None
        self.selected_node: Optional[int] = None
        self.selected_link: Optional[int] = None
        
        # ID counters
        self.next_node_id = 1
        self.next_link_id = 1
        self.next_lane_id = 1
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_world_pos = (0, 0)
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        # UI state
        self.show_node_ids = True
        self.show_link_ids = False
        self.lane_count = 1  # lanes per direction
        
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
            
            elif event.type == pygame.KEYDOWN:
                self._handle_key_down(event)
        
        return True
    
    def _handle_mouse_down(self, event):
        """Handle mouse button down events."""
        self.mouse_pos = event.pos
        self.mouse_world_pos = self.camera.screen_to_world(*event.pos)
        
        if event.button == 1:  # Left click
            if self.current_tool == Tool.DRAW_ROAD:
                self._start_draw_road()
            elif self.current_tool == Tool.SELECT:
                self._select_at_mouse()
        
        elif event.button == 2:  # Middle click - start pan
            self.dragging = True
            self.last_mouse_pos = event.pos
        
        elif event.button == 3:  # Right click
            if self.current_tool == Tool.DRAW_ROAD and self.is_drawing:
                self._cancel_draw_road()
    
    def _handle_mouse_up(self, event):
        """Handle mouse button up events."""
        if event.button == 1:  # Left click
            if self.current_tool == Tool.DRAW_ROAD and self.is_drawing:
                self._finish_draw_road()
        
        elif event.button == 2:  # Middle click - stop pan
            self.dragging = False
    
    def _handle_mouse_motion(self, event):
        """Handle mouse motion events."""
        self.mouse_pos = event.pos
        self.mouse_world_pos = self.camera.screen_to_world(*event.pos)
        
        if self.dragging:
            dx = event.pos[0] - self.last_mouse_pos[0]
            dy = event.pos[1] - self.last_mouse_pos[1]
            self.camera.pan(dx, dy)
            self.last_mouse_pos = event.pos
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        zoom_factor = 1.1 if event.y > 0 else 0.9
        self.camera.zoom_at(*self.mouse_pos, zoom_factor)
    
    def _handle_key_down(self, event):
        """Handle key down events."""
        if event.key == pygame.K_1:
            self.current_tool = Tool.DRAW_ROAD
        elif event.key == pygame.K_2:
            self.current_tool = Tool.SELECT
        elif event.key == pygame.K_3:
            self.current_tool = Tool.ERASER
        elif event.key == pygame.K_ESCAPE:
            if self.is_drawing:
                self._cancel_draw_road()
        elif event.key == pygame.K_i:
            self.show_node_ids = not self.show_node_ids
        elif event.key == pygame.K_l:
            self.show_link_ids = not self.show_link_ids
        elif event.key == pygame.K_MINUS:
            self.lane_count = max(1, self.lane_count - 1)
        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            self.lane_count = min(4, self.lane_count + 1)
        elif event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._save_network()
        elif event.key == pygame.K_o and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._load_network()
    
    def _start_draw_road(self):
        """Start drawing a new road."""
        self.is_drawing = True
        self.draw_start = self.mouse_world_pos
    
    def _finish_draw_road(self):
        """Finish drawing the current road."""
        if not self.is_drawing or not self.draw_start:
            return
        
        start_x, start_y = self.draw_start
        end_x, end_y = self.mouse_world_pos
        
        # Check if we're close to existing nodes
        start_node = self._find_nearby_node(start_x, start_y)
        end_node = self._find_nearby_node(end_x, end_y)
        
        # Create nodes if needed
        if start_node is None:
            start_node = Node(self.next_node_id, start_x, start_y)
            self.network.add_node(start_node)
            self.next_node_id += 1
        
        if end_node is None:
            end_node = Node(self.next_node_id, end_x, end_y)
            self.network.add_node(end_node)
            self.next_node_id += 1
        
        # Calculate link length and create link
        length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if length > 5.0:  # Minimum link length
            link = Link(
                id=self.next_link_id,
                a=start_node.id,
                b=end_node.id,
                length=length,
                speed_mps=13.89  # 50 km/h default
            )
            
            # Create lanes for this link
            for i in range(self.lane_count):
                lane = Lane(
                    id=self.next_lane_id,
                    link_id=link.id,
                    lane_type=LaneType.DRIVING
                )
                lane.set_length(length)
                link.lanes.append(lane.id)
                self.network.add_lane(lane)
                self.next_lane_id += 1
            
            self.network.add_link(link)
            self.next_link_id += 1
        
        self.is_drawing = False
        self.draw_start = None
    
    def _cancel_draw_road(self):
        """Cancel current road drawing."""
        self.is_drawing = False
        self.draw_start = None
    
    def _find_nearby_node(self, x: float, y: float, threshold: float = 20.0) -> Optional[Node]:
        """Find node within threshold distance (in world coordinates)."""
        threshold_world = threshold / self.camera.zoom
        
        for node in self.network.nodes.values():
            dist = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist <= threshold_world:
                return node
        return None
    
    def _select_at_mouse(self):
        """Select node or link at mouse position."""
        # First try to select a node
        node = self._find_nearby_node(*self.mouse_world_pos)
        if node:
            self.selected_node = node.id
            self.selected_link = None
            return
        
        # Then try to select a link
        # TODO: Implement link selection based on distance to line segment
        self.selected_node = None
        self.selected_link = None
    
    def _save_network(self):
        """Save current network to file."""
        try:
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()
            
            filename = filedialog.asksaveasfilename(
                title="Save Network",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            
            if filename:
                NetworkSerializer.save_network(self.network, filename)
                print(f"Network saved to {filename}")
        except Exception as e:
            print(f"Error saving network: {e}")
    
    def _load_network(self):
        """Load network from file."""
        try:
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()
            
            filename = filedialog.askopenfilename(
                title="Load Network",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            
            if filename:
                self.network = NetworkSerializer.load_network(filename)
                
                # Update ID counters to avoid conflicts
                if self.network.nodes:
                    self.next_node_id = max(self.network.nodes.keys()) + 1
                if self.network.links:
                    self.next_link_id = max(self.network.links.keys()) + 1
                if self.network.lanes:
                    self.next_lane_id = max(self.network.lanes.keys()) + 1
                
                print(f"Network loaded from {filename}")
        except Exception as e:
            print(f"Error loading network: {e}")
    
    def draw(self):
        """Draw the entire scene."""
        self.screen.fill(Colors.WHITE)
        
        # Draw grid
        self._draw_grid()
        
        # Draw network
        self._draw_links()
        self._draw_nodes()
        
        # Draw current drawing operation
        if self.is_drawing and self.draw_start:
            self._draw_current_road()
        
        # Draw UI
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw background grid."""
        grid_size = 50.0  # meters
        screen_grid_size = int(grid_size * self.camera.zoom)
        
        if screen_grid_size < 10:  # Don't draw if too small
            return
        
        # Calculate grid offset
        offset_x = int((-self.camera.x * self.camera.zoom) % screen_grid_size)
        offset_y = int((-self.camera.y * self.camera.zoom) % screen_grid_size)
        
        # Draw vertical lines
        for x in range(offset_x, self.width, screen_grid_size):
            pygame.draw.line(self.screen, Colors.LIGHT_GRAY, 
                           (x, 0), (x, self.height), 1)
        
        # Draw horizontal lines
        for y in range(offset_y, self.height, screen_grid_size):
            pygame.draw.line(self.screen, Colors.LIGHT_GRAY,
                           (0, y), (self.width, y), 1)
    
    def _draw_links(self):
        """Draw all links in the network."""
        for link in self.network.links.values():
            node_a = self.network.nodes[link.a]
            node_b = self.network.nodes[link.b]
            
            start_screen = self.camera.world_to_screen(node_a.x, node_a.y)
            end_screen = self.camera.world_to_screen(node_b.x, node_b.y)
            
            # Calculate road width based on number of lanes
            lane_width = 3.5 * self.camera.zoom
            road_width = max(4, len(link.lanes) * lane_width)
            
            # Draw road background
            if road_width > 2:
                self._draw_thick_line(start_screen, end_screen, 
                                    int(road_width), Colors.ROAD_FILL)
            
            # Draw lane dividers
            if len(link.lanes) > 1 and lane_width > 2:
                for i in range(1, len(link.lanes)):
                    # Calculate position of lane divider
                    t = i / len(link.lanes)
                    # This is simplified - would need proper perpendicular offset
                    pygame.draw.line(self.screen, Colors.LANE_DIVIDER,
                                   start_screen, end_screen, 1)
            
            # Draw centerline for single lane roads
            if len(link.lanes) == 1:
                pygame.draw.line(self.screen, Colors.CENTERLINE,
                               start_screen, end_screen, 2)
            
            # Highlight selected link
            if self.selected_link == link.id:
                pygame.draw.line(self.screen, Colors.BLUE,
                               start_screen, end_screen, 3)
            
            # Draw link ID if enabled
            if self.show_link_ids:
                mid_x = (node_a.x + node_b.x) / 2
                mid_y = (node_a.y + node_b.y) / 2
                mid_screen = self.camera.world_to_screen(mid_x, mid_y)
                self._draw_text(str(link.id), mid_screen, Colors.BLACK)
    
    def _draw_nodes(self):
        """Draw all nodes in the network."""
        for node in self.network.nodes.values():
            screen_pos = self.camera.world_to_screen(node.x, node.y)
            
            # Draw node circle
            color = Colors.RED if self.selected_node == node.id else Colors.DARK_GRAY
            radius = max(3, int(5 * self.camera.zoom))
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            
            # Draw control type indicator
            if node.control == ControlType.STOP:
                pygame.draw.circle(self.screen, Colors.RED, screen_pos, radius + 3, 2)
            elif node.control == ControlType.YIELD:
                pygame.draw.circle(self.screen, Colors.YELLOW, screen_pos, radius + 3, 2)
            elif node.control == ControlType.SIGNAL:
                pygame.draw.circle(self.screen, Colors.GREEN, screen_pos, radius + 3, 2)
            
            # Draw node ID if enabled
            if self.show_node_ids:
                text_pos = (screen_pos[0] + 10, screen_pos[1] - 10)
                self._draw_text(str(node.id), text_pos, Colors.BLACK)
    
    def _draw_current_road(self):
        """Draw the road currently being drawn."""
        if not self.draw_start:
            return
        
        start_screen = self.camera.world_to_screen(*self.draw_start)
        end_screen = self.camera.world_to_screen(*self.mouse_world_pos)
        
        # Draw preview line
        pygame.draw.line(self.screen, Colors.GRAY, start_screen, end_screen, 2)
        
        # Draw length
        length = math.sqrt(
            (self.mouse_world_pos[0] - self.draw_start[0])**2 +
            (self.mouse_world_pos[1] - self.draw_start[1])**2
        )
        mid_screen = (
            (start_screen[0] + end_screen[0]) // 2,
            (start_screen[1] + end_screen[1]) // 2
        )
        self._draw_text(f"{length:.1f}m", mid_screen, Colors.BLACK)
    
    def _draw_thick_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                        width: int, color: Tuple[int, int, int]):
        """Draw a thick line using polygon."""
        if width <= 2:
            pygame.draw.line(self.screen, color, start, end, width)
            return
        
        # Calculate perpendicular vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Normalize and get perpendicular
        dx /= length
        dy /= length
        px = -dy * width // 2
        py = dx * width // 2
        
        # Create rectangle points
        points = [
            (start[0] + px, start[1] + py),
            (start[0] - px, start[1] - py),
            (end[0] - px, end[1] - py),
            (end[0] + px, end[1] + py)
        ]
        
        pygame.draw.polygon(self.screen, color, points)
    
    def _draw_text(self, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw text at position."""
        font = pygame.font.Font(None, 24)
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)
    
    def _draw_ui(self):
        """Draw UI elements."""
        # Tool info
        tool_text = f"Tool: {self.current_tool.value} (Keys: 1=Road, 2=Select, 3=Erase)"
        self._draw_text(tool_text, (10, 10), Colors.BLACK)
        
        # Lane count
        if self.current_tool == Tool.DRAW_ROAD:
            lane_text = f"Lanes per direction: {self.lane_count} (+/- to change)"
            self._draw_text(lane_text, (10, 35), Colors.BLACK)
        
        # Instructions
        instructions = [
            "Middle mouse: Pan",
            "Wheel: Zoom",
            "I: Toggle node IDs",
            "L: Toggle link IDs",
            "Ctrl+S: Save network",
            "Ctrl+O: Load network",
            "ESC: Cancel current operation"
        ]
        
        for i, instruction in enumerate(instructions):
            self._draw_text(instruction, (10, self.height - 120 + i * 25), Colors.DARK_GRAY)
        
        # Network stats
        stats = [
            f"Nodes: {len(self.network.nodes)}",
            f"Links: {len(self.network.links)}",
            f"Lanes: {len(self.network.lanes)}"
        ]
        
        for i, stat in enumerate(stats):
            self._draw_text(stat, (self.width - 150, 10 + i * 25), Colors.BLACK)
    
    def run(self):
        """Main application loop."""
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = RoadBuilder()
    app.run()