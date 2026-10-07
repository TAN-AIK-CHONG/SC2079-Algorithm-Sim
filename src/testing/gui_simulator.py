"""Tkinter simulator for the 20x20 arena: generate or hand-place obstacles,
plan a route over them, and watch the robot drive it.

Run from src/:
    python testing/gui_simulator.py
"""

import json
import math
import queue
import random
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import ttk

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from algorithms.graph import Graph
from algorithms.hamiltonian import exhaustive_search, path_length
from collision import footprint_in_collision
from planner import Command, PlanningError, _apply_command, plan_mission
from model import (
    ARENA_LENGTH_CM,
    AXLE_TO_REAR_CM,
    GRID_LENGTH_CM,
    NUM_GRIDS,
    OBSTACLE_FOOTPRINT_LENGTH_CM,
    ROBOT_WIDTH_CM,
    Corners,
    Direction,
    Obstacle,
    Point,
    Robot,
    parse_scenario,
)

CELL_PX = 30
MARGIN_PX = 26
PX_PER_CM = CELL_PX / GRID_LENGTH_CM
CANVAS_PX = NUM_GRIDS * CELL_PX + 2 * MARGIN_PX

# The 40cm x 40cm start area, and where the robot's own footprint sits inside it.
# The cell is the footprint's BOTTOM-LEFT corner (see Robot.from_grid), so (0, 0)
# parks the robot flush into the corner of the arena.
START_ZONE_CELLS = 4
ROBOT_START_CELL = (0, 0, Direction.NORTH)

MIN_OBSTACLES = 4
MAX_OBSTACLES = 8
# 4 cells between obstacle origins leaves a 30cm gap between their TRUE
# footprints - but obstacles are inflated by OBSTACLE_MARGIN_CM on every side
# for collision, so the drivable corridor is really 20cm against a 19cm robot.
# It fits, but only just; gaps this tight are rarely usable in practice.
MIN_OBSTACLE_SEPARATION_CELLS = 4
IMAGE_IDS = tuple(range(11, 41))  # the target IDs the camera can come back with

# Clicking an obstacle that is already placed turns its image side clockwise.
HEADING_CYCLE = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
# Tk on macOS reports the right mouse button as Button-2; elsewhere it is Button-3.
RIGHT_CLICK_EVENTS = ("<Button-2>", "<Button-3>")
JSON_INDENT = 4

ANIMATION_MS = 40  # one path step (5cm of driving) per tick
POLL_MS = 50  # how often the UI drains the planner thread's queue
VIEWING_PAUSE_MS = 3000  # how long the robot sits at a viewing pose "taking a photo"

# Arc-length spacing between poses sampled along a leg's true (smoothed)
# trajectory - independent of hybrid_astar's STEP_CM, which only governs the
# search's own discretisation, not how finely we render its result.
SAMPLE_STEP_CM = 5

# Speed multipliers divide ANIMATION_MS, so larger is faster. The top of the
# range is the pace the simulator used to run at by default.
MIN_SPEED = 0.1
MAX_SPEED = 1.0
DEFAULT_SPEED = MIN_SPEED

COLOURS = {
    "grid": "#e4e4e4",
    "arena": "#ffffff",
    "border": "#333333",
    "start_zone": "#dff0d8",
    "start_edge": "#4a9a4a",
    "obstacle": "#3c3c3c",
    "obstacle_margin": "#b07070",
    "image_side": "#d64545",
    "axle": "#123f66",
    "turn_centre": "#8e44ad",
    "robot": "#4a90d9",
    "robot_front": "#f5a623",
    "pose": "#e08a00",
    "pose_outline": "#b0b0b0",
    "path": "#1f77b4",
    "trail": "#d64545",
    "label": "#777777",
}


def _footprint_centre_cm(robot: Robot) -> Point:
    """The robot's pose is its rear-axle midpoint; markers read better at the centre."""
    corners = robot.footprint_corners_cm()
    return (
        sum(x for x, _ in corners) / 4,
        sum(y for _, y in corners) / 4,
    )


def _front_edge_cm(robot: Robot) -> tuple[Point, Point]:
    """The footprint's two front corners: the robot's face."""
    corners = robot.footprint_corners_cm()
    return corners[1], corners[2]


def _left_unit(theta_rad: float) -> Point:
    """Unit vector pointing to the robot's left at heading theta_rad."""
    return -math.sin(theta_rad), math.cos(theta_rad)


def _rear_axle_cm(robot: Robot) -> tuple[Point, Point]:
    """The rear axle's two ends, where the rear wheels sit: across the full
    width of the footprint, through the pose."""
    half = ROBOT_WIDTH_CM / 2
    left_x, left_y = _left_unit(robot.theta_rad)
    return (
        (robot.x_cm + half * left_x, robot.y_cm + half * left_y),
        (robot.x_cm - half * left_x, robot.y_cm - half * left_y),
    )


def _turn_centre_cm(start: Robot, command: Command) -> Point | None:
    """The point an arc Command pivots about: level with the rear axle, one
    turning radius out on the steered side (the same side whether driving
    forward or reversing). None for a straight."""
    if command.turn == "STRAIGHT":
        return None
    radius_cm = command.distance_cm / math.radians(command.swept_angle_deg)
    side_sign = 1 if command.turn == "LEFT" else -1
    left_x, left_y = _left_unit(start.theta_rad)
    return (
        start.x_cm + side_sign * radius_cm * left_x,
        start.y_cm + side_sign * radius_cm * left_y,
    )


@dataclass
class Leg:
    """One planned hop, from the previous node to `goal_id`."""

    goal_id: str | int
    poses: list[Robot]
    turn_centres: list[Point | None]  # per pose: what its arc pivots about, None on a straight
    length_cm: float


@dataclass
class Plan:
    order: list[str | int]
    node_robots: dict[str | int, Robot]
    legs: list[Leg] = field(default_factory=list)
    length_cm: float = 0.0
    skipped_ids: list[str | int] = field(default_factory=list)  # unreachable - not visited


# --------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------


def _viewing_pose_ok(obstacle: Obstacle, footprints: list[Corners]) -> bool:
    """The robot has to be able to stand at the pose without clipping anything."""
    return not footprint_in_collision(obstacle.cm_viewing_position(), footprints)


def _placement_ok(candidate: Obstacle, placed: list[Obstacle]) -> bool:
    if candidate.x_coord <= START_ZONE_CELLS and candidate.y_coord <= START_ZONE_CELLS:
        return False  # keep the start area clear

    return all(
        max(
            abs(candidate.x_coord - other.x_coord),
            abs(candidate.y_coord - other.y_coord),
        )
        >= MIN_OBSTACLE_SEPARATION_CELLS
        for other in placed
    )


def generate_obstacles(count: int, rng: random.Random) -> list[Obstacle]:
    """Place `count` obstacles that are spread out and each actually viewable.

    Unlike generate_maps.py this does not path-plan while generating - the GUI
    has to stay responsive, and an unreachable obstacle shows up in the event
    log when the route is planned.
    """
    while True:
        obstacles: list[Obstacle] = []
        for _ in range(count * 200):  # attempt budget, then start the layout over
            if len(obstacles) == count:
                break
            candidate = Obstacle(
                len(obstacles),
                rng.randint(0, NUM_GRIDS - 1),
                rng.randint(0, NUM_GRIDS - 1),
                rng.choice(list(Direction)),
            )
            if _placement_ok(candidate, obstacles):
                obstacles.append(candidate)

        if len(obstacles) != count:
            continue

        footprints = [obstacle.inflated_footprint_corners_cm() for obstacle in obstacles]
        if all(_viewing_pose_ok(obstacle, footprints) for obstacle in obstacles):
            return obstacles


def scenario_dict(start_cell: tuple[int, int, Direction], obstacles: list[Obstacle]) -> dict:
    """The layout in the scenario JSON shape model.parse_scenario reads."""
    x_coord, y_coord, facing = start_cell
    return {
        "robot": {"x_coord": x_coord, "y_coord": y_coord, "facing": facing.name},
        "obstacles": [
            {
                "id": obstacle.id,
                "x_coord": obstacle.x_coord,
                "y_coord": obstacle.y_coord,
                "image_side": obstacle.image_side.name,
            }
            for obstacle in obstacles
        ],
    }


def _renumbered(obstacles: list[Obstacle]) -> list[Obstacle]:
    """Ids 0..n-1 in placement order, so removing one leaves no gap."""
    return [
        Obstacle(index, obstacle.x_coord, obstacle.y_coord, obstacle.image_side)
        for index, obstacle in enumerate(obstacles)
    ]


def _next_heading(heading: Direction) -> Direction:
    return HEADING_CYCLE[(HEADING_CYCLE.index(heading) + 1) % len(HEADING_CYCLE)]


# --------------------------------------------------------------------------
# Planning (runs off the UI thread)
# --------------------------------------------------------------------------


def _arc_poses(start: Robot, command: Command, step_cm: float) -> list[Robot]:
    """Sample poses along a Command's true path via planner._apply_command -
    a straight line, or a true circular arc at the radius implied by
    distance_cm/swept_angle_deg, not hybrid_astar's per-step
    chord-then-rotate approximation. This is what the robot will actually
    drive, since Command is exactly what planner.py hands to the STM32."""
    n_samples = max(1, round(command.distance_cm / step_cm))
    return [
        _apply_command(start, command, command.distance_cm * i / n_samples)
        for i in range(1, n_samples + 1)
    ]


def _leg_poses(
    start: Robot, commands: list[Command]
) -> tuple[list[Robot], list[Point | None]]:
    """The full smoothed trajectory for a leg: straights and true arcs
    chained end to end, one Command at a time - plus, for every pose, the
    centre of the arc it is on (None on a straight)."""
    poses = [start]
    turn_centres: list[Point | None] = [None]
    for command in commands:
        command_poses = _arc_poses(poses[-1], command, SAMPLE_STEP_CM)
        turn_centres.extend([_turn_centre_cm(poses[-1], command)] * len(command_poses))
        poses.extend(command_poses)
    return poses, turn_centres


def plan_route(start: Robot, obstacles: list[Obstacle], emit) -> Plan:
    """Plan with planner.plan_mission() - the same function rpi/main.py calls
    for the real run, leg recovery included - so the simulator drives the
    route the robot actually would. An obstacle no leg can reach is skipped,
    not visited."""
    graph = Graph.build(start, obstacles)
    order = exhaustive_search(graph)
    emit(
        f"Order (exhaustive search): {' -> '.join(str(i) for i in order)}"
        f"  [{path_length(graph, order):.0f}cm as Reeds-Shepp hops]"
    )
    plan = Plan(order=order, node_robots={node.id: node.viewing_pose for node in graph.nodes})

    try:
        mission = plan_mission(start, obstacles)
    except PlanningError:
        plan.skipped_ids = order[1:]
        return plan

    current_pose = start
    for mission_leg in mission.legs:
        poses, turn_centres = _leg_poses(current_pose, mission_leg.commands)
        length_cm = sum(command.distance_cm for command in mission_leg.commands)
        plan.legs.append(Leg(mission_leg.to_id, poses, turn_centres, length_cm))
        plan.length_cm += length_cm
        emit(
            f"Leg {mission_leg.from_id} -> {mission_leg.to_id}: {length_cm}cm "
            f"over {len(mission_leg.commands)} commands"
        )
        current_pose = poses[-1]

    plan.skipped_ids = mission.skipped_ids
    for skipped_id in mission.skipped_ids:
        emit(f"NO PATH FOUND to obstacle {skipped_id}, skipping it")
    return plan


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SC2079 Arena Simulator")

        self.rng = random.Random()
        self.start_cell = ROBOT_START_CELL
        self.start_robot = Robot.from_grid(*self.start_cell)
        self.obstacles: list[Obstacle] = []
        self.image_ids: dict[int, int] = {}
        self.plan: Plan | None = None

        self.robot = self.start_robot
        self.trail_px: list[float] = []
        self.trail_item: int | None = None
        self.animation_job: str | None = None
        self.leg_index = 0
        self.pose_index = 0
        self.running = False

        self.events: queue.Queue = queue.Queue()

        self._build_widgets()
        self._draw_static()
        self._draw_robot()
        self._update_buttons()
        self._log(
            "Ready. Robot footprint's bottom-left at grid "
            f"({ROBOT_START_CELL[0]}, {ROBOT_START_CELL[1]}) facing "
            f"{ROBOT_START_CELL[2].name}."
        )
        self._log(
            "Rear axle (dark line and dot) at "
            f"({self.start_robot.x_cm:.0f}, {self.start_robot.y_cm:.0f})cm, "
            f"{AXLE_TO_REAR_CM}cm in from the back of the footprint. While "
            "turning, the purple dot is the point the robot pivots about."
        )
        self.root.after(POLL_MS, self._drain_events)

    # -- layout ------------------------------------------------------------

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_PX,
            height=CANVAS_PX,
            background=COLOURS["arena"],
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nw")

        side = ttk.Frame(frame, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(side, text="Controls", padding=8)
        controls.pack(fill=tk.X)

        count_row = ttk.Frame(controls)
        count_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(count_row, text="Obstacles:").pack(side=tk.LEFT)
        self.count_var = tk.IntVar(value=MIN_OBSTACLES)
        ttk.Spinbox(
            count_row,
            from_=MIN_OBSTACLES,
            to=MAX_OBSTACLES,
            textvariable=self.count_var,
            width=4,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        self.generate_button = ttk.Button(
            controls, text="Generate obstacles", command=self.on_generate
        )
        self.generate_button.pack(fill=tk.X, pady=2)

        self._build_manual_controls(controls)

        self.plan_button = ttk.Button(controls, text="Plan path", command=self.on_plan)
        self.plan_button.pack(fill=tk.X, pady=2)

        self.run_button = ttk.Button(controls, text="Start robot", command=self.on_run)
        self.run_button.pack(fill=tk.X, pady=2)

        self.reset_button = ttk.Button(
            controls, text="Reset robot", command=self.on_reset
        )
        self.reset_button.pack(fill=tk.X, pady=2)

        speed_row = ttk.Frame(controls)
        speed_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(speed_row, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=DEFAULT_SPEED)
        ttk.Scale(
            speed_row,
            from_=MIN_SPEED,
            to=MAX_SPEED,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.status_var = tk.StringVar(value="No obstacles yet.")
        ttk.Label(side, textvariable=self.status_var, wraplength=320).pack(
            fill=tk.X, pady=(8, 4)
        )

        log_frame = ttk.LabelFrame(side, text="Event log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(
            log_frame, width=44, height=24, wrap=tk.WORD, state=tk.DISABLED
        )
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_manual_controls(self, parent: ttk.Frame) -> None:
        manual = ttk.LabelFrame(parent, text="Manual placement", padding=6)
        manual.pack(fill=tk.X, pady=(6, 6))

        self.manual_var = tk.BooleanVar(value=False)
        self.manual_check = ttk.Checkbutton(
            manual,
            text="Place obstacles by clicking",
            variable=self.manual_var,
            command=self._on_manual_toggled,
        )
        self.manual_check.pack(anchor="w")

        heading_row = ttk.Frame(manual)
        heading_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(heading_row, text="Image side:").pack(side=tk.LEFT)
        self.heading_var = tk.StringVar(value=Direction.NORTH.name)
        ttk.Combobox(
            heading_row,
            textvariable=self.heading_var,
            values=[heading.name for heading in HEADING_CYCLE],
            width=7,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            manual,
            text=(
                f"Left-click an empty cell to place (max {MAX_OBSTACLES}), "
                "left-click an obstacle to rotate its image side, "
                "right-click to remove."
            ),
            wraplength=300,
            foreground=COLOURS["label"],
        ).pack(fill=tk.X, pady=(2, 4))

        self.clear_button = ttk.Button(
            manual, text="Clear obstacles", command=self.on_clear
        )
        self.clear_button.pack(fill=tk.X, pady=2)

        self.json_button = ttk.Button(manual, text="Show JSON", command=self.on_show_json)
        self.json_button.pack(fill=tk.X, pady=2)

        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        for event in RIGHT_CLICK_EVENTS:
            self.canvas.bind(event, self.on_canvas_right_click)

    # -- canvas ------------------------------------------------------------

    def _to_px(self, x_cm: float, y_cm: float) -> Point:
        """cm in arena coordinates -> canvas pixels (y flipped: canvas grows down)."""
        return (
            MARGIN_PX + x_cm * PX_PER_CM,
            MARGIN_PX + (ARENA_LENGTH_CM - y_cm) * PX_PER_CM,
        )

    def _px_to_cell(self, x_px: float, y_px: float) -> tuple[int, int] | None:
        """Canvas pixels -> the grid cell under them, or None off the arena."""
        x_cm = (x_px - MARGIN_PX) / PX_PER_CM
        y_cm = ARENA_LENGTH_CM - (y_px - MARGIN_PX) / PX_PER_CM
        if not (0 <= x_cm < ARENA_LENGTH_CM and 0 <= y_cm < ARENA_LENGTH_CM):
            return None
        return int(x_cm // GRID_LENGTH_CM), int(y_cm // GRID_LENGTH_CM)

    def _polygon_px(self, corners) -> list[float]:
        return [value for corner in corners for value in self._to_px(*corner)]

    def _draw_static(self) -> None:
        self.canvas.delete("static")

        left, bottom = self._to_px(0, 0)
        right, top = self._to_px(ARENA_LENGTH_CM, ARENA_LENGTH_CM)
        for i in range(NUM_GRIDS + 1):
            offset = i * GRID_LENGTH_CM
            x_px, y_px = self._to_px(offset, offset)
            self.canvas.create_line(
                x_px, top, x_px, bottom, fill=COLOURS["grid"], tags="static"
            )
            self.canvas.create_line(
                left, y_px, right, y_px, fill=COLOURS["grid"], tags="static"
            )

        # Cell indices along the bottom and left edges.
        for cell in range(NUM_GRIDS):
            centre = cell * GRID_LENGTH_CM + GRID_LENGTH_CM / 2
            x_px, _ = self._to_px(centre, 0)
            _, y_px = self._to_px(0, centre)
            self.canvas.create_text(
                x_px,
                CANVAS_PX - MARGIN_PX / 2,
                text=str(cell),
                font=("TkDefaultFont", 7),
                fill=COLOURS["label"],
                tags="static",
            )
            self.canvas.create_text(
                MARGIN_PX / 2,
                y_px,
                text=str(cell),
                font=("TkDefaultFont", 7),
                fill=COLOURS["label"],
                tags="static",
            )

        # The 40cm x 40cm start area.
        zone_cm = START_ZONE_CELLS * GRID_LENGTH_CM
        x0, y0 = self._to_px(0, zone_cm)
        x1, y1 = self._to_px(zone_cm, 0)
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=COLOURS["start_zone"],
            outline=COLOURS["start_edge"],
            width=2,
            dash=(4, 3),
            tags="static",
        )
        self.canvas.create_text(
            (x0 + x1) / 2,
            y0 + 10,
            text="START 40x40",
            font=("TkDefaultFont", 7, "bold"),
            fill=COLOURS["start_edge"],
            tags="static",
        )
        self.canvas.tag_lower("static")

        # Arena wall.
        self.canvas.create_rectangle(
            left, top, right, bottom, outline=COLOURS["border"], width=2, tags="static"
        )

    def _draw_obstacles(self) -> None:
        self.canvas.delete("obstacle")
        size = OBSTACLE_FOOTPRINT_LENGTH_CM

        for obstacle in self.obstacles:
            # The inflated outline first, so the true block sits on top of it.
            # This washed-out square is what every collision check actually sees.
            self.canvas.create_polygon(
                self._polygon_px(obstacle.inflated_footprint_corners_cm()),
                fill=COLOURS["obstacle_margin"],
                stipple="gray25",
                outline=COLOURS["obstacle_margin"],
                dash=(3, 3),
                tags="obstacle",
            )
            corners = obstacle.footprint_corners_cm()
            self.canvas.create_polygon(
                self._polygon_px(corners),
                fill=COLOURS["obstacle"],
                outline="black",
                tags="obstacle",
            )
            centre_cm = (
                obstacle.x_coord * GRID_LENGTH_CM + size / 2,
                obstacle.y_coord * GRID_LENGTH_CM + size / 2,
            )
            self.canvas.create_text(
                *self._to_px(*centre_cm),
                text=str(obstacle.id),
                fill="white",
                font=("TkDefaultFont", 8, "bold"),
                tags="obstacle",
            )

            # The face carrying the image: a thick bar on that edge, plus an
            # arrow pointing the way the camera has to look from.
            dx, dy = obstacle.image_side.value
            edge = self._image_side_edge(corners, obstacle.image_side)
            self.canvas.create_line(
                *self._to_px(*edge[0]),
                *self._to_px(*edge[1]),
                fill=COLOURS["image_side"],
                width=4,
                tags="obstacle",
            )
            self.canvas.create_line(
                *self._to_px(*centre_cm),
                *self._to_px(centre_cm[0] + dx * size, centre_cm[1] + dy * size),
                fill=COLOURS["image_side"],
                width=2,
                arrow=tk.LAST,
                tags="obstacle",
            )

    @staticmethod
    def _image_side_edge(corners: Corners, side: Direction) -> tuple[Point, Point]:
        """Obstacle corners run anticlockwise from the bottom-left."""
        bottom_left, bottom_right, top_right, top_left = corners
        return {
            Direction.SOUTH: (bottom_left, bottom_right),
            Direction.EAST: (bottom_right, top_right),
            Direction.NORTH: (top_left, top_right),
            Direction.WEST: (bottom_left, top_left),
        }[side]

    def _draw_plan(self) -> None:
        self.canvas.delete("plan")
        if self.plan is None:
            return

        for leg in self.plan.legs:
            # The planner's poses are the robot's REAR-AXLE MIDPOINT, so that is
            # what the planned path traces - not the robot's centre. It is the
            # only point whose motion hybrid_astar actually models.
            points = [
                value
                for pose in leg.poses
                for value in self._to_px(pose.x_cm, pose.y_cm)
            ]
            if len(points) >= 4:
                self.canvas.create_line(
                    points, fill=COLOURS["path"], width=2, tags="plan"
                )

        for visit_index, node_id in enumerate(self.plan.order):
            pose = self.plan.node_robots[node_id]
            self.canvas.create_polygon(
                self._polygon_px(pose.footprint_corners_cm()),
                fill="",
                outline=COLOURS["pose_outline"],
                dash=(3, 3),
                tags="plan",
            )
            x_px, y_px = self._to_px(pose.x_cm, pose.y_cm)
            colour = COLOURS["start_edge"] if node_id == "S" else COLOURS["pose"]
            self.canvas.create_oval(
                x_px - 4,
                y_px - 4,
                x_px + 4,
                y_px + 4,
                fill=colour,
                outline="",
                tags="plan",
            )
            front = _front_edge_cm(pose)
            front_mid = (
                (front[0][0] + front[1][0]) / 2,
                (front[0][1] + front[1][1]) / 2,
            )
            # The heading arrow stays on the footprint's centre line; only the
            # marker itself sits on the pose.
            self.canvas.create_line(
                *self._to_px(*_footprint_centre_cm(pose)),
                *self._to_px(*front_mid),
                fill=colour,
                width=2,
                arrow=tk.LAST,
                tags="plan",
            )
            self.canvas.create_text(
                x_px + 12,
                y_px - 10,
                text=f"{visit_index}:{node_id}",
                font=("TkDefaultFont", 7, "bold"),
                fill=colour,
                tags="plan",
            )

    def _draw_robot(self, turn_centre_cm: Point | None = None) -> None:
        """Draws self.robot. Pass the arc's centre while it is turning, to
        show the point it pivots about."""
        self.canvas.delete("robot")
        corners = self.robot.footprint_corners_cm()

        # The robot's true footprint, ROBOT_LENGTH_CM x ROBOT_WIDTH_CM...
        self.canvas.create_polygon(
            self._polygon_px(corners),
            fill=COLOURS["robot"],
            stipple="gray50",
            outline=COLOURS["robot"],
            width=2,
            tags="robot",
        )
        # ...with the leading edge and a heading arrow so the front is obvious.
        front_left, front_right = _front_edge_cm(self.robot)
        self.canvas.create_line(
            *self._to_px(*front_left),
            *self._to_px(*front_right),
            fill=COLOURS["robot_front"],
            width=4,
            tags="robot",
        )
        # The rear axle, wheel to wheel, AXLE_TO_REAR_CM in from the back.
        axle_left, axle_right = _rear_axle_cm(self.robot)
        self.canvas.create_line(
            *self._to_px(*axle_left),
            *self._to_px(*axle_right),
            fill=COLOURS["axle"],
            width=3,
            tags="robot",
        )
        # Axle -> nose, so the arrow gives the heading AND shows where in the
        # body the tracked point actually sits.
        front_mid = (
            (front_left[0] + front_right[0]) / 2,
            (front_left[1] + front_right[1]) / 2,
        )
        axle_x_px, axle_y_px = self._to_px(self.robot.x_cm, self.robot.y_cm)
        self.canvas.create_line(
            axle_x_px,
            axle_y_px,
            *self._to_px(*front_mid),
            fill=COLOURS["robot_front"],
            width=2,
            arrow=tk.LAST,
            tags="robot",
        )
        if turn_centre_cm is not None:
            self._draw_turn_centre(turn_centre_cm)
        # The rear-axle midpoint itself, drawn last so it stays visible: this is
        # the pose hybrid_astar propagates, and what the path and trail trace.
        self.canvas.create_oval(
            axle_x_px - 4,
            axle_y_px - 4,
            axle_x_px + 4,
            axle_y_px + 4,
            fill=COLOURS["axle"],
            outline="white",
            tags="robot",
        )

    def _draw_turn_centre(self, centre_cm: Point) -> None:
        """The point the robot is pivoting about, joined to its rear-axle
        midpoint. A car pivots about a point level with its rear axle, so
        this dashed line should always run straight along the axle."""
        centre_x_px, centre_y_px = self._to_px(*centre_cm)
        self.canvas.create_line(
            centre_x_px,
            centre_y_px,
            *self._to_px(self.robot.x_cm, self.robot.y_cm),
            fill=COLOURS["turn_centre"],
            dash=(4, 3),
            tags="robot",
        )
        self.canvas.create_oval(
            centre_x_px - 4,
            centre_y_px - 4,
            centre_x_px + 4,
            centre_y_px + 4,
            fill=COLOURS["turn_centre"],
            outline="",
            tags="robot",
        )

    def _extend_trail(self) -> None:
        # Same point the planned path traces: the robot's rear-axle midpoint.
        self.trail_px.extend(self._to_px(self.robot.x_cm, self.robot.y_cm))
        if len(self.trail_px) < 4:
            return
        if self.trail_item is None:
            self.trail_item = self.canvas.create_line(
                self.trail_px, fill=COLOURS["trail"], width=2, tags="trail"
            )
        else:
            self.canvas.coords(self.trail_item, *self.trail_px)

    def _clear_trail(self) -> None:
        self.canvas.delete("trail")
        self.trail_px = []
        self.trail_item = None

    # -- actions -----------------------------------------------------------

    def on_generate(self) -> None:
        count = self.count_var.get()
        self._set_obstacles(generate_obstacles(count, self.rng))

        self.status_var.set(f"{count} obstacles placed. Plan a path next.")
        self._log(f"Generated {count} obstacles:")
        for obstacle in self.obstacles:
            pose = obstacle.cm_viewing_position()
            self._log(
                f"  #{obstacle.id} at ({obstacle.x_coord}, {obstacle.y_coord}), "
                f"image faces {obstacle.image_side.name}, "
                f"viewing axle ({pose.x_cm:.0f}, {pose.y_cm:.0f})cm "
                f"facing {obstacle.image_side.opposite.name}"
            )

    def on_clear(self) -> None:
        self._set_obstacles([])
        self.status_var.set("Obstacles cleared.")
        self._log("Obstacles cleared.")

    def on_canvas_left_click(self, event: tk.Event) -> None:
        cell = self._editable_cell(event)
        if cell is None:
            return

        existing = self._obstacle_at(cell)
        if existing is not None:
            self._rotate_obstacle(existing)
            return
        self._place_obstacle(cell)

    def on_canvas_right_click(self, event: tk.Event) -> None:
        cell = self._editable_cell(event)
        if cell is None:
            return

        existing = self._obstacle_at(cell)
        if existing is None:
            return
        self._set_obstacles(_renumbered([o for o in self.obstacles if o is not existing]))
        self._log(f"Removed obstacle at {cell}; {len(self.obstacles)} left.")
        self.status_var.set(f"{len(self.obstacles)} obstacles placed.")

    def on_show_json(self) -> None:
        text = json.dumps(scenario_dict(self.start_cell, self.obstacles), indent=JSON_INDENT)
        print(text, flush=True)

        window = tk.Toplevel(self.root)
        window.title("Scenario JSON")
        box = tk.Text(window, width=48, height=30, wrap=tk.NONE)
        box.insert("1.0", text)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        def copy() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(box.get("1.0", "end-1c"))
            self._log("Scenario JSON copied to clipboard.")

        def import_json() -> None:
            if self._import_scenario(box.get("1.0", "end-1c")):
                window.destroy()

        buttons = ttk.Frame(window)
        buttons.pack(pady=(0, 8))
        ttk.Button(buttons, text="Copy to clipboard", command=copy).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Import JSON", command=import_json).pack(side=tk.LEFT, padx=4)

    def _import_scenario(self, text: str) -> bool:
        """Replace the robot start and obstacles with a pasted scenario.
        Returns whether the import went through."""
        if self._is_layout_locked():
            self._log("Can't import while planning or driving; wait or pause first.")
            return False
        try:
            data = json.loads(text)
            start_robot, obstacles = parse_scenario(data)
        except (ValueError, KeyError, TypeError) as exc:
            self._log(f"Import failed, not a valid scenario: {exc!r}")
            self.status_var.set("Import failed: not a valid scenario JSON.")
            return False

        robot = data["robot"]
        self.start_cell = (robot["x_coord"], robot["y_coord"], Direction[robot["facing"]])
        self.start_robot = start_robot
        self._set_obstacles(obstacles)
        self._log(
            f"Imported scenario: robot at ({self.start_cell[0]}, {self.start_cell[1]}) "
            f"facing {self.start_cell[2].name}, {len(obstacles)} obstacles."
        )
        self.status_var.set(f"{len(obstacles)} obstacles imported. Plan a path next.")
        return True

    def _on_manual_toggled(self) -> None:
        manual = self.manual_var.get()
        self.canvas.configure(cursor="crosshair" if manual else "")
        if manual:
            self.status_var.set("Manual placement: click the arena to place obstacles.")

    def _editable_cell(self, event: tk.Event) -> tuple[int, int] | None:
        """The clicked cell, or None when the layout must not change right now."""
        if not self.manual_var.get() or self._is_layout_locked():
            return None
        return self._px_to_cell(event.x, event.y)

    def _is_layout_locked(self) -> bool:
        return self.running or getattr(self, "busy", False)

    def _obstacle_at(self, cell: tuple[int, int]) -> Obstacle | None:
        return next(
            (o for o in self.obstacles if (o.x_coord, o.y_coord) == cell), None
        )

    def _rotate_obstacle(self, obstacle: Obstacle) -> None:
        heading = _next_heading(obstacle.image_side)
        rotated = Obstacle(obstacle.id, obstacle.x_coord, obstacle.y_coord, heading)
        self._set_obstacles([rotated if o is obstacle else o for o in self.obstacles])
        self._log(f"Obstacle {obstacle.id} image side -> {heading.name}.")

    def _place_obstacle(self, cell: tuple[int, int]) -> None:
        x_coord, y_coord = cell
        if x_coord < START_ZONE_CELLS and y_coord < START_ZONE_CELLS:
            self._log(f"Cell {cell} is inside the start area; keep it clear.")
            return
        if len(self.obstacles) >= MAX_OBSTACLES:
            self._log(f"Already {MAX_OBSTACLES} obstacles; remove one first.")
            return

        heading = Direction[self.heading_var.get()]
        # Imported ids need not be 0..n-1, so take the next free one.
        next_id = max((o.id for o in self.obstacles), default=-1) + 1
        obstacle = Obstacle(next_id, x_coord, y_coord, heading)
        self._set_obstacles(self.obstacles + [obstacle])
        self._log(f"Placed obstacle {obstacle.id} at {cell}, image faces {heading.name}.")
        self.status_var.set(f"{len(self.obstacles)} obstacles placed.")

    def _set_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Swap in a new layout: any plan or drive over the old one is void."""
        self._stop_animation()
        self.obstacles = obstacles
        self.image_ids = {
            obstacle.id: self.image_ids.get(obstacle.id, self.rng.choice(IMAGE_IDS))
            for obstacle in obstacles
        }
        self.plan = None
        self.leg_index = 0
        self.pose_index = 0
        self.robot = self.start_robot

        self._clear_trail()
        self.canvas.delete("plan")
        self._draw_obstacles()
        self._draw_robot()
        self._update_buttons()

    def on_plan(self) -> None:
        self._stop_animation()
        self.plan = None
        self._clear_trail()
        self.canvas.delete("plan")
        self.robot = self.start_robot
        self._draw_robot()

        self.status_var.set("Planning...")
        self._log(f"Planning route over {len(self.obstacles)} obstacles...")
        self._set_busy(True)

        obstacles = list(self.obstacles)
        start = self.start_robot

        def work() -> None:
            emit = lambda text: self.events.put(("log", text))
            try:
                self.events.put(("plan", plan_route(start, obstacles, emit)))
            except Exception as exc:  # a planner crash must not kill the UI
                self.events.put(("error", f"Planning failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def on_run(self) -> None:
        if self.running:
            self._stop_animation()
            self._log("Paused.")
            self.status_var.set("Paused.")
            self._update_buttons()
            return

        if self.plan is None or not self.plan.legs:
            return

        if self.leg_index >= len(self.plan.legs):  # finished route: start over
            self.on_reset()

        self.running = True
        self._update_buttons()
        self.status_var.set("Robot running...")
        if self.leg_index == 0 and self.pose_index == 0:
            self._log("Robot started.")
        self._step()

    def on_reset(self) -> None:
        self._stop_animation()
        self.leg_index = 0
        self.pose_index = 0
        self.robot = self.start_robot
        self._clear_trail()
        self._draw_robot()
        self._update_buttons()
        self.status_var.set("Robot back at the start.")

    # -- animation ---------------------------------------------------------

    def _step(self) -> None:
        self.animation_job = None
        if not self.running or self.plan is None:
            return

        leg = self.plan.legs[self.leg_index]
        self.robot = leg.poses[self.pose_index]
        self._draw_robot(leg.turn_centres[self.pose_index])
        self._extend_trail()
        self.pose_index += 1

        at_viewing_pose = self.pose_index >= len(leg.poses)
        if at_viewing_pose:
            self._on_leg_complete(leg)
            self.leg_index += 1
            self.pose_index = 1  # the next leg starts where this one ended
            if self.leg_index >= len(self.plan.legs):
                self._on_route_complete()
                return
            # Sit still for a moment, the way the real robot waits on the
            # camera before driving off to the next obstacle.
            self.status_var.set(
                f"Capturing at obstacle {leg.goal_id} ({VIEWING_PAUSE_MS / 1000:.0f}s)..."
            )
            self.animation_job = self.root.after(VIEWING_PAUSE_MS, self._resume_driving)
            return

        delay = max(1, int(ANIMATION_MS / self.speed_var.get()))
        self.animation_job = self.root.after(delay, self._step)

    def _resume_driving(self) -> None:
        if not self.running:
            return
        self.status_var.set("Robot running...")
        self._step()

    def _on_leg_complete(self, leg: Leg) -> None:
        obstacle = next((o for o in self.obstacles if o.id == leg.goal_id), None)
        if obstacle is None:
            return

        pose = obstacle.cm_viewing_position()
        self._log(
            f"Obstacle {obstacle.id}: reached viewing pose "
            f"({pose.x_cm:.0f}, {pose.y_cm:.0f})cm facing "
            f"{obstacle.image_side.opposite.name} after {leg.length_cm:.0f}cm"
        )
        self._log(
            f"Obstacle {obstacle.id}: image recognised -> ID {self.image_ids[obstacle.id]}"
        )

    def _on_route_complete(self) -> None:
        self.running = False
        self._update_buttons()
        recognised = len(self.plan.legs) if self.plan else 0
        self.status_var.set(f"Route complete: {recognised} images recognised.")
        self._log(
            f"Route complete. {recognised}/{len(self.obstacles)} obstacles visited, "
            f"{self.plan.length_cm:.0f}cm driven."
        )

    def _stop_animation(self) -> None:
        self.running = False
        if self.animation_job is not None:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None

    # -- plumbing ----------------------------------------------------------

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._log(payload)
            elif kind == "error":
                self._log(payload)
                self.status_var.set(payload)
                self._set_busy(False)
            elif kind == "plan":
                self._on_plan_ready(payload)

        self.root.after(POLL_MS, self._drain_events)

    def _on_plan_ready(self, plan: Plan) -> None:
        self.plan = plan
        self.leg_index = 0
        self.pose_index = 0
        self._draw_plan()
        self._draw_robot()
        self._set_busy(False)

        if not plan.legs:
            self.status_var.set("Planning failed: no obstacle is reachable at all.")
            self._log("Planning failed: not one obstacle in the layout is reachable.")
            return

        if plan.skipped_ids:
            self.status_var.set(
                f"Path planned: {len(plan.legs)} legs, {plan.length_cm:.0f}cm total "
                f"({len(plan.skipped_ids)} obstacle(s) skipped)."
            )
            self._log(
                f"Path planned: {len(plan.legs)} legs, {plan.length_cm:.0f}cm total. "
                f"Skipped unreachable obstacles: {plan.skipped_ids}."
            )
            return

        self.status_var.set(
            f"Path planned: {len(plan.legs)} legs, {plan.length_cm:.0f}cm total."
        )
        self._log(f"Path planned: {len(plan.legs)} legs, {plan.length_cm:.0f}cm total.")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = getattr(self, "busy", False)
        has_plan = self.plan is not None and bool(self.plan.legs)

        self.generate_button.configure(
            state=tk.DISABLED if busy or self.running else tk.NORMAL
        )
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.obstacles and not busy and not self.running
                else tk.DISABLED
            )
        )
        self.run_button.configure(
            state=tk.NORMAL if has_plan and not busy else tk.DISABLED,
            text="Pause robot" if self.running else "Start robot",
        )
        self.reset_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        editable = tk.DISABLED if busy or self.running else tk.NORMAL
        self.manual_check.configure(state=editable)
        self.clear_button.configure(state=editable)

    def _log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
