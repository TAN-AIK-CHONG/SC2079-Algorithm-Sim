"""Tkinter simulator for the 20x20 arena: generate obstacles, plan a route
over them, and watch the robot drive it.

Run from src/:
    python testing/gui_simulator.py
"""

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
from algorithms.hybrid_astar import hybrid_astar
from collision import footprint_in_collision
from model import (
    ARENA_LENGTH_CM,
    GRID_LENGTH_CM,
    NUM_GRIDS,
    OBSTACLE_FOOTPRINT_LENGTH_CM,
    Corners,
    Direction,
    Obstacle,
    Point,
    Robot,
)

CELL_PX = 30
MARGIN_PX = 26
PX_PER_CM = CELL_PX / GRID_LENGTH_CM
CANVAS_PX = NUM_GRIDS * CELL_PX + 2 * MARGIN_PX

# The 40cm x 40cm start area, and where the robot's own footprint sits inside it.
START_ZONE_CELLS = 4
ROBOT_START_CELL = (1, 1, Direction.NORTH)

MIN_OBSTACLES = 4
MAX_OBSTACLES = 8
# 4 cells between obstacle origins leaves a 30cm gap between their footprints -
# exactly the robot's width, so every gap is a corridor it can actually drive.
# Tighter than this and roughly half of all layouts have an unreachable obstacle.
MIN_OBSTACLE_SEPARATION_CELLS = 4
IMAGE_IDS = tuple(range(11, 41))  # the target IDs the camera can come back with

ANIMATION_MS = 40  # one path step (5cm of driving) per tick
POLL_MS = 50  # how often the UI drains the planner thread's queue
VIEWING_PAUSE_MS = 3000  # how long the robot sits at a viewing pose "taking a photo"

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
    "image_side": "#d64545",
    "robot": "#4a90d9",
    "robot_front": "#f5a623",
    "pose": "#e08a00",
    "pose_outline": "#b0b0b0",
    "path": "#1f77b4",
    "trail": "#d64545",
    "label": "#777777",
}


def _footprint_centre_cm(robot: Robot) -> Point:
    """The robot's pose is its rear-left corner; markers read better at the centre."""
    corners = robot.footprint_corners_cm()
    return (
        sum(x for x, _ in corners) / 4,
        sum(y for _, y in corners) / 4,
    )


def _front_edge_cm(robot: Robot) -> tuple[Point, Point]:
    """The two corners one footprint-length ahead of the pose: the robot's face."""
    corners = robot.footprint_corners_cm()
    return corners[1], corners[2]


@dataclass
class Leg:
    """One planned hop, from the previous node to `goal_id`."""

    goal_id: str | int
    poses: list[Robot]
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

        footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]
        if all(_viewing_pose_ok(obstacle, footprints) for obstacle in obstacles):
            return obstacles


# --------------------------------------------------------------------------
# Planning (runs off the UI thread)
# --------------------------------------------------------------------------


def plan_route(start: Robot, obstacles: list[Obstacle], emit) -> Plan:
    """Plan a route over obstacles in exhaustive_search's order. If an
    obstacle turns out to be unreachable from wherever the robot currently
    is, it's skipped - not visited - and planning continues toward the next
    obstacle in the order, still from the last position actually reached.
    One unreachable obstacle no longer cancels every obstacle after it."""
    graph = Graph.build(start, obstacles)
    order = exhaustive_search(graph)
    emit(
        f"Order (exhaustive search): {' -> '.join(str(i) for i in order)}"
        f"  [{path_length(graph, order):.0f}cm as Reeds-Shepp hops]"
    )

    node_robots = {node.id: node.viewing_pose for node in graph.nodes}
    footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]
    plan = Plan(order=order, node_robots=node_robots)

    current_id = order[0]  # "S"
    for target_id in order[1:]:
        result = hybrid_astar(node_robots[current_id], node_robots[target_id], footprints)
        if result is None:
            plan.skipped_ids.append(target_id)
            emit(f"Leg {current_id} -> {target_id}: NO PATH FOUND, skipping obstacle {target_id}")
            continue  # stay at current_id, try the next obstacle in the order instead

        plan.legs.append(Leg(target_id, result.path, result.length))
        plan.length_cm += result.length
        emit(
            f"Leg {current_id} -> {target_id}: {result.length:.0f}cm over {len(result.path)} steps"
        )
        current_id = target_id

    return plan


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SC2079 Arena Simulator")

        self.rng = random.Random()
        self.start_robot = Robot.from_grid(*ROBOT_START_CELL)
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
            "Ready. Robot parked at grid "
            f"({ROBOT_START_CELL[0]}, {ROBOT_START_CELL[1]}) facing "
            f"{ROBOT_START_CELL[2].name}."
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

    # -- canvas ------------------------------------------------------------

    def _to_px(self, x_cm: float, y_cm: float) -> Point:
        """cm in arena coordinates -> canvas pixels (y flipped: canvas grows down)."""
        return (
            MARGIN_PX + x_cm * PX_PER_CM,
            MARGIN_PX + (ARENA_LENGTH_CM - y_cm) * PX_PER_CM,
        )

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
            # The planner's poses are the footprint's bottom-left corner, so that
            # is what the planned path traces - not the robot's centre.
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

    def _draw_robot(self) -> None:
        self.canvas.delete("robot")
        corners = self.robot.footprint_corners_cm()

        # Full 30cm x 30cm footprint...
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
        # The pose the planner actually tracks, so it is visibly on the path.
        pose_x_px, pose_y_px = self._to_px(self.robot.x_cm, self.robot.y_cm)
        self.canvas.create_oval(
            pose_x_px - 3,
            pose_y_px - 3,
            pose_x_px + 3,
            pose_y_px + 3,
            fill=COLOURS["robot"],
            outline="",
            tags="robot",
        )
        centre = _footprint_centre_cm(self.robot)
        front_mid = (
            (front_left[0] + front_right[0]) / 2,
            (front_left[1] + front_right[1]) / 2,
        )
        self.canvas.create_line(
            *self._to_px(*centre),
            *self._to_px(*front_mid),
            fill=COLOURS["robot_front"],
            width=2,
            arrow=tk.LAST,
            tags="robot",
        )

    def _extend_trail(self) -> None:
        # Same point the planned path traces: the footprint's bottom-left corner.
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
        self._stop_animation()
        count = self.count_var.get()
        self.obstacles = generate_obstacles(count, self.rng)
        self.image_ids = {
            obstacle.id: self.rng.choice(IMAGE_IDS) for obstacle in self.obstacles
        }
        self.plan = None
        self.robot = self.start_robot

        self._clear_trail()
        self.canvas.delete("plan")
        self._draw_obstacles()
        self._draw_robot()
        self._update_buttons()

        self.status_var.set(f"{count} obstacles placed. Plan a path next.")
        self._log(f"Generated {count} obstacles:")
        for obstacle in self.obstacles:
            x, y, facing = obstacle.grid_viewing_position()
            self._log(
                f"  #{obstacle.id} at ({obstacle.x_coord}, {obstacle.y_coord}), "
                f"image faces {obstacle.image_side.name}, "
                f"viewing pose ({x}, {y}) facing {facing.name}"
            )

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
        self._draw_robot()
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

        x, y, facing = obstacle.grid_viewing_position()
        self._log(
            f"Obstacle {obstacle.id}: reached viewing pose ({x}, {y}) "
            f"facing {facing.name} after {leg.length_cm:.0f}cm"
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
