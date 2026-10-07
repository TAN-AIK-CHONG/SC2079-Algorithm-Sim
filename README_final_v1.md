# final_v1 — RPi Group 24 working planner

This is the planner the **RPi side is currently running** (vendored into the Pi
`full-pipeline-v2`). It is the branch to build against for integration.

## What it is
`final_v1` = base commit **`f9835a0`** ("update inflated obstacle") **+ one fix**.
`f9835a0` is the version we vendored; it already exists in the repo history
(reachable from `main` and `finegrain-turns-fallback`), so you have it — this
branch just adds the single change below on top of it.

## The one change — viewing-pose standoff (`src/model.py`)
The STM drives the **rear-axle midpoint**, but `cm_viewing_position()` was
building the standoff with `ROBOT_LENGTH_CM` (i.e. placing the **rear bumper**).
Result: when the STM drove the axle to the planned pose, the robot's **nose
stopped ~2 cm too far** from the box.

Fix: add `AXLE_TO_FRONT_CM = 21` (real rear-axle → nose) and use it in the
standoff instead of `ROBOT_LENGTH_CM`:

```python
standoff = OBSTACLE_FOOTPRINT_LENGTH_CM / 2   # 5  (half the 10 cm box)
         + CAMERA_CLEARANCE_LENGTH_CM         # 20 (nose clearance we want)
         + AXLE_TO_FRONT_CM                   # 21 (rear axle -> nose)  [was ROBOT_LENGTH_CM]
```

Now the rear axle is placed **46 cm** from the box centre → **41 cm** from the
face → the **nose lands exactly 20 cm (`CAMERA_CLEARANCE_LENGTH_CM`) off the box
face**, which is where recognition expects it. Verified on our 7-obstacle map.

## What was intentionally NOT changed
We left the **collision footprint** as the existing `23 × 19` rear-bumper box.
Switching it to the true rear-axle `25 × 20` footprint is more correct, but it
made this (hybrid A*) planner fail to find paths on a cluttered map, so it's out
of scope here. (The `finegrain-turns-fallback` branch already models the robot
that way and plans fine — see that branch if you want the full-geometry version.)

## Diff summary
- `src/model.py`: `+ AXLE_TO_FRONT_CM = 21`; `cm_viewing_position` standoff uses
  `AXLE_TO_FRONT_CM` instead of `ROBOT_LENGTH_CM`.
- Everything else: identical to `f9835a0`, except the shared tooling and
  collision check below.

## Shared with `astar_90deg`
These are kept identical to `astar_90deg` so all branches are tested the same way:
- `src/testing/gui_simulator.py`, `generate_maps.py`, `visualize_map.py`: copied
  verbatim. They plan through `plan_mission`, like `rpi/main.py`.
- `src/collision.py`: copied verbatim. The SAT test skips the car's own axes when
  it faces along x or y, and the arena wall allows `WALL_TOLERANCE_CM` (3 mm).
- `src/algorithms/hybrid_astar.py`: each move only checks obstacles within
  `MOVE_COLLISION_RADIUS_CM` (85 cm) of the car.
- `src/model.py`: `AXLE_TO_REAR_CM = 0`. The tools need it, and 0 keeps the
  23 x 19 collision box measured from the pose, unchanged.

Compare: `git diff f9835a0..final_v1`
