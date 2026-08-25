# Algorithm API — how to call it from the RPi

This describes `src/api.py`: a small HTTP server that takes the robot's
starting position and the list of obstacles, and returns the order to visit
them in plus the exact coordinates to drive to. This doc is written for
whoever is writing the RPi-side code that calls it (Aasim).

## 1. Starting the server

Someone (whoever's running the algorithm code, on their laptop) needs to
have this running before the RPi can call it:

```powershell
cd src
python api.py
```

When it starts, it prints something like:

```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.20.10.2:5000
```

**That last IP address (`172.20.10.2` here) is what the RPi needs to
connect to** — copy it from whatever the terminal actually prints, don't
reuse this example value. It changes depending on what network the laptop
is connected to, so check it fresh each time you set up for a test run.

Both the laptop running `api.py` and the RPi need to be on the **same
network** (e.g. both connected to the RPi's own hotspot, or both connected
to the same router). If requests time out even though the server is clearly
running, check Windows Firewall isn't blocking the incoming connection.

## 2. Checking it's reachable: `GET /health`

Before wiring up the real call, confirm the RPi can actually reach the
server at all:

```python
import requests
requests.get("http://172.20.10.2:5000/health", timeout=5).json()
# {"status": "ok"}
```

If this fails (connection refused / timeout), the problem is network
reachability, not the planning logic — fix that first.

## 3. Getting a plan: `POST /plan`

### Request

Send a JSON body with the robot's position and the obstacle list:

```json
{
  "robot": {
    "x_coord": 1,
    "y_coord": 1,
    "facing": "NORTH"
  },
  "obstacles": [
    {"id": "1", "x_coord": 5, "y_coord": 15, "image_side": "SOUTH"},
    {"id": "2", "x_coord": 10, "y_coord": 5, "image_side": "WEST"}
  ],
  "algorithm": "exhaustive_search",
  "radius": 25
}
```

| Field | Required? | Notes |
|---|---|---|
| `robot.x_coord`, `robot.y_coord` | yes | grid cell coordinates |
| `robot.facing` | yes | one of `"NORTH"`, `"SOUTH"`, `"EAST"`, `"WEST"` |
| `obstacles` | yes | list of obstacles; must not be empty |
| `obstacles[].id` | yes | any string, just needs to be unique per obstacle |
| `obstacles[].x_coord`, `.y_coord` | yes | grid cell coordinates |
| `obstacles[].image_side` | yes | which face has the image — same 4 direction strings |
| `algorithm` | no | `"nearest_neighbour"`, `"pairwise_swap"`, or `"exhaustive_search"` (default) |
| `radius` | no | turning radius in cm (default: 25) |

### Response

```json
{
  "algorithm": "exhaustive_search",
  "order": ["S", "2", "1"],
  "waypoints": [
    {"id": "S", "x_cm": 10, "y_cm": 10, "theta_rad": 1.5708},
    {"id": "2", "x_cm": 50, "y_cm": 40, "theta_rad": 0.0},
    {"id": "1", "x_cm": 40, "y_cm": 100, "theta_rad": 1.5708}
  ],
  "legs": [
    {
      "to": "2",
      "commands": [
        {"direction": "forward", "turn": "right", "distance_cm": 31.23, "degrees": 71.57},
        {"direction": "forward", "turn": "straight", "distance_cm": 15.81, "degrees": 0.0},
        {"direction": "forward", "turn": "right", "distance_cm": 8.04, "degrees": 18.43}
      ]
    },
    {
      "to": "1",
      "commands": [
        {"direction": "backward", "turn": "right", "distance_cm": 8.87, "degrees": 20.32},
        {"direction": "forward", "turn": "left", "distance_cm": 52.07, "degrees": 119.34},
        {"direction": "forward", "turn": "right", "distance_cm": 21.67, "degrees": 49.66}
      ]
    }
  ],
  "length_cm": 137.69,
  "time_ms": 0.02
}
```

| Field | Meaning |
|---|---|
| `order` | obstacle IDs in the order to visit them, starting with `"S"` (the robot's own start position) |
| `waypoints` | the target pose for each entry in `order` — useful for logging/debugging, but **`legs` is what you actually drive** |
| `legs` | one entry per obstacle in `order` (skipping `"S"`), each with the list of drive commands to get there from wherever the previous leg ended |
| `legs[].to` | which obstacle this leg's commands drive to — trigger image recognition once these commands finish executing |
| `legs[].commands` | ordered list of commands to execute, one after another, to complete this leg |
| `length_cm` | total path length for the whole route |
| `time_ms` | how long the server took to compute it (not travel time) |

Each **command** is one of two things:

- **A straight run**: `"turn": "straight"`, `"degrees": 0.0` — drive `distance_cm` in a straight line, forward or backward per `"direction"`.
- **A turn**: `"turn": "left"` or `"right"` — drive a constant-radius arc (at the `radius` from the request, 25cm by default) that turns the robot by `"degrees"`, forward or backward per `"direction"`. `distance_cm` here is the arc length actually driven, not a straight-line distance.

Commands within a leg are meant to be executed **in order**, back to back — the end pose of one command is the start pose of the next. Note that `"direction": "backward"` does show up sometimes (see leg `"1"` above) — the underlying planner (Reeds-Shepp) allows the robot to reverse when that's the shorter route, which matches the real robot's ability to drive in reverse.

### Calling it from Python

```python
import requests

ALGO_SERVER = "http://172.20.10.2:5000"  # update to match what api.py printed


def get_plan(robot, obstacles, algorithm="exhaustive_search", radius=25):
    response = requests.post(
        f"{ALGO_SERVER}/plan",
        json={
            "robot": robot,
            "obstacles": obstacles,
            "algorithm": algorithm,
            "radius": radius,
        },
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


plan = get_plan(
    robot={"x_coord": 1, "y_coord": 1, "facing": "NORTH"},
    obstacles=[
        {"id": "1", "x_coord": 5, "y_coord": 15, "image_side": "SOUTH"},
        {"id": "2", "x_coord": 10, "y_coord": 5, "image_side": "WEST"},
    ],
)

for leg in plan["legs"]:
    for cmd in leg["commands"]:
        if cmd["turn"] == "straight":
            print(f"drive {cmd['direction']} {cmd['distance_cm']}cm")
        else:
            print(f"drive {cmd['direction']}, turning {cmd['turn']} by {cmd['degrees']} degrees")
    print(f"-> arrived at obstacle {leg['to']}, recognize image now")
```

### Calling it from the command line (for quick testing)

```bash
curl -X POST http://172.20.10.2:5000/plan \
  -H "Content-Type: application/json" \
  -d '{"robot":{"x_coord":1,"y_coord":1,"facing":"NORTH"},"obstacles":[{"id":"1","x_coord":5,"y_coord":15,"image_side":"SOUTH"}]}'
```

## 4. Error responses

If something's wrong with the request, you get a `400` status with a JSON
body explaining what:

```json
{"error": "invalid robot/obstacles: 'NORTHEAST'"}
```

Common causes: a typo'd direction (must be exactly `NORTH`/`SOUTH`/`EAST`/`WEST`,
all caps), a missing field, or an empty `obstacles` list. Check
`response.status_code` before trusting `response.json()` to be a real plan —
`requests.raise_for_status()` (used in the example above) will raise a
Python exception automatically if the server returned an error.
