# RoboSort — CLAUDE.md

Repo memory for the **RoboSort** workspace (`~/Documents/RoboSort`). This file is
the source of truth for the JETANK robot's ROS 2 control interface. Keep it
factual and reusable — agents replay it instead of re-deriving the layout.

## What RoboSort is

RoboSort is the **real-robot / Gazebo / control-app side** of the JETANK project.
JETANK is a small tracked rover with a 4-DOF arm + gripper (Waveshare JETANK form
factor, originally Jetson Nano based). This repo is the **CONTROLLER and the source
of truth for the robot's ROS 2 interface**: the `jetank_control_gui` app publishes
the control topics, and downstream consumers (Gazebo bridges, and the Isaac Sim
`jetank-dt` digital twin) subscribe to them.

The control flow mirrors how SO-ARM101 is driven from its GUI:

```
jetank_control_gui  ──/cmd_vel (Twist)──────────►  consumer (Gazebo bridge | Isaac jetank-dt)
   (this repo)      ──/joint_commands (JointState)►
```

This repo is the publisher/controller. **Isaac Sim subscribes; it does not lead.**

## Colcon workspace layout

This repo *is* a colcon workspace, but packages live at the **repo root** (not in a
`src/` subdir). Build artifacts (`build/ install/ log/`) are git-ignored.

| Package | Type | Role |
|---|---|---|
| `JETANK_description` | ament_python | The main package: the control GUI app, the Gazebo bridges, pose publishers, URDF/xacro, meshes, worlds, launch files. **This is where almost everything lives.** |
| `jetank_control` | ament_python | **Real-hardware drivers ONLY** (servo / motor / imu / camera). Runs on the Jetson. NOT launched in the headless/sim flow. |
| `drop_container_description` | ament_python | URDF + ArUco helpers for the drop-off container (the bin legos get sorted into). |
| `dds_config` | not a ROS package | DDS/Tailscale profiles + media-streaming helper scripts. Reference material, not built by colcon. |

### `JETANK_description/JETANK_description/` (the nodes)

| File | Entry point | Purpose |
|---|---|---|
| `jetank_control_gui.py` | `jetank_control_gui` | **The app.** Tkinter control GUI. Publishes `/cmd_vel` + `/joint_commands`. Has a headless agent backdoor (see below). 150KB+; the central node. |
| `cmd_vel_to_wheels.py` | `cmd_vel_to_wheels` | Gazebo bridge: `/cmd_vel` (Twist) → per-wheel `/cmd/Revolute_DRIVING_WHEEL_{R,L}` (Float64). Custom diff-drive (URDF wheel axes point opposite ways, so stock DiffDrive can't be used). |
| `joint_command_bridge.py` | `joint_command_bridge` | Gazebo bridge: `/joint_commands` (JointState) → per-joint `/cmd/<joint>` (Float64) for Ignition JointPositionController plugins. |
| `ee_pose_publisher.py` | `ee_pose_publisher` | Publishes gripper-center link pose (PoseStamped) from TF. |
| `camera_pose_publisher.py` | `camera_pose_publisher` | Publishes `CAMERA_1` link pose (PoseStamped). |
| `camera_to_ee_pose_publisher.py` | `camera_to_ee_pose_publisher` | Publishes EE pose expressed in the camera frame. |
| `world_tf_publisher.py` | `world_tf_publisher` | Publishes `world → base_footprint` TF from the Gazebo model pose (closes the Gazebo-world ↔ ROS-TF-tree gap). |
| `randomize_legos.py` | `randomize_legos` | Randomizes lego positions in Gazebo (keeps them in camera frame). Sim-data tooling. |
| `verify_detections.py` | `verify_detections` | Compares YOLOE-detected `/objects_poses` against Gazebo ground truth (`lego_world.sdf`). |

Other assets: `urdf/` (`JETANK.urdf`, `JETANK.xacro`, `.gazebo`, `.trans`,
`materials.xacro`), `meshes/` (20 STLs), `worlds/lego_world.sdf`,
`config/display.rviz`, `scripts/{fk,ik}.py` (arm kinematics helpers).

### Launch files (`JETANK_description/launch/`)

| Launch | Use |
|---|---|
| `tkinter.launch.py` | **Full real-robot stack**: robot_state_publisher + `jetank_control` servo/motor/imu drivers + GUI + RViz + pose publishers. Needs the Jetson hardware. |
| `gazebo.launch.py` | Gazebo (Ignition) sim: spawns the robot, GUI, ros_gz bridges, `joint_command_bridge`, `cmd_vel_to_wheels`, pose + world_tf publishers. The Gazebo reference the Isaac twin replaces. |
| `display.launch.py` | URDF in RViz only (no control). |

## VERIFIED headless launch recipe

The GUI runs **headless on dual-a4500** against virtual display `:1` (it uses
`tk.Tk()` + `mainloop()`, so it needs an X display — `:1` is a virtual one, this is
not a true displayless run). **`use_real_hardware=False` by default**, and the
servo/motor/imu hardware nodes live in the separate `jetank_control` package which
this run does NOT launch — so no Jetson hardware is required.

```bash
cd ~/Documents/RoboSort
source /opt/ros/humble/setup.bash
source install/setup.bash
source ~/Documents/isaac-sim-mcp/config/ros_dds.env   # Domain 7, CycloneDDS, localhost-only
export DISPLAY=:1
ros2 run JETANK_description jetank_control_gui
```

Node name once up: `/jetank_gripper_control_gui`. Log line on success:
`JETANK Control GUI initialized`.

### DDS / domain (IMPORTANT — two different configs exist)

- **Sim/headless flow (what the recipe uses):** sources
  `~/Documents/isaac-sim-mcp/config/ros_dds.env` →
  `ROS_DOMAIN_ID=7`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `ROS_LOCALHOST_ONLY=1`, `CYCLONEDDS_URI` → isaac's `cyclonedds.xml`
  (AllowMulticast off, loopback only). This is the config the Isaac `jetank-dt`
  twin shares. `ros-humble-rmw-cyclonedds-cpp` is apt-installed (the profile pins
  Cyclone). **`ROS_LOCALHOST_ONLY` must be identical (=1) on every node sharing
  the domain** — a mixed 0/1 setup gives one-way discovery.
- **Multi-machine flow (this repo's own `dds_config/cyclonedds.xml`):** a DIFFERENT
  profile — `Domain id=45`, `tailscale0` interface, explicit Tailscale peers. This
  is for cross-machine ROS over Tailscale and is NOT used by the headless recipe.
  Don't confuse the two.

## Published control topics (the interface the Isaac twin consumes)

Both topics are **GLOBAL (no namespace)**. Verified live (see proof below).

### `/cmd_vel` — `geometry_msgs/Twist` (base drive)

Published by the GUI drive buttons AND the headless agent backdoor. The GUI also
emits a periodic zero-Twist "brake" when no motion is active (expect a stream of
zeros between commands — that is normal, not a bug).

### `/joint_commands` — `sensor_msgs/JointState` (12 joints, arm + gripper + wheels)

Published on a **10 Hz timer** (always streaming, even idle). **Joint order is
fixed — index position matters, consumers must respect this exact order:**

| idx | joint | idx | joint |
|---|---|---|---|
| 0 | `revolute_BEARING` | 6 | `Revolute_SERVO_LOWER` |
| 1 | `revolute_FREE_WHEEL_LEFT` | 7 | `Revolute_DRIVING_WHEEL_R` |
| 2 | `revolute_FREE_WHEEL_RIGHT` | 8 | `Revolute_DRIVING_WHEEL_L` |
| 3 | `revolute_GRIPPER_L1` | 9 | `Revolute_GRIPPER_R2` |
| 4 | `revolute_GRIPPER_L2` | 10 | `Revolute_GRIPPER_R1` |
| 5 | `Revolute_SERVO_UPPER` | 11 | `revolute_CAMERA_HOLDER_ARM_LOWER` |

(Note the casing inconsistency in the URDF/source — `revolute_` vs `Revolute_` —
is intentional and must be matched exactly.)

## Headless agent backdoor — `/jetank_agent_cmd` (`std_msgs/String`)

Added in commit `34ec104`. Lets an external orchestrator (the Isaac twin, a
verifier) drive the base **without a Tk window / without touching the GUI**. The
callback maps the String to a Twist and publishes the **same `/cmd_vel`** as the
drive buttons. Accepted values:

- `forward` | `backward` | `left` | `right` | `stop` (aliases: `f/b/l/r/s`, `fwd`, `back`)
- an explicit `"<linear>,<angular>"` pair, e.g. `"0.5,0.0"`

**Wheel-inversion gotcha:** `self.wheels_inverted = True` by default, so the named
commands are sign-flipped relative to the raw axis:
`forward → linear.x = -0.5`, `backward → +0.5`. To get an exact, inversion-free
Twist, send the explicit pair (`"0.5,0.0"` → `linear.x = +0.5`).

Example:
```bash
ros2 topic pub --once /jetank_agent_cmd std_msgs/String "{data: forward}"
ros2 topic pub --once /jetank_agent_cmd std_msgs/String "{data: '0.5,0.0'}"
```

## Connection to the Isaac `jetank-dt` digital twin

The Isaac twin (in the separate `~/Documents/isaac-sim-mcp` repo — **do not edit
that repo from here; coordinate via /send-to-agent**) builds OmniGraph action
graphs that **subscribe to `/cmd_vel` and `/joint_commands`** from this repo,
replacing the Gazebo bridges (`cmd_vel_to_wheels.py`, `joint_command_bridge.py`)
which are the working *reference* for how those topics map to actuation:

- **Diff-drive reference** (`cmd_vel_to_wheels.py`): `WHEEL_SEPARATION=0.1478 m`,
  `WHEEL_RADIUS=0.0273 m`. URDF wheel names are swapped vs ROS convention —
  `WHEEL_R` is physically LEFT (axis ≈ −Y), `WHEEL_L` is physically RIGHT
  (axis ≈ +Y). The bridge negates `linear.x`, computes standard diff-drive, then
  applies axis-sign corrections: `wheel_r = phys_left`, `wheel_l = −phys_right`.
- **Joint reference** (`joint_command_bridge.py`): splits `/joint_commands` into
  per-joint `/cmd/<name>` Float64 for the Gazebo position controllers.

For the same DDS reasons above, the Isaac side must join **Domain 7 + CycloneDDS +
localhost_only=1** to see these topics.

## Verified publish proof (re-confirmed 2026-06-04, dual-a4500, DISPLAY=:1)

GUI launched headless via the recipe; node `/jetank_gripper_control_gui` came up
(`JETANK Control GUI initialized`). Topics present: `/cmd_vel`, `/joint_commands`,
`/joint_states`, `/jetank_agent_cmd`.

`/cmd_vel` while driving the backdoor (`ros2 topic echo /cmd_vel`):
```
# agent_cmd 'forward'  ->  linear.x: -0.5   (wheels_inverted=True)
# agent_cmd '0.5,0.0'  ->  linear.x:  0.5
```
GUI log confirmed:
```
agent_cmd 'forward' -> cmd_vel linear.x=-0.5 angular.z=0.0
agent_cmd '0.5,0.0' -> cmd_vel linear.x=0.5 angular.z=0.0
```
`/joint_commands` confirmed streaming at 10 Hz with the exact 12-joint order above.

## Conventions for agents working here

- Scope is **this repo only.** Don't touch `~/Documents/isaac-sim-mcp` — coordinate
  with the orchestrator via the `/send-to-agent` skill.
- This agent OWNS the repo: organize, document, and COMMIT (no `Co-Authored-By`).
- Hardware (Jetson servo/motor/imu) is in `jetank_control` and is intentionally NOT
  part of the headless/sim flow — `use_real_hardware=False`.
- Engineering defaults: wait on real signals (a topic/log line) not blind `sleep`;
  fail loudly at the source; prefer the real fix over a silent fallback.
