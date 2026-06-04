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
replacing the Gazebo actuation, which is the working *reference* for how those
topics map to actuation:

- **Diff-drive reference**: as of the `02e642f` drift fix, Gazebo drives the base
  with the **Ignition DiffDrive plugin** (in `JETANK.gazebo`): `left_joint=Revolute_DRIVING_WHEEL_R`,
  `right_joint=Revolute_DRIVING_WHEEL_L`, `wheel_separation=0.1240 m`,
  `wheel_radius=0.0273 m`, `topic=/cmd_vel`, **`max_linear_acceleration=2.0`** (an
  accel limit that tames the velocity-step pitch — see contract point 7). Both
  drive-wheel axes are `(0 1 0)`, so forward = both wheels same sign. The custom
  `cmd_vel_to_wheels.py` node is now **superseded** by that plugin (no longer
  launched) but is kept, updated to the same-axis convention, as a non-DiffDrive
  reference. URDF wheel names are swapped vs side: `WHEEL_R` is physically LEFT,
  `WHEEL_L` is physically RIGHT.
- **Joint reference** (`joint_command_bridge.py`): splits `/joint_commands` into
  per-joint `/cmd/<name>` Float64 for the Gazebo position controllers.

For the same DDS reasons above, the Isaac side must join **Domain 7 + CycloneDDS +
localhost_only=1** to see these topics.

### Isaac action-graph reimplementation contract (MUST match the bridges)

The Isaac twin subscribes to the **original app topics** (`/cmd_vel` Twist,
`/joint_commands` JointState) and reimplements the bridge split **internally**.
Things the OmniGraph reimplementation must get right (extracted from the bridges +
`gazebo.launch.py` + URDF — these are easy to miss):

1. **Two control modes in one pipeline — do NOT position-control the wheels:**
   - Drive wheels (`Revolute_DRIVING_WHEEL_R/L`): **VELOCITY** control, units
     **rad/s**, source = `/cmd_vel` via diff-drive.
   - Arm/gripper (the 8 joints below): **POSITION** control, units **radians**
     (absolute), source = `/joint_commands`.
   - Free wheels (`revolute_FREE_WHEEL_LEFT/RIGHT`): **NO drive at all** — passive,
     roll with ground friction. Set them free-spinning (zero stiffness/damping) in
     Isaac or the rover binds.

2. **`/joint_commands` carries 12 names but only 8 are actuated.** `joint_command_bridge.py`
   forwards exactly these `CONTROLLED_JOINTS` (position): `revolute_BEARING`,
   `Revolute_SERVO_LOWER`, `Revolute_SERVO_UPPER`, `revolute_CAMERA_HOLDER_ARM_LOWER`,
   `revolute_GRIPPER_L1`, `revolute_GRIPPER_L2`, `Revolute_GRIPPER_R1`,
   `Revolute_GRIPPER_R2`. It **skips both free wheels AND both drive wheels** (drive
   wheels come from the `/cmd_vel` path). The position graph should consume only
   those 8 entries from the 12-element array and ignore the wheel entries.

3. **The per-joint `/cmd/<name>` Float64 topics are a Gazebo-only artifact**
   (`ros_gz_bridge` `std_msgs/Float64 ]ignition.msgs.Double`). They are **not** part
   of the app interface — don't recreate them in Isaac; do the split internally.

4. **Drive-wheel axes are NORMALIZED to a common `(0 1 0)`** (URDF-confirmed, as of
   the `02e642f` drift fix). Both
   `Revolute_DRIVING_WHEEL_R` (at y=+0.043, physically LEFT) and
   `Revolute_DRIVING_WHEEL_L` (at y=−0.043, physically RIGHT) now have axis
   `(0 1 0)`, so **positive joint velocity rolls each wheel FORWARD** and **pure
   forward drives BOTH wheels the SAME sign** (≈ +18.3 rad/s each at 0.5 m/s). The
   diff-drive mapping is therefore negation-free: `phys_left=(v−ω·SEP/2)/R`,
   `phys_right=(v+ω·SEP/2)/R`, publish `wheel_r=phys_left`, `wheel_l=phys_right`
   (`SEP=0.1240 m`, `R=0.0273 m`). **An Isaac Velocity_Controller must drive both
   wheels the same sign for forward** — if it applies opposing signs the robot spins.
   Treat `msg.linear.x` sign as the single source of truth and use the explicit
   `"<lin>,<ang>"` backdoor form for unambiguous tests.
   ⚠️ **HISTORY / gotcha:** before `02e642f`, `WHEEL_R` was the *opposing* `(0,−1,0)`
   axis, which forced the old `cmd_vel_to_wheels.py` sign gymnastics (negated
   `linear_x` + negated `wheel_l`) and made forward come out *opposite-signed*. That
   opposing-axis layout is what caused veer and is now gone. Any USD built from a
   pre-`02e642f` URDF (or that snapshot's stale generated `.urdf`) will still have the
   old opposing axes — rebuild the USD from the current `JETANK.urdf` to get the
   `(0 1 0)(0 1 0)` convention.

5. **Make drive + free wheels CONTINUOUS in Isaac.** URDF types them `revolute` with
   limit `0..6.283185`, but functionally they spin continuously — no position limit
   for velocity drive.

6. **Neither bridge clamps to joint limits.** Limits live only in the URDF (e.g.
   `revolute_BEARING` −1.5708..1.5708, camera tilt ±0.785398, per-joint gripper
   limits). **Enforce them on the Isaac articulation drives** — the app/bridges won't.

7. **The control path applies an INSTANTANEOUS velocity STEP — zero acceleration /
   slew limiting anywhere.** The backdoor/GUI publishes `/cmd_vel` as a step (0 →
   0.5 in one message) and `cmd_vel_to_wheels.py` has **no ramp/accel/slew logic**
   (confirmed by inspection) — it converts `/cmd_vel` straight to wheel rad/s, so the
   drive wheels are commanded to jump `0 → ±18.3 rad/s` in a single tick. On a robot
   with a raised CoM this hard accel impulse pitches the base forward (observed: a
   short forward drive tipped the Isaac twin to ~59° with `ROBOT_BASE_Z=0.11`, while
   it is rock-stable undriven). **The fix belongs on the physics/actuation side**
   (Isaac wheel-drive max-acceleration / velocity ramp / capped joint max force,
   and/or lower CoM), not necessarily in this repo — real motors can't step-change
   velocity; the sim just lets them. An app-side acceleration-limited `/cmd_vel`
   option could be added here if source-side shaping is preferred.

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
