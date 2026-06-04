# proof/ — collision-only URDF for the Isaac drive/upright proof

**Throwaway branch `proof/collision-only-urdf`** (do not merge to main). Exists so
the artifact survives `/tmp` cleanup for the next Isaac-rebuild session.

## `JETANK_collision_only.urdf`

Offline/headless **standalone-isaacsim import source** for the physics-only
drive/upright proof. Derived from `JETANK_description/urdf/JETANK.urdf` @ `7f5bbb8`.

Verified contents:
- 24 links + 23 joints preserved verbatim (incl. `base_footprint→base_link` root).
- **Zero mesh refs** (`<mesh>` / `.stl` / `package://` all absent) → no package://
  resolution, no importer stall.
- Inertials/masses intact: 23 mass blocks, total **2.3803 kg** (CoM/inertia faithful).
- Collisions = 5 primitives: 4 wheel **cylinders** (FREE r=0.02486 l=0.04476,
  DRIVER r=0.02734 l=0.03903) + 1 gripper-center box.
- Drive wheels `type=continuous`, axis `(0 1 0)` both (same-sign-forward convention).

By-design caveat: the 16 body/arm links have **no collider** (only wheels +
gripper box). Faithful for a flat-ground drive; see `make_collision_only.py` header
for the full rationale.

## Regenerate on demand

Deterministic — reproduce from `7f5bbb8` any time:

```bash
python3 proof/make_collision_only.py   # reads JETANK_description/urdf/JETANK.urdf -> proof/JETANK_collision_only.urdf
```

## Next steps (Isaac side, separate repo)

Import this URDF offline (standalone Isaac python, NOT in-process on the live
socket) → bake corrected `jetank.usd` → wire same-sign forward + accel ramp
≤2.0 m/s² → drive-test. Then RoboSort agent re-drives a ~3–4s steady-state forward
(`/jetank_agent_cmd "0.5,0.0"`) for the upright/wheel-velocity check.
