#!/usr/bin/env python3
"""Generate the collision-only / visuals-stripped JETANK URDF.

Provenance: derived from JETANK_description/urdf/JETANK.urdf @ commit 7f5bbb8
(the regenerated cylinder-collision + same-axis (0 1 0) URDF).

Purpose: an OFFLINE/headless standalone-isaacsim import source for the
physics-only drive/upright proof. It has ZERO mesh refs, so the Isaac URDF
importer never hits package:// mesh resolution (which previously wedged the live
MCP worker). Import this -> bake corrected jetank.usd -> wire same-sign +
accel-ramp control -> drive-test.

Transform (deterministic):
  - keep every <link> (inertial/mass) and every <joint> verbatim
  - keep collisions that are PRIMITIVES (cylinder/box/sphere) -> 4 wheel
    cylinders + 1 gripper-center box
  - drop all <visual>, all MESH-based <collision>, all <gazebo>, all <transmission>

Caveat (by design): the 16 body/arm links lose their colliders — only the 4
wheels + gripper box remain. Faithful for a flat-ground drive (the real robot's
body never touches the ground; only wheels do) and keeps tipping measurable from
the root pose without spurious body-ground contacts. No self-collision / no
body-catches-ground past vertical — neither affects the upright/drive proof. For
a production bake with body colliders, use the full mesh URDF (@7f5bbb8) or
convex-decompose the bodies.

Regenerate on demand:
    git checkout 7f5bbb8 -- JETANK_description/urdf/JETANK.urdf   # if needed
    python3 proof/make_collision_only.py
"""
import xml.etree.ElementTree as ET

SRC = 'JETANK_description/urdf/JETANK.urdf'
OUT = 'proof/JETANK_collision_only.urdf'


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    dropped = {'visual': 0, 'mesh_collision': 0, 'gazebo': 0, 'transmission': 0}

    for tag in ('gazebo', 'transmission'):
        for e in root.findall(tag):
            root.remove(e)
            dropped[tag] += 1

    for link in root.findall('link'):
        for v in link.findall('visual'):
            link.remove(v)
            dropped['visual'] += 1
        for c in link.findall('collision'):
            g = c.find('geometry')
            if g is not None and g.find('mesh') is not None:
                link.remove(c)
                dropped['mesh_collision'] += 1

    try:
        ET.indent(tree)  # py3.9+
    except Exception:
        pass
    tree.write(OUT, encoding='utf-8', xml_declaration=True)
    print('wrote', OUT, 'dropped', dropped)


if __name__ == '__main__':
    main()
