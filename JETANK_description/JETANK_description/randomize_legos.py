#!/usr/bin/env python3
"""Randomize lego positions in Gazebo via ign service.

Reads live camera pose/intrinsics from ROS2 and the world->BEARING_1
transform from TF2 (published by world_tf_publisher). Uses the same
forward projection as verify_detections.py to guarantee objects are in frame.

Usage:
    ros2 run JETANK_description randomize_legos              # all fully in frame
    ros2 run JETANK_description randomize_legos --edge       # one partially outside
    ros2 run JETANK_description randomize_legos --reset      # default SDF positions
"""
import argparse
import random
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
import tf2_ros

# Default SDF positions (world frame)
DEFAULTS = {
    "red_lego_2x4":   (0.170, -0.020, 0.0055),
    "green_lego_2x3": (0.180,  0.020, 0.0055),
    "blue_lego_2x2":  (0.200,  0.040, 0.0055),
}

# Lego half-sizes in meters (length/2, width/2)
HALF_SIZES = {
    "red_lego_2x4":   (0.016, 0.008),
    "green_lego_2x3": (0.012, 0.008),
    "blue_lego_2x2":  (0.008, 0.008),
}

TABLE_Z = 0.0055
MIN_SPACING = 0.03
SEARCH_BOUNDS = {"x": (0.08, 0.30), "y": (-0.15, 0.15)}

OPENCV_TO_CAMERA_QUAT = np.array([-0.5, -0.5, 0.5, 0.5])


def forward_project(point_b1, cam_pos, cam_quat, K):
    """Forward-project a BEARING_1 point to pixel. Same math as verify_detections.py."""
    R_wc = R.from_quat(cam_quat).as_matrix()
    R_o2c = R.from_quat(OPENCV_TO_CAMERA_QUAT).as_matrix()
    v_world = point_b1 - cam_pos
    v_csf = R_wc.T @ v_world
    v_opencv = R_o2c.T @ v_csf
    if v_opencv[2] <= 0:
        return None
    u = K[0, 0] * v_opencv[0] / v_opencv[2] + K[0, 2]
    v = K[1, 1] * v_opencv[1] / v_opencv[2] + K[1, 2]
    return u, v


class CameraHelper:
    def __init__(self):
        self.cam_pos = None
        self.cam_quat = None
        self.K = None
        self.img_w = self.img_h = None
        # world -> BEARING_1 transform from TF2
        self._world_to_b1_pos = None
        self._world_to_b1_rot = None

        rclpy.init()
        self.node = Node("randomize_legos")
        self.node.create_subscription(PoseStamped, "/camera_pose", self._pose_cb, 1)
        self.node.create_subscription(CameraInfo, "/camera/camera_info", self._info_cb, 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def _pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.cam_pos = np.array([p.x, p.y, p.z])
        self.cam_quat = np.array([q.x, q.y, q.z, q.w])

    def _info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.img_w = msg.width
        self.img_h = msg.height

    def _lookup_world_to_b1(self):
        """Look up world -> BEARING_1 transform from TF2."""
        try:
            t = self.tf_buffer.lookup_transform(
                "BEARING_1", "world", rclpy.time.Time())
            p = t.transform.translation
            q = t.transform.rotation
            self._world_to_b1_pos = np.array([p.x, p.y, p.z])
            self._world_to_b1_rot = R.from_quat([q.x, q.y, q.z, q.w])
            return True
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return False

    def wait_for_data(self, timeout=15.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if (self.cam_pos is not None and self.K is not None
                    and self._lookup_world_to_b1()):
                return True
        return False

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def world_to_bearing1(self, wx, wy, wz):
        """Convert world point to BEARING_1 frame using live TF."""
        p_world = np.array([wx, wy, wz])
        return self._world_to_b1_rot.apply(p_world - (-self._world_to_b1_pos))

    def project_world(self, wx, wy, wz):
        """Project a world-frame point to pixel coordinates."""
        # Use TF-based transform: world -> BEARING_1
        p_world = np.array([wx, wy, wz])
        # TF gives BEARING_1 <- world, so: p_b1 = R * p_world + t
        p_b1 = self._world_to_b1_rot.apply(p_world) + self._world_to_b1_pos
        return forward_project(p_b1, self.cam_pos, self.cam_quat, self.K)

    def object_fully_in_frame(self, wx, wy, half_l, half_w, margin=20):
        for dx in (-half_l, half_l):
            for dy in (-half_w, half_w):
                pix = self.project_world(wx + dx, wy + dy, TABLE_Z)
                if pix is None:
                    return False
                u, v = pix
                if u < margin or u > self.img_w - margin or v < margin or v > self.img_h - margin:
                    return False
        return True

    def object_partially_in_frame(self, wx, wy, half_l, half_w):
        in_count = 0
        for dx in (-half_l, half_l):
            for dy in (-half_w, half_w):
                pix = self.project_world(wx + dx, wy + dy, TABLE_Z)
                if pix is not None:
                    u, v = pix
                    if 0 <= u <= self.img_w and 0 <= v <= self.img_h:
                        in_count += 1
        return 0 < in_count < 4


def set_pose(name, x, y, z):
    req = f"name: '{name}', position: {{x: {x}, y: {y}, z: {z}}}"
    result = subprocess.run(
        ["ign", "service", "-s", "/world/lego_world/set_pose",
         "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
         "--timeout", "3000", "-r", req],
        capture_output=True, text=True
    )
    return "true" in result.stdout


def random_positions_in_view(cam, names):
    positions = []
    for name in names:
        hl, hw = HALF_SIZES[name]
        for _attempt in range(500):
            x = random.uniform(*SEARCH_BOUNDS["x"])
            y = random.uniform(*SEARCH_BOUNDS["y"])
            if not cam.object_fully_in_frame(x, y, hl, hw):
                continue
            if all(((x - px)**2 + (y - py)**2)**0.5 > MIN_SPACING
                   for px, py in positions):
                positions.append((x, y))
                break
        else:
            print(f"  WARNING: could not place {name}, using default")
            positions.append((DEFAULTS[name][0], DEFAULTS[name][1]))
    return positions


def random_positions_edge(cam, names):
    positions = []
    need_edge = random.randint(0, len(names) - 1)
    for i, name in enumerate(names):
        hl, hw = HALF_SIZES[name]
        want_edge = (i == need_edge)
        for _attempt in range(1000):
            x = random.uniform(*SEARCH_BOUNDS["x"])
            y = random.uniform(*SEARCH_BOUNDS["y"])
            if want_edge:
                if not cam.object_partially_in_frame(x, y, hl, hw):
                    continue
            else:
                if not cam.object_fully_in_frame(x, y, hl, hw):
                    continue
            if all(((x - px)**2 + (y - py)**2)**0.5 > MIN_SPACING
                   for px, py in positions):
                positions.append((x, y))
                break
        else:
            print(f"  WARNING: could not place {name}, using default")
            positions.append((DEFAULTS[name][0], DEFAULTS[name][1]))
    return positions


def main():
    parser = argparse.ArgumentParser(description="Randomize lego positions in Gazebo")
    parser.add_argument("--edge", action="store_true",
                        help="At least one object partially outside camera frame")
    parser.add_argument("--reset", action="store_true",
                        help="Reset to default SDF positions")
    args = parser.parse_args()

    if args.reset:
        print("Resetting legos to default positions...")
        for name, (x, y, z) in DEFAULTS.items():
            ok = set_pose(name, x, y, z)
            print(f"  {name}: ({x:.3f}, {y:.3f}, {z:.4f}) {'OK' if ok else 'FAIL'}")
        return

    cam = CameraHelper()
    print("Reading camera data and TF from ROS2...")
    if not cam.wait_for_data():
        print("ERROR: Could not read camera/TF data. Is the sim + world_tf_publisher running?")
        cam.shutdown()
        sys.exit(1)

    print(f"  Camera pos (B1): ({cam.cam_pos[0]:.4f}, {cam.cam_pos[1]:.4f}, {cam.cam_pos[2]:.4f})")
    print(f"  K: fx={cam.K[0,0]:.1f} fy={cam.K[1,1]:.1f} [{cam.img_w}x{cam.img_h}]")

    # Sanity check with known default red lego
    test = cam.project_world(0.170, -0.020, TABLE_Z)
    if test:
        print(f"  Projection sanity (red default): px=({test[0]:.1f}, {test[1]:.1f})")

    names = list(DEFAULTS.keys())
    if args.edge:
        positions = random_positions_edge(cam, names)
    else:
        positions = random_positions_in_view(cam, names)

    mode = "with edge case" if args.edge else "all fully in frame"
    print(f"\nRandomizing legos ({mode})...")
    for name, (x, y) in zip(names, positions):
        z = DEFAULTS[name][2]
        ok = set_pose(name, x, y, z)
        hl, hw = HALF_SIZES[name]
        full = cam.object_fully_in_frame(x, y, hl, hw)
        partial = cam.object_partially_in_frame(x, y, hl, hw)
        label = "FULL" if full else ("EDGE" if partial else "OUT")
        pix = cam.project_world(x, y, TABLE_Z)
        px_str = f"px=({pix[0]:.0f},{pix[1]:.0f})" if pix else "px=N/A"
        print(f"  {name}: world=({x:.3f}, {y:.3f}) {px_str} {'OK' if ok else 'FAIL'} [{label}]")

    cam.shutdown()


if __name__ == "__main__":
    main()
