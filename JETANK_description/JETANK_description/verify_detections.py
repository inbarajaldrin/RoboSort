#!/usr/bin/env python3
"""
Verify YOLOE detected poses against Gazebo ground truth.

Subscribes to /objects_poses (TFMessage) and compares detected positions
against known ground truth from lego_world.sdf.

Enhanced with forward-projection diagnostic to isolate Y-axis bias source.

Usage:
    ros2 run JETANK_description verify_detections
"""

import re
import subprocess

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
import time
import os

# Lego model names in Gazebo -> color keys for detection matching
LEGO_MODELS = {
    'red_lego_2x4':   'red',
    'green_lego_2x3': 'green',
    'blue_lego_2x2':  'blue',
}

def query_lego_world_positions():
    """Read actual lego positions from Gazebo's dynamic_pose topic."""
    try:
        result = subprocess.run(
            ["ign", "topic", "-e", "-t",
             "/world/lego_world/pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    positions = {}
    blocks = re.split(r'(?=^pose\s*\{)', result.stdout, flags=re.MULTILINE)
    for block in blocks:
        for model_name, color in LEGO_MODELS.items():
            if f'"{model_name}"' in block:
                def _val(txt, key):
                    m = re.search(rf'{key}:\s*([-\d.e+]+)', txt)
                    return float(m.group(1)) if m else 0.0

                pos_m = re.search(r'position\s*\{([^}]*)\}', block)
                pb = pos_m.group(1) if pos_m else ""
                positions[color] = np.array([
                    _val(pb, 'x'), _val(pb, 'y'), _val(pb, 'z')])
                break
    return positions

# Lego top surface Z (ground truth Z + half height)
LEGO_HALF_HEIGHT = 0.0055  # 11mm / 2

# opencv_to_camera quaternion from robot_config.yaml
OPENCV_TO_CAMERA_QUAT = np.array([-0.5, -0.5, 0.5, 0.5])


def forward_project(point_b1, cam_pos, cam_quat, camera_matrix, opencv_to_camera_quat):
    """Forward-project a 3D point in BEARING_1 frame to pixel coordinates.

    Returns (u, v) pixel and euclidean/z-depth distances.
    """
    R_wc = R.from_quat(cam_quat).as_matrix()
    R_o2c = R.from_quat(opencv_to_camera_quat).as_matrix()

    # Vector from camera to point in BEARING_1 frame
    v_world = point_b1 - cam_pos

    # To camera sensor frame: R_wc^T
    v_csf = R_wc.T @ v_world

    # To OpenCV frame: R_o2c^T
    v_opencv = R_o2c.T @ v_csf

    if v_opencv[2] <= 0:
        return None  # behind camera

    # Project to pixel
    u = camera_matrix[0, 0] * v_opencv[0] / v_opencv[2] + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * v_opencv[1] / v_opencv[2] + camera_matrix[1, 2]

    # Euclidean and Z-depth
    euclidean = np.linalg.norm(v_world)
    # Z-depth = projection onto optical axis
    optical_axis_world = R_wc @ R_o2c @ np.array([0, 0, 1])
    z_depth = np.dot(v_world, optical_axis_world)

    return u, v, euclidean, z_depth


def backproject(u, v, depth, cam_pos, cam_quat, camera_matrix, opencv_to_camera_quat,
                treat_as_euclidean=True):
    """Back-project a pixel + depth to 3D point in BEARING_1 frame.

    Args:
        treat_as_euclidean: If True, normalize ray and scale by depth (Euclidean).
                           If False, scale raw ray by depth (Z-depth).
    """
    K_inv = np.linalg.inv(camera_matrix)
    pixel = np.array([u, v, 1.0])
    ray_opencv = K_inv @ pixel

    R_o2c = R.from_quat(opencv_to_camera_quat)
    ray_cam = R_o2c.apply(ray_opencv)

    R_wc = R.from_quat(cam_quat).as_matrix()
    ray_world = R_wc @ ray_cam

    if treat_as_euclidean:
        ray_normalized = ray_world / np.linalg.norm(ray_world)
        return cam_pos + ray_normalized * depth
    else:
        # Z-depth: scale raw ray by depth
        # raw ray has optical-axis component = 1 (since K_inv gives z=1)
        return cam_pos + ray_world * depth


class DetectionVerifier(Node):
    def __init__(self):
        super().__init__('detection_verifier')
        self.subscription = self.create_subscription(
            TFMessage, '/objects_poses', self.objects_callback, 10
        )
        self.cam_pose_sub = self.create_subscription(
            PoseStamped, '/camera_pose', self.camera_pose_callback, 10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/depth_camera/image_raw', self.depth_callback, 10
        )

        self.detections = {}
        self.detection_count = 0
        self.start_time = time.time()

        # Camera state
        self.cam_pos = None
        self.cam_quat = None
        self.cam_info = None
        self.depth_image = None
        self.diagnostic_done = False

        # TF2 for world -> BEARING_1 (handles sim drift)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ground_truth = {}  # populated from TF

        self.get_logger().info('Detection Verifier started (with TF-based ground truth)')
        self.get_logger().info(f'Waiting for world->BEARING_1 TF and detections...')

        # Timer to print comparison every 5 seconds
        self.timer = self.create_timer(5.0, self.print_results)
        # Timer for diagnostic (run once after camera data arrives)
        self.diag_timer = self.create_timer(2.0, self.run_diagnostic)

    def camera_pose_callback(self, msg):
        self.cam_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self.cam_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

    def camera_info_callback(self, msg):
        if self.cam_info is None:
            self.cam_info = msg

    def depth_callback(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self.depth_image = arr

    def run_diagnostic(self):
        """Run forward-projection diagnostic once camera data is available."""
        if self.diagnostic_done:
            return
        if (self.cam_pos is None or self.cam_info is None
                or self.depth_image is None or not self.ground_truth):
            return

        self.diagnostic_done = True
        self.get_logger().info('=' * 70)
        self.get_logger().info('Y-AXIS BIAS DIAGNOSTIC')
        self.get_logger().info('=' * 70)

        # Camera info from Gazebo
        ci = self.cam_info
        self.get_logger().info(f'Gazebo camera_info: fx={ci.k[0]:.2f} fy={ci.k[4]:.2f} '
                              f'cx={ci.k[2]:.2f} cy={ci.k[5]:.2f}')

        # Config camera matrix
        import math
        hfov, vfov = 69.4, 54.9  # current config values
        fx_cfg = 640 / (2 * math.tan(math.radians(hfov / 2)))
        fy_cfg = 480 / (2 * math.tan(math.radians(vfov / 2)))
        self.get_logger().info(f'Config camera_matrix: fx={fx_cfg:.2f} fy={fy_cfg:.2f} '
                              f'cx=320.0 cy=240.0  (vfov={vfov}°)')

        K_config = np.array([[fx_cfg, 0, 320], [0, fy_cfg, 240], [0, 0, 1]], dtype=np.float64)
        K_gazebo = np.array([[ci.k[0], 0, ci.k[2]], [0, ci.k[4], ci.k[5]], [0, 0, 1]], dtype=np.float64)

        # Camera pose
        self.get_logger().info(f'Camera pose (BEARING_1→CAMERA_SENSOR_FRAME):')
        self.get_logger().info(f'  pos=({self.cam_pos[0]:.5f}, {self.cam_pos[1]:.5f}, {self.cam_pos[2]:.5f})')
        self.get_logger().info(f'  quat=({self.cam_quat[0]:.5f}, {self.cam_quat[1]:.5f}, '
                              f'{self.cam_quat[2]:.5f}, {self.cam_quat[3]:.5f})')
        euler = R.from_quat(self.cam_quat).as_euler('xyz', degrees=True)
        self.get_logger().info(f'  euler(xyz)=({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)')

        # Forward-project each ground truth object
        self.get_logger().info('')
        self.get_logger().info('FORWARD PROJECTION (GT → pixel):')
        for name, gt in self.ground_truth.items():
            gt_center = np.array([gt['x'], gt['y'], gt['z']])
            gt_top = gt_center.copy()
            gt_top[2] += LEGO_HALF_HEIGHT  # top surface

            # Forward-project with config K
            result_cfg = forward_project(gt_top, self.cam_pos, self.cam_quat,
                                         K_config, OPENCV_TO_CAMERA_QUAT)
            result_gz = forward_project(gt_top, self.cam_pos, self.cam_quat,
                                        K_gazebo, OPENCV_TO_CAMERA_QUAT)

            if result_cfg is None:
                self.get_logger().info(f'  {name}: BEHIND CAMERA')
                continue

            u_cfg, v_cfg, euclid, zdepth = result_cfg
            u_gz, v_gz, _, _ = result_gz

            # Read actual depth at the projected pixel
            iu, iv = int(round(u_cfg)), int(round(v_cfg))
            actual_depth = float('nan')
            if 0 <= iv < self.depth_image.shape[0] and 0 <= iu < self.depth_image.shape[1]:
                # Sample 5x5 region around the pixel
                y1 = max(0, iv - 2)
                y2 = min(self.depth_image.shape[0], iv + 3)
                x1 = max(0, iu - 2)
                x2 = min(self.depth_image.shape[1], iu + 3)
                patch = self.depth_image[y1:y2, x1:x2]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if len(valid) > 0:
                    actual_depth = float(np.median(valid))

            self.get_logger().info(f'  {name} (top surface):')
            self.get_logger().info(f'    3D(B1): ({gt_top[0]:.4f}, {gt_top[1]:.4f}, {gt_top[2]:.4f})')
            self.get_logger().info(f'    Pixel(config K): ({u_cfg:.1f}, {v_cfg:.1f})')
            self.get_logger().info(f'    Pixel(Gazebo K): ({u_gz:.1f}, {v_gz:.1f})')
            self.get_logger().info(f'    Euclidean dist:  {euclid*1000:.1f}mm')
            self.get_logger().info(f'    Z-depth:         {zdepth*1000:.1f}mm')
            self.get_logger().info(f'    Gazebo depth@px: {actual_depth*1000:.1f}mm')

            # Back-project with different assumptions
            for label, K, depth, as_euclid in [
                ('ConfigK+GazDepth+Euclid', K_config, actual_depth, True),
                ('ConfigK+GazDepth+Zdepth', K_config, actual_depth, False),
                ('ConfigK+TrueEuclid',      K_config, euclid,       True),
                ('GazeboK+GazDepth+Euclid', K_gazebo, actual_depth, True),
                ('GazeboK+GazDepth+Zdepth', K_gazebo, actual_depth, False),
                ('GazeboK+TrueEuclid',      K_gazebo, euclid,       True),
            ]:
                if np.isnan(depth):
                    continue
                pt = backproject(u_cfg, v_cfg, depth, self.cam_pos, self.cam_quat,
                                 K, OPENCV_TO_CAMERA_QUAT, treat_as_euclidean=as_euclid)
                err = (pt - gt_center) * 1000
                total = np.linalg.norm(err)
                self.get_logger().info(
                    f'    {label:30s}: err=({err[0]:+.1f}, {err[1]:+.1f}, {err[2]:+.1f})mm  '
                    f'total={total:.1f}mm')

        self.get_logger().info('=' * 70)

    def objects_callback(self, msg):
        self.detection_count += 1
        for transform in msg.transforms:
            name = transform.child_frame_id
            pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])
            quat = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ])
            self.detections[name] = {
                'pos': pos,
                'quat': quat,
                'time': time.time(),
            }

    def match_detection_to_gt(self, det_name):
        """Match a detection name to a ground truth entry by color."""
        det_lower = det_name.lower()
        for gt_name in self.ground_truth:
            if gt_name in det_lower:
                return gt_name
        return None

    def _update_ground_truth_from_tf(self):
        """Refresh ground truth positions using live Gazebo poses + TF (handles drift + randomization)."""
        try:
            t = self.tf_buffer.lookup_transform(
                "BEARING_1", "world", rclpy.time.Time())
            p = t.transform.translation
            q = t.transform.rotation
            tf_pos = np.array([p.x, p.y, p.z])
            tf_rot = R.from_quat([q.x, q.y, q.z, q.w])

            # Query live lego positions from Gazebo
            world_positions = query_lego_world_positions()
            if not world_positions:
                return len(self.ground_truth) > 0  # keep stale if query fails

            for color, p_world in world_positions.items():
                p_b1 = tf_rot.apply(p_world) + tf_pos
                self.ground_truth[color] = {
                    'x': p_b1[0], 'y': p_b1[1], 'z': p_b1[2]}
            return True
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return False

    def print_results(self):
        elapsed = time.time() - self.start_time

        # Refresh ground truth from TF
        if not self._update_ground_truth_from_tf():
            if not self.ground_truth:
                self.get_logger().info(
                    f'[{elapsed:.0f}s] Waiting for world->BEARING_1 TF '
                    f'(is world_tf_publisher running?)...')
                return

        if not self.detections:
            self.get_logger().info(
                f'[{elapsed:.0f}s] No detections yet '
                f'(received {self.detection_count} messages)...'
            )
            return

        lines = []
        lines.append('=' * 70)
        lines.append(f'DETECTION vs GROUND TRUTH  [{elapsed:.0f}s elapsed, '
                      f'{self.detection_count} msgs, '
                      f'{len(self.detections)} objects]')
        lines.append('=' * 70)

        matched = 0
        total_error = 0.0

        for det_name, det_data in sorted(self.detections.items()):
            det_pos = det_data['pos']
            age = time.time() - det_data['time']

            gt_name = self.match_detection_to_gt(det_name)

            if gt_name is not None:
                gt = self.ground_truth[gt_name]
                gt_pos = np.array([gt['x'], gt['y'], gt['z']])
                error_vec = det_pos - gt_pos
                error_mm = np.linalg.norm(error_vec) * 1000

                lines.append(
                    f'  {det_name:20s} -> {gt_name:8s}  '
                    f'det=({det_pos[0]:.4f}, {det_pos[1]:.4f}, {det_pos[2]:.4f})  '
                    f'gt=({gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f})  '
                    f'err={error_mm:.1f}mm  '
                    f'(dx={error_vec[0]*1000:.1f}, dy={error_vec[1]*1000:.1f}, '
                    f'dz={error_vec[2]*1000:.1f})  '
                    f'age={age:.1f}s'
                )
                matched += 1
                total_error += error_mm
            else:
                lines.append(
                    f'  {det_name:20s} -> NO MATCH  '
                    f'pos=({det_pos[0]:.4f}, {det_pos[1]:.4f}, {det_pos[2]:.4f})  '
                    f'age={age:.1f}s'
                )

        if matched > 0:
            avg_error = total_error / matched
            lines.append(f'\n  Matched: {matched}/{len(self.ground_truth)} | '
                          f'Avg error: {avg_error:.1f}mm | '
                          f'{"PASS" if avg_error < 50 else "FAIL"} (threshold: 50mm)')

        lines.append('=' * 70)

        for line in lines:
            self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
