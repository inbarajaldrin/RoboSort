#!/usr/bin/env python3
"""
Verify YOLOE detected poses against Gazebo ground truth.

Subscribes to /objects_poses (TFMessage) and compares detected positions
against known ground truth from lego_world.sdf.

Usage:
    ros2 run JETANK_description verify_detections
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import numpy as np
import time
import os

# Ground truth positions from lego_world.sdf (world frame)
# Phase A: Large blocks (5cm cubes)
GROUND_TRUTH_PHASE_A = {
    'red': {'x': 0.20, 'y': -0.10, 'z': 0.025, 'yaw': 0.3},
    'green': {'x': 0.15, 'y': 0.08, 'z': 0.025, 'yaw': -0.5},
    'blue': {'x': 0.25, 'y': 0.05, 'z': 0.025, 'yaw': 0.8},
}

# Phase B: Real lego bricks (uncomment when switching)
# GROUND_TRUTH_PHASE_B = {
#     'red': {'x': 0.18, 'y': -0.08, 'z': 0.0055, 'yaw': 0.3},
#     'green': {'x': 0.14, 'y': 0.06, 'z': 0.0055, 'yaw': -0.5},
#     'blue': {'x': 0.22, 'y': 0.04, 'z': 0.0055, 'yaw': 0.8},
# }

GROUND_TRUTH = GROUND_TRUTH_PHASE_A


class DetectionVerifier(Node):
    def __init__(self):
        super().__init__('detection_verifier')
        self.subscription = self.create_subscription(
            TFMessage, '/objects_poses', self.objects_callback, 10
        )
        self.detections = {}
        self.detection_count = 0
        self.start_time = time.time()
        self.log_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'verification_results.log'
        )

        self.get_logger().info('Detection Verifier started')
        self.get_logger().info(f'Waiting for detections on /objects_poses...')
        self.get_logger().info(f'Ground truth: {len(GROUND_TRUTH)} objects')
        for name, gt in GROUND_TRUTH.items():
            self.get_logger().info(
                f'  {name}: ({gt["x"]:.3f}, {gt["y"]:.3f}, {gt["z"]:.3f})'
            )

        # Timer to print comparison every 5 seconds
        self.timer = self.create_timer(5.0, self.print_results)

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
        for gt_name in GROUND_TRUTH:
            if gt_name in det_lower:
                return gt_name
        return None

    def print_results(self):
        elapsed = time.time() - self.start_time

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
                gt = GROUND_TRUTH[gt_name]
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
            lines.append(f'\n  Matched: {matched}/{len(GROUND_TRUTH)} | '
                          f'Avg error: {avg_error:.1f}mm | '
                          f'{"PASS" if avg_error < 50 else "FAIL"} (threshold: 50mm)')

        lines.append('=' * 70)

        for line in lines:
            self.get_logger().info(line)

        # Append to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f'\n--- {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
                for line in lines:
                    f.write(line + '\n')
        except Exception:
            pass


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
