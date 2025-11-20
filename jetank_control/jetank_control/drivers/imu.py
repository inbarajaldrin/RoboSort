#!/usr/bin/env python3
"""
Standalone BNO055 IMU Publisher for Jetank Control

Based on: https://github.com/flynneva/bno055
Original Copyright: 2021 AUTHORS (flynneva)
License: BSD

Configured for I2C (bus 0, address 0x28).
"""

import sys
import threading
import struct
import json
from math import sqrt
from time import sleep

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter

# ROS2 message types
from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu, MagneticField, Temperature
from std_msgs.msg import String

# I2C communication
try:
    from smbus import SMBus
except ImportError:
    print("Error: smbus module not found. Install with: pip install smbus")
    sys.exit(1)


# ============================================================================
# BNO055 Register Definitions (embedded from bno055 package)
# ============================================================================

BNO055_ADDRESS_A = 0x28
BNO055_ADDRESS_B = 0x29
BNO055_ID = 0xA0

# Register addresses
BNO055_CHIP_ID_ADDR = 0x00
BNO055_ACCEL_DATA_X_LSB_ADDR = 0x08
BNO055_MAG_DATA_X_LSB_ADDR = 0x0E
BNO055_GYRO_DATA_X_LSB_ADDR = 0x14
BNO055_EULER_H_LSB_ADDR = 0x1A
BNO055_QUATERNION_DATA_W_LSB_ADDR = 0x20
BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR = 0x28
BNO055_GRAVITY_DATA_X_LSB_ADDR = 0x2E
BNO055_TEMP_ADDR = 0x34
BNO055_CALIB_STAT_ADDR = 0x35

BNO055_PAGE_ID_ADDR = 0x07
BNO055_OPR_MODE_ADDR = 0x3D
BNO055_PWR_MODE_ADDR = 0x3E
BNO055_SYS_TRIGGER_ADDR = 0x3F
BNO055_UNIT_SEL_ADDR = 0x3B
BNO055_AXIS_MAP_CONFIG_ADDR = 0x41
BNO055_AXIS_MAP_SIGN_ADDR = 0x42

# Operation modes
OPERATION_MODE_CONFIG = 0x00
OPERATION_MODE_NDOF = 0x0C
POWER_MODE_NORMAL = 0x00

# Default values
DEFAULT_VARIANCE_ACC = [0.017, 0.017, 0.017]
DEFAULT_VARIANCE_ANGULAR_VEL = [0.04, 0.04, 0.04]
DEFAULT_VARIANCE_ORIENTATION = [0.0159, 0.0159, 0.0159]
DEFAULT_VARIANCE_MAG = [0.0, 0.0, 0.0]


# ============================================================================
# I2C Connector Class (embedded from bno055 package)
# ============================================================================

class I2CConnector:
    """I2C connector for BNO055 sensor."""
    
    def __init__(self, node, i2c_bus=0, i2c_addr=BNO055_ADDRESS_A):
        self.node = node
        self.bus = SMBus(i2c_bus)
        self.address = i2c_addr
    
    def connect(self):
        """Connect and verify sensor."""
        returned_id = self.bus.read_byte_data(self.address, BNO055_CHIP_ID_ADDR)
        if returned_id != BNO055_ID:
            raise IOError(f'Could not get BNO055 chip ID via I2C. Got: {returned_id}')
    
    def receive(self, reg_addr, length):
        """Read data from sensor."""
        buffer = bytearray()
        bytes_left = length
        while bytes_left > 0:
            read_len = min(bytes_left, 32)
            read_off = length - bytes_left
            response = self.bus.read_i2c_block_data(
                self.address, reg_addr + read_off, read_len)
            buffer += bytearray(response)
            bytes_left -= read_len
        return buffer
    
    def transmit(self, reg_addr, length, data):
        """Write data to sensor."""
        bytes_left = length
        while bytes_left > 0:
            write_len = min(bytes_left, 32)
            write_off = length - bytes_left
            datablock = list(data[write_off:write_off + write_len])
            self.bus.write_i2c_block_data(
                self.address, reg_addr + write_off, datablock)
            bytes_left -= write_len
        return True


# ============================================================================
# Sensor Service Class (simplified from bno055 package)
# ============================================================================

class SensorService:
    """Service for reading and publishing BNO055 sensor data."""
    
    def __init__(self, node, connector, config):
        self.node = node
        self.con = connector
        self.config = config
        
        prefix = config.get('ros_topic_prefix', 'imu/')
        qos = QoSProfile(depth=10)
        
        # Create publishers
        self.pub_imu_raw = node.create_publisher(Imu, prefix + 'data_raw', qos)
        self.pub_imu = node.create_publisher(Imu, prefix + 'data', qos)
        self.pub_mag = node.create_publisher(MagneticField, prefix + 'mag', qos)
        self.pub_grav = node.create_publisher(Vector3, prefix + 'grav', qos)
        self.pub_temp = node.create_publisher(Temperature, prefix + 'temp', qos)
        self.pub_calib_status = node.create_publisher(String, prefix + 'calib_status', qos)
    
    def configure(self):
        """Configure the IMU sensor."""
        self.node.get_logger().info('Configuring BNO055 device...')
        
        # Verify chip ID
        data = self.con.receive(BNO055_CHIP_ID_ADDR, 1)
        if data[0] != BNO055_ID:
            raise IOError(f'Device ID incorrect: {data[0]}')
        
        # Set config mode
        self.con.transmit(BNO055_OPR_MODE_ADDR, 1, bytes([OPERATION_MODE_CONFIG]))
        sleep(0.025)
        
        # Set power mode
        self.con.transmit(BNO055_PWR_MODE_ADDR, 1, bytes([POWER_MODE_NORMAL]))
        sleep(0.01)
        
        # Set page 0
        self.con.transmit(BNO055_PAGE_ID_ADDR, 1, bytes([0x00]))
        sleep(0.01)
        
        # Reset system
        self.con.transmit(BNO055_SYS_TRIGGER_ADDR, 1, bytes([0x00]))
        sleep(0.05)
        
        # Set units (m/s^2, rad/s, degrees)
        self.con.transmit(BNO055_UNIT_SEL_ADDR, 1, bytes([0x83]))
        sleep(0.01)
        
        # Set axis remap (P1 default)
        placement = self.config.get('placement_axis_remap', 'P1')
        mount_positions = {
            'P0': bytes(b'\x21\x04'), 'P1': bytes(b'\x24\x00'),
            'P2': bytes(b'\x24\x06'), 'P3': bytes(b'\x21\x02'),
            'P4': bytes(b'\x24\x03'), 'P5': bytes(b'\x21\x02'),
            'P6': bytes(b'\x21\x07'), 'P7': bytes(b'\x24\x05')
        }
        if placement in mount_positions:
            self.con.transmit(BNO055_AXIS_MAP_CONFIG_ADDR, 2, mount_positions[placement])
            sleep(0.01)
        
        # Set operation mode (NDOF)
        device_mode = self.config.get('operation_mode', OPERATION_MODE_NDOF)
        self.con.transmit(BNO055_OPR_MODE_ADDR, 1, bytes([device_mode]))
        sleep(0.05)
        
        self.node.get_logger().info('BNO055 IMU configuration complete.')
    
    def unpack_bytes_to_float(self, lsb, msb):
        """Convert two bytes to signed 16-bit integer."""
        return float(struct.unpack('h', struct.pack('BB', lsb, msb))[0])
    
    def get_sensor_data(self):
        """Read and publish sensor data."""
        # Read 45 bytes starting from accelerometer data
        buf = self.con.receive(BNO055_ACCEL_DATA_X_LSB_ADDR, 45)
        
        frame_id = self.config.get('frame_id', 'bno055')
        acc_factor = self.config.get('acc_factor', 100.0)
        gyr_factor = self.config.get('gyr_factor', 900.0)
        mag_factor = self.config.get('mag_factor', 16000000.0)
        grav_factor = self.config.get('grav_factor', 100.0)
        
        variance_acc = self.config.get('variance_acc', DEFAULT_VARIANCE_ACC)
        variance_angular_vel = self.config.get('variance_angular_vel', DEFAULT_VARIANCE_ANGULAR_VEL)
        variance_orientation = self.config.get('variance_orientation', DEFAULT_VARIANCE_ORIENTATION)
        variance_mag = self.config.get('variance_mag', DEFAULT_VARIANCE_MAG)
        
        now = self.node.get_clock().now().to_msg()
        
        # Publish raw IMU data
        imu_raw_msg = Imu()
        imu_raw_msg.header.stamp = now
        imu_raw_msg.header.frame_id = frame_id
        imu_raw_msg.orientation_covariance = [
            variance_orientation[0], 0.0, 0.0,
            0.0, variance_orientation[1], 0.0,
            0.0, 0.0, variance_orientation[2]
        ]
        imu_raw_msg.linear_acceleration.x = self.unpack_bytes_to_float(buf[0], buf[1]) / acc_factor
        imu_raw_msg.linear_acceleration.y = self.unpack_bytes_to_float(buf[2], buf[3]) / acc_factor
        imu_raw_msg.linear_acceleration.z = self.unpack_bytes_to_float(buf[4], buf[5]) / acc_factor
        imu_raw_msg.linear_acceleration_covariance = [
            variance_acc[0], 0.0, 0.0,
            0.0, variance_acc[1], 0.0,
            0.0, 0.0, variance_acc[2]
        ]
        imu_raw_msg.angular_velocity.x = self.unpack_bytes_to_float(buf[12], buf[13]) / gyr_factor
        imu_raw_msg.angular_velocity.y = self.unpack_bytes_to_float(buf[14], buf[15]) / gyr_factor
        imu_raw_msg.angular_velocity.z = self.unpack_bytes_to_float(buf[16], buf[17]) / gyr_factor
        imu_raw_msg.angular_velocity_covariance = [
            variance_angular_vel[0], 0.0, 0.0,
            0.0, variance_angular_vel[1], 0.0,
            0.0, 0.0, variance_angular_vel[2]
        ]
        self.pub_imu_raw.publish(imu_raw_msg)
        
        # Publish filtered IMU data with quaternion
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = frame_id
        imu_msg.orientation_covariance = imu_raw_msg.orientation_covariance
        
        # Quaternion (w, x, y, z)
        q_w = self.unpack_bytes_to_float(buf[24], buf[25])
        q_x = self.unpack_bytes_to_float(buf[26], buf[27])
        q_y = self.unpack_bytes_to_float(buf[28], buf[29])
        q_z = self.unpack_bytes_to_float(buf[30], buf[31])
        
        # Normalize quaternion
        norm = sqrt(q_x * q_x + q_y * q_y + q_z * q_z + q_w * q_w)
        if norm > 0:
            imu_msg.orientation.x = q_x / norm
            imu_msg.orientation.y = q_y / norm
            imu_msg.orientation.z = q_z / norm
            imu_msg.orientation.w = q_w / norm
        
        # Linear acceleration (from linear accel registers)
        imu_msg.linear_acceleration.x = self.unpack_bytes_to_float(buf[32], buf[33]) / acc_factor
        imu_msg.linear_acceleration.y = self.unpack_bytes_to_float(buf[34], buf[35]) / acc_factor
        imu_msg.linear_acceleration.z = self.unpack_bytes_to_float(buf[36], buf[37]) / acc_factor
        imu_msg.linear_acceleration_covariance = imu_raw_msg.linear_acceleration_covariance
        imu_msg.angular_velocity = imu_raw_msg.angular_velocity
        imu_msg.angular_velocity_covariance = imu_raw_msg.angular_velocity_covariance
        self.pub_imu.publish(imu_msg)
        
        # Publish magnetometer data
        mag_msg = MagneticField()
        mag_msg.header.stamp = now
        mag_msg.header.frame_id = frame_id
        mag_msg.magnetic_field.x = self.unpack_bytes_to_float(buf[6], buf[7]) / mag_factor
        mag_msg.magnetic_field.y = self.unpack_bytes_to_float(buf[8], buf[9]) / mag_factor
        mag_msg.magnetic_field.z = self.unpack_bytes_to_float(buf[10], buf[11]) / mag_factor
        mag_msg.magnetic_field_covariance = [
            variance_mag[0], 0.0, 0.0,
            0.0, variance_mag[1], 0.0,
            0.0, 0.0, variance_mag[2]
        ]
        self.pub_mag.publish(mag_msg)
        
        # Publish gravity vector
        grav_msg = Vector3()
        grav_msg.x = self.unpack_bytes_to_float(buf[38], buf[39]) / grav_factor
        grav_msg.y = self.unpack_bytes_to_float(buf[40], buf[41]) / grav_factor
        grav_msg.z = self.unpack_bytes_to_float(buf[42], buf[43]) / grav_factor
        self.pub_grav.publish(grav_msg)
        
        # Publish temperature
        temp_msg = Temperature()
        temp_msg.header.stamp = now
        temp_msg.header.frame_id = frame_id
        temp_msg.temperature = float(buf[44])
        self.pub_temp.publish(temp_msg)
    
    def get_calib_status(self):
        """Read and publish calibration status."""
        calib_status = self.con.receive(BNO055_CALIB_STAT_ADDR, 1)
        sys_cal = (calib_status[0] >> 6) & 0x03
        gyro_cal = (calib_status[0] >> 4) & 0x03
        accel_cal = (calib_status[0] >> 2) & 0x03
        mag_cal = calib_status[0] & 0x03
        
        calib_dict = {'sys': sys_cal, 'gyro': gyro_cal, 'accel': accel_cal, 'mag': mag_cal}
        calib_str = String()
        calib_str.data = json.dumps(calib_dict)
        self.pub_calib_status.publish(calib_str)


# ============================================================================
# ROS2 Node Class
# ============================================================================

class StandaloneIMU(Node):
    """Standalone IMU publisher node - completely self-contained."""
    
    def __init__(self):
        super().__init__('imu_publisher')
        self.sensor = None
        self.lock = threading.Lock()
        
        # Configuration (hardcoded - modify as needed)
        self.config = {
            'i2c_bus': 0,
            'i2c_addr': 40,  # 0x28 in decimal
            'ros_topic_prefix': 'imu/',
            'frame_id': 'bno055',
            'data_query_frequency': 10.0,  # Hz
            'calib_status_frequency': 0.1,  # Hz
            'operation_mode': OPERATION_MODE_NDOF,
            'placement_axis_remap': 'P1',
            'acc_factor': 100.0,
            'mag_factor': 16000000.0,
            'gyr_factor': 900.0,
            'grav_factor': 100.0,
            'variance_acc': DEFAULT_VARIANCE_ACC,
            'variance_angular_vel': DEFAULT_VARIANCE_ANGULAR_VEL,
            'variance_orientation': DEFAULT_VARIANCE_ORIENTATION,
            'variance_mag': DEFAULT_VARIANCE_MAG,
        }
        
        # Setup sensor
        self.setup()
        
        # Start timers
        self.start_timers()
    
    def setup(self):
        """Initialize sensor connection."""
        # Create I2C connector
        connector = I2CConnector(self, self.config['i2c_bus'], self.config['i2c_addr'])
        
        # Connect
        connector.connect()
        self.get_logger().info(
            f'Connected to BNO055 on I2C bus {self.config["i2c_bus"]}, '
            f'address 0x{self.config["i2c_addr"]:02X}'
        )
        
        # Create sensor service
        self.sensor = SensorService(self, connector, self.config)
        
        # Configure
        self.sensor.configure()
        self.get_logger().info('BNO055 IMU ready')
    
    def start_timers(self):
        """Start data publishing timers."""
        data_freq = 1.0 / self.config['data_query_frequency']
        self.data_timer = self.create_timer(data_freq, self.read_data)
        
        calib_freq = 1.0 / self.config['calib_status_frequency']
        self.calib_timer = self.create_timer(calib_freq, self.log_calibration_status)
    
    def read_data(self):
        """Read sensor data."""
        if self.lock.locked():
            return
        self.lock.acquire()
        try:
            self.sensor.get_sensor_data()
        except Exception as e:
            self.get_logger().warn(f'Data read failed: {type(e).__name__}: {e}')
        finally:
            self.lock.release()
    
    def log_calibration_status(self):
        """Log calibration status."""
        if self.lock.locked():
            return
        self.lock.acquire()
        try:
            self.sensor.get_calib_status()
        except Exception as e:
            self.get_logger().warn(f'Calibration status failed: {type(e).__name__}: {e}')
        finally:
            self.lock.release()
    
    def destroy_node(self):
        """Cleanup."""
        try:
            self.destroy_timer(self.data_timer)
            self.destroy_timer(self.calib_timer)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    node = None
    try:
        rclpy.init(args=args)
        node = StandaloneIMU()
        node.get_logger().info('IMU node started. Press Ctrl+C to stop.')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down IMU node...')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

