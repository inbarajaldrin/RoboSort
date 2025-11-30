# ROS2 FastDDS Tailscale Setup

Configure ROS2 to communicate across machines using FastDDS over Tailscale.

## Setup

### 1. Install Tailscale
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale status  # Note all Tailscale IPs
```

### 2. Configure FastDDS
Update IPs in `fast_ts.xml`, then:
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fast_ts.xml
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=45  # Same on all machines
ros2 daemon stop && ros2 daemon start
```

Add to `~/.bashrc` to make persistent:
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fast_ts.xml
```

### 3. Docker (if using)

**One-time setup:**
```bash
apt update
apt install -y tailscale
tailscale up --hostname=<hostname>  # Authenticates device and sets name in Tailscale network

# Create Tailscale state directory 
mkdir -p /var/lib/tailscale

# Start the Tailscale daemon in the background
tailscaled --state=/var/lib/tailscale/tailscaled.state > /dev/null 2>&1 &

# Export variables
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fast_ts.xml
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=45  # Same on all machines
ros2 daemon stop && ros2 daemon start
```

**Every time you start the container:**
```bash
# Start the Tailscale daemon
tailscaled --state=/var/lib/tailscale/tailscaled.state > /dev/null 2>&1 &
```

## Test
```bash
# Machine 1
ros2 run demo_nodes_cpp talker

# Machine 2
ros2 topic list  # Should see /chatter
ros2 run demo_nodes_cpp listener
```

## Notes
- FastDDS calculates ports automatically based on `ROS_DOMAIN_ID`
- All machines need same `ROS_DOMAIN_ID` and `fast_ts.xml` with all peer IPs
- Restart ROS2 daemon after config changes
- Start nodes AFTER setting env vars and restarting daemon

