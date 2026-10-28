# MQTT Integration for LEGO Sorting System

The web GUI now supports MQTT in addition to ROS2 for database updates. Both systems can work simultaneously.

## Configuration

MQTT configuration is set via environment variables:

Create a `.env` file or set environment variables:

```bash
export MQTT_BROKER=broker.hivemq.com  # MQTT broker address (default: broker.hivemq.com)
export MQTT_PORT=1883                 # MQTT broker port (default: 1883)
export MQTT_TOPIC=lego_sorting/sql_update  # Topic name (default: lego_sorting/sql_update)
export MQTT_CLIENT_ID=lego_gui_client # Client ID (default: lego_gui_client)
export MQTT_USERNAME=                # Optional: MQTT username
export MQTT_PASSWORD=                # Optional: MQTT password
```

**Security Note**: Never commit actual passwords or credentials to git. Use environment variables or `.env` files (which should be in `.gitignore`).

## Message Format

MQTT messages use the same JSON format as ROS2 `/sql_update` topic:

### Insert Action
```json
{
    "action": "insert",
    "aruco_marker_id": 1,
    "color": "Red",
    "status": "Processing",
    "count": 3,
    "timestamp": "2024-01-01 12:00:00"  // Optional
}
```

### Update Action (Processing -> Completed)
```json
{
    "action": "update",
    "aruco_marker_id": 1,
    "color": "Red"  // Optional but recommended
}
```

### Update by ID Action
```json
{
    "action": "update_by_id",
    "record_id": 123,
    "status": "Completed"
}
```

## Installation

1. Install the MQTT client library:
```bash
pip install paho-mqtt
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the GUI with MQTT

1. Start your MQTT broker (e.g., Mosquitto):
```bash
# On Ubuntu/Debian
sudo systemctl start mosquitto

# Or run locally
mosquitto -p 1883
```

2. Start the web GUI:
```bash
python3 web_gui.py
```

The GUI will automatically connect to the MQTT broker and subscribe to the configured topic.

## Testing MQTT

Use the provided test script:

```bash
python3 mqtt_publisher.py
```

Or use the `mosquitto_pub` command:

```bash
# Insert a Processing record
mosquitto_pub -h localhost -p 1883 -t lego_sorting/sql_update -m '{"action":"insert","aruco_marker_id":1,"color":"Red","status":"Processing","count":3}'

# Update to Completed
mosquitto_pub -h localhost -p 1883 -t lego_sorting/sql_update -m '{"action":"update","aruco_marker_id":1,"color":"Red"}'
```

## ROS2 and MQTT Together

Both ROS2 and MQTT can work simultaneously:
- ROS2 subscribes to `/sql_update` topic
- MQTT subscribes to `lego_sorting/sql_update` topic
- Both use the same message format and processing logic
- Database updates from either source are handled identically

## Troubleshooting

1. **MQTT connection failed**: Check that the MQTT broker is running and accessible
2. **Messages not processed**: Check the topic name matches between publisher and subscriber
3. **Import error**: Install paho-mqtt: `pip install paho-mqtt`

## Example: Python MQTT Publisher

See `mqtt_publisher.py` for a complete example of how to publish MQTT messages.


