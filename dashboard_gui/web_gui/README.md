# LEGO Sorting System Web GUI

A web-based GUI for monitoring and managing a LEGO sorting system with real-time camera feeds, PostgreSQL database integration, and MQTT/ROS2 support.

## Features

- **Dual Camera Streams**: Real-time video feeds from two cameras
- **PostgreSQL Database**: Persistent storage for sorting records
- **Real-time Statistics**: Live updates of sorting statistics and bar charts
- **MQTT Support**: Receive database updates via MQTT
- **ROS2 Support**: Receive camera feeds and database updates via ROS2 topics
- **Modern Web Interface**: Responsive, dark-themed UI

## Quick Start

### Option 1: Using the Run Script (Recommended)

```bash
./run.sh
```

### Option 2: Manual Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the application:**
   ```bash
   python3 web_gui.py
   ```

3. **Open in browser:**
   ```
   http://localhost:5001
   ```

## Prerequisites

### Required
- Python 3.8+
- PostgreSQL (for database features)
- Web browser

### Optional
- Mosquitto MQTT broker (for MQTT features)
- ROS2 (for ROS2 camera feeds)

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example` for template) or set environment variables:

```bash
# Database
export DB_NAME=lego_sorting_db
export DB_USER=postgres
export DB_PASSWORD=your_database_password  # Required - no default for security
export DB_HOST=localhost
export DB_PORT=5432

# MQTT
export MQTT_BROKER=broker.hivemq.com  # or localhost
export MQTT_PORT=1883
export MQTT_TOPIC=lego_sorting/sql_update
export MQTT_USERNAME=  # Optional
export MQTT_PASSWORD=  # Optional
```

**Important**: Never commit `.env` files or hardcoded passwords to git!

## Usage

### Starting the System

```bash
./run.sh
```

The script will:
1. Check Python and dependencies
2. Verify PostgreSQL, MQTT, and ROS2 availability
3. Free port 5001 if needed
4. Start the web GUI

### Testing MQTT

```bash
# In one terminal: Start web_gui.py
./run.sh

# In another terminal: Send test messages
python3 mqtt_publisher.py

# Or use mosquitto_pub
mosquitto_pub -h broker.hivemq.com -p 1883 -t 'lego_sorting/sql_update' \
  -m '{"action":"insert","aruco_marker_id":1,"color":"Red","status":"Processing","count":3}'
```

### Viewing Database

```bash
python3 view_database.py
```

### Testing MQTT Updates

```bash
python3 mqtt_publisher.py
```

## Project Structure

```
robot_gui_final/
├── web_gui.py              # Main Flask application
├── mqtt_publisher.py       # MQTT publisher for testing
├── view_database.py        # Database viewer utility
├── run.sh                  # Startup script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── static/                 # Static web assets
│   ├── css/
│   └── js/
├── templates/              # HTML templates
│   └── index.html
└── docs/                   # Documentation
    ├── SQL_UPDATE_TOPIC_FORMAT.md
    ├── MQTT_SETUP.md
    └── FIX_CAMERA_PERMISSIONS.md
```

## Documentation

- [SQL Update Topic Format](SQL_UPDATE_TOPIC_FORMAT.md) - Message format for database updates
- [MQTT Setup](MQTT_SETUP.md) - MQTT configuration and usage
- [Fix Camera Permissions](FIX_CAMERA_PERMISSIONS.md) - Camera troubleshooting guide

## Troubleshooting

### Port 5001 Already in Use

```bash
# Kill the process
lsof -ti :5001 | xargs kill -9
```

### Database Connection Failed

- Check PostgreSQL is running: `systemctl status postgresql`
- Verify credentials in environment variables
- Check database exists: `psql -U postgres -l`

### MQTT Not Working

- Check mosquitto is running: `systemctl status mosquitto`
- Verify broker address and port
- Test connection: `mosquitto_sub -h localhost -p 1883 -t 'test' -v`

### Camera Not Working

- Check camera permissions
- Verify camera ID in `web_gui.py`
- For ROS2 camera: Ensure `/camera_annotated` topic is publishing

## License

This project is for educational/research purposes.


