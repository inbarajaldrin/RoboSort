#!/bin/bash
# Run script for LEGO Sorting System Web GUI
# This script handles all setup and starts the application

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LEGO Sorting System - Startup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$1 " || ss -tuln 2>/dev/null | grep -q ":$1 "
}

# Function to kill process on port
kill_port() {
    local port=$1
    if port_in_use "$port"; then
        echo -e "${YELLOW}⚠ Port $port is in use. Attempting to free it...${NC}"
        lsof -ti :"$port" 2>/dev/null | xargs kill -9 2>/dev/null || true
        sleep 1
        if port_in_use "$port"; then
            echo -e "${RED}✗ Failed to free port $port${NC}"
            return 1
        else
            echo -e "${GREEN}✓ Port $port freed${NC}"
        fi
    fi
    return 0
}

# Check Python
echo -e "${BLUE}[1/6] Checking Python...${NC}"
if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Check dependencies
echo -e "${BLUE}[2/6] Checking Python dependencies...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
else
    echo "Installing/updating dependencies..."
    pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true
    echo -e "${GREEN}✓ Dependencies checked${NC}"
fi

# Check PostgreSQL (optional but recommended)
echo -e "${BLUE}[3/6] Checking PostgreSQL...${NC}"
if command_exists psql; then
    if pgrep -x postgres > /dev/null || systemctl is-active --quiet postgresql 2>/dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is running${NC}"
    else
        echo -e "${YELLOW}⚠ PostgreSQL is not running (database features may not work)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ PostgreSQL not found (database features may not work)${NC}"
fi

# Check MQTT broker (optional)
echo -e "${BLUE}[4/6] Checking MQTT broker...${NC}"
if command_exists mosquitto; then
    if pgrep -x mosquitto > /dev/null || systemctl is-active --quiet mosquitto 2>/dev/null; then
        echo -e "${GREEN}✓ Mosquitto MQTT broker is running${NC}"
    else
        echo -e "${YELLOW}⚠ Mosquitto is not running (MQTT features may not work)${NC}"
        echo "  Start with: sudo systemctl start mosquitto"
    fi
else
    echo -e "${YELLOW}⚠ Mosquitto not found (MQTT features may not work)${NC}"
fi

# Check ROS2 (optional)
echo -e "${BLUE}[5/6] Checking ROS2...${NC}"
if command_exists ros2; then
    echo -e "${GREEN}✓ ROS2 is available${NC}"
else
    echo -e "${YELLOW}⚠ ROS2 not found (ROS2 features may not work)${NC}"
fi

# Check and free port 5001
echo -e "${BLUE}[6/6] Checking port 5001...${NC}"
if ! kill_port 5001; then
    echo -e "${RED}✗ Cannot free port 5001. Please stop the process manually.${NC}"
    echo "  Find process: lsof -i :5001"
    exit 1
fi

# Set environment variables (optional)
export MQTT_BROKER="${MQTT_BROKER:-broker.hivemq.com}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_TOPIC="${MQTT_TOPIC:-lego_sorting/sql_update}"

# Display configuration
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "${BLUE}========================================${NC}"
echo "  Web GUI: http://localhost:5001"
echo "  MQTT Broker: $MQTT_BROKER:$MQTT_PORT"
echo "  MQTT Topic: $MQTT_TOPIC"
echo ""

# Start the application
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting LEGO Sorting System Web GUI...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Trap Ctrl+C for clean shutdown
trap 'echo ""; echo -e "${YELLOW}Shutting down...${NC}"; exit 0' INT TERM

# Run web_gui.py
python3 web_gui.py


