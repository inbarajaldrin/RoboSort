#!/usr/bin/env python3
"""
MQTT Publisher for LEGO Sorting System
Uses public broker.hivemq.com for testing
Based on the example pattern with proper callback handling
"""

import time
import random
import paho.mqtt.client as mqtt
import json

BROKER = "broker.hivemq.com"   # Public test broker (do NOT send secrets)
PORT = 1883                    # Unencrypted MQTT port
CLIENT_ID = "LEGO_Sorting_Publisher"  # Make unique if many clients run this

# Topic configuration
TOPIC = "lego_sorting/sql_update"

def on_connect(client, userdata, flags, rc, properties=None):
    """Called when the client connects to the broker.
    rc == 0 means success; non-zero indicates a failure code.
    """
    if rc == 0:
        print(f"✓ Connected to {BROKER} (rc={rc})")
    else:
        print(f"⚠ Connection failed (rc={rc})")

def on_publish(client, userdata, mid, reason_codes=None, properties=None):
    """Called when a publish completes.
    - QoS 0: fired after the packet is sent by the client (no broker ack).
    - QoS 1/2: fired after broker ack (PUBACK/PUBCOMP).
    'mid' is the message ID for the published message.
    """
    print(f"✓ Publish confirmed (mid={mid})")

# Create client with VERSION2 callback API
client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_publish = on_publish

# Connect to broker
print(f"Connecting to {BROKER}:{PORT}...")
client.connect(BROKER, PORT, keepalive=60)  # 'keepalive' is the maximum period (in seconds) between communications.
client.loop_start()  # Start the background network loop so MQTT I/O and callbacks run asynchronously.
print("Loop started. Publishing messages...")
time.sleep(1)  # Wait for connection

try:
    # Example 1: Insert a Processing record
    print("\n--- Example 1: Insert Processing Record ---")
    message1 = {
        "action": "insert",
        "aruco_marker_id": 1,
        "color": "Red",
        "status": "Processing",
        "count": 3
    }
    payload1 = json.dumps(message1)
    info = client.publish(TOPIC, payload=payload1, qos=1, retain=False)
    print(f"Published {payload1} → {TOPIC}")
    time.sleep(1)
    
    # Example 2: Update Processing to Completed
    print("\n--- Example 2: Update to Completed ---")
    message2 = {
        "action": "update",
        "aruco_marker_id": 1,
        "color": "Red"
    }
    payload2 = json.dumps(message2)
    info = client.publish(TOPIC, payload=payload2, qos=1, retain=False)
    print(f"Published {payload2} → {TOPIC}")
    time.sleep(1)
    
    # Example 3: Insert another Processing record
    print("\n--- Example 3: Insert Another Processing Record ---")
    message3 = {
        "action": "insert",
        "aruco_marker_id": 2,
        "color": "Blue",
        "status": "Processing",
        "count": 5
    }
    payload3 = json.dumps(message3)
    info = client.publish(TOPIC, payload=payload3, qos=1, retain=False)
    print(f"Published {payload3} → {TOPIC}")
    time.sleep(1)
    
    # Example 4: Update to Completed
    print("\n--- Example 4: Update to Completed ---")
    message4 = {
        "action": "update",
        "aruco_marker_id": 2,
        "color": "Blue"
    }
    payload4 = json.dumps(message4)
    info = client.publish(TOPIC, payload=payload4, qos=1, retain=False)
    print(f"Published {payload4} → {TOPIC}")
    time.sleep(1)
    
    # Example 5: Simulate multiple readings (like sensor data)
    print("\n--- Example 5: Simulate Multiple Readings ---")
    colors = ["Red", "Blue", "Green", "Yellow"]
    aruco_ids = [1, 2, 3, 4]
    
    for i in range(4):
        color = colors[i]
        aruco_id = aruco_ids[i]
        count = random.randint(1, 5)
        
        # Insert Processing
        message = {
            "action": "insert",
            "aruco_marker_id": aruco_id,
            "color": color,
            "status": "Processing",
            "count": count
        }
        payload = json.dumps(message)
        info = client.publish(TOPIC, payload=payload, qos=1, retain=False)
        print(f"Published {payload} → {TOPIC}")
        time.sleep(0.5)
        
        # Update to Completed
        update_message = {
            "action": "update",
            "aruco_marker_id": aruco_id,
            "color": color
        }
        update_payload = json.dumps(update_message)
        info = client.publish(TOPIC, payload=update_payload, qos=1, retain=False)
        print(f"Published {update_payload} → {TOPIC}")
        time.sleep(0.5)
    
    print("\n✓ All messages published. Waiting for delivery confirmation...")
    time.sleep(2)  # Wait for all publish confirmations

finally:
    # Always stop the loop and disconnect cleanly (even on errors/KeyboardInterrupt).
    client.loop_stop()
    client.disconnect()
    print("✓ Disconnected")
