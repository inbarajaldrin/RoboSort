# /sql_update ROS2 Topic Format

## Overview
The `/sql_update` topic is used to send database update messages. The GUI subscribes to this topic and automatically updates the PostgreSQL database when messages are received.

## Topic Details
- **Topic Name**: `/sql_update`
- **Message Type**: `std_msgs/String`
- **Message Format**: JSON string

## Expected Message Format

The message format depends on the action type. The `action` field determines whether to insert a new record or update an existing one.

### Insert New Record (Default)

```json
{
    "action": "insert",
    "aruco_marker_id": 1,
    "color": "Red",
    "status": "Processing",
    "count": 5,
    "timestamp": "2025-12-01 12:30:45"
}
```

### Update Processing to Completed

To update an existing "Processing" record to "Completed" without creating a new entry:

```json
{
    "action": "update",
    "aruco_marker_id": 1,
    "color": "Red"
}
```

This will find the most recent "Processing" record for the given ArUco marker ID and update it to "Completed".

### Update by Record ID

To update a specific record by its database ID:

```json
{
    "action": "update_by_id",
    "record_id": 123,
    "status": "Completed"
}
```

### Fields by Action Type

#### Insert Action (`action: "insert"`)

**Required Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `aruco_marker_id` | integer | ArUco marker ID associated with the color | `1`, `2`, `3` |
| `color` | string | Color of the LEGO piece | `"Red"`, `"Blue"`, `"Green"` |
| `status` | string | Status of the sorting operation | `"Processing"` or `"Completed"` |
| `count` | integer | Number of LEGO pieces | `1`, `5`, `10` |

**Optional Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `action` | string | Action type (default: "insert") | `"insert"` |
| `timestamp` | string | Timestamp in format `YYYY-MM-DD HH:MM:SS`. If not provided, current timestamp is used | `"2025-12-01 12:30:45"` |

#### Update Action (`action: "update"`)

**Required Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `action` | string | Must be `"update"` | `"update"` |
| `aruco_marker_id` | integer | ArUco marker ID to find the Processing record | `1`, `2`, `3` |

**Optional Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `color` | string | Color to help identify the record (optional but recommended) | `"Red"`, `"Blue"` |

This action finds the most recent "Processing" record for the given ArUco marker ID and updates it to "Completed".

#### Update by ID Action (`action: "update_by_id"`)

**Required Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `action` | string | Must be `"update_by_id"` | `"update_by_id"` |
| `record_id` | integer | Database record ID to update | `123`, `456` |
| `status` | string | New status (default: "Completed") | `"Completed"` |

## Valid Values

### Status Values
- `"Processing"` - Item is currently being processed
- `"Completed"` - Item processing is complete

### Color Values
- `"Red"` (ArUco ID: 1)
- `"Blue"` (ArUco ID: 2)
- `"Green"` (ArUco ID: 3)
- `"Yellow"` (ArUco ID: 4)
- `"Orange"` (ArUco ID: 5)
- `"Black"` (ArUco ID: 6)
- `"White"` (ArUco ID: 7)
- `"Purple"` (ArUco ID: 8)

## Example Messages

### Example 1: Insert New Processing Record
```json
{
    "action": "insert",
    "aruco_marker_id": 1,
    "color": "Red",
    "status": "Processing",
    "count": 3
}
```

### Example 2: Insert Completed Record with Timestamp
```json
{
    "action": "insert",
    "aruco_marker_id": 2,
    "color": "Blue",
    "status": "Completed",
    "count": 5,
    "timestamp": "2025-12-01 14:25:30"
}
```

### Example 3: Update Processing to Completed
```json
{
    "action": "update",
    "aruco_marker_id": 1,
    "color": "Red"
}
```

This finds the most recent "Processing" record for ArUco ID 1 and updates it to "Completed" without creating a new entry.

### Example 4: Update by Record ID
```json
{
    "action": "update_by_id",
    "record_id": 123,
    "status": "Completed"
}
```

### Example 5: Insert Multiple Pieces (Backward Compatible)
```json
{
    "aruco_marker_id": 3,
    "color": "Green",
    "status": "Processing",
    "count": 10
}
```

If `action` is not specified, it defaults to "insert" for backward compatibility.

## Publishing Messages

### Python Example
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class SQLUpdatePublisher(Node):
    def __init__(self):
        super().__init__('sql_update_publisher')
        self.publisher = self.create_publisher(String, '/sql_update', 10)
    
    def publish_insert(self, aruco_id, color, status, count, timestamp=None):
        """Insert a new record"""
        message_data = {
            'action': 'insert',
            'aruco_marker_id': aruco_id,
            'color': color,
            'status': status,
            'count': count
        }
        
        if timestamp:
            message_data['timestamp'] = timestamp
        
        msg = String()
        msg.data = json.dumps(message_data)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published insert: {msg.data}')
    
    def publish_update(self, aruco_id, color=None):
        """Update Processing to Completed"""
        message_data = {
            'action': 'update',
            'aruco_marker_id': aruco_id
        }
        
        if color:
            message_data['color'] = color
        
        msg = String()
        msg.data = json.dumps(message_data)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published update: {msg.data}')
    
    def publish_update_by_id(self, record_id, status='Completed'):
        """Update specific record by ID"""
        message_data = {
            'action': 'update_by_id',
            'record_id': record_id,
            'status': status
        }
        
        msg = String()
        msg.data = json.dumps(message_data)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published update_by_id: {msg.data}')

# Usage
rclpy.init()
node = SQLUpdatePublisher()

# Insert a new Processing record
node.publish_insert(aruco_id=1, color='Red', status='Processing', count=3)

# Later, update it to Completed (no new entry created)
node.publish_update(aruco_id=1, color='Red')

rclpy.spin_once(node)
```

### C++ Example
```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>

class SQLUpdatePublisher : public rclcpp::Node {
public:
    SQLUpdatePublisher() : Node("sql_update_publisher") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("/sql_update", 10);
    }
    
    void publish_update(int aruco_id, const std::string& color, 
                       const std::string& status, int count) {
        nlohmann::json json_data;
        json_data["aruco_marker_id"] = aruco_id;
        json_data["color"] = color;
        json_data["status"] = status;
        json_data["count"] = count;
        
        auto message = std_msgs::msg::String();
        message.data = json_data.dump();
        publisher_->publish(message);
    }
    
private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

## Database Behavior

When a message is received on `/sql_update`:
1. The message is parsed and validated
2. A new record is inserted into the `sorting_records` table
3. The record is automatically published back to `/sql_update` (for confirmation)
4. The GUI automatically refreshes to show the new data

## Error Handling

- Invalid JSON: Message is logged and ignored
- Missing required fields: Warning is logged, message is ignored
- Invalid status: Status is set to "Processing" by default
- Database error: Error is logged, message is ignored

## Notes

- The GUI automatically subscribes to `/sql_update` when ROS2 is available
- Messages are processed asynchronously
- The database is updated immediately upon receiving a valid message
- The GUI refreshes every 2 seconds to show new data

