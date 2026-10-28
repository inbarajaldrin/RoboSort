#!/usr/bin/env python3
"""
Simple script to view the LEGO sorting database
"""
import psycopg2
import os
from psycopg2.extras import RealDictCursor

# Database configuration (same as web_gui.py)
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'lego_sorting_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),  # Require environment variable for security
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

def view_database():
    """View database contents"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM sorting_records")
        total = cursor.fetchone()['total']
        print(f"\n{'='*60}")
        print(f"LEGO Sorting Database - Total Records: {total}")
        print(f"{'='*60}\n")
        
        # Get statistics
        cursor.execute("""
            SELECT 
                status,
                COUNT(*) as count,
                SUM(count) as total_pieces
            FROM sorting_records
            GROUP BY status
            ORDER BY status
        """)
        print("Statistics by Status:")
        print("-" * 60)
        for row in cursor.fetchall():
            print(f"  {row['status']:15} | Records: {row['count']:5} | Pieces: {row['total_pieces']:5}")
        print()
        
        # Get recent records
        cursor.execute("""
            SELECT id, timestamp, color, aruco_marker_id, count, status
            FROM sorting_records
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        
        print("Recent Records (Last 20):")
        print("-" * 60)
        print(f"{'ID':<5} {'Timestamp':<20} {'Color':<10} {'ArUco':<7} {'Count':<6} {'Status':<12}")
        print("-" * 60)
        
        for record in cursor.fetchall():
            print(f"{record['id']:<5} {str(record['timestamp']):<20} {record['color']:<10} "
                  f"{record['aruco_marker_id']:<7} {record['count']:<6} {record['status']:<12}")
        
        # Get color distribution
        cursor.execute("""
            SELECT 
                color,
                COUNT(*) as record_count,
                SUM(count) as total_pieces,
                AVG(count) as avg_pieces
            FROM sorting_records
            GROUP BY color
            ORDER BY total_pieces DESC
        """)
        
        print("\nColor Distribution:")
        print("-" * 60)
        print(f"{'Color':<10} {'Records':<10} {'Total Pieces':<15} {'Avg Pieces':<12}")
        print("-" * 60)
        for row in cursor.fetchall():
            print(f"{row['color']:<10} {row['record_count']:<10} {row['total_pieces']:<15} "
                  f"{row['avg_pieces']:.1f}")
        
        cursor.close()
        conn.close()
        
        print(f"\n{'='*60}")
        print("Database connection closed.")
        print(f"{'='*60}\n")
        
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print("You can create it by running web_gui.py first.")

if __name__ == '__main__':
    view_database()


