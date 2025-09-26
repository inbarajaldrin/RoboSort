#!/usr/bin/env python3

import cv2
import numpy as np
import os

def generate_aruco_markers():
    """
    Generate ArUco markers from ID 1 to 15 using 4x4 dictionary.
    Each marker is 3x3 cm with no white border.
    """
    
    # Create output directory if it doesn't exist
    output_dir = "aruco_markers"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Marker size in pixels (adjust as needed for 3x3 cm)
    # Assuming 300 DPI: 3cm = 1.18 inches = 354 pixels
    marker_size = 354
    
    print("Generating ArUco markers (ID 1-15) with 4x4 dictionary...")
    print(f"Marker size: {marker_size}x{marker_size} pixels (3x3 cm)")
    print(f"Output directory: {output_dir}/")
    
    # Try different approaches for ArUco dictionary creation
    aruco_dict = None
    
    try:
        # Method 1: Modern OpenCV 4.x
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        print("Using getPredefinedDictionary method")
    except AttributeError:
        try:
            # Method 2: Alternative modern API
            aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_4X4_50)
            print("Using Dictionary.get method")
        except AttributeError:
            try:
                # Method 3: Older API
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                print("Using Dictionary_get method")
            except AttributeError:
                print("Error: Could not create ArUco dictionary with any known method")
                return
    
    if aruco_dict is None:
        print("Error: Failed to create ArUco dictionary")
        return
    
    for marker_id in range(1, 16):  # IDs 1 to 15
        try:
            # Try different marker generation methods
            marker = None
            
            # Method 1: generateImageMarker (modern)
            try:
                marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
                print(f"Generated marker {marker_id} using generateImageMarker")
            except:
                # Method 2: drawMarker (older API)
                try:
                    marker = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
                    print(f"Generated marker {marker_id} using drawMarker")
                except Exception as e:
                    print(f"Failed to generate marker {marker_id}: {e}")
                    continue
            
            if marker is None:
                print(f"Skipping marker {marker_id} - generation failed")
                continue
            
            # Save marker as PNG
            filename = f"{output_dir}/aruco_marker_{marker_id:02d}.png"
            cv2.imwrite(filename, marker)
            
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error processing marker {marker_id}: {e}")
            continue
    
    print(f"\nArUco marker generation completed!")
    print(f"Dictionary: 4x4")
    print(f"Dimensions: 3x3 cm each")
    print(f"No white border")
    
    # Verify markers were created
    created_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"Successfully created {len(created_files)} marker files")

if __name__ == "__main__":
    generate_aruco_markers()