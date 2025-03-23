#!/usr/bin/env python3
"""
Create a custom classes file for wood imperfection detection
This allows you to map YOLOv8 outputs to specific wood imperfection classes
"""

import os
import argparse
from pathlib import Path

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def create_classes_file(classes, output_path):
    """
    Create a classes.txt file with the specified wood imperfection classes
    
    Args:
        classes (list): List of class names
        output_path (str): Path to save the classes file
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        ensure_dir_exists(parent_dir)
    
    # Write classes to file
    with open(output_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    print(f"Created classes file at {output_path} with {len(classes)} classes")

def main():
    parser = argparse.ArgumentParser(description='Create a custom classes file for wood imperfection detection')
    parser.add_argument('--output', type=str, default='models/classes.txt',
                        help='Path to save the classes file')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=["damage", "crack", "hole", "stain", "dent", "scratch", "split", "tear"],
                        help='List of class names for wood imperfection detection')
    
    args = parser.parse_args()
    
    create_classes_file(args.classes, args.output)
    
    print(f"To use these classes, set the environment variable:")
    print(f"YOLOV8_CLASS_FILE={args.output}")

if __name__ == "__main__":
    main() 