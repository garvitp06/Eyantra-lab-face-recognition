# app_bundler.py - Script to create an executable using PyInstaller

import os
import sys
import argparse
import subprocess
import shutil
import platform


def create_executable(app_path, output_dir, icon_path=None, one_file=False, console=False):
    """
    Create an executable using PyInstaller

    Args:
        app_path (str): Path to the main Python file to bundle
        output_dir (str): Directory to place the bundled executable
        icon_path (str, optional): Path to application icon (.ico)
        one_file (bool): Whether to bundle as a single file
        console (bool): Whether to show console window
    """
    print("Creating executable with PyInstaller...")

    # Create command
    cmd = ["pyinstaller"]

    # Add icon if provided
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])

    # Add one-file/one-dir option
    if one_file:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Add windowed/console option
    if not console:
        cmd.append("--windowed")

    # Add name and output directory
    app_name = os.path.splitext(os.path.basename(app_path))[0]
    cmd.extend(["--name", app_name])
    cmd.extend(["--distpath", output_dir])

    # Add required data files
    # OpenVINO models
    cmd.extend(["--add-data", "*.xml;."])
    cmd.extend(["--add-data", "*.bin;."])

    # Add face detection cascade
    if platform.system() == "Windows":
        try:
            import cv2
            cv2_path = os.path.dirname(cv2.__file__)
            cascade_path = os.path.join(cv2_path, "data")
            if os.path.exists(cascade_path):
                cmd.extend(["--add-data", f"{cascade_path};cv2/data"])
        except ImportError:
            print("Warning: OpenCV package not found, cascade files may not be included.")
    else:
        # For Linux/Mac, use colon instead of semicolon
        try:
            import cv2
            cv2_path = os.path.dirname(cv2.__file__)
            cascade_path = os.path.join(cv2_path, "data")
            if os.path.exists(cascade_path):
                cmd.extend(["--add-data", f"{cascade_path}:cv2/data"])
        except ImportError:
            print("Warning: OpenCV package not found, cascade files may not be included.")

    # Add application entrypoint
    cmd.append(app_path)

    # Run PyInstaller
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

    print(f"Executable created in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create executable for Facial Attendance System")
    parser.add_argument("app_file", help="Path to the main Python file")
    parser.add_argument("--output", "-o", default="dist", help="Output directory for executable")
    parser.add_argument("--icon", "-i", help="Path to icon file (.ico)")
    parser.add_argument("--onefile", action="store_true", help="Create a single file executable")
    parser.add_argument("--console", action="store_true", help="Show console window")

    args = parser.parse_args()

    if not os.path.exists(args.app_file):
        print(f"Error: Main Python file not found: {args.app_file}")
        sys.exit(1)

    if args.icon and not os.path.exists(args.icon):
        print(f"Warning: Icon file not found: {args.icon}")
        args.icon = None

    create_executable(args.app_file, args.output, args.icon, args.onefile, args.console)