#!/usr/bin/env python3
"""
Script to run all demonstration scripts in the examples directory.

This script will:
1. Run each Python script in the examples directory one by one
2. Capture all output and write it to output_of_complete_battery_of_demos.txt
3. Continue running even if a script fails
4. Log errors with traceback for debugging
"""

import os
import sys
import time
import traceback
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    examples_dir = Path(__file__).parent.absolute()
    
    # Get the project root directory (parent of examples dir)
    project_root = examples_dir.parent
    
    # Path to the output file
    output_file = examples_dir / "output_of_complete_battery_of_demos.txt"
    
    # Get all Python files in the examples directory
    python_files = [f for f in os.listdir(examples_dir) if f.endswith('.py') and f != os.path.basename(__file__)]
    
    # Sort files alphabetically
    python_files.sort()
    
    print(f"Found {len(python_files)} Python scripts to run")
    print(f"Output will be written to {output_file}")
    print(f"Running scripts from project root: {project_root}")
    
    # Open the output file (overwrite if it exists)
    with open(output_file, 'w') as f:
        f.write("=== Running Complete Battery of Demonstration Scripts ===\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Found {len(python_files)} scripts to run\n")
        f.write(f"Running from project root: {project_root}\n\n")
        
        # Run each script
        for i, script in enumerate(python_files, 1):
            script_path = examples_dir / script
            
            separator = "=" * 80
            header = f"\n\n{separator}\n({i}/{len(python_files)}) RUNNING: {script}\n{separator}\n"
            
            print(f"Running {i}/{len(python_files)}: {script}")
            f.write(header)
            f.flush()  # Ensure header is written immediately
            
            try:
                # Run the script and capture output
                start_time = time.time()
                
                # Use subprocess to run the script
                process = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=project_root  # Run from project root instead of examples dir
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Write stdout if any
                if process.stdout:
                    f.write("\n--- STDOUT ---\n")
                    f.write(process.stdout)
                
                # Write stderr if any
                if process.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(process.stderr)
                
                # Write execution result
                if process.returncode == 0:
                    result = f"\n--- SUCCESS: Script completed in {duration:.2f} seconds ---\n"
                else:
                    result = f"\n--- FAILED: Script exited with code {process.returncode} after {duration:.2f} seconds ---\n"
                
                f.write(result)
                
            except Exception as e:
                # Catch any exceptions that might occur when trying to run the script
                f.write("\n--- ERROR: Failed to execute script ---\n")
                f.write(f"Exception: {type(e).__name__}: {e}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            
            f.flush()  # Ensure output is written immediately
        
        # Write summary
        f.write("\n\n=== Complete Battery of Demonstration Scripts Finished ===\n")
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main() 