"""
Script to run habitat get-habitat command
Execute habitat analysis using the configuration file in current directory
"""

import subprocess
import sys
from pathlib import Path


def main():
    """
    Run habitat get-habitat command with the configuration file
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    config_file = current_dir / "config_getting_habitat.yaml"
    
    # Check if configuration file exists
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    print(f"Running habitat get-habitat with config: {config_file}")
    print("=" * 60)
    
    # Build command
    cmd = ["habit", "get-habitat", "-c", str(config_file)]
    
    try:
        # Run command
        result = subprocess.run(
            cmd,
            cwd=current_dir,
            check=True,
            text=True
        )
        
        print("=" * 60)
        print("Habitat analysis completed successfully!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"Error: Command failed with return code {e.returncode}")
        sys.exit(e.returncode)
        
    except FileNotFoundError:
        print("Error: 'habit' command not found. Please make sure habit is installed.")
        print("You can install it with: pip install -e /path/to/habit")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
