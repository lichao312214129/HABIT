#!/usr/bin/env python3
"""
Setup script for Sphinx documentation.

This script helps set up the Sphinx documentation environment.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for documentation."""
    base_dir = Path(__file__).parent / 'docs'
    
    dirs = [
        'source/getting_started',
        'source/user_guide',
        'source/api',
        'source/tutorials',
        'source/architecture',
        'build',
        '_static',
        '_templates',
    ]
    
    for dir_path in dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {full_path}")

def check_dependencies():
    """Check if required packages are installed."""
    required = ['sphinx', 'sphinx_rtd_theme', 'myst_parser']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("All dependencies are installed.")
    return True

def main():
    """Main setup function."""
    print("Setting up Sphinx documentation...")
    print()
    
    create_directories()
    print()
    
    if not check_dependencies():
        print("\nPlease install missing dependencies and run again.")
        sys.exit(1)
    
    print()
    print("Documentation setup complete!")
    print()
    print("Next steps:")
    print("  1. cd docs")
    print("  2. make html")
    print("  3. Open build/html/index.html in your browser")

if __name__ == '__main__':
    main()
