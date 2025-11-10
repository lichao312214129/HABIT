# run_debug_menu.py
"""Interactive menu to run debug scripts"""
import sys
from pathlib import Path


def show_menu():
    """Display menu of available debug scripts"""
    
    print("=" * 80)
    print("HABIT Debug Script Menu")
    print("=" * 80)
    print()
    
    scripts = {
        '1': ('debug_preprocess.py', 'Image Preprocessing Pipeline'),
        '2': ('debug_habitat.py', 'Habitat Analysis'),
        '3': ('debug_extract_features.py', 'Feature Extraction'),
        '4': ('debug_radiomics.py', 'Radiomics Extraction'),
        '5': ('debug_ml.py', 'Machine Learning Pipeline'),
        '6': ('debug_kfold.py', 'K-Fold Cross Validation'),
        '7': ('debug_icc.py', 'ICC Analysis'),
        '8': ('debug_test_retest.py', 'Test-Retest Reliability'),
        '9': ('debug_compare.py', 'Model Comparison'),
        'q': (None, 'Quit'),
    }
    
    for key, (script, description) in scripts.items():
        print(f"{key}. {description}")
    
    print()
    print("=" * 80)
    
    return scripts


def run_script(script_name: str):
    """Run a debug script"""
    
    tests_dir = Path(__file__).parent
    script_path = tests_dir / script_name
    
    if not script_path.exists():
        print(f"Error: Script {script_name} not found!")
        return
    
    print(f"\nRunning {script_name}...")
    print("=" * 80)
    
    # Import and run the script
    # We use exec instead of subprocess to keep it in the same Python process
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Save current sys.path and restore after execution
    old_path = sys.path.copy()
    try:
        # Add parent directory to path
        sys.path.insert(0, str(tests_dir.parent))
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        print(f"\nError running script: {e}")
    finally:
        sys.path = old_path


def main():
    """Main menu loop"""
    
    while True:
        scripts = show_menu()
        choice = input("Select an option: ").strip().lower()
        
        if choice == 'q':
            print("\nExiting...")
            break
        
        if choice in scripts and scripts[choice][0] is not None:
            run_script(scripts[choice][0])
            input("\nPress Enter to continue...")
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

