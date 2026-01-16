import os
import shutil

base_dir = r'f:\work\habit_project\habit\core\habitat_analysis'
source = os.path.join(base_dir, 'features')
dest = os.path.join(base_dir, 'feature_construction')

print("=== Directory Rename Script ===")
print(f"Source: {source}")
print(f"Destination: {dest}")
print(f"Source exists: {os.path.exists(source)}")
print(f"Dest exists: {os.path.exists(dest)}")

# Check if source exists
if not os.path.exists(source):
    print("ERROR: Source directory does not exist!")
    sys.exit(1)

# Remove destination if it exists
if os.path.exists(dest):
    print("Removing existing destination directory...")
    shutil.rmtree(dest)

# Rename
print("Renaming directory...")
os.rename(source, dest)
print("SUCCESS: Directory renamed!")

# Verify
if os.path.exists(dest) and not os.path.exists(source):
    print("VERIFICATION: Rename successful!")
else:
    print("ERROR: Rename verification failed!")
