import os
import sys

base_dir = r'f:\work\habit_project\habit\core\habitat_analysis'
source = os.path.join(base_dir, 'features')
dest = os.path.join(base_dir, 'feature_construction')

print(f"Source: {source}")
print(f"Destination: {dest}")
print(f"Source exists: {os.path.exists(source)}")
print(f"Dest exists: {os.path.exists(dest)}")

if os.path.exists(dest):
    print("Removing existing destination...")
    import shutil
    shutil.rmtree(dest)

if os.path.exists(source):
    print("Renaming directory...")
    os.rename(source, dest)
    print("Done!")
else:
    print("Source directory does not exist!")
