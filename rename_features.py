import os
import shutil

source = r'f:\work\habit_project\habit\core\habitat_analysis\features'
dest = r'f:\work\habit_project\habit\core\habitat_analysis\feature_construction'

if os.path.exists(dest):
    print(f"Destination {dest} already exists, removing...")
    shutil.rmtree(dest)

print(f"Moving {source} to {dest}...")
shutil.move(source, dest)
print("Done!")
