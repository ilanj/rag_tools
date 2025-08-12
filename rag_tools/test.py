from pathlib import Path

# Get current working directory
current_dir = Path.cwd()
current_dir = current_dir / "data"
print("Current Directory:", current_dir)

# List files
files = [f.name for f in current_dir.iterdir()]
print("Files:", files)
