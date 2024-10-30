import os
import json
from datetime import datetime

def create_run_directory(base_dir='runs', prefix='run', suffix='', parent_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{prefix}_{timestamp}{suffix}"
    
    if parent_dir:
        run_dir = os.path.join(parent_dir, dir_name)
    else:
        run_dir = os.path.join(base_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Find the most recent run directory
def get_latest_run_dir(base_dir='runs', include_hyperparam=True):
    all_dirs = []
    
    for d in os.listdir(base_dir):
        if d.startswith('run_'):
            full_path = os.path.join(base_dir, d)
            all_dirs.append(full_path)
            
            if d.endswith('_hyperparam') and include_hyperparam:
                # If it's a hyperparam directory, add its trial subdirectories
                trial_dirs = [os.path.join(full_path, td) for td in os.listdir(full_path) if td.startswith('trial_')]
                all_dirs.extend(trial_dirs)
    
    if not all_dirs:
        raise ValueError(f"No run directories found in {base_dir}")
    
    # Sort directories by timestamp in the directory name
    return max(all_dirs, key=get_dir_timestamp)

def get_run_file(filename, run_dir=None, required=True):
    """Get a file from a run directory
    
    Args:
        filename: Name of file to get (e.g., 'best_model.pth', 'config.json')
        run_dir: Run directory path (uses latest if None)
        required: Whether to raise an error if file not found
    
    Returns:
        str: Path to the file
        dict: Loaded JSON data if file ends with .json
    """
    if run_dir is None:
        run_dir = get_latest_run_dir()
    
    file_path = os.path.join(run_dir, filename)
    
    if not os.path.exists(file_path):
        if required:
            raise FileNotFoundError(f"{filename} not found in {run_dir}")
        return None
    
    # Load JSON files automatically
    if filename.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    return file_path

def get_latest_model_path(run_dir=None):
    """Get path to best_model.pth"""
    return get_run_file('best_model.pth', run_dir)

def get_config(run_dir=None):
    """Get config from run directory"""
    return get_run_file('config.json', run_dir)

# Helper function to parse directory name and get timestamp
def get_dir_timestamp(dir_path):
    dir_name = os.path.basename(dir_path)
    try:
        # Extract timestamp from directory name
        timestamp_str = dir_name.split('_')[1]  # Assumes format is always prefix_timestamp or prefix_timestamp_suffix
        return datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
    except (IndexError, ValueError):
        # If parsing fails, return the earliest possible date
        return datetime.min
