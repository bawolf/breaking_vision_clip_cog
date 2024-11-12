import os
from pathlib import Path
from visualize import run_visualization

def get_opposite_dataset_path(run_folder):
    # Map run folders to their corresponding opposite dataset training files
    dataset_mapping = {
        'search_bryant_adjusted': './data/blog/datasets/youtube/adjusted',
        'search_bryant_random': './data/blog/datasets/youtube/random',
        'search_youtube_adjusted': './data/blog/datasets/bryant/adjusted',
        'search_youtube_random': './data/blog/datasets/bryant/random'
    }
    
    for folder_prefix, dataset_path in dataset_mapping.items():
        if run_folder.startswith(folder_prefix):
            return dataset_path
    return None

def process_runs(base_dir):
    # Get the full path to the runs directory
    runs_dir = Path(base_dir)
    
    # Process each search directory
    for search_dir in runs_dir.iterdir():
        if not search_dir.is_dir() or search_dir.name == 'visualization':
            continue
            
        # Get the opposite dataset path for this search directory
        opposite_dataset = get_opposite_dataset_path(search_dir.name)
        
        if opposite_dataset is not None:
            print(f"Skipping {search_dir.name} - no matching dataset mapping")
            continue
        
        # Process each trial directory within the search directory
        for trial_dir in search_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
                continue
                
            print(f"Processing {trial_dir} with {opposite_dataset}")
            try:
                vis_dir, cm = run_visualization(
                    run_dir=str(trial_dir),
                    data_path=opposite_dataset,
                    test_csv=os.path.join(opposite_dataset, "train.csv")
                )
                print(f"Visualization complete: {vis_dir}")
            except Exception as e:
                print(f"Error processing {trial_dir}: {e}")

if __name__ == "__main__":
    # Example usage
    runs_path = "runs_hyperparam/hyperparam_20241106_124214"
    process_runs(runs_path)