import optuna
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
import math

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.train import train_and_evaluate
from src.utils.utils import create_run_directory

def create_hyperparam_directory():
    """Create a parent directory for all hyperparameter searches"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "runs_hyperparam"
    hyperparam_dir = os.path.join(base_dir, f"hyperparam_{timestamp}")
    os.makedirs(hyperparam_dir, exist_ok=True)
    return hyperparam_dir

def objective(trial, hyperparam_run_dir, data_path):
    """Objective function for a single dataset"""
    
    # Then suggest parameters using the model-specific ranges
    config = {
        "clip_model":  trial.suggest_categorical("clip_model", ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]),
        "batch_size": trial.suggest_categorical("batch_size", [8,16,32]),
        "unfreeze_layers": trial.suggest_int("unfreeze_layers", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True),
        "gradient_clip_max_norm": trial.suggest_float("gradient_clip_max_norm", 0.1, 1.0),
        "augmentation_strength": trial.suggest_float("augmentation_strength", 0.0, 1.0),
        "crop_scale_min": trial.suggest_float("crop_scale_min", 0.6, 0.9),
        "max_frames": trial.suggest_int("max_frames", 5, 15),
        "sigma": trial.suggest_float("sigma", 0.1, 0.5),
    }

    class_labels = ["windmill", "halo", "swipe", "baby_mill"][:3]

    # Fixed configurations
    config.update({
        "class_labels": class_labels,
        "num_classes": len(class_labels),
        "data_path": data_path,
        "num_epochs": 50,
        "patience": 10,
        "image_size": 224,
        "crop_scale_max": 1.0,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "overfitting_threshold": 10,
    })

    # Derive augmentation parameters
    config.update({
        "flip_probability": 0.5 * config["augmentation_strength"],
        "rotation_degrees": int(15 * config["augmentation_strength"]),
        "brightness_jitter": 0.2 * config["augmentation_strength"],
        "contrast_jitter": 0.2 * config["augmentation_strength"],
        "saturation_jitter": 0.2 * config["augmentation_strength"],
        "hue_jitter": 0.1 * config["augmentation_strength"],
    })

    # Create dataset-specific run directory
    dataset_label = '_'.join(Path(data_path).parts[-2:])  # Get last two parts of path
    trial_dir = create_run_directory(
        prefix=f"trial_{dataset_label}", 
        parent_dir=hyperparam_run_dir
    )
    config["run_dir"] = trial_dir


    # Run training and evaluation with device cleanup
    try:
        val_accuracy, vis_dir = train_and_evaluate(config)
        
        if val_accuracy is None or math.isnan(val_accuracy) or math.isinf(val_accuracy):
            raise ValueError(f"Invalid accuracy value: {val_accuracy}")
            
        # Save trial info
        trial_info = {
            'dataset': data_path,
            'dataset_label': dataset_label,
            'trial_number': trial.number,
            'parameters': trial.params,
            'accuracy': val_accuracy,
            'visualization_dir': vis_dir,
            'trial_dir': trial_dir
        }
        
        with open(os.path.join(trial_dir, 'trial_info.json'), 'w') as f:
            json.dump(trial_info, f, indent=4)
            
        return val_accuracy
    
    except Exception as e:
        print(f"Error in trial for {data_path}: {str(e)}")
        # Log detailed error information
        error_log_path = os.path.join(hyperparam_run_dir, 'error_log.txt')
        with open(error_log_path, 'a') as f:
            f.write(f"\nError in trial at {datetime.now()}:\n")
            f.write(f"Dataset: {data_path}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Trial params: {trial.params}\n")
            f.write("Stack trace:\n")
            import traceback
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n")
        
        return float('-inf')

def run_hyperparameter_search(data_paths, n_trials=100):
    """Run hyperparameter search for multiple datasets"""
    
    # Create parent directory for all searches
    parent_hyperparam_dir = create_hyperparam_directory()
    
    # Store results for all datasets
    all_results = {}
    
    for data_path in data_paths:
        print(f"\nStarting hyperparameter search for dataset: {data_path}")
        
        # Create dataset-specific directory
        dataset_label = '_'.join(Path(data_path).parts[-2:])
        dataset_dir = os.path.join(parent_hyperparam_dir, f"search_{dataset_label}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create and run study with explicit trial count tracking
        study = optuna.create_study(direction="maximize")
        completed_trials = 0
        failed_trials = []
        total_attempts = 0
        max_attempts =  n_trials * 2
        while completed_trials < n_trials and total_attempts < max_attempts:
            try:
                total_attempts += 1
                study.optimize(
                    lambda trial: objective(trial, dataset_dir, data_path),
                    n_trials=1
                )
                # Only increment if the trial actually succeeded
                if study.trials[-1].value != float('-inf'):
                    completed_trials += 1
                    print(f"Completed trial {completed_trials}/{n_trials} for {dataset_label}")
                else:
                    error_info = {
                        'trial_number': completed_trials + len(failed_trials) + 1,
                        'error': "Trial returned -inf",
                        'timestamp': datetime.now().isoformat()
                    }
                    failed_trials.append(error_info)
                    print(f"Failed trial for {dataset_label}: returned -inf")
                
            except Exception as e:
                error_info = {
                    'trial_number': completed_trials + len(failed_trials) + 1,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                failed_trials.append(error_info)
                print(f"Error in trial for {dataset_label}: {str(e)}")
                
                # Log the error
                with open(os.path.join(dataset_dir, 'failed_trials.json'), 'w') as f:
                    json.dump(failed_trials, f, indent=4)
            if total_attempts >= max_attempts:
                print(f"Warning: Reached maximum attempts ({max_attempts}) for {dataset_label}")
                
        # Save study results
        results_df = study.trials_dataframe()
        results_df.to_csv(os.path.join(dataset_dir, 'study_results.csv'))
        
        # Save trial statistics
        trial_stats = {
            'completed_trials': completed_trials,
            'failed_trials': len(failed_trials),
            'total_attempts': completed_trials + len(failed_trials)
        }
        with open(os.path.join(dataset_dir, 'trial_statistics.json'), 'w') as f:
            json.dump(trial_stats, f, indent=4)
        
        # Save best trial info
        best_trial = study.best_trial
        best_params_path = os.path.join(dataset_dir, 'best_params.txt')
        with open(best_params_path, 'w') as f:
            f.write(f"Best trial value: {best_trial.value}\n\n")
            f.write("Best parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # Store results
        all_results[data_path] = {
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'study': study,
            'results_df': results_df,
            'failed_trials': failed_trials,
            'trial_stats': trial_stats
        }
        
        print(f"\nResults for {data_path}:")
        print(f"Completed trials: {completed_trials}")
        print(f"Failed trials: {len(failed_trials)}")
        print(f"Best trial value: {best_trial.value}")
        print("Best parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    
    # Create overall summary with additional statistics
    summary_data = []
    for data_path, result in all_results.items():
        summary_data.append({
            'dataset': data_path,
            'best_accuracy': result['best_value'],
            'completed_trials': result['trial_stats']['completed_trials'],
            'failed_trials': result['trial_stats']['failed_trials'],
            **result['best_params']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(parent_hyperparam_dir, 'overall_summary.csv'), index=False)
    
    return parent_hyperparam_dir, all_results

if __name__ == "__main__":
    # List of dataset paths to optimize
    data_paths = [
        './data/blog/datasets/bryant/random',
        './data/blog/datasets/bryant/adjusted',
        './data/blog/datasets/youtube/random',
        './data/blog/datasets/youtube/adjusted',
        './data/blog/datasets/combined/random',
        './data/blog/datasets/combined/adjusted',
        './data/blog/datasets/bryant_train_youtube_val/default'
    ]
    
    # Run hyperparameter search
    hyperparam_dir, results = run_hyperparameter_search(
        data_paths,
        n_trials=8  # Adjust as needed
    )
    
    print(f"\nHyperparameter search complete!")
    print(f"Results are saved in: {hyperparam_dir}")
