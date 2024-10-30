import optuna
import os

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.train import train_and_evaluate
from src.utils.utils import create_run_directory

def objective(trial, hyperparam_run_dir):
    config = {
        "clip_model": trial.suggest_categorical("clip_model", ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-8, 1e-1),
        "unfreeze_layers": trial.suggest_int("unfreeze_layers", 1, 6),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gradient_clip_max_norm": trial.suggest_uniform("gradient_clip_max_norm", 0.1, 1.0),
        "augmentation_strength": trial.suggest_float("augmentation_strength", 0.0, 1.0),
        "crop_scale_min": trial.suggest_float("crop_scale_min", 0.6, 0.9),
        "max_frames": trial.suggest_int("max_frames", 5, 15),
        "sigma": trial.suggest_uniform("sigma", 0.1, 0.5),
    }

    class_labels = ["windmill", "halo", "swipe", "baby_mill"][:3]

    # Fixed configurations
    config.update({
        "class_labels": class_labels,
        "num_classes": len(class_labels),
        "data_path": '../finetune/3moves_test',
        "num_epochs": 50,  # Reduced for faster trials
        "patience": 10,    # Adjusted for faster trials
        "image_size": 224,
        "crop_scale_max": 1.0,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "overfitting_threshold": 10,
    })

    # Derive augmentation parameters from augmentation_strength
    config.update({
        "flip_probability": 0.5 * config["augmentation_strength"],
        "rotation_degrees": int(15 * config["augmentation_strength"]),
        "brightness_jitter": 0.2 * config["augmentation_strength"],
        "contrast_jitter": 0.2 * config["augmentation_strength"],
        "saturation_jitter": 0.2 * config["augmentation_strength"],
        "hue_jitter": 0.1 * config["augmentation_strength"],
    })

    # Create a unique run directory for this trial
    config["run_dir"] = create_run_directory(prefix=f"trial", parent_dir=hyperparam_run_dir)

    # Run training and evaluation
    val_accuracy = train_and_evaluate(config)
    return val_accuracy

def main():
    # Set up the study and optimize
    hyperparam_run_dir = create_run_directory(suffix='_hyperparam')
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, hyperparam_run_dir), n_trials=100)  # Adjust the number of trials as needed

    # Save the study results
    study.trials_dataframe().to_csv(os.path.join(hyperparam_run_dir, 'study_results.csv'))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best trial parameters
    with open(os.path.join(hyperparam_run_dir, 'best_params.txt'), 'w') as f:
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
