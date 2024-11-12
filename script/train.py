import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import logging
import csv
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.utils import create_run_directory
from src.dataset.dataset import VideoDataset
from src.models.model import create_model
from src.dataset.video_utils import create_transform
from visualization.visualize import run_visualization
from visualization.miscalculations_report import analyze_misclassifications

def train_and_evaluate(config):
    try:
        # Create a run directory if it doesn't exist
        if "run_dir" not in config:
            config["run_dir"] = create_run_directory()
        
        # Update paths based on run_dir
        config.update({
            "best_model_path": os.path.join(config["run_dir"], 'best_model.pth'),
            "final_model_path": os.path.join(config["run_dir"], 'final_model.pth'),
            "csv_path": os.path.join(config["run_dir"], 'training_log.csv'),
            "misclassifications_dir": os.path.join(config["run_dir"], 'misclassifications'),
        })

        config_path = os.path.join(config["run_dir"], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(config["run_dir"], 'training.log')),
                                    logging.StreamHandler()])
        logger = logging.getLogger(__name__)

        # Use device from config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
            print(f"Currently allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        model = create_model(config["num_classes"], config["clip_model"])
        # Unfreeze the last 2 layers of the vision encoder
        model.unfreeze_vision_encoder(num_layers=config["unfreeze_layers"])
        model = model.to(device)

        # Ensure criterion is on the same device
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # logger.info(f"Model architecture:\n{model}")

        # Load datasets
        train_dataset = VideoDataset(
            os.path.join(config['data_path'], 'train.csv'), 
            config=config
        )
        
        # For validation, create a new config with training=False for transforms
        val_config = config.copy()
        val_dataset = VideoDataset(
            os.path.join(config['data_path'], 'val.csv'), 
            config=val_config,
            transform=create_transform(config, training=False)
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

        # Open a CSV file to log training progress
        with open(config["csv_path"], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

        # Function to calculate accuracy
        def calculate_accuracy(outputs, labels):
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            return correct / total

        def log_misclassifications(outputs, labels, video_paths, dataset, misclassified_videos):
            _, predicted = torch.max(outputs, 1)
            for pred, label, video_path in zip(predicted, labels, video_paths):
                if pred != label:
                    true_label = dataset.label_map[label.item()]
                    predicted_label = dataset.label_map[pred.item()]
                    misclassified_videos.append({
                        'video_path': video_path,
                        'true_label': true_label,
                        'predicted_label': predicted_label
                    })

        # Create a subfolder for misclassification logs
        os.makedirs(config["misclassifications_dir"], exist_ok=True)

        # Training loop
        for epoch in range(config["num_epochs"]):
            model.train()
            total_loss = 0
            total_accuracy = 0
            for frames, labels, video_paths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
                frames = frames.to(device)
                labels = labels.to(device)
                
                logits = model(frames)
                
                loss = criterion(logits, labels)
                accuracy = calculate_accuracy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip_max_norm"])
                optimizer.step()
                
                total_loss += loss.item()
                total_accuracy += accuracy
            
            avg_train_loss = total_loss / len(train_loader)
            avg_train_accuracy = total_accuracy / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            misclassified_videos = []
            with torch.no_grad():
                for frames, labels, video_paths in val_loader:
                    frames = frames.to(device)
                    labels = labels.to(device)
                    
                    logits = model(frames)
                    
                    loss = criterion(logits, labels)
                    accuracy = calculate_accuracy(logits, labels)
                    
                    val_loss += loss.item()
                    val_accuracy += accuracy
                    
                    # Log misclassifications
                    log_misclassifications(logits, labels, video_paths, val_dataset, misclassified_videos)
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            
            # Log misclassified videos
            if misclassified_videos:
                misclassified_log_path = os.path.join(config["misclassifications_dir"], f'epoch_{epoch+1}.json')
                with open(misclassified_log_path, 'w') as f:
                    json.dump(misclassified_videos, f, indent=2)
                logger.info(f"Logged {len(misclassified_videos)} misclassified videos to {misclassified_log_path}")

            # Log the metrics
            logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], "
                        f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy*100:.2f}%, "
                        f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy*100:.2f}%")

            # Write to CSV
            with open(config["csv_path"], 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, avg_train_loss, avg_train_accuracy*100, avg_val_loss, avg_val_accuracy*100])

            # Learning rate scheduling
            scheduler.step()

            # Save the best model and check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config["best_model_path"])
                logger.info(f"Saved best model to {config['best_model_path']}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if epochs_without_improvement >= config["patience"]:
                logger.info(f"Early stopping triggered after {config['patience']} epochs without improvement")
                break

            # Overfitting detection
            if avg_train_accuracy - avg_val_accuracy > config["overfitting_threshold"]:
                logger.warning("Possible overfitting detected")

        logger.info("Training finished!")

        # Save the final model
        torch.save(model.state_dict(), config["final_model_path"])
        logger.info(f"Saved final model to {config['final_model_path']}")

        # Save run information
        with open(os.path.join(config["run_dir"], 'run_info.txt'), 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
            f.write(f"Scheduler: {scheduler.__class__.__name__}\n")
            f.write(f"Loss function: CrossEntropyLoss\n")
            f.write(f"Data augmentation: RandomHorizontalFlip, RandomRotation(5), ColorJitter\n")
            f.write(f"Mixed precision training: {'Enabled' if 'scaler' in locals() else 'Disabled'}\n")
            f.write(f"Train dataset size: {len(train_dataset)}\n")
            f.write(f"Validation dataset size: {len(val_dataset)}\n")
            f.write(f"Vision encoder frozen: {'Partially' if hasattr(model, 'unfreeze_vision_encoder') else 'Unknown'}\n")

        # Run visualization
        try:
            logger.info("Running visualization...")
            vis_dir, confusion_matrix = run_visualization(config["run_dir"])
            logger.info(f"Visualization complete! Check the output directory: {vis_dir}")
            
            # Log confusion matrix results
            class_accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
            overall_accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()
            
            logger.info("\nConfusion Matrix Results:")
            for i, (label, accuracy) in enumerate(zip(config['class_labels'], class_accuracies)):
                logger.info(f"{label}: {accuracy:.2%}")
            logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error running visualization: {str(e)}")

        # Run misclassification analysis
        try:
            analyze_misclassifications(config["run_dir"])
            logger.info(f"Misclassification analysis complete! Check the output directory: {config['run_dir']}")
        except Exception as e:
            logger.error(f"Error running misclassification analysis: {str(e)}")

            
        if math.isnan(avg_val_accuracy) or math.isinf(avg_val_accuracy):
                raise ValueError(f"Invalid validation accuracy: {avg_val_accuracy}")
        
        print("Script finished.")

        return avg_val_accuracy, vis_dir
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise  # Re-raise the exception to be caught by the hyperparameter tuning

def main():
    # Create run directory
    run_dir = create_run_directory()
    class_labels = ["windmill", "halo", "swipe", "baby_mill"][:3]

    # Write configuration
    config = {
        "class_labels": class_labels,
        "num_classes": len(class_labels),
        "data_path": './data/blog/datasets/bryant/random',
        "batch_size": 8,
        "learning_rate": 2e-6,
        "weight_decay": 0.007,
        "num_epochs": 2,
        "patience": 10,  # for early stopping
        "max_frames": 10,
        "sigma": 0.3,
        "image_size": 224,
        "flip_probability": 0.5,
        "rotation_degrees": 15,
        "brightness_jitter": 0.2,
        "contrast_jitter": 0.2,
        "saturation_jitter": 0.2,
        "hue_jitter": 0.1,
        "crop_scale_min": 0.8,
        "crop_scale_max": 1.0,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "unfreeze_layers": 3,
        # "clip_model": "openai/clip-vit-large-patch14",
        "clip_model": "openai/clip-vit-base-patch32",
        "gradient_clip_max_norm": 1.0,
        "overfitting_threshold": 10,
        "run_dir": run_dir,
    }
    train_and_evaluate(config)

if __name__ == "__main__":
    main()
