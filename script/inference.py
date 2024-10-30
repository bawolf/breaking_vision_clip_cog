import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.utils import get_latest_run_dir, get_latest_model_path, get_config
from src.models.model import load_model
from src.data.video_utils import create_transform, extract_frames

def setup_model(run_dir=None):
    """Setup model and configuration"""
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get run directory
    if run_dir is None:
        run_dir = get_latest_run_dir()
    print(f"Using run directory: {run_dir}")
    
    try:
        # Load configuration
        config = get_config(run_dir)
        print(f"Loaded configuration from: {run_dir}")

        # Load the model
        model_path = get_latest_model_path(run_dir)
        print(f"Loading model from: {model_path}")
        
        model = load_model(
            config['num_classes'],
            model_path,
            device,
            config['clip_model']
        )
        model.eval()
        
        return model, config, device
        
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading model: {str(e)}")
        exit(1)

def predict(video_path, model, config, device):
    """Predict class for a video using the model"""
    transform = create_transform(config, training=False)
    
    try:
        frames, success = extract_frames(video_path, 
                                      config,
                                      transform)
        if not success:
            raise ValueError(f"Failed to process video: {video_path}")
            
        frames = frames.to(device)
        
        # Add batch dimension correctly
        frames = frames.unsqueeze(0)  # Add batch dimension at the start
        
        with torch.no_grad():
            try:
                outputs = model(frames)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            except Exception as e:
                print(f"Error during model forward pass: {str(e)}")
                print(f"Model input shape: {frames.shape}")
                raise
        
        
        # Get predictions
        avg_probabilities = probabilities[0].cpu().numpy()
        predicted_class = np.argmax(avg_probabilities)
        
        # Create a dictionary of class labels and their probabilities
        class_probabilities = {
            label: float(prob)
            for label, prob in zip(config['class_labels'], avg_probabilities)
        }
        
        return config['class_labels'][predicted_class], class_probabilities
        
    except Exception as e:
        raise ValueError(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on a video file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the video file')
    parser.add_argument('--run-dir', type=str,
                        help='Path to specific run directory (optional)')
    
    args = parser.parse_args()
    
    # Setup model and config
    model, config, device = setup_model(args.run_dir)
    
    try:
        predicted_label, class_probabilities = predict(args.video, model, config, device)
        print(f"\nPredicted label: {predicted_label}")
        print("\nClass probabilities:")
        for label, prob in class_probabilities.items():
            print(f"  {label}: {prob:.4f}")
    except ValueError as e:
        print(f"Error: {str(e)}")
