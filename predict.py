import os
from cog import BasePredictor, Input, Path
import torch
import json
from src.models.model import load_model
from src.data.video_utils import create_transform, extract_frames

CHECKPOINT_DIR = "runs/run_20241024-150232_otherpeopleval_large_model/"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration from JSON
        with open(
            os.path.join(CHECKPOINT_DIR, "config.json"), 'r') as f:
            self.config = json.load(f)
        
        # Create transform
        self.transform = create_transform(self.config, training=False)
        
        # Load model
        self.model = load_model(
            self.config['num_classes'],
            os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            self.device,
            self.config['clip_model']
        )
        self.model.eval()

    def predict(self, video: Path = Input(description="Input video file")) -> dict:
        """Run a single prediction on the model"""
        try:
            # Extract frames using shared function with config
            frames, success = extract_frames(
                str(video), 
                self.config, 
                self.transform
            )
            
            if not success or frames is None:
                raise ValueError(f"Failed to process video: {video}")
            
            # Now frames is a tensor, not a tuple
            frames = frames.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(frames)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Get all class confidences
                all_confidences = {
                    label: probabilities[0][i].item()
                    for i, label in enumerate(self.config['class_labels'])
                }
            
            return {
                "class": self.config['class_labels'][predicted_class],
                "confidence": confidence,
                "all_confidences": all_confidences
            }
            
        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")