from transformers import CLIPProcessor
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
import torch
from src.models.model import create_model

def upload_model_to_hub(hf_username):
    # Initialize huggingface api
    api = HfApi()
    
    # Load your custom model
    num_classes = 3  # windmills, halos, and swipes
    model = create_model(num_classes, "openai/clip-vit-large-patch14")
    
    # Load your trained weights
    state_dict = torch.load("./checkpoints/model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    # Get the processor from the base CLIP model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    repo_id = f"{hf_username}/breaking-vision-clip-classifier"
    
    # Save model configuration and architecture
    config = {
        "num_classes": num_classes,
        "base_model": "openai/clip-vit-large-patch14",
        "class_labels": ["windmill", "halo", "swipe"],
        "model_type": "VariableLengthCLIP"
    }
    
    # Push to hub with config
    model.push_to_hub(
        repo_id,
        config_dict=config,
        commit_message="Upload custom CLIP-based dance classifier"
    )
    processor.push_to_hub(repo_id)

if __name__ == "__main__":
    load_dotenv()
    hf_username = os.getenv("HF_USERNAME")
    upload_model_to_hub(hf_username) 
