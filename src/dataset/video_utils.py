import cv2
import numpy as np
import torch
from torchvision import transforms
from scipy.stats import norm
import os

def create_transform(config, training=False):
    """Create transform pipeline based on config"""
    # Validate base required keys
    required_keys = {
        "image_size",
        "normalization_mean",
        "normalization_std"
    }
    
    # Add training-specific required keys
    if training:
        required_keys.update({
            "flip_probability",
            "rotation_degrees",
            "brightness_jitter",
            "contrast_jitter",
            "saturation_jitter",
            "hue_jitter",
            "crop_scale_min",
            "crop_scale_max"
        })
    
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    # Build transform list
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((config["image_size"], config["image_size"]))
    ]
    
    # Add training augmentations if needed
    if training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=config["flip_probability"]),
            transforms.RandomRotation(config["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=config["brightness_jitter"],
                contrast=config["contrast_jitter"],
                saturation=config["saturation_jitter"],
                hue=config["hue_jitter"]
            ),
            transforms.RandomResizedCrop(
                config["image_size"],
                scale=(config["crop_scale_min"], config["crop_scale_max"])
            )
        ])
    
    # Add final transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["normalization_mean"],
            std=config["normalization_std"]
        )
    ])
    
    return transforms.Compose(transform_list)

def extract_frames(video_path: str, config: dict, transform) -> tuple[torch.Tensor, bool]:
    """Extract and process frames from video using Gaussian sampling
    Returns:
        tuple: (frames tensor, success boolean)
    """
    # Validate required config keys
    required_keys = {"max_frames", "sigma"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required config keys for frame extraction: {missing_keys}")
        
    frames = []
    success = True
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return None, False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video has no frames: {video_path}")
        cap.release()
        return None, False
    
    # Create a normal distribution centered at the middle of the video
    x = np.linspace(0, 1, total_frames)
    probabilities = norm.pdf(x, loc=0.5, scale=config["sigma"])
    probabilities /= probabilities.sum()

    # Sample frame indices based on this distribution
    frame_indices = np.sort(np.random.choice(
        total_frames, 
        size=min(config["max_frames"], total_frames), 
        replace=False, 
        p=probabilities
    ))

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx} from video: {video_path}")
            success = False
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"No frames extracted from video: {video_path}")
        return None, False

    # Pad with zeros if we don't have enough frames
    while len(frames) < config["max_frames"]:
        frames.append(torch.zeros_like(frames[0]))

    return torch.stack(frames), success