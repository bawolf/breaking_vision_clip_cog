import torch
from torch.utils.data import Dataset
import csv
from .video_utils import create_transform, extract_frames
import os

class VideoDataset(Dataset):
    def __init__(self, file_path, config, transform=None):
        self.data = []
        self.label_map = {}
        # Use create_transform if no custom transform is provided
        self.transform = transform or create_transform(config)
        
        # Validate required config keys
        required_keys = {"max_frames", "sigma", "class_labels"}
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        self.max_frames = config['max_frames']
        self.sigma = config['sigma']
        
        # Create label map from class_labels list
        self.label_map = {i: label for i, label in enumerate(config['class_labels'])}
        
        # Read the CSV file and parse the data
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) != 2:
                    print(f"Skipping invalid row: {row}")
                    continue
                relative_video_path, label = row
                video_path = os.path.join(config['data_path'], relative_video_path)
                try:
                    label = int(label)
                except ValueError:
                    print(f"Skipping row with invalid label: {row}")
                    continue
                self.data.append((video_path, label))

        if not self.data:
            raise ValueError(f"No valid data found in the CSV file: {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}")
            print(f"Absolute path attempt: {os.path.abspath(video_path)}")
            raise FileNotFoundError(f"File not found: {video_path}")
        
        frames, success = extract_frames(video_path, 
                                      {"max_frames": self.max_frames, "sigma": self.sigma}, 
                                      self.transform)
        
        if not success:
            frames = self._get_error_tensor()
            
        return frames, label, video_path

    def _get_error_tensor(self):
        return torch.zeros((self.max_frames, 3, 224, 224))
