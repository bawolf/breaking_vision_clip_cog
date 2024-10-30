import torch
import torch.nn as nn
from transformers import CLIPModel

class VariableLengthCLIP(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.visual_projection = nn.Linear(clip_model.visual_projection.in_features, num_classes)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.clip_model.vision_model(x).pooler_output
        features = features.view(batch_size, num_frames, -1)
        features = torch.mean(features, dim=1)  # Average over frames
        return self.visual_projection(features)

    def unfreeze_vision_encoder(self, num_layers=2):
        # Freeze the entire vision encoder
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        # Unfreeze the last few layers of the vision encoder
        for param in self.clip_model.vision_model.encoder.layers[-num_layers:].parameters():
            param.requires_grad = True

def create_model(num_classes, pretrained_model_name="openai/clip-vit-base-patch32"):
    clip_model = CLIPModel.from_pretrained(pretrained_model_name)
    return VariableLengthCLIP(clip_model, num_classes)

def load_model(num_classes, model_path, device, pretrained_model_name="openai/clip-vit-base-patch32"):
    # Create the model
    model = create_model(num_classes, pretrained_model_name)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Load the state dict, ignoring mismatched keys
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)  # Move the model to the appropriate device
    return model
