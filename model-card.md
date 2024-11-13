---
language: en
tags:
  - clip
  - breakdance
  - video-classification
  - dance
  - pytorch
  - vision-encoder
license: MIT
datasets:
  - custom
library_name: transformers
base_model: openai/clip-vit-large-patch14
pipeline_tag: video-classification
model-index:
  - name: CLIP-Based Break Dance Move Classifier
    results:
      - task:
          type: video-classification
        dataset:
          name: custom_breakdance
          type: custom
        metrics:
          - name: Overall Accuracy
            type: accuracy
            value: [specify %]
          - name: Windmill Precision
            type: precision
            value: [specify %]
          - name: Halo Precision
            type: precision
            value: [specify %]
          - name: Swipe Precision
            type: precision
            value: [specify %]
---

# CLIP-Based Break Dance Move Classifier

This model is a fine-tuned version of CLIP (ViT-Large/14) specialized in classifying break dance power moves from video frames, including windmills, halos, and swipes.

## Model Description

- **Model Type:** Custom CLIP-based architecture (VariableLengthCLIP)
- **Base Model:** CLIP ViT-Large/14 (for feature extraction)
- **Architecture:**
  - Uses CLIP's vision encoder for frame-level feature extraction
  - Processes multiple frames from a video
  - Averages frame features
  - Projects to 3 classes via a learned linear layer
- **Task:** Video Classification
- **Training Data:** Custom break dance video dataset
- **Output:** 3 classes of break dance moves (windmill, halo, swipe)

## Usage

```python
import torch
from transformers import CLIPProcessor
from PIL import Image
import cv2
import numpy as np
from src.models.model import create_model

# Load model and processor
model = create_model(num_classes=3, pretrained_model_name="openai/clip-vit-large-patch14")
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Process video
def process_video(video_path, model, processor):
    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        processed = processor(images=frame_pil, return_tensors="pt")
        frames.append(processed.pixel_values)

    video.release()

    # Stack frames and process
    frames_tensor = torch.cat(frames, dim=0)
    with torch.no_grad():
        predictions = model(frames_tensor.unsqueeze(0))

    return predictions
```

## Limitations

- Model performance may vary with video quality and lighting conditions
- Best results are achieved with clear, centered shots of the dance moves
- May have difficulty distinguishing between similar power moves
- Performance may be affected by unusual camera angles or partial views
- Currently only supports three specific power moves (windmills, halos, and swipes)

## Training Procedure

- Fine-tuned on CLIP ViT-Large/14 architecture
- Training dataset: Custom dataset of break dance videos
- Dataset size: [specify number] frames from [specify number] different videos
- Training epochs: [specify number]
- Learning rate: [specify rate]
- Batch size: [specify size]
- Hardware used: [specify GPU/CPU details]

## Evaluation Results

- Overall accuracy: [specify %]
  Per-class performance:
- Windmills: [specify precision/recall]
- Halos: [specify precision/recall]
- Swipes: [specify precision/recall]

## Citation

If you use this model in your research or project, please cite:

```bibtex
@misc{clip-breakdance-classifier,
  author = {Bryant Wolf},
  title = {CLIP-Based Break Dance Move Classifier},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/bawolf/clip-breakdance-classifier}}
}
```
