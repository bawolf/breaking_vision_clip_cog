---
language: en
tags:
  - clip
  - breakdance
  - video-classification
  - dance
license: MIT
datasets:
  - custom
---

# CLIP-Based Break Dance Move Classifier

This model is a fine-tuned version of CLIP (ViT-Large/14) specialized in classifying break dance power moves from video frames, including windmills, halos, and swipes.

## Model Description

- **Model Type:** Fine-tuned CLIP model
- **Base Model:** ViT-Large/14
- **Task:** Video Classification
- **Training Data:** Custom break dance video dataset
- **Output:** 3 classes of break dance moves

## Usage

```python
from transformers import CLIPProcessor, CLIPModel
import torch
import cv2
from PIL import Image

# Load model and processor
processor = CLIPProcessor.from_pretrained("[your-username]/clip-breakdance-classifier")
model = CLIPModel.from_pretrained("[your-username]/clip-breakdance-classifier")

# Load video and process frames
video = cv2.VideoCapture("breakdance_move.mp4")
predictions = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert BGR to RGB and to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Process frame
    inputs = processor(images=frame_pil, return_tensors="pt")
    outputs = model(**inputs)
    predictions.append(outputs)

video.release()
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
  howpublished = {\url{https://huggingface.co/[your-username]/clip-breakdance-classifier}}
}
```
