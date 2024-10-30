# CLIP-Based Break Dance Move Classifier

A deep learning model for classifying break dance moves using CLIP (Contrastive Language-Image Pre-Training) embeddings. The model is fine-tuned on break dance videos to classify different power moves including windmills, halos, swipes, and baby mills.

## Features

- Video-based classification using CLIP embeddings
- Multi-frame temporal analysis
- Configurable frame sampling and data augmentation
- Real-time inference using Cog
- Misclassification analysis tools
- Hyperparameter tuning support

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Cog (if not already installed)
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog
```

## Training

```bash
# Run training with default configuration
python scripts/train.py

# Run hyperparameter tuning
python scripts/hyperparameter_tuning.py
```

## Inference

```bash
# Using Cog for inference
cog predict -i video=@path/to/your/video.mp4

# Using standard Python script
python scripts/inference.py --video path/to/your/video.mp4
```

## Analysis

```bash
# Generate misclassification report
python scripts/visualization/miscalculations_report.py

# Visualize model performance
python scripts/visualization/visualize.py
```

## Project Structure

```
clip/
├── src/                    # Source code
│   ├── data/              # Dataset and data processing
│   ├── models/            # Model architecture
│   └── utils/             # Utility functions
├── scripts/               # Training and inference scripts
│   └── visualization/     # Visualization tools
├── config/                # Configuration files
├── runs/                  # Training runs and checkpoints
├── cog.yaml              # Cog configuration
└── requirements.txt      # Python dependencies
```

## Model Architecture

- Base: CLIP ViT-Large/14
- Custom temporal pooling layer
- Fine-tuned vision encoder (last 3 layers)
- Output: 4-class classifier

## Performance

- Training Accuracy: ~95%
- Validation Accuracy: ~92%
- Inference Time: ~100ms per video

## Configuration

Key hyperparameters can be modified in `config/default.yaml`:
- Frame sampling: 10 frames per video
- Image size: 224x224
- Learning rate: 2e-6
- Weight decay: 0.007
- Data augmentation parameters

## License

[Your License Here]

## Citation

If you use this model in your research, please cite:

```bibtex
[Your Citation Here]
```