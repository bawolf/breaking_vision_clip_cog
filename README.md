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

## Cog

download the weights

```bash
gdown https://drive.google.com/uc?id=1Gn3UdoKffKJwz84GnGx-WMFTwZuvDsuf -O ./checkpoints/
```

build the image

```bash
cog build --separate-weights
```

push a new image

```bash
cog push
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

## Training Data

To run training on your own, you can find the training data [here](https://drive.google.com/drive/folders/11M6nSuSuvoU2wpcV_-6KFqCzEMGP75q6?usp=drive_link) and put it in the a directory at the root of the project called `./data`.

## Checkpoints

To run predictions with cog or locally on an existing checkpoint, you can find a checkpoint and configuration files [here](https://drive.google.com/drive/folders/1Gn3UdoKffKJwz84GnGx-WMFTwZuvDsuf?usp=sharing) and put them in the a directory at the root of the project called `./checkpoints`.

## Model Architecture

- Base: CLIP ViT-Large/14
- Custom temporal pooling layer
- Fine-tuned vision encoder (last 3 layers)
- Output: 4-class classifier

## License

[Your License Here]

## Citation

If you use this model in your research, please cite:

```bibtex
[Your Citation Here]
```