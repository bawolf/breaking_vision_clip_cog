import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import VideoDataset
from src.utils.utils import get_latest_model_path, get_latest_run_dir, get_config
from src.models.model import load_model
import json

def plot_training_curves(log_file, output_dir):
    data = pd.read_csv(log_file)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['train_accuracy'], label='Train Accuracy')
    plt.plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def generate_evaluation_metrics(model, data_loader, device, output_dir, class_labels, data_info):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels, _ in data_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.title(f'Confusion Matrix\n{data_info}')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']

    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        average_precision = average_precision_score(all_labels == i, all_probs[:, i])
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f'{class_label} (AP = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{data_info}')
    plt.legend(loc="lower left")
    plt.savefig(f'{output_dir}/precision_recall_curve.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\n{data_info}')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()

    return cm

if __name__ == "__main__":
    # Find the most recent run directory
    #
    run_dir = get_latest_run_dir()
    # run_dir= "/home/bawolf/workspace/break/clip/runs/run_20241024-150232_otherpeopleval_large_model"
    # run_dir = "/home/bawolf/workspace/break/clip/runs/run_20241022-122939_3moves_balanced"

    # Load configuration
    config = get_config(run_dir)
    
    class_labels = config['class_labels']
    num_classes = config['num_classes']
    data_path = config['data_path']
    # data_path= '../finetune/3moves_otherpeopleval'
    # data_path = '../finetune/otherpeople3moves'

    # Paths
    log_file = os.path.join(run_dir, 'training_log.csv')
    model_path = get_latest_model_path(run_dir)
    test_csv = os.path.join(data_path, 'test.csv')
    # test_csv = os.path.join(data_path, 'val.csv')
    # test_csv = os.path.join(data_path, 'train.csv')
    
    # Get the last directory of data_path and the file name
    last_dir = os.path.basename(os.path.normpath(data_path))
    file_name = os.path.basename(test_csv)
    
    # Create a directory for visualization outputs
    vis_dir = os.path.join(run_dir, f'visualization_{last_dir}_{file_name.split(".")[0]}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create data_info string for chart headers
    data_info = f'Data: {last_dir}, File: {file_name}'
    
    # Plot training curves
    plot_training_curves(log_file, vis_dir)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(num_classes, model_path, device, config['clip_model'])
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = VideoDataset(test_csv, config)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Generate evaluation metrics
    cm = generate_evaluation_metrics(model, test_loader, device, vis_dir, class_labels, data_info)
    
    print(f"Visualization complete! Check the output directory: {vis_dir}")
