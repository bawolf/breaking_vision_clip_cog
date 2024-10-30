import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.utils import get_latest_run_dir

def analyze_misclassifications(run_dir=None):
    if run_dir is None:
        # run_dir = "/home/bawolf/workspace/break/clip/runs/run_20241022-122939_3moves_balanced"
        run_dir =  get_latest_run_dir()
    
    misclassifications_dir = os.path.join(run_dir, 'misclassifications')
    all_misclassifications = {}
    
    # Collect all misclassifications across epochs
    for file in os.listdir(misclassifications_dir):
        if file.endswith('.json'):
            with open(os.path.join(misclassifications_dir, file), 'r') as f:
                epoch_misclassifications = json.load(f)
                for item in epoch_misclassifications:
                    video_path = item['video_path']
                    if video_path not in all_misclassifications:
                        all_misclassifications[video_path] = []
                    all_misclassifications[video_path].append(item)

    # Determine the total number of epochs from the files
    epoch_files = [f for f in os.listdir(misclassifications_dir) if f.startswith('epoch_') and f.endswith('.json')]
    total_epochs = len(epoch_files)

    # Count misclassifications per video
    misclassification_counts = {video: len(misclassifications) 
                                for video, misclassifications in all_misclassifications.items()}

    # Calculate percentage of epochs each video was misclassified
    misclassification_percentages = {video: (count / total_epochs) * 100 
                                     for video, count in misclassification_counts.items()}

    # Sort videos by misclassification percentage
    sorted_videos = sorted(misclassification_percentages.items(), key=lambda x: x[1], reverse=True)

    # Prepare report
    report = "Misclassification Analysis Report\n"
    report += "=================================\n\n"

    # Top N most misclassified videos
    N = 20
    report += f"Top {N} Most Misclassified Videos:\n"
    for video, percentage in sorted_videos[:N]:
        report += f"{Path(video).name}: Misclassified in {percentage:.2f}% of epochs ({misclassification_counts[video]} out of {total_epochs})\n"
        misclassifications = all_misclassifications[video]
        true_label = misclassifications[0]['true_label']
        predicted_labels = Counter(m['predicted_label'] for m in misclassifications)
        report += f"  True Label: {true_label}\n"
        report += f"  Predicted Labels: {dict(predicted_labels)}\n\n"

    # Overall statistics
    total_misclassifications = sum(misclassification_counts.values())
    total_videos = len(misclassification_counts)
    report += "Overall Statistics:\n"
    report += f"Total misclassified videos: {total_videos}\n"
    report += f"Total misclassifications: {total_misclassifications}\n"
    report += f"Average misclassification percentage per video: {sum(misclassification_percentages.values()) / total_videos:.2f}%\n"
    report += f"Total epochs: {total_epochs}\n"

    # Save report
    report_path = os.path.join(run_dir, 'misclassification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_videos)), [percentage for _, percentage in sorted_videos])
    plt.title(f'Videos Ranked by Misclassification Percentage (Total Epochs: {total_epochs})')
    plt.xlabel('Video Rank')
    plt.ylabel('Misclassification Percentage')
    plt.ylim(0, 100)  # Set y-axis limit to 0-100%
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'misclassification_distribution.png'))

    print(f"Analysis complete. Report saved to {report_path}")
    print(f"Visualization saved to {os.path.join(run_dir, 'misclassification_distribution.png')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        print("Usage: python analyze_misclassifications.py [path_to_run_directory]")
        sys.exit(1)
    
    run_dir = sys.argv[1] if len(sys.argv) == 2 else None
    analyze_misclassifications(run_dir)
