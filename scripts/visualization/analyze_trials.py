import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def parse_error_analysis(vis_dir):
    """Parse the error_analysis.txt file to get accuracy and misclassification details"""
    metrics = {}
    class_accuracies = {}
    misclassified_files = []
    
    with open(os.path.join(vis_dir, 'error_analysis.txt'), 'r') as f:
        lines = f.readlines()
        parsing_errors = False
        header_found = False
        
        for line in lines:
            # Get overall accuracy
            if line.startswith("Overall Accuracy:"):
                metrics['overall_accuracy'] = float(line.split(":")[1].strip().rstrip('%')) / 100
            
            # Parse per-class accuracy
            if "samples)" in line and ":" in line:
                class_name = line.split(":")[0].strip()
                accuracy = float(line.split(":")[1].split("%")[0].strip()) / 100
                samples = int(line.split("(")[1].split(" ")[0])
                class_accuracies[class_name] = {
                    'accuracy': accuracy,
                    'samples': samples
                }
            
            # Parse misclassified files
            if "Misclassified Videos:" in line:
                parsing_errors = True
                continue
            if "Filename" in line and "True Class" in line:
                header_found = True
                continue
            if parsing_errors and header_found and line.strip() and not line.startswith("-"):
                try:
                    # Split the line while preserving filename with spaces
                    parts = line.strip().split()
                    # Find the confidence value (last element with %)
                    confidence_idx = next(i for i, part in enumerate(parts) if part.endswith('%'))
                    # Everything before the last three elements is the filename
                    filename = ' '.join(parts[:confidence_idx-2])
                    true_class = parts[confidence_idx-2]
                    pred_class = parts[confidence_idx-1]
                    confidence = float(parts[confidence_idx].rstrip('%')) / 100
                    
                    misclassified_files.append({
                        'filename': filename,
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue
    
    metrics['class_accuracies'] = class_accuracies
    metrics['misclassified_files'] = misclassified_files
    return metrics

def analyze_trial(trial_dir):
    """Analyze all visualization directories in a trial and aggregate results"""
    trial_metrics = {
        'overall_accuracy': 0,
        'total_samples': 0,
        'class_accuracies': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'misclassified_files': []
    }
    
    # Find all visualization directories
    vis_dirs = [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith('visualization_')]
    if not vis_dirs:
        return None
        
    for vis_dir in vis_dirs:
        try:
            metrics = parse_error_analysis(vis_dir)
            
            # Add to total samples and weighted accuracy
            samples = sum(m['samples'] for m in metrics['class_accuracies'].values())
            trial_metrics['total_samples'] += samples
            trial_metrics['overall_accuracy'] += metrics['overall_accuracy'] * samples
            
            # Aggregate per-class metrics
            for class_name, class_metrics in metrics['class_accuracies'].items():
                trial_metrics['class_accuracies'][class_name]['correct'] += (
                    class_metrics['accuracy'] * class_metrics['samples']
                )
                trial_metrics['class_accuracies'][class_name]['total'] += class_metrics['samples']
            
            # Collect misclassified files with visualization directory info
            for error in metrics['misclassified_files']:
                error['vis_dir'] = vis_dir.name
                trial_metrics['misclassified_files'].append(error)
                
        except Exception as e:
            print(f"Error processing visualization directory {vis_dir}: {e}")
    
    # Calculate final metrics
    if trial_metrics['total_samples'] > 0:
        trial_metrics['overall_accuracy'] /= trial_metrics['total_samples']
        
        for class_metrics in trial_metrics['class_accuracies'].values():
            if class_metrics['total'] > 0:
                class_metrics['accuracy'] = class_metrics['correct'] / class_metrics['total']
    
    return trial_metrics

def analyze_trials(hyperparam_dir):
    results = {
        'search_dirs': defaultdict(lambda: {
            'best_overall': {'accuracy': 0, 'trial': None},
            'best_per_class': defaultdict(lambda: {'accuracy': 0, 'trial': None}),
            'misclassified_files': []
        })
    }
    
    # Process each search directory
    for search_dir in Path(hyperparam_dir).iterdir():
        if not search_dir.is_dir() or not search_dir.name.startswith('search_'):
            continue
            
        # Process each trial directory
        for trial_dir in search_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
                continue
            
            trial_metrics = analyze_trial(trial_dir)
            if trial_metrics is None:
                continue
                
            search_results = results['search_dirs'][search_dir.name]
            
            # Update overall best for this search directory
            if trial_metrics['overall_accuracy'] > search_results['best_overall']['accuracy']:
                search_results['best_overall']['accuracy'] = trial_metrics['overall_accuracy']
                search_results['best_overall']['trial'] = trial_dir.name
            
            # Update per-class bests for this search directory
            for class_name, class_metrics in trial_metrics['class_accuracies'].items():
                if class_metrics['accuracy'] > search_results['best_per_class'][class_name]['accuracy']:
                    search_results['best_per_class'][class_name]['accuracy'] = class_metrics['accuracy']
                    search_results['best_per_class'][class_name]['trial'] = trial_dir.name
            
            # Collect misclassified files
            search_results['misclassified_files'].extend(trial_metrics['misclassified_files'])
    
    return results

def save_analysis_report(results, hyperparam_dir):
    output_file = os.path.join(hyperparam_dir, 'trial_analysis_report.txt')
    
    with open(output_file, 'w') as f:
        for search_dir, search_results in results['search_dirs'].items():
            f.write(f"\n=== Results for {search_dir} ===\n")
            f.write("-" * 80 + "\n")
            
            # Best overall model
            f.write("\nBest Overall Model:\n")
            f.write(f"Trial: {search_results['best_overall']['trial']}\n")
            f.write(f"Accuracy: {search_results['best_overall']['accuracy']:.2%}\n")
            
            # Best model per class
            f.write("\nBest Model Per Class:\n")
            f.write(f"{'Class':<20} {'Accuracy':<10} {'Trial'}\n")
            f.write("-" * 60 + "\n")
            for class_name, data in search_results['best_per_class'].items():
                f.write(f"{class_name:<20} {data['accuracy']:.2%}    {data['trial']}\n")
            
            # Most frequently misclassified files
            f.write("\nMost Frequently Misclassified Files:\n")
            f.write(f"{'Filename':<40} {'True Class':<15} {'Predicted':<15} {'Confidence':<10} {'Dataset'}\n")
            f.write("-" * 100 + "\n")
            
            # Sort misclassified files by confidence (ascending) to show most problematic cases first
            misclassified = sorted(search_results['misclassified_files'], 
                                 key=lambda x: x['confidence'])
            for error in misclassified[:10]:  # Show top 10 most problematic
                f.write(f"{error['filename']:<40} {error['true_class']:<15} "
                       f"{error['predicted_class']:<15} {error['confidence']:<10.2%} {error['vis_dir']}\n")
            
            f.write("\n" + "=" * 80 + "\n")

def print_results(results):
    """Print a summary of the analysis results"""
    for search_dir, search_results in results['search_dirs'].items():
        print(f"\n=== Results for {search_dir} ===")
        print("-" * 80)
        
        # Best overall model
        print(f"\nBest Overall Model:")
        print(f"Trial: {search_results['best_overall']['trial']}")
        print(f"Accuracy: {search_results['best_overall']['accuracy']:.2%}")
        
        # Best model per class
        print(f"\nBest Model Per Class:")
        print(f"{'Class':<20} {'Accuracy':<10} {'Trial'}")
        print("-" * 60)
        for class_name, data in search_results['best_per_class'].items():
            print(f"{class_name:<20} {data['accuracy']:.2%}    {data['trial']}")
        
        # Most frequently misclassified files (top 5)
        print(f"\nTop 5 Most Problematic Files:")
        print(f"{'Filename':<40} {'True Class':<15} {'Predicted':<15} {'Confidence'}")
        print("-" * 80)
        misclassified = sorted(search_results['misclassified_files'], 
                             key=lambda x: x['confidence'])[:5]
        for error in misclassified:
            print(f"{error['filename']:<40} {error['true_class']:<15} "
                  f"{error['predicted_class']:<15} {error['confidence']:.2%}")

if __name__ == "__main__":
    hyperparam_dir = "runs_hyperparam/hyperparam_20241106_124214"
    results = analyze_trials(hyperparam_dir)
    
    # Print summary to console
    print_results(results)
    
    # Save detailed results to file
    save_analysis_report(results, hyperparam_dir) 