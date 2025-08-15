import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def load_data(data_dir="core"):
    """Load all S1 and S2 result data"""
    s1_results = {}
    s2_results = {}
    
    # Load S1 results
    for i in range(1, 4):
        s1_dir = Path(data_dir) / f"s1_result_{i}"
        if s1_dir.exists():
            s1_results[f"run_{i}"] = {}
            for file in s1_dir.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    s1_results[f"run_{i}"][file.stem] = data
    
    # Load S2 results
    for i in range(1, 4):
        s2_dir = Path(data_dir) / f"s2_result_{i}"
        if s2_dir.exists():
            s2_results[f"run_{i}"] = {}
            for file in s2_dir.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    s2_results[f"run_{i}"][file.stem] = data
    
    return s1_results, s2_results

def extract_metrics(data_list):
    """Extract key metrics from data"""
    accuracy = []
    time_tokens = []
    pass_count = 0
    total_count = len(data_list)
    
    for item in data_list:
        if 'result' in item:
            result = item['result']
            # Get agent type (s1_agent or s2_agent)
            agent_key = None
            for k in result.keys():
                if k.endswith('_agent'):
                    agent_key = k
                    break
            
            if agent_key:
                agent_result = result[agent_key]
                
                # Accuracy
                pass_or_fail = int(agent_result.get('pass_or_fail', 0))
                accuracy.append(pass_or_fail)
                if pass_or_fail == 1:
                    pass_count += 1
                
                # Time consumption
                time_token = result.get('time_token', 0)
                time_tokens.append(time_token)
    
    return {
        'accuracy': np.mean(accuracy) if accuracy else 0,
        'time': np.mean(time_tokens) if time_tokens else 0,
        'pass_rate': pass_count / total_count if total_count > 0 else 0
    }

def generate_comparison_data(s1_results, s2_results):
    """Generate comparison data"""
    comparison_data = []
    datasets = ['results_crt1', 'results_crt2', 'results_crt3', 'results_si']
    
    for dataset in datasets:
        for run in ['run_1', 'run_2', 'run_3']:
            if run in s1_results and dataset in s1_results[run]:
                s1_metrics = extract_metrics(s1_results[run][dataset])
                
                if run in s2_results and dataset in s2_results[run]:
                    s2_metrics = extract_metrics(s2_results[run][dataset])
                    
                    comparison_data.append({
                        'dataset': dataset.replace('results_', ''),
                        'run': run.replace('run_', ''),
                        's1_accuracy': s1_metrics['accuracy'],
                        's2_accuracy': s2_metrics['accuracy'],
                        's1_time': s1_metrics['time'],
                        's2_time': s2_metrics['time'],
                        's1_pass_rate': s1_metrics['pass_rate'],
                        's2_pass_rate': s2_metrics['pass_rate']
                    })
    
    return comparison_data

def plot_comparison(comparison_data, save_path="figures"):
    """Create performance comparison charts"""
    Path(save_path).mkdir(exist_ok=True)
    
    # Group by dataset and calculate averages
    datasets = {}
    for item in comparison_data:
        dataset = item['dataset']
        if dataset not in datasets:
            datasets[dataset] = {'s1_acc': [], 's2_acc': [], 's1_time': [], 's2_time': []}
        
        datasets[dataset]['s1_acc'].append(item['s1_accuracy'])
        datasets[dataset]['s2_acc'].append(item['s2_accuracy'])
        datasets[dataset]['s1_time'].append(item['s1_time'])
        datasets[dataset]['s2_time'].append(item['s2_time'])
    
    # Calculate averages
    dataset_names = list(datasets.keys())
    s1_acc_means = [np.mean(datasets[d]['s1_acc']) for d in dataset_names]
    s2_acc_means = [np.mean(datasets[d]['s2_acc']) for d in dataset_names]
    s1_time_means = [np.mean(datasets[d]['s1_time']) for d in dataset_names]
    s2_time_means = [np.mean(datasets[d]['s2_time']) for d in dataset_names]
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('S1 vs S2 Performance Comparison Analysis', fontsize=16, fontweight='bold')
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, s1_acc_means, width, label='S1', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, s2_acc_means, width, label='S2', alpha=0.8, color='lightcoral')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in dataset_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time consumption comparison
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, s1_time_means, width, label='S1', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x + width/2, s2_time_means, width, label='S2', alpha=0.8, color='lightcoral')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Average Time Consumption (tokens)')
    ax2.set_title('Time Consumption Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in dataset_names])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance improvement
    ax3 = axes[1, 0]
    improvement = [s2 - s1 for s1, s2 in zip(s1_acc_means, s2_acc_means)]
    colors = ['red' if imp > 0 else 'orange' for imp in improvement]
    
    bars5 = ax3.bar(x, improvement, color=colors, alpha=0.8)
    
    for bar in bars5:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Accuracy Improvement')
    ax3.set_title('Performance Improvement')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.upper() for d in dataset_names])
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot
    ax4 = axes[1, 1]
    s1_acc_all = [item['s1_accuracy'] for item in comparison_data]
    s2_acc_all = [item['s2_accuracy'] for item in comparison_data]
    
    ax4.scatter(s1_acc_all, s2_acc_all, alpha=0.6, s=100, c='purple')
    
    # Add diagonal line
    min_val = min(min(s1_acc_all), min(s2_acc_all))
    max_val = max(max(s1_acc_all), max(s2_acc_all))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Diagonal')
    
    ax4.set_xlabel('S1 Accuracy')
    ax4.set_ylabel('S2 Accuracy')
    ax4.set_title('S1 vs S2 Accuracy Scatter Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(f"{save_path}/s1_s2_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_data

def generate_summary(comparison_data):
    """Generate statistical summary"""
    s1_acc_all = [item['s1_accuracy'] for item in comparison_data]
    s2_acc_all = [item['s2_accuracy'] for item in comparison_data]
    s1_time_all = [item['s1_time'] for item in comparison_data]
    s2_time_all = [item['s2_time'] for item in comparison_data]
    
    summary = {
        'S1_Average_Accuracy': np.mean(s1_acc_all),
        'S2_Average_Accuracy': np.mean(s2_acc_all),
        'Accuracy_Improvement': np.mean(s2_acc_all) - np.mean(s1_acc_all),
        'S1_Average_Time': np.mean(s1_time_all),
        'S2_Average_Time': np.mean(s2_time_all),
        'Time_Efficiency_Ratio': np.mean(s1_time_all) / np.mean(s2_time_all) if np.mean(s2_time_all) > 0 else float('inf')
    }
    
    return summary

def save_results(comparison_data, summary, output_dir="analysis_results"):
    """Save analysis results"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save comparison data as JSON
    with open(f"{output_dir}/comparison_data.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    # Save statistical summary
    with open(f"{output_dir}/summary_stats.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Generate report
    report = f"""
# S1 vs S2 Performance Analysis Report

## Overall Performance Comparison
- S1 Average Accuracy: {summary['S1_Average_Accuracy']:.4f}
- S2 Average Accuracy: {summary['S2_Average_Accuracy']:.4f}
- Accuracy Difference: {summary['Accuracy_Improvement']:.4f}

- S1 Average Time: {summary['S1_Average_Time']:.4f} tokens
- S2 Average Time: {summary['S2_Average_Time']:.4f} tokens
- Time Efficiency Ratio: {summary['Time_Efficiency_Ratio']:.4f}

## Dataset-Specific Performance
"""
    
    # Group by dataset
    datasets = {}
    for item in comparison_data:
        dataset = item['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(item)
    
    for dataset, items in datasets.items():
        s1_acc = np.mean([item['s1_accuracy'] for item in items])
        s2_acc = np.mean([item['s2_accuracy'] for item in items])
        s1_time = np.mean([item['s1_time'] for item in items])
        s2_time = np.mean([item['s2_time'] for item in items])
        
        report += f"""
### {dataset.upper()}
- S1 Accuracy: {s1_acc:.4f}
- S2 Accuracy: {s2_acc:.4f}
- S1 Time: {s1_time:.4f}
- S2 Time: {s2_time:.4f}
"""
    
    with open(f"{output_dir}/performance_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Analysis results saved to {output_dir} directory")

def main():
    """Main function"""
    print("Starting S1 and S2 performance data analysis...")
    
    # Load data
    print("Loading data...")
    s1_results, s2_results = load_data()
    
    # Generate comparison data
    print("Generating comparison data...")
    comparison_data = generate_comparison_data(s1_results, s2_results)
    
    if not comparison_data:
        print("No sufficient data found for comparison")
        return
    
    # Create charts
    print("Generating performance comparison charts...")
    plot_comparison(comparison_data)
    
    # Generate statistical summary
    print("Generating statistical summary...")
    summary = generate_summary(comparison_data)
    
    # Save results
    print("Saving analysis results...")
    save_results(comparison_data, summary)
    
    # Print statistical summary
    print("\n=== Statistical Summary ===")
    print(f"S1 Average Accuracy: {summary['S1_Average_Accuracy']:.4f}")
    print(f"S2 Average Accuracy: {summary['S2_Average_Accuracy']:.4f}")
    print(f"Accuracy Improvement: {summary['Accuracy_Improvement']:.4f}")
    print(f"S1 Average Time: {summary['S1_Average_Time']:.4f}")
    print(f"S2 Average Time: {summary['S2_Average_Time']:.4f}")
    print(f"Time Efficiency Ratio: {summary['Time_Efficiency_Ratio']:.4f}")
    
    print("\nAnalysis complete! All charts and reports have been saved.")

if __name__ == "__main__":
    main() 
