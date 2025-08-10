import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class S1S2DataAnalyzer:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.results = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Load all S1 and S2 result files"""
        # Find all result files
        result_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.json') and 'results' in file:
                    result_files.append(os.path.join(root, file))
        
        print(f"Found {len(result_files)} result files")
        
        # Load and parse results
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract system type and dataset from filename
                path_parts = Path(file_path).parts
                system_type = None
                dataset = None
                
                for part in path_parts:
                    if part.startswith('s1_result') or part.startswith('s2_result'):
                        system_type = 'S1' if 's1' in part else 'S2'
                        result_num = part.split('_')[-1]
                        break
                
                if 'crt1' in file_path:
                    dataset = 'CRT1'
                elif 'crt2' in file_path:
                    dataset = 'CRT2'
                elif 'crt3' in file_path:
                    dataset = 'CRT3'
                elif 'si' in file_path:
                    dataset = 'SI'
                
                if system_type and dataset:
                    key = f"{system_type}_{dataset}"
                    self.results[key] = {
                        'system': system_type,
                        'dataset': dataset,
                        'data': data,
                        'file_path': file_path
                    }
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def analyze_results(self):
        """Analyze all loaded results"""
        analysis_results = {}
        
        for key, result in self.results.items():
            data = result['data']
            system = result['system']
            dataset = result['dataset']
            
            # Calculate statistics
            total_tasks = len(data)
            correct_answers = 0
            processing_times = []
            
            for item in data:
                # Check if the item has the expected structure
                if 'result' in item:
                    result_data = item['result']
                    
                    # Check for pass_or_fail in the appropriate agent result
                    if f'{system.lower()}_agent' in result_data:
                        agent_result = result_data[f'{system.lower()}_agent']
                        if 'pass_or_fail' in agent_result:
                            if agent_result['pass_or_fail'] == '1':
                                correct_answers += 1
                    
                    # Extract processing time
                    if 'time_token' in result_data:
                        processing_times.append(result_data['time_token'])
            
            accuracy = (correct_answers / total_tasks) * 100 if total_tasks > 0 else 0
            avg_time = np.mean(processing_times) if processing_times else 0
            
            analysis_results[key] = {
                'system': system,
                'dataset': dataset,
                'total_tasks': total_tasks,
                'correct_answers': correct_answers,
                'accuracy': accuracy,
                'avg_processing_time': avg_time,
                'processing_times': processing_times
            }
        
        return analysis_results
    
    def create_comparison_chart(self, analysis_results):
        """Create comprehensive comparison chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('S1 vs S2 System Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        datasets = ['CRT1', 'CRT2', 'CRT3', 'SI']
        s1_accuracies = []
        s2_accuracies = []
        s1_times = []
        s2_times = []
        
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            if s1_key in analysis_results:
                s1_accuracies.append(analysis_results[s1_key]['accuracy'])
                s1_times.append(analysis_results[s1_key]['avg_processing_time'])
            else:
                s1_accuracies.append(0)
                s1_times.append(0)
            
            if s2_key in analysis_results:
                s2_accuracies.append(analysis_results[s2_key]['accuracy'])
                s2_times.append(analysis_results[s2_key]['avg_processing_time'])
            else:
                s2_accuracies.append(0)
                s2_times.append(0)
        
        # 1. Accuracy Comparison
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, s1_accuracies, width, label='S1 (Intuitive)', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, s2_accuracies, width, label='S2 (Rational)', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. Processing Time Comparison
        bars3 = ax2.bar(x - width/2, s1_times, width, label='S1 (Intuitive)', color='skyblue', alpha=0.8)
        bars4 = ax2.bar(x + width/2, s2_times, width, label='S2 (Rational)', color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Average Processing Time (seconds)')
        ax2.set_title('Processing Time Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # 3. Accuracy vs Time Scatter Plot
        ax3.scatter(s1_times, s1_accuracies, s=100, alpha=0.7, label='S1 (Intuitive)', color='skyblue')
        ax3.scatter(s2_times, s2_accuracies, s=100, alpha=0.7, label='S2 (Rational)', color='lightcoral')
        
        # Add dataset labels
        for i, dataset in enumerate(datasets):
            ax3.annotate(f'S1-{dataset}', (s1_times[i], s1_accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax3.annotate(f'S2-{dataset}', (s2_times[i], s2_accuracies[i]), 
                        xytext=(5, -5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Average Processing Time (seconds)')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy vs Processing Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Summary Table
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            if s1_key in analysis_results and s2_key in analysis_results:
                s1_acc = analysis_results[s1_key]['accuracy']
                s2_acc = analysis_results[s2_key]['accuracy']
                s1_time = analysis_results[s1_key]['avg_processing_time']
                s2_time = analysis_results[s2_key]['avg_processing_time']
                
                table_data.append([
                    dataset,
                    f"{s1_acc:.1f}%",
                    f"{s2_acc:.1f}%",
                    f"{s1_time:.2f}s",
                    f"{s2_time:.2f}s"
                ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Dataset', 'S1 Acc (%)', 'S2 Acc (%)', 'S1 Time (s)', 'S2 Time (s)'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Dataset column
                        table[(i, j)].set_facecolor('#E8F5E8')
                    else:
                        table[(i, j)].set_facecolor('#F8F9FA')
        
        plt.tight_layout()
        plt.savefig('s1_s2_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_analysis(self, analysis_results):
        """Create detailed analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        datasets = ['CRT1', 'CRT2', 'CRT3', 'SI']
        
        # Prepare data
        s1_data = {dataset: [] for dataset in datasets}
        s2_data = {dataset: [] for dataset in datasets}
        
        for key, result in analysis_results.items():
            dataset = result['dataset']
            system = result['system']
            processing_times = result['processing_times']
            
            if system == 'S1':
                s1_data[dataset].extend(processing_times)
            elif system == 'S2':
                s2_data[dataset].extend(processing_times)
        
        # 1. Processing Time Distribution
        ax1 = axes[0, 0]
        all_s1_times = []
        all_s2_times = []
        
        for dataset in datasets:
            all_s1_times.extend(s1_data[dataset])
            all_s2_times.extend(s2_data[dataset])
        
        ax1.hist(all_s1_times, bins=20, alpha=0.7, label='S1 (Intuitive)', color='skyblue')
        ax1.hist(all_s2_times, bins=20, alpha=0.7, label='S2 (Rational)', color='lightcoral')
        ax1.set_xlabel('Processing Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Processing Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy by Dataset
        ax2 = axes[0, 1]
        s1_accuracies = []
        s2_accuracies = []
        
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            s1_acc = analysis_results.get(s1_key, {}).get('accuracy', 0)
            s2_acc = analysis_results.get(s2_key, {}).get('accuracy', 0)
            
            s1_accuracies.append(s1_acc)
            s2_accuracies.append(s2_acc)
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, s1_accuracies, width, label='S1 (Intuitive)', color='skyblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, s2_accuracies, width, label='S2 (Rational)', color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy by Dataset')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Processing Time by Dataset
        ax3 = axes[1, 0]
        s1_times = []
        s2_times = []
        
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            s1_time = analysis_results.get(s1_key, {}).get('avg_processing_time', 0)
            s2_time = analysis_results.get(s2_key, {}).get('avg_processing_time', 0)
            
            s1_times.append(s1_time)
            s2_times.append(s2_time)
        
        bars3 = ax3.bar(x - width/2, s1_times, width, label='S1 (Intuitive)', color='skyblue', alpha=0.8)
        bars4 = ax3.bar(x + width/2, s2_times, width, label='S2 (Rational)', color='lightcoral', alpha=0.8)
        
        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Average Processing Time (seconds)')
        ax3.set_title('Processing Time by Dataset')
        ax3.set_xticks(x)
        ax3.set_xticklabels(datasets)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Efficiency (Accuracy/Time ratio)
        ax4 = axes[1, 1]
        s1_efficiency = []
        s2_efficiency = []
        
        for i, dataset in enumerate(datasets):
            if s1_times[i] > 0:
                s1_efficiency.append(s1_accuracies[i] / s1_times[i])
            else:
                s1_efficiency.append(0)
            
            if s2_times[i] > 0:
                s2_efficiency.append(s2_accuracies[i] / s2_times[i])
            else:
                s2_efficiency.append(0)
        
        bars5 = ax4.bar(x - width/2, s1_efficiency, width, label='S1 (Intuitive)', color='skyblue', alpha=0.8)
        bars6 = ax4.bar(x + width/2, s2_efficiency, width, label='S2 (Rational)', color='lightcoral', alpha=0.8)
        
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Efficiency (Accuracy/Time)')
        ax4.set_title('Performance Efficiency')
        ax4.set_xticks(x)
        ax4.set_xticklabels(datasets)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_paper_charts(self, analysis_results):
        """Create publication-ready charts for the paper"""
        
        # 1. Performance Comparison Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('S1 vs S2 System Performance Analysis', fontsize=16, fontweight='bold')
        
        datasets = ['CRT1', 'CRT2', 'CRT3', 'SI']
        s1_accuracies = []
        s2_accuracies = []
        s1_times = []
        s2_times = []
        
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            s1_acc = analysis_results.get(s1_key, {}).get('accuracy', 0)
            s2_acc = analysis_results.get(s2_key, {}).get('accuracy', 0)
            s1_time = analysis_results.get(s1_key, {}).get('avg_processing_time', 0)
            s2_time = analysis_results.get(s2_key, {}).get('avg_processing_time', 0)
            
            s1_accuracies.append(s1_acc)
            s2_accuracies.append(s2_acc)
            s1_times.append(s1_time)
            s2_times.append(s2_time)
        
        x = np.arange(len(datasets))
        width = 0.35
        
        # Accuracy comparison
        bars1 = ax1.bar(x - width/2, s1_accuracies, width, label='S1 (Intuitive)', 
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, s2_accuracies, width, label='S2 (Rational)', 
                        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Dataset Type', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Processing time comparison
        bars3 = ax2.bar(x - width/2, s1_times, width, label='S1 (Intuitive)', 
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars4 = ax2.bar(x + width/2, s2_times, width, label='S2 (Rational)', 
                        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Dataset Type', fontsize=12)
        ax2.set_ylabel('Average Processing Time (seconds)', fontsize=12)
        ax2.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('paper_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Efficiency vs Accuracy Trade-off Chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate efficiency (accuracy/time)
        s1_efficiency = []
        s2_efficiency = []
        
        for i, dataset in enumerate(datasets):
            if s1_times[i] > 0:
                s1_efficiency.append(s1_accuracies[i] / s1_times[i])
            else:
                s1_efficiency.append(0)
            
            if s2_times[i] > 0:
                s2_efficiency.append(s2_accuracies[i] / s2_times[i])
            else:
                s2_efficiency.append(0)
        
        # Create scatter plot
        scatter1 = ax.scatter(s1_times, s1_accuracies, s=200, alpha=0.7, 
                             label='S1 (Intuitive)', color='#3498db', edgecolors='black', linewidth=1)
        scatter2 = ax.scatter(s2_times, s2_accuracies, s=200, alpha=0.7, 
                             label='S2 (Rational)', color='#e74c3c', edgecolors='black', linewidth=1)
        
        # Add dataset labels
        for i, dataset in enumerate(datasets):
            ax.annotate(f'S1-{dataset}', (s1_times[i], s1_accuracies[i]), 
                       xytext=(10, 10), textcoords='offset points', 
                       fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.annotate(f'S2-{dataset}', (s2_times[i], s2_accuracies[i]), 
                       xytext=(10, -10), textcoords='offset points', 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Average Processing Time (seconds)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Efficiency vs Accuracy Trade-off Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add efficiency contours
        efficiency_levels = [20, 40, 60, 80]
        for level in efficiency_levels:
            x_vals = np.linspace(1, 5, 100)
            y_vals = level * x_vals
            ax.plot(x_vals, y_vals, '--', alpha=0.3, color='gray')
            ax.text(4.5, level * 4.5, f'Efficiency={level}', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('paper_efficiency_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self, analysis_results):
        """Generate summary statistics table"""
        summary_data = []
        
        datasets = ['CRT1', 'CRT2', 'CRT3', 'SI']
        
        for dataset in datasets:
            s1_key = f"S1_{dataset}"
            s2_key = f"S2_{dataset}"
            
            s1_stats = analysis_results.get(s1_key, {})
            s2_stats = analysis_results.get(s2_key, {})
            
            summary_data.append({
                'Dataset': dataset,
                'S1_Accuracy': f"{s1_stats.get('accuracy', 0):.1f}%",
                'S2_Accuracy': f"{s2_stats.get('accuracy', 0):.1f}%",
                'S1_AvgTime': f"{s1_stats.get('avg_processing_time', 0):.3f}s",
                'S2_AvgTime': f"{s2_stats.get('avg_processing_time', 0):.3f}s",
                'S1_TotalTasks': s1_stats.get('total_tasks', 0),
                'S2_TotalTasks': s2_stats.get('total_tasks', 0)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv('summary_statistics.csv', index=False)
        
        print("Summary Statistics:")
        print(df.to_string(index=False))
        
        return df

def main():
    print("Starting S1/S2 Data Analysis...")
    
    # Initialize analyzer
    analyzer = S1S2DataAnalyzer()
    
    # Analyze results
    analysis_results = analyzer.analyze_results()
    
    print(f"\nAnalyzed {len(analysis_results)} result sets")
    
    # Create visualizations
    print("\nCreating comparison chart...")
    analyzer.create_comparison_chart(analysis_results)
    
    print("Creating detailed analysis...")
    analyzer.create_detailed_analysis(analysis_results)
    
    print("Creating paper-ready charts...")
    analyzer.create_paper_charts(analysis_results)
    
    print("Generating summary table...")
    analyzer.generate_summary_table(analysis_results)
    
    print("\nAnalysis complete! Generated files:")
    print("- s1_s2_comparison.png")
    print("- detailed_analysis.png") 
    print("- paper_performance_comparison.png")
    print("- paper_efficiency_accuracy_tradeoff.png")
    print("- summary_statistics.csv")

if __name__ == "__main__":
    main() 