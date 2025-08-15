import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set academic paper style charts with English fonts
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class PaperFigureGenerator:
    def __init__(self, data_dir="core"):
        self.data_dir = Path(data_dir)
        self.s1_results = {}
        self.s2_results = {}
        self.load_all_data()
        self.colors = {
            's1': '#2E86AB',  # Deep blue
            's2': '#A23B72',  # Deep purple
            'baseline': '#F18F01',  # Orange
            'improvement': '#C73E1D'  # Red
        }
    
    def load_all_data(self):
        """Load all S1 and S2 result data"""
        # Load S1 results
        for i in range(1, 4):
            s1_dir = self.data_dir / f"s1_result_{i}"
            if s1_dir.exists():
                self.s1_results[f"run_{i}"] = {}
                for file in s1_dir.glob("*.json"):
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.s1_results[f"run_{i}"][file.stem] = data
        
        # Load S2 results
        for i in range(1, 4):
            s2_dir = self.data_dir / f"s2_result_{i}"
            if s2_dir.exists():
                self.s2_results[f"run_{i}"] = {}
                for file in s2_dir.glob("*.json"):
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.s2_results[f"run_{i}"][file.stem] = data
    
    def extract_metrics(self, data_list):
        """Extract key metrics from data"""
        metrics = {
            'accuracy': [],
            'time_tokens': [],
            'pass_count': 0,
            'fail_count': 0,
            'total_count': len(data_list)
        }
        
        for item in data_list:
            if 'result' in item:
                result = item['result']
                agent_key = [k for k in result.keys() if k.endswith('_agent')][0]
                agent_result = result[agent_key]
                
                pass_or_fail = int(agent_result.get('pass_or_fail', 0))
                metrics['accuracy'].append(pass_or_fail)
                if pass_or_fail == 1:
                    metrics['pass_count'] += 1
                else:
                    metrics['fail_count'] += 1
                
                time_token = result.get('time_token', 0)
                metrics['time_tokens'].append(time_token)
        
        return metrics
    
    def generate_comparison_data(self):
        """Generate comparison data"""
        comparison_data = []
        datasets = ['results_crt1', 'results_crt2', 'results_crt3', 'results_si']
        
        for dataset in datasets:
            for run in ['run_1', 'run_2', 'run_3']:
                if run in self.s1_results and dataset in self.s1_results[run]:
                    s1_metrics = self.extract_metrics(self.s1_results[run][dataset])
                    
                    if run in self.s2_results and dataset in self.s2_results[run]:
                        s2_metrics = self.extract_metrics(self.s2_results[run][dataset])
                        
                        comparison_data.append({
                            'dataset': dataset.replace('results_', ''),
                            'run': run.replace('run_', ''),
                            's1_accuracy': np.mean(s1_metrics['accuracy']),
                            's2_accuracy': np.mean(s2_metrics['accuracy']),
                            's1_time': np.mean(s1_metrics['time_tokens']),
                            's2_time': np.mean(s2_metrics['time_tokens']),
                            's1_pass_rate': s1_metrics['pass_count'] / s1_metrics['total_count'],
                            's2_pass_rate': s2_metrics['pass_count'] / s2_metrics['total_count']
                        })
        
        return pd.DataFrame(comparison_data)
    
    def plot_main_comparison(self, save_path="paper_figures"):
        """Generate main performance comparison chart (suitable for papers)"""
        df = self.generate_comparison_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Comparison: S1 vs S2', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        datasets = df['dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.35
        
        s1_acc = [df[df['dataset'] == d]['s1_accuracy'].mean() for d in datasets]
        s2_acc = [df[df['dataset'] == d]['s2_accuracy'].mean() for d in datasets]
        
        bars1 = ax1.bar(x - width/2, s1_acc, width, label='S1', color=self.colors['s1'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, s2_acc, width, label='S2', color=self.colors['s2'], alpha=0.8)
        
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
        ax1.set_ylabel('Accuracy')
        ax1.set_title('(a) Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.upper() for d in datasets])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time efficiency comparison
        ax2 = axes[0, 1]
        s1_time = [df[df['dataset'] == d]['s1_time'].mean() for d in datasets]
        s2_time = [df[df['dataset'] == d]['s2_time'].mean() for d in datasets]
        
        bars3 = ax2.bar(x - width/2, s1_time, width, label='S1', color=self.colors['s1'], alpha=0.8)
        bars4 = ax2.bar(x + width/2, s2_time, width, label='S2', color=self.colors['s2'], alpha=0.8)
        
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Time (tokens)')
        ax2.set_title('(b) Time Efficiency')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.upper() for d in datasets])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance improvement
        ax3 = axes[1, 0]
        improvement = [s2 - s1 for s1, s2 in zip(s1_acc, s2_acc)]
        colors = [self.colors['improvement'] if imp > 0 else self.colors['baseline'] for imp in improvement]
        
        bars5 = ax3.bar(x, improvement, color=colors, alpha=0.8)
        
        for bar in bars5:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_title('(c) Performance Improvement')
        ax3.set_xticks(x)
        ax3.set_xticklabels([d.upper() for d in datasets])
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot
        ax4 = axes[1, 1]
        ax4.scatter(df['s1_accuracy'], df['s2_accuracy'], 
                   alpha=0.7, s=80, c=self.colors['s2'], edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_val = min(df['s1_accuracy'].min(), df['s2_accuracy'].min())
        max_val = max(df['s1_accuracy'].max(), df['s2_accuracy'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.7, linewidth=1)
        
        ax4.set_xlabel('S1 Accuracy')
        ax4.set_ylabel('S2 Accuracy')
        ax4.set_title('(d) S1 vs S2 Correlation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save image
        Path(save_path).mkdir(exist_ok=True)
        plt.savefig(f"{save_path}/main_comparison.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{save_path}/main_comparison.png", format='png', bbox_inches='tight')
        plt.show()
        
        return df
    
    def plot_time_analysis_paper(self, save_path="paper_figures"):
        """Generate time analysis chart (paper quality)"""
        df = self.generate_comparison_data()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Time Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time distribution
        ax1 = axes[0]
        all_s1_times = []
        all_s2_times = []
        
        for run in ['run_1', 'run_2', 'run_3']:
            for dataset in ['results_crt1', 'results_crt2', 'results_crt3', 'results_si']:
                if run in self.s1_results and dataset in self.s1_results[run]:
                    s1_metrics = self.extract_metrics(self.s1_results[run][dataset])
                    all_s1_times.extend(s1_metrics['time_tokens'])
                
                if run in self.s2_results and dataset in self.s2_results[run]:
                    s2_metrics = self.extract_metrics(self.s2_results[run][dataset])
                    all_s2_times.extend(s2_metrics['time_tokens'])
        
        # Use kernel density estimation
        from scipy import stats
        
        x_range = np.linspace(min(min(all_s1_times), min(all_s2_times)), 
                             max(max(all_s1_times), max(all_s2_times)), 100)
        
        kde_s1 = stats.gaussian_kde(all_s1_times)
        kde_s2 = stats.gaussian_kde(all_s2_times)
        
        ax1.plot(x_range, kde_s1(x_range), label='S1', color=self.colors['s1'], linewidth=2)
        ax1.plot(x_range, kde_s2(x_range), label='S2', color=self.colors['s2'], linewidth=2)
        ax1.fill_between(x_range, kde_s1(x_range), alpha=0.3, color=self.colors['s1'])
        ax1.fill_between(x_range, kde_s2(x_range), alpha=0.3, color=self.colors['s2'])
        
        ax1.set_xlabel('Time (tokens)')
        ax1.set_ylabel('Density')
        ax1.set_title('(a) Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time vs accuracy
        ax2 = axes[1]
        scatter1 = ax2.scatter(df['s1_time'], df['s1_accuracy'], 
                              alpha=0.7, s=100, label='S1', color=self.colors['s1'], 
                              edgecolors='black', linewidth=0.5)
        scatter2 = ax2.scatter(df['s2_time'], df['s2_accuracy'], 
                              alpha=0.7, s=100, label='S2', color=self.colors['s2'], 
                              edgecolors='black', linewidth=0.5)
        
        # Add trend lines
        z1 = np.polyfit(df['s1_time'], df['s1_accuracy'], 1)
        p1 = np.poly1d(z1)
        ax2.plot(df['s1_time'], p1(df['s1_time']), "--", color=self.colors['s1'], alpha=0.8)
        
        z2 = np.polyfit(df['s2_time'], df['s2_accuracy'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(df['s2_time'], p2(df['s2_time']), "--", color=self.colors['s2'], alpha=0.8)
        
        ax2.set_xlabel('Time (tokens)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('(b) Time vs Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save image
        Path(save_path).mkdir(exist_ok=True)
        plt.savefig(f"{save_path}/time_analysis.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{save_path}/time_analysis.png", format='png', bbox_inches='tight')
        plt.show()
    
    def plot_dataset_breakdown(self, save_path="paper_figures"):
        """Generate dataset-specific analysis chart"""
        df = self.generate_comparison_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        datasets = df['dataset'].unique()
        
        for i, dataset in enumerate(datasets):
            ax = axes[i//2, i%2]
            dataset_df = df[df['dataset'] == dataset]
            
            if not dataset_df.empty:
                runs = dataset_df['run'].values
                s1_acc = dataset_df['s1_accuracy'].values
                s2_acc = dataset_df['s2_accuracy'].values
                
                x = np.arange(len(runs))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, s1_acc, width, label='S1', 
                              color=self.colors['s1'], alpha=0.8)
                bars2 = ax.bar(x + width/2, s2_acc, width, label='S2', 
                              color=self.colors['s2'], alpha=0.8)
                
                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Run')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'({chr(97+i)}) {dataset.upper()} Dataset')
                ax.set_xticks(x)
                ax.set_xticklabels([f'Run {r}' for r in runs])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save image
        Path(save_path).mkdir(exist_ok=True)
        plt.savefig(f"{save_path}/dataset_breakdown.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{save_path}/dataset_breakdown.png", format='png', bbox_inches='tight')
        plt.show()
    
    def generate_latex_table(self, save_path="paper_figures"):
        """Generate LaTeX table"""
        df = self.generate_comparison_data()
        
        # Calculate summary statistics
        summary_stats = df.groupby('dataset').agg({
            's1_accuracy': ['mean', 'std'],
            's2_accuracy': ['mean', 'std'],
            's1_time': ['mean', 'std'],
            's2_time': ['mean', 'std']
        }).round(4)
        
        # Generate LaTeX table
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison between S1 and S2 Systems}
\label{tab:performance_comparison}
\begin{tabular}{lcccccc}
\toprule
Dataset & \multicolumn{2}{c}{S1 Accuracy} & \multicolumn{2}{c}{S2 Accuracy} & \multicolumn{2}{c}{Time (tokens)} \\
& Mean & Std & Mean & Std & S1 & S2 \\
\midrule
"""
        
        for dataset in summary_stats.index:
            s1_acc_mean = summary_stats.loc[dataset, ('s1_accuracy', 'mean')]
            s1_acc_std = summary_stats.loc[dataset, ('s1_accuracy', 'std')]
            s2_acc_mean = summary_stats.loc[dataset, ('s2_accuracy', 'mean')]
            s2_acc_std = summary_stats.loc[dataset, ('s2_accuracy', 'std')]
            s1_time = summary_stats.loc[dataset, ('s1_time', 'mean')]
            s2_time = summary_stats.loc[dataset, ('s2_time', 'mean')]
            
            latex_table += f"{dataset.upper()} & {s1_acc_mean:.4f} & {s1_acc_std:.4f} & {s2_acc_mean:.4f} & {s2_acc_std:.4f} & {s1_time:.2f} & {s2_time:.2f} \\\\\n"
        
        latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save LaTeX table
        Path(save_path).mkdir(exist_ok=True)
        with open(f"{save_path}/performance_table.tex", 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to {save_path}/performance_table.tex")
        return latex_table
    
    def generate_all_paper_figures(self, save_path="paper_figures"):
        """Generate all paper figures"""
        print("Generating paper quality figures...")
        
        # Generate main comparison chart
        print("1. Generating main performance comparison chart...")
        df = self.plot_main_comparison(save_path)
        
        # Generate time analysis chart
        print("2. Generating time analysis chart...")
        self.plot_time_analysis_paper(save_path)
        
        # Generate dataset-specific analysis chart
        print("3. Generating dataset-specific analysis chart...")
        self.plot_dataset_breakdown(save_path)
        
        # Generate LaTeX table
        print("4. Generating LaTeX table...")
        self.generate_latex_table(save_path)
        
        # Generate statistical summary
        print("5. Generating statistical summary...")
        summary = {
            'S1_Average_Accuracy': df['s1_accuracy'].mean(),
            'S2_Average_Accuracy': df['s2_accuracy'].mean(),
            'Accuracy_Improvement': df['s2_accuracy'].mean() - df['s1_accuracy'].mean(),
            'S1_Average_Time': df['s1_time'].mean(),
            'S2_Average_Time': df['s2_time'].mean(),
            'Time_Efficiency_Ratio': df['s1_time'].mean() / df['s2_time'].mean()
        }
        
        with open(f"{save_path}/summary_stats.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nAll paper figures saved to {save_path} directory")
        print(f"Statistical summary: {summary}")
        
        return summary

def main():
    """Main function"""
    print("Starting paper quality figure generation...")
    
    # Create figure generator
    generator = PaperFigureGenerator()
    
    # Generate all figures
    summary = generator.generate_all_paper_figures()
    
    print("\n=== Paper Figure Generation Complete ===")
    print("Generated files include:")
    print("- main_comparison.pdf/png: Main performance comparison chart")
    print("- time_analysis.pdf/png: Time analysis chart")
    print("- dataset_breakdown.pdf/png: Dataset-specific analysis chart")
    print("- performance_table.tex: LaTeX table")
    print("- summary_stats.json: Statistical summary")

if __name__ == "__main__":
    main() 