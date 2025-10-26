#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Academic Visualization Script for Elliot Recommendation Results
Generates publication-quality figures and LaTeX tables

Author: Auto-generated for academic analysis
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class ResultsVisualizer:
    """Visualizer for recommendation system experiment results"""
    
    def __init__(self, results_dir: str = 'results/movielens_small'):
        self.results_dir = Path(results_dir)
        self.performance_dir = self.results_dir / 'performance'
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = ['nDCG', 'Recall', 'MAP', 'MRR']
        self.cutoffs = [10, 20]
        self.colors = sns.color_palette("colorblind", 10)
        
    def load_performance_data(self) -> Dict[str, pd.DataFrame]:
        """Load all performance TSV files"""
        data = {}
        
        for cutoff in self.cutoffs:
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
                pattern = f'rec_{model_type}_cutoff_{cutoff}_relthreshold_0_*.tsv'
                files = list(self.performance_dir.glob(pattern))
                
                if files:
                    df = pd.read_csv(files[0], sep='\t')
                    key = f'{model_type}_cutoff_{cutoff}'
                    data[key] = df
                    print(f"Loaded {key}: {len(df)} configurations")
        
        return data
    
    def load_best_models(self) -> List[Dict]:
        """Load best model parameters from JSON"""
        pattern = 'bestmodelparams_cutoff_10_*.json'
        files = list(self.performance_dir.glob(pattern))
        
        if files:
            with open(files[0], 'r') as f:
                best_models = json.load(f)
            return best_models
        return []
    
    def extract_model_type(self, model_name: str) -> str:
        """Extract base model type from full model name"""
        if 'VSM' in model_name:
            return 'VSM'
        elif 'AttributeItemKNN' in model_name:
            return 'AttributeItemKNN'
        elif 'AttributeUserKNN' in model_name:
            return 'AttributeUserKNN'
        elif 'RP3beta' in model_name:
            return 'RP3beta'
        return 'Unknown'
    
    def create_best_models_comparison(self, data: Dict[str, pd.DataFrame]):
        """Figure 1: Best models performance comparison across cutoffs"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Get best performing config for each model at cutoff 10
        best_configs = {}
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
            key = f'{model_type}_cutoff_10'
            if key in data:
                df = data[key]
                best_idx = df['nDCG'].idxmax()
                best_configs[model_type] = df.loc[best_idx, 'model']
        
        # Collect results for both cutoffs
        results = []
        for cutoff in self.cutoffs:
            for model_type, best_model in best_configs.items():
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    df = data[key]
                    row = df[df['model'] == best_model]
                    if not row.empty:
                        for metric in self.metrics:
                            results.append({
                                'Model': model_type,
                                'Cutoff': f'@{cutoff}',
                                'Metric': metric,
                                'Value': row[metric].values[0]
                            })
        
        results_df = pd.DataFrame(results)
        
        # Plot each metric
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            metric_data = results_df[results_df['Metric'] == metric]
            
            # Pivot for grouped bar chart
            pivot_data = metric_data.pivot(index='Model', columns='Cutoff', values='Value')
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_data['@10'], width, 
                          label='@10', color=self.colors[0], alpha=0.8)
            bars2 = ax.bar(x + width/2, pivot_data['@20'], width, 
                          label='@20', color=self.colors[1], alpha=0.8)
            
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_data.index, rotation=15, ha='right')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_best_models_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_best_models_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: Figure 1 - Best Models Comparison")
        plt.close()
    
    def create_hyperparameter_analysis(self, data: Dict[str, pd.DataFrame]):
        """Figure 2: Hyperparameter sensitivity analysis"""
        
        # Focus on AttributeUserKNN and RP3beta (most configurable)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # AttributeUserKNN: neighbors impact
        ax = axes[0, 0]
        key = 'AttributeUserKNN_cutoff_10'
        if key in data:
            df = data[key].copy()
            # Extract neighbors
            df['neighbors'] = df['model'].str.extract(r'nn=(\d+)')[0].astype(int)
            df['similarity'] = df['model'].str.extract(r'sim=(\w+)')[0]
            
            for sim in df['similarity'].unique():
                sim_data = df[df['similarity'] == sim].groupby('neighbors')['nDCG'].max()
                ax.plot(sim_data.index, sim_data.values, marker='o', 
                       label=sim, linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Neighbors', fontweight='bold')
            ax.set_ylabel('nDCG@10', fontweight='bold')
            ax.set_title('AttributeUserKNN: Neighbor Size Impact', fontweight='bold')
            ax.legend(title='Similarity', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # AttributeUserKNN: profile type impact
        ax = axes[0, 1]
        if key in data:
            df = data[key].copy()
            df['profile'] = df['model'].str.extract(r'profile=(\w+)')[0]
            
            profile_perf = df.groupby('profile')[self.metrics].max()
            profile_perf.plot(kind='bar', ax=ax, width=0.7, alpha=0.8)
            
            ax.set_xlabel('Profile Type', fontweight='bold')
            ax.set_ylabel('Performance', fontweight='bold')
            ax.set_title('AttributeUserKNN: Profile Type Impact', fontweight='bold')
            ax.legend(title='Metrics', framealpha=0.9, loc='best')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # RP3beta: Neighborhood size vs nDCG
        ax = axes[1, 0]
        key = 'RP3beta_cutoff_10'
        if key in data:
            df = data[key].copy()
            # Extract neighborhood size
            df['neighborhood'] = df['model'].str.extract(r'neighborhood=(\d+)')[0].astype(int)
            
            # Group by neighborhood and show mean/max performance
            neighborhood_perf = df.groupby('neighborhood')['nDCG'].agg(['mean', 'max', 'min'])
            
            x = neighborhood_perf.index
            ax.plot(x, neighborhood_perf['mean'], marker='o', label='Mean', 
                   linewidth=2, markersize=10, color=self.colors[0])
            ax.plot(x, neighborhood_perf['max'], marker='s', label='Max', 
                   linewidth=2, markersize=8, color=self.colors[1], linestyle='--')
            ax.fill_between(x, neighborhood_perf['min'], neighborhood_perf['max'], 
                           alpha=0.2, color=self.colors[0])
            
            ax.set_xlabel('Neighborhood Size', fontweight='bold')
            ax.set_ylabel('nDCG@10', fontweight='bold')
            ax.set_title('RP3beta: Neighborhood Size Impact', fontweight='bold')
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Model complexity: All models performance distribution
        ax = axes[1, 1]
        all_ndcg = []
        labels = []
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
            key = f'{model_type}_cutoff_10'
            if key in data:
                all_ndcg.append(data[key]['nDCG'].values)
                labels.append(model_type)
        
        if all_ndcg:
            bp = ax.boxplot(all_ndcg, labels=labels, patch_artist=True, 
                           showmeans=True, meanline=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel('nDCG@10', fontweight='bold')
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_title('Performance Distribution Across Configurations', fontweight='bold')
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_hyperparameter_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: Figure 2 - Hyperparameter Analysis")
        plt.close()
    
    def create_metric_correlation_heatmap(self, data: Dict[str, pd.DataFrame]):
        """Figure 3: Metric correlation analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, cutoff in enumerate(self.cutoffs):
            # Combine all models for this cutoff
            all_results = []
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    all_results.append(data[key][self.metrics])
            
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                corr_matrix = combined_df[self.metrics].corr()
                
                ax = axes[idx]
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                           center=0, vmin=-1, vmax=1, square=True, ax=ax,
                           cbar_kws={'label': 'Correlation'})
                ax.set_title(f'Metric Correlation @{cutoff}', fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_metric_correlation.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_metric_correlation.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: Figure 3 - Metric Correlation Heatmap")
        plt.close()
    
    def create_cutoff_improvement_analysis(self, data: Dict[str, pd.DataFrame]):
        """Figure 4: Performance improvement from cutoff 10 to 20"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        improvements = []
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
            key_10 = f'{model_type}_cutoff_10'
            key_20 = f'{model_type}_cutoff_20'
            
            if key_10 in data and key_20 in data:
                # Get best config from cutoff 10
                best_model = data[key_10].loc[data[key_10]['nDCG'].idxmax(), 'model']
                
                # Find same config in both cutoffs
                perf_10 = data[key_10][data[key_10]['model'] == best_model][self.metrics].values[0]
                row_20 = data[key_20][data[key_20]['model'] == best_model]
                
                if not row_20.empty:
                    perf_20 = row_20[self.metrics].values[0]
                    improvement = ((perf_20 - perf_10) / perf_10 * 100)
                    
                    for metric_idx, metric in enumerate(self.metrics):
                        improvements.append({
                            'Model': model_type,
                            'Metric': metric,
                            'Improvement (%)': improvement[metric_idx]
                        })
        
        imp_df = pd.DataFrame(improvements)
        pivot_imp = imp_df.pivot(index='Model', columns='Metric', values='Improvement (%)')
        
        pivot_imp.plot(kind='bar', ax=ax, width=0.75, alpha=0.8)
        ax.set_ylabel('Improvement (%)', fontweight='bold')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_title('Performance Improvement: @10 → @20', fontweight='bold', pad=10)
        ax.legend(title='Metrics', framealpha=0.9, loc='best')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_cutoff_improvement.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_cutoff_improvement.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: Figure 4 - Cutoff Improvement Analysis")
        plt.close()
    
    def create_latex_table(self, data: Dict[str, pd.DataFrame]):
        """Generate LaTeX table for publication"""
        latex_output = []
        
        latex_output.append(r"\begin{table}[htbp]")
        latex_output.append(r"\centering")
        latex_output.append(r"\caption{Performance Comparison of Best Model Configurations}")
        latex_output.append(r"\label{tab:best_models}")
        latex_output.append(r"\begin{tabular}{l|cccc|cccc}")
        latex_output.append(r"\hline")
        latex_output.append(r"\multirow{2}{*}{Model} & \multicolumn{4}{c|}{@10} & \multicolumn{4}{c}{@20} \\")
        latex_output.append(r" & nDCG & Recall & MAP & MRR & nDCG & Recall & MAP & MRR \\")
        latex_output.append(r"\hline")
        
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
            row_data = [model_type]
            
            for cutoff in [10, 20]:
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    df = data[key]
                    best_idx = df['nDCG'].idxmax()
                    best_row = df.loc[best_idx]
                    
                    for metric in self.metrics:
                        row_data.append(f"{best_row[metric]:.4f}")
            
            if len(row_data) == 9:  # Model + 8 metrics
                latex_output.append(" & ".join(row_data) + r" \\")
        
        latex_output.append(r"\hline")
        latex_output.append(r"\end{tabular}")
        latex_output.append(r"\end{table}")
        
        # Save to file
        latex_file = self.output_dir / 'table_best_models.tex'
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_output))
        
        print(f"✓ Saved: LaTeX Table - {latex_file}")
        
        # Also create a summary CSV
        summary_data = []
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'RP3beta']:
            row = {'Model': model_type}
            for cutoff in [10, 20]:
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    df = data[key]
                    best_idx = df['nDCG'].idxmax()
                    best_row = df.loc[best_idx]
                    for metric in self.metrics:
                        row[f'{metric}@{cutoff}'] = best_row[metric]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'summary_best_models.csv', index=False, float_format='%.4f')
        print(f"✓ Saved: Summary CSV - summary_best_models.csv")
    
    def generate_all_visualizations(self):
        """Main method to generate all visualizations"""
        print("\n" + "="*60)
        print("Academic Visualization Generation Started")
        print("="*60 + "\n")
        
        # Load data
        print("Loading performance data...")
        data = self.load_performance_data()
        
        if not data:
            print("❌ No performance data found!")
            return
        
        print(f"✓ Loaded {len(data)} performance files\n")
        
        # Generate figures
        print("Generating visualizations...")
        self.create_best_models_comparison(data)
        self.create_hyperparameter_analysis(data)
        self.create_metric_correlation_heatmap(data)
        self.create_cutoff_improvement_analysis(data)
        
        print("\nGenerating LaTeX table...")
        self.create_latex_table(data)
        
        print("\n" + "="*60)
        print(f"✓ All visualizations saved to: {self.output_dir.absolute()}")
        print("="*60 + "\n")
        
        print("Generated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  • {file.name}")
        
        print("\n📊 Ready for academic presentation!")


def main():
    """Main execution"""
    visualizer = ResultsVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
