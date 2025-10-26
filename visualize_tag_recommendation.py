#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签推荐系统学术可视化脚本
Tag-based Recommendation System Academic Visualization

研究目标：基于用户标签的推荐算法验证与对比分析
算法类型：
1. VSM (Vector Space Model) - 基于向量空间的算法
2. AttributeItemKNN - 基于物品属性的协同过滤
3. AttributeUserKNN - 基于用户属性的协同过滤
4. DeepFM - 基于深度学习的因子分解机
5. RP3beta - 基于图的随机游走算法

Author: 学术研究可视化
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

# 设置中文字体和学术风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 配置matplotlib以支持中文
# 尝试使用系统中可用的中文字体
import matplotlib.font_manager as fm

# 查找可用的中文字体
def get_chinese_font():
    """获取系统中可用的中文字体"""
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',           # 黑体
        'SimSun',           # 宋体
        'Microsoft YaHei',  # 微软雅黑
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',    # 文泉驿正黑 (Linux)
        'Noto Sans CJK SC',     # 思源黑体
        'Noto Sans CJK TC',
        'DejaVu Sans',          # 备用字体
        'Arial Unicode MS',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            return font
    
    # 如果没找到中文字体，返回默认字体
    print("⚠️  未找到中文字体，使用默认字体")
    return 'DejaVu Sans'

chinese_font = get_chinese_font()
print(f"使用字体: {chinese_font}")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [chinese_font, 'DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.unicode_minus': False,  # 解决负号显示问题
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


class TagRecommendationVisualizer:
    """标签推荐系统可视化器"""
    
    def __init__(self, results_dir: str = 'results/movielens_small'):
        self.results_dir = Path(results_dir)
        self.performance_dir = self.results_dir / 'performance'
        self.output_dir = Path('academic_visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = ['nDCG', 'Recall', 'MAP', 'MRR']
        self.metric_names_cn = {
            'nDCG': 'nDCG (归一化折损累计增益)',
            'Recall': 'Recall (召回率)',
            'MAP': 'MAP (平均精度均值)',
            'MRR': 'MRR (平均倒数排名)'
        }
        self.cutoffs = [10, 20]
        
        # 算法分类
        self.algorithm_types = {
            'VSM': 'VSM',
            'AttributeItemKNN': 'AttributeItemKNN',
            'AttributeUserKNN': 'AttributeUserKNN',
            'DeepFM': 'DeepFM',
            'RP3beta': 'RP3beta'
        }
        
        # 用于长标题的算法名称（用于总结表等）
        self.algorithm_types_long = {
            'VSM': 'VSM (Vector Space Model)',
            'AttributeItemKNN': 'AttributeItemKNN',
            'AttributeUserKNN': 'AttributeUserKNN',
            'DeepFM': 'DeepFM (Deep Factorization Machine)',
            'RP3beta': 'RP3beta (Random Walk with Restart)'
        }
        
        self.colors = sns.color_palette("Set2", 10)
        
    def load_all_performance_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有性能数据"""
        data = {}
        
        # 查找最新的实验结果
        all_files = list(self.performance_dir.glob('rec_*_cutoff_*_relthreshold_0_*.tsv'))
        
        if not all_files:
            print("❌ 未找到性能数据文件！")
            return data
        
        # 按时间戳排序，获取最新的
        timestamps = set()
        for f in all_files:
            # 文件名格式: rec_ModelName_cutoff_X_relthreshold_0_YYYY_MM_DD_HH_MM_SS.tsv
            parts = f.stem.split('_')
            # 提取最后6个部分作为时间戳
            if len(parts) >= 6:
                timestamp = '_'.join(parts[-6:])
                timestamps.add(timestamp)
        
        if not timestamps:
            print("❌ 无法解析时间戳！")
            return data
            
        latest_timestamp = sorted(timestamps)[-1]
        print(f"📅 使用最新实验数据: {latest_timestamp}\n")
        
        for cutoff in self.cutoffs:
            # 总体结果
            pattern = f'rec_cutoff_{cutoff}_relthreshold_0_{latest_timestamp}.tsv'
            files = list(self.performance_dir.glob(pattern))
            if files:
                df = pd.read_csv(files[0], sep='\t')
                data[f'overall_cutoff_{cutoff}'] = df
                print(f"✓ 加载总体结果 @{cutoff}: {len(df)} 个配置")
            
            # 各模型单独结果
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
                pattern = f'rec_{model_type}_cutoff_{cutoff}_relthreshold_0_{latest_timestamp}.tsv'
                files = list(self.performance_dir.glob(pattern))
                
                if files:
                    df = pd.read_csv(files[0], sep='\t')
                    key = f'{model_type}_cutoff_{cutoff}'
                    data[key] = df
                    print(f"✓ 加载 {model_type} @{cutoff}: {len(df)} 个配置")
        
        return data
    
    def load_best_models(self) -> List[Dict]:
        """加载最佳模型参数"""
        pattern = 'bestmodelparams_cutoff_10_*.json'
        files = list(self.performance_dir.glob(pattern))
        
        if files:
            # 使用最新的
            latest_file = sorted(files)[-1]
            with open(latest_file, 'r') as f:
                best_models = json.load(f)
            print(f"\n✓ 加载最佳模型配置: {latest_file.name}")
            return best_models
        return []
    
    def get_best_config_per_model(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """获取每个模型的最佳配置"""
        best_configs = []
        
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
            for cutoff in self.cutoffs:
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    df = data[key]
                    if len(df) > 0:
                        # 按nDCG选择最佳
                        best_idx = df['nDCG'].idxmax()
                        best_row = df.loc[best_idx].copy()
                        best_row['model_type'] = model_type
                        best_row['cutoff'] = cutoff
                        best_configs.append(best_row)
        
        return pd.DataFrame(best_configs)
    
    def fig1_algorithm_comparison_by_metric(self, data: Dict[str, pd.DataFrame]):
        """图1：不同算法在各评价指标上的性能对比（4个子图，每个指标一个）"""
        print("\n生成图1：算法性能对比...")
        
        best_configs = self.get_best_config_per_model(data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            
            # 准备数据
            plot_data = []
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
                for cutoff in self.cutoffs:
                    subset = best_configs[
                        (best_configs['model_type'] == model_type) & 
                        (best_configs['cutoff'] == cutoff)
                    ]
                    if not subset.empty:
                        plot_data.append({
                            'Algorithm': self.algorithm_types[model_type],
                            'Cutoff': f'@{cutoff}',
                            'Value': subset[metric].values[0]
                        })
            
            df_plot = pd.DataFrame(plot_data)
            
            # 绘制分组柱状图
            x_labels = [self.algorithm_types[m] for m in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']]
            x_pos = np.arange(len(x_labels))
            width = 0.35
            
            values_10 = []
            values_20 = []
            for label in x_labels:
                v10 = df_plot[(df_plot['Algorithm'] == label) & (df_plot['Cutoff'] == '@10')]
                v20 = df_plot[(df_plot['Algorithm'] == label) & (df_plot['Cutoff'] == '@20')]
                values_10.append(v10['Value'].values[0] if not v10.empty else 0)
                values_20.append(v20['Value'].values[0] if not v20.empty else 0)
            
            bars1 = ax.bar(x_pos - width/2, values_10, width, label='Top-10', 
                          color=self.colors[0], alpha=0.85, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x_pos + width/2, values_20, width, label='Top-20', 
                          color=self.colors[1], alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel(metric, fontsize=13, fontweight='bold')
            ax.set_title(f'{self.metric_names_cn[metric]}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=9, ha='center', rotation=0)
            ax.legend(loc='best', framealpha=0.95, fontsize=11)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('基于标签的推荐算法性能对比分析', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig1_algorithm_performance_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def fig2_cutoff_sensitivity_analysis(self, data: Dict[str, pd.DataFrame]):
        """图2：不同Top-K截断对算法性能的影响"""
        print("\n生成图2：Top-K截断敏感性分析...")
        
        best_configs = self.get_best_config_per_model(data)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        model_types = ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']
        x_pos = np.arange(len(model_types))
        width = 0.2
        
        for metric_idx, metric in enumerate(self.metrics):
            improvements = []
            
            for model_type in model_types:
                perf_10 = best_configs[
                    (best_configs['model_type'] == model_type) & 
                    (best_configs['cutoff'] == 10)
                ]
                perf_20 = best_configs[
                    (best_configs['model_type'] == model_type) & 
                    (best_configs['cutoff'] == 20)
                ]
                
                if not perf_10.empty and not perf_20.empty:
                    v10 = perf_10[metric].values[0]
                    v20 = perf_20[metric].values[0]
                    improvement = ((v20 - v10) / v10 * 100) if v10 > 0 else 0
                    improvements.append(improvement)
                else:
                    improvements.append(0)
            
            offset = width * (metric_idx - 1.5)
            bars = ax.bar(x_pos + offset, improvements, width, 
                         label=metric, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=9, fontweight='bold')
        
        ax.set_ylabel('性能提升百分比 (%)', fontsize=13, fontweight='bold')
        ax.set_xlabel('推荐算法', fontsize=13, fontweight='bold')
        ax.set_title('Top-10 → Top-20 性能提升分析', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.algorithm_types[m] for m in model_types], 
                          fontsize=10, ha='center', rotation=15)
        ax.legend(title='评价指标', framealpha=0.95, fontsize=11, title_fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'fig2_cutoff_sensitivity_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def fig3_hyperparameter_impact(self, data: Dict[str, pd.DataFrame]):
        """图3：超参数对算法性能的影响（多子图分析）"""
        print("\n生成图3：超参数影响分析...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 3.1 AttributeUserKNN - neighbors影响
        ax1 = fig.add_subplot(gs[0, 0])
        key = 'AttributeUserKNN_cutoff_10'
        if key in data:
            df = data[key].copy()
            df['neighbors'] = df['model'].str.extract(r'nn=(\d+)')[0]
            df = df.dropna(subset=['neighbors'])
            df['neighbors'] = df['neighbors'].astype(int)
            
            neighbor_perf = df.groupby('neighbors')[self.metrics].mean()
            
            for metric in self.metrics:
                ax1.plot(neighbor_perf.index, neighbor_perf[metric], 
                        marker='o', label=metric, linewidth=2, markersize=8)
            
            ax1.set_xlabel('邻居数量 (K)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('性能指标值', fontsize=12, fontweight='bold')
            ax1.set_title('AttributeUserKNN: 邻居数量影响', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
        
        # 3.2 AttributeUserKNN - similarity影响
        ax2 = fig.add_subplot(gs[0, 1])
        if key in data:
            df = data[key].copy()
            df['similarity'] = df['model'].str.extract(r'sim=(\w+)')[0]
            
            sim_perf = df.groupby('similarity')[self.metrics].mean()
            
            x_pos = np.arange(len(sim_perf.index))
            width = 0.2
            
            for i, metric in enumerate(self.metrics):
                offset = width * (i - 1.5)
                ax2.bar(x_pos + offset, sim_perf[metric], width, 
                       label=metric, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            ax2.set_xlabel('相似度度量', fontsize=12, fontweight='bold')
            ax2.set_ylabel('性能指标值', fontsize=12, fontweight='bold')
            ax2.set_title('AttributeUserKNN: 相似度度量影响', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(sim_perf.index, fontsize=11)
            ax2.legend(fontsize=10, framealpha=0.9)
            ax2.grid(axis='y', alpha=0.3)
        
        # 3.3 AttributeUserKNN - profile影响
        ax3 = fig.add_subplot(gs[0, 2])
        if key in data:
            df = data[key].copy()
            df['profile'] = df['model'].str.extract(r'profile=(\w+)')[0]
            
            profile_perf = df.groupby('profile')[self.metrics].mean()
            
            x_pos = np.arange(len(profile_perf.index))
            width = 0.2
            
            for i, metric in enumerate(self.metrics):
                offset = width * (i - 1.5)
                ax3.bar(x_pos + offset, profile_perf[metric], width, 
                       label=metric, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            ax3.set_xlabel('用户画像表示', fontsize=12, fontweight='bold')
            ax3.set_ylabel('性能指标值', fontsize=12, fontweight='bold')
            ax3.set_title('AttributeUserKNN: 画像表示影响', fontsize=12, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(profile_perf.index, fontsize=11)
            ax3.legend(fontsize=10, framealpha=0.9)
            ax3.grid(axis='y', alpha=0.3)
        
        # 3.4 RP3beta - neighborhood影响
        ax4 = fig.add_subplot(gs[1, 0])
        key = 'RP3beta_cutoff_10'
        if key in data:
            df = data[key].copy()
            df['neighborhood'] = df['model'].str.extract(r'neighborhood=(\d+)')[0]
            df = df.dropna(subset=['neighborhood'])
            df['neighborhood'] = df['neighborhood'].astype(int)
            
            neighborhood_perf = df.groupby('neighborhood')[self.metrics].mean()
            
            for metric in self.metrics:
                ax4.plot(neighborhood_perf.index, neighborhood_perf[metric], 
                        marker='s', label=metric, linewidth=2, markersize=8)
            
            ax4.set_xlabel('邻域大小', fontsize=12, fontweight='bold')
            ax4.set_ylabel('性能指标值', fontsize=12, fontweight='bold')
            ax4.set_title('RP3beta: 邻域大小影响', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10, framealpha=0.9)
            ax4.grid(True, alpha=0.3)
        
        # 3.5 VSM - user_profile影响
        ax5 = fig.add_subplot(gs[1, 1])
        key = 'VSM_cutoff_10'
        if key in data:
            df = data[key].copy()
            df['user_profile'] = df['model'].str.extract(r'up=(\w+)')[0]
            
            up_perf = df.groupby('user_profile')[self.metrics].mean()
            
            x_pos = np.arange(len(up_perf.index))
            width = 0.2
            
            for i, metric in enumerate(self.metrics):
                offset = width * (i - 1.5)
                ax5.bar(x_pos + offset, up_perf[metric], width, 
                       label=metric, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            ax5.set_xlabel('用户画像类型', fontsize=12, fontweight='bold')
            ax5.set_ylabel('性能指标值', fontsize=12, fontweight='bold')
            ax5.set_title('VSM: 用户画像类型影响', fontsize=12, fontweight='bold')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(up_perf.index, fontsize=11)
            ax5.legend(fontsize=10, framealpha=0.9)
            ax5.grid(axis='y', alpha=0.3)
        
        # 3.6 所有算法配置数量分布
        ax6 = fig.add_subplot(gs[1, 2])
        config_counts = []
        model_names = []
        
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
            key = f'{model_type}_cutoff_10'
            if key in data:
                config_counts.append(len(data[key]))
                model_names.append(model_type)
        
        bars = ax6.barh(model_names, config_counts, color=self.colors[:len(model_names)], 
                       alpha=0.85, edgecolor='black', linewidth=0.5)
        
        for i, (bar, count) in enumerate(zip(bars, config_counts)):
            ax6.text(count, i, f'  {count}', va='center', fontsize=11, fontweight='bold')
        
        ax6.set_xlabel('配置数量', fontsize=12, fontweight='bold')
        ax6.set_ylabel('算法', fontsize=12, fontweight='bold')
        ax6.set_title('各算法超参数配置空间', fontsize=12, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
        
        plt.suptitle('超参数对推荐性能的影响分析', fontsize=16, fontweight='bold', y=0.995)
        
        output_file = self.output_dir / 'fig3_hyperparameter_impact_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def fig4_metric_correlation_heatmap(self, data: Dict[str, pd.DataFrame]):
        """图4：评价指标相关性热图"""
        print("\n生成图4：评价指标相关性分析...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, cutoff in enumerate(self.cutoffs):
            ax = axes[idx]
            
            # 收集所有模型的数据
            all_data = []
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
                key = f'{model_type}_cutoff_{cutoff}'
                if key in data:
                    all_data.append(data[key][self.metrics])
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                corr_matrix = combined.corr()
                
                # 绘制热图
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                           center=0.5, vmin=0, vmax=1, square=True, ax=ax,
                           cbar_kws={'label': '相关系数'},
                           annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                           linewidths=1, linecolor='white')
                
                ax.set_title(f'评价指标相关性 @{cutoff}', fontsize=13, fontweight='bold', pad=10)
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
        
        plt.suptitle('推荐系统评价指标相关性分析', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig4_metric_correlation_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def fig5_algorithm_performance_distribution(self, data: Dict[str, pd.DataFrame]):
        """图5：算法性能分布箱线图"""
        print("\n生成图5：算法性能分布分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            
            # 收集数据
            plot_data = []
            labels = []
            
            for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
                key = f'{model_type}_cutoff_10'
                if key in data and len(data[key]) > 0:
                    plot_data.append(data[key][metric].values)
                    labels.append(model_type)
            
            # 绘制箱线图
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=False,
                           meanprops=dict(marker='D', markerfacecolor='red', 
                                        markeredgecolor='red', markersize=8),
                           medianprops=dict(color='blue', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='gray', 
                                         markersize=5, alpha=0.5))
            
            # 着色
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric, fontsize=13, fontweight='bold')
            ax.set_xlabel('算法', fontsize=13, fontweight='bold')
            ax.set_title(f'{self.metric_names_cn[metric]} 性能分布', 
                        fontsize=13, fontweight='bold', pad=10)
            ax.set_xticklabels(labels, fontsize=10, rotation=20, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加图例说明
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='white', edgecolor='blue', linewidth=2, label='中位数'),
                    plt.Line2D([0], [0], marker='D', color='w', 
                              markerfacecolor='red', markersize=8, label='均值')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.suptitle('基于标签的推荐算法性能分布对比 (Top-10)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig5_algorithm_performance_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def fig6_best_model_radar_chart(self, data: Dict[str, pd.DataFrame]):
        """图6：最佳模型雷达图对比"""
        print("\n生成图6：最佳模型雷达图对比...")
        
        best_configs = self.get_best_config_per_model(data)
        
        # 只使用cutoff=10的数据
        best_10 = best_configs[best_configs['cutoff'] == 10]
        
        if len(best_10) == 0:
            print("  ⚠️  未找到cutoff=10的数据，跳过...")
            return
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(self.metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 归一化数据到0-1范围
        normalized_data = {}
        for metric in self.metrics:
            max_val = best_10[metric].max()
            min_val = best_10[metric].min()
            if max_val > min_val:
                normalized_data[metric] = (best_10[metric] - min_val) / (max_val - min_val)
            else:
                normalized_data[metric] = best_10[metric] / max_val if max_val > 0 else best_10[metric]
        
        # 绘制每个模型
        for idx, model_type in enumerate(['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']):
            model_data = best_10[best_10['model_type'] == model_type]
            
            if not model_data.empty:
                values = [normalized_data[metric].loc[model_data.index[0]] for metric in self.metrics]
                values += values[:1]  # 闭合
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=self.algorithm_types[model_type], 
                       color=self.colors[idx], markersize=8)
                ax.fill(angles, values, alpha=0.15, color=self.colors[idx])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.metrics, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax.set_title('最佳模型配置综合性能对比 (Top-10, 归一化)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig6_best_model_radar_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_file}")
        plt.close()
    
    def create_summary_table(self, data: Dict[str, pd.DataFrame]):
        """生成总结表格（CSV格式）"""
        print("\n生成总结表格...")
        
        best_configs = self.get_best_config_per_model(data)
        
        # 创建详细表格
        summary_rows = []
        
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
            row = {'算法': self.algorithm_types_long[model_type].replace('\n', ' ')}
            
            for cutoff in self.cutoffs:
                subset = best_configs[
                    (best_configs['model_type'] == model_type) & 
                    (best_configs['cutoff'] == cutoff)
                ]
                
                if not subset.empty:
                    for metric in self.metrics:
                        row[f'{metric}@{cutoff}'] = f"{subset[metric].values[0]:.4f}"
                    
                    # 添加最佳配置信息
                    if cutoff == 10:
                        model_name = subset['model'].values[0]
                        row['最佳配置'] = model_name
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # 保存CSV
        output_file = self.output_dir / 'summary_table_best_models.csv'
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 保存: {output_file}")
        
        # 创建简化的对比表
        comparison_rows = []
        for model_type in ['VSM', 'AttributeItemKNN', 'AttributeUserKNN', 'DeepFM', 'RP3beta']:
            for cutoff in self.cutoffs:
                subset = best_configs[
                    (best_configs['model_type'] == model_type) & 
                    (best_configs['cutoff'] == cutoff)
                ]
                
                if not subset.empty:
                    row = {
                        '算法': model_type,
                        'Top-K': cutoff,
                        'nDCG': f"{subset['nDCG'].values[0]:.4f}",
                        'Recall': f"{subset['Recall'].values[0]:.4f}",
                        'MAP': f"{subset['MAP'].values[0]:.4f}",
                        'MRR': f"{subset['MRR'].values[0]:.4f}"
                    }
                    comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        output_file = self.output_dir / 'summary_comparison_table.csv'
        comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 保存: {output_file}")
        
        # 打印最佳结果摘要
        print("\n" + "="*70)
        print("最佳模型性能摘要 (Top-10)")
        print("="*70)
        
        best_10 = best_configs[best_configs['cutoff'] == 10].sort_values('nDCG', ascending=False)
        
        for idx, row in best_10.iterrows():
            print(f"\n{idx+1}. {self.algorithm_types_long[row['model_type']].replace(chr(10), ' ')}")
            print(f"   nDCG: {row['nDCG']:.4f} | Recall: {row['Recall']:.4f} | "
                  f"MAP: {row['MAP']:.4f} | MRR: {row['MRR']:.4f}")
        
        print("\n" + "="*70)
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("\n" + "="*70)
        print("基于标签的推荐算法学术可视化生成")
        print("="*70)
        
        # 加载数据
        print("\n📊 加载实验数据...")
        data = self.load_all_performance_data()
        
        if not data:
            print("❌ 未找到数据，退出...")
            return
        
        print(f"\n✓ 成功加载 {len(data)} 个数据文件")
        
        # 生成所有图表
        print("\n" + "="*70)
        print("开始生成可视化图表...")
        print("="*70)
        
        self.fig1_algorithm_comparison_by_metric(data)
        self.fig2_cutoff_sensitivity_analysis(data)
        self.fig3_hyperparameter_impact(data)
        self.fig4_metric_correlation_heatmap(data)
        self.fig5_algorithm_performance_distribution(data)
        self.fig6_best_model_radar_chart(data)
        
        # 生成总结表格
        self.create_summary_table(data)
        
        # 完成
        print("\n" + "="*70)
        print(f"✅ 所有可视化已保存到: {self.output_dir.absolute()}")
        print("="*70)
        
        print("\n📁 生成的文件列表:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  • {file.name}")
        
        print("\n✨ 学术可视化生成完成！")
        print("="*70)


def main():
    """主函数"""
    visualizer = TagRecommendationVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
