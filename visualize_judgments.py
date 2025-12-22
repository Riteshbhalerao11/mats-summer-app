#!/usr/bin/env python3
"""
Visualize judgment outputs across different vec types, datasets, and configurations.

This script loads judgment data from the judge_outputs directory and generates
production-grade plots comparing different steering approaches.

Usage:
    python visualize_judgments.py --judge-model openai_gpt-oss-120b --model llama
    python visualize_judgments.py --dataset wildjb_benign --output-dir my_plots/
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for production-grade plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

# Color-blind friendly palette
COLORS = sns.color_palette("colorblind", 10)
VEC_TYPE_COLORS = {
    'control': COLORS[0],
    'refusal': COLORS[1],
    'trust_complete': COLORS[2],
    'trust_orthogonal': COLORS[3],
}


def load_judgment_data(base_dir: Path, judge_model: Optional[str] = None,
                       model: Optional[str] = None, dataset: Optional[str] = None,
                       vec_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load all judgment YAML files from the directory structure.
    
    Args:
        base_dir: Base directory containing judge_outputs
        judge_model: Filter by judge model (e.g., 'openai_gpt-oss-120b')
        model: Filter by model (e.g., 'llama')
        dataset: Filter by dataset (e.g., 'wildjb_benign', 'harmful')
        vec_types: Filter by vec types (e.g., ['control', 'refusal'])
    
    Returns:
        DataFrame with all judgment data
    """
    records = []
    
    # Find all judgment YAML files
    pattern = "*_judgment.yaml"
    yaml_files = list(base_dir.rglob(pattern))
    
    print(f"Found {len(yaml_files)} judgment files")
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Apply filters
            if judge_model and data.get('judge_model', '').replace('/', '_').replace(':', '_') != judge_model:
                continue
            if model and data.get('model') != model:
                continue
            if dataset and data.get('dataset') != dataset:
                continue
            if vec_types and data.get('vec_type') not in vec_types:
                continue
            
            # Extract judgment fields
            judgment = data.get('judgment', {})
            
            record = {
                'judge_model': data.get('judge_model'),
                'model': data.get('model'),
                'vec_type': data.get('vec_type'),
                'dataset': data.get('dataset'),
                'layer_id': data.get('layer_id'),
                'strength': data.get('strength'),
                'prompt_id': data.get('prompt_id'),
                'category': data.get('category'),
                'file_path': str(yaml_file),
            }
            
            # Add all judgment fields
            record.update(judgment)
            
            records.append(record)
        
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} judgments after filtering")
    
    return df


def is_benign_dataset(dataset_name: str) -> bool:
    """Check if dataset is benign based on name."""
    return 'benign' in dataset_name.lower()


def get_metrics_for_dataset(dataset_name: str) -> List[str]:
    """Get relevant metrics based on dataset type."""
    if is_benign_dataset(dataset_name):
        return ['is_refusal', 'is_faithful_answer']
    else:
        return ['is_refusal', 'contains_restricted_content']


def create_comparison_plots(df: pd.DataFrame, output_dir: Path, file_format: str = 'png'):
    """
    Create comparison plots showing metrics across vec types for each dataset.
    Aggregates all layer-strength configurations for each vec type.
    
    Args:
        df: DataFrame with judgment data
        output_dir: Directory to save plots
        file_format: 'png' or 'pdf' (default: 'png')
    """
    # Group by dataset
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        metrics = get_metrics_for_dataset(dataset)
        
        # Get all vec types present in this dataset
        vec_types_present = sorted(dataset_df['vec_type'].unique())
        
        if len(vec_types_present) == 0:
            continue
        
        # Create plot with subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Calculate metric rates by vec_type (aggregating across all configs)
            metric_data = []
            
            for vec_type in vec_types_present:
                vec_df = dataset_df[dataset_df['vec_type'] == vec_type]
                if metric in vec_df.columns:
                    rate = vec_df[metric].mean() * 100  # Convert to percentage
                    count = len(vec_df)
                    metric_data.append({
                        'vec_type': vec_type,
                        'rate': rate,
                        'count': count
                    })
            
            if not metric_data:
                continue
            
            metric_df = pd.DataFrame(metric_data)
            
            # Create bar plot
            bars = ax.bar(
                metric_df['vec_type'],
                metric_df['rate'],
                color=[VEC_TYPE_COLORS.get(vt, COLORS[4]) for vt in metric_df['vec_type']],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels on bars
            for bar, row in zip(bars, metric_data):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 1,
                    f'{height:.1f}%\n(n={row["count"]})',
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold'
                )
            
            # Formatting
            ax.set_ylabel('Rate (%)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Vec Type', fontsize=13, fontweight='bold')
            ax.set_title(
                metric.replace('_', ' ').title(),
                fontsize=15,
                fontweight='bold',
                pad=15
            )
            ax.set_ylim(0, 110)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Rotate x-axis labels if needed
            if len(vec_types_present) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        # Overall title
        fig.suptitle(
            f'Vec Type Comparison - {dataset}',
            fontsize=18,
            fontweight='bold',
            y=1.00
        )
        
        plt.tight_layout()
        
        # Save plot
        safe_dataset = dataset.replace('/', '_')
        filename = f'vec_comparison_{safe_dataset}'
        
        if file_format in ['png', 'both']:
            plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        if file_format in ['pdf', 'both']:
            plt.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
        
        plt.close()
        print(f"Saved: {filename}.{file_format}")



def create_category_breakdown(df: pd.DataFrame, output_dir: Path, file_format: str = 'png'):
    """
    Create category breakdown plots if category data exists.
    Shows all vec types compared across categories for each dataset.
    
    Args:
        df: DataFrame with judgment data
        output_dir: Directory to save plots
        file_format: 'png' or 'pdf' (default: 'png')
    """
    # Check if category column exists and has non-null values
    if 'category' not in df.columns or df['category'].isna().all():
        print("No category data found, skipping category breakdown plots")
        return
    
    # Filter out rows without categories
    df_with_cat = df[df['category'].notna()]
    
    if len(df_with_cat) == 0:
        print("No category data found, skipping category breakdown plots")
        return
    
    # Group by dataset
    datasets = df_with_cat['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df_with_cat[df_with_cat['dataset'] == dataset]
        metrics = get_metrics_for_dataset(dataset)
        
        vec_types_present = sorted(dataset_df['vec_type'].unique())
        categories = sorted(dataset_df['category'].unique())
        
        # Create plot for each metric
        for metric in metrics:
            if metric not in dataset_df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.5), 6))
            
            # Calculate rates by category and vec_type (aggregating all configs)
            data_matrix = np.zeros((len(vec_types_present), len(categories)))
            counts_matrix = np.zeros((len(vec_types_present), len(categories)))
            
            for i, vec_type in enumerate(vec_types_present):
                for j, category in enumerate(categories):
                    subset = dataset_df[
                        (dataset_df['vec_type'] == vec_type) & 
                        (dataset_df['category'] == category)
                    ]
                    if len(subset) > 0:
                        data_matrix[i, j] = subset[metric].mean() * 100
                        counts_matrix[i, j] = len(subset)
            
            # Create grouped bar chart
            x = np.arange(len(categories))
            width = 0.8 / len(vec_types_present)
            
            for i, vec_type in enumerate(vec_types_present):
                offset = (i - len(vec_types_present) / 2) * width + width / 2
                bars = ax.bar(
                    x + offset,
                    data_matrix[i],
                    width,
                    label=vec_type,
                    color=VEC_TYPE_COLORS.get(vec_type, COLORS[4]),
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1.0
                )
            
            # Formatting
            ax.set_ylabel('Rate (%)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Category', fontsize=13, fontweight='bold')
            ax.set_title(
                f'{metric.replace("_", " ").title()} by Category - {dataset}',
                fontsize=15,
                fontweight='bold',
                pad=15
            )
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_ylim(0, 110)
            ax.legend(title='Vec Type', fontsize=11, title_fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save plot
            safe_dataset = dataset.replace('/', '_')
            safe_metric = metric.replace('/', '_')
            filename = f'category_{safe_dataset}_{safe_metric}'
            
            if file_format in ['png', 'both']:
                plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
            if file_format in ['pdf', 'both']:
                plt.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
            
            plt.close()
            print(f"Saved: {filename}.{file_format}")



def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """
    Create a summary statistics table aggregated by vec_type.
    
    Args:
        df: DataFrame with judgment data
        output_dir: Directory to save table
    """
    summary_data = []
    
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        metrics = get_metrics_for_dataset(dataset)
        
        # Get all unique vec_types
        vec_types = sorted(dataset_df['vec_type'].unique())
        
        for vec_type in vec_types:
            vec_df = dataset_df[dataset_df['vec_type'] == vec_type]
            
            if len(vec_df) == 0:
                continue
            
            row = {
                'dataset': dataset,
                'vec_type': vec_type,
                'n_samples': len(vec_df)
            }
            
            # Add metrics
            for metric in metrics:
                if metric in vec_df.columns:
                    rate = vec_df[metric].mean() * 100
                    row[f'{metric}_rate'] = f'{rate:.1f}%'
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary table: {csv_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)



def main():
    parser = argparse.ArgumentParser(
        description='Visualize judgment outputs across vec types and configurations'
    )
    parser.add_argument('--judge-model', type=str, default=None,
                        help='Filter by judge model (e.g., openai_gpt-oss-120b)')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model (e.g., llama)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset (e.g., wildjb_benign, harmful)')
    parser.add_argument('--vec-type', type=str, nargs='+', default=None,
                        help='Filter by vec types (e.g., control refusal)')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots/)')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'both'], default='png',
                        help='Output format (default: png)')
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path('judge_outputs')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("JUDGMENT VISUALIZATION")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    if args.judge_model:
        print(f"Judge model filter: {args.judge_model}")
    if args.model:
        print(f"Model filter: {args.model}")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")
    if args.vec_type:
        print(f"Vec type filter: {args.vec_type}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading judgment data...")
    df = load_judgment_data(
        base_dir,
        judge_model=args.judge_model,
        model=args.model,
        dataset=args.dataset,
        vec_types=args.vec_type
    )
    
    if len(df) == 0:
        print("No data found matching filters. Exiting.")
        return 1
    
    print(f"\nDatasets: {df['dataset'].unique().tolist()}")
    print(f"Vec types: {df['vec_type'].unique().tolist()}")
    print(f"Models: {df['model'].unique().tolist()}")
    print()
    
    # Generate plots
    print("Generating comparison plots...")
    create_comparison_plots(df, output_dir, args.format)
    
    print("\nGenerating category breakdown plots...")
    create_category_breakdown(df, output_dir, args.format)
    
    print("\nGenerating summary table...")
    create_summary_table(df, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Plots saved to: {output_dir}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())
