"""Automated visualization suite for ILI forecasting results."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def plot_national_trajectories(forecasts: pd.DataFrame, truth: pd.DataFrame, output_dir: str, horizon: int = 4):
    """Plot median forecasts and truth for the national series."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for national and median
    df_plot = forecasts[forecasts['horizon'] == horizon].copy()
    df_plot = df_plot[np.isclose(df_plot['quantile'], 0.5)]
    
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=truth, x='target_date', y='true_value', label='Actual ILI', color='black', linewidth=2)
    
    # Plot top models
    for model in df_plot['model'].unique():
        m_df = df_plot[df_plot['model'] == model].sort_values('target_date')
        plt.plot(m_df['target_date'], m_df['value'], label=model, alpha=0.7)
        
    plt.title(f"National ILI Forecasts (Horizon: {horizon} weeks)")
    plt.xlabel("Date")
    plt.ylabel("Incidence")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"national_trajectories_h{horizon}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"national_trajectories_h{horizon}.pdf"))
    plt.close()

def plot_best_model_heatmap(metrics: pd.DataFrame, output_dir: str, metric_name: str = 'MAE'):
    """Heatmap of model performance across horizons."""
    os.makedirs(output_dir, exist_ok=True)
    
    pivot_df = metrics.pivot(index='model', columns='horizon', values=metric_name)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGn_r")
    plt.title(f"Model Performance Heatmap ({metric_name})")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"best_model_heatmap_{metric_name.lower()}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"best_model_heatmap_{metric_name.lower()}.pdf"))
    plt.close()

def plot_regional_performance(regional_metrics_dir: str, output_dir: str):
    """Aggregate and plot regional performance heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_reg_metrics = []
    for f in os.listdir(regional_metrics_dir):
        if f.endswith("_metrics.csv") and not f.endswith("_peak_metrics.csv") and not f.startswith("all_regions"):
            df = pd.read_csv(os.path.join(regional_metrics_dir, f))
            all_reg_metrics.append(df)
            
    if not all_reg_metrics:
        return
        
    full_df = pd.concat(all_reg_metrics, ignore_index=True)
    
    # Average across horizons for simplicity in the heatmap
    pivot_df = full_df.groupby(['region', 'model'])['MAE'].mean().unstack()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGn_r")
    plt.title("Average MAE by Region and Model")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "regional_performance_mae.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "regional_performance_mae.pdf"))
    plt.close()

def plot_best_model_map(regional_metrics_dir: str, output_dir: str):
    """Generates a synthetic grid map of Italy showing the best model per region."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_reg_metrics = []
    for f in os.listdir(regional_metrics_dir):
        if f.endswith("_metrics.csv") and not f.endswith("_peak_metrics.csv") and not f.startswith("best_model_per_region") and not f.startswith("all_regions"):
            df = pd.read_csv(os.path.join(regional_metrics_dir, f))
            all_reg_metrics.append(df)
            
    if not all_reg_metrics:
        return
        
    full_df = pd.concat(all_reg_metrics, ignore_index=True)
    
    # Identify the best model (lowest average MAE) for each region
    region_best = full_df.groupby(['region', 'model'])['MAE'].mean().reset_index()
    best_idx = region_best.groupby('region')['MAE'].idxmin()
    best_models = region_best.loc[best_idx].copy()
    
    # Save a CSV with best model per region as requested
    best_models.to_csv(os.path.join(regional_metrics_dir, "best_model_per_region.csv"), index=False)
    
    # Schematic grid map layout of Italy
    grid_coords = {
        'valle_d_aosta': (0, 0),
        'pa_bolzano': (0, 2),
        'friuli_venezia_giulia': (0, 4),
        'piemonte': (1, 0),
        'lombardia': (1, 1),
        'pa_trento': (1, 2),
        'veneto': (1, 3),
        'liguria': (2, 0),
        'emilia_romagna': (2, 2),
        'toscana': (3, 1),
        'umbria': (3, 2),
        'marche': (3, 3),
        'sardegna': (4, 0),
        'lazio': (4, 2),
        'abruzzo': (4, 3),
        'molise': (5, 3),
        'campania': (5, 2),
        'puglia': (5, 4),
        'basilicata': (6, 3),
        'calabria': (7, 3),
        'sicilia': (8, 1)
    }
    
    rows = 9
    cols = 5
    grid_data = np.full((rows, cols), np.nan)
    grid_labels = np.full((rows, cols), "", dtype=object)
    
    # Unique models for coloring
    unique_models = sorted(best_models['model'].unique())
    model_to_idx = {model: idx for idx, model in enumerate(unique_models)}
    
    for _, row in best_models.iterrows():
        reg = row['region']
        model = row['model']
        if reg in grid_coords:
            r, c = grid_coords[reg]
            grid_data[r, c] = model_to_idx[model]
            grid_labels[r, c] = f"{reg.replace('_', ' ').title()}\n({model})"
            
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    # Use distinct colors from qualitative map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    cmap = ListedColormap(colors)
    
    # Custom drawing with patches to show the grid of regions
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5) # Invert y-axis to match coordinate grid
    
    for r in range(rows):
        for c in range(cols):
            val = grid_data[r, c]
            if not np.isnan(val):
                color = colors[int(val)]
                rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                ax.text(c, r, grid_labels[r, c], ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            else:
                rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#f8f9fa', edgecolor='#e9ecef', linewidth=1)
                ax.add_patch(rect)
                
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title("Synthetic Map: Best Model per Region (Lowest Average MAE)", fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend_patches = [mpatches.Patch(color=colors[idx], label=model) for model, idx in model_to_idx.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_model_map.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "best_model_map.pdf"), bbox_inches='tight')
    plt.close()
