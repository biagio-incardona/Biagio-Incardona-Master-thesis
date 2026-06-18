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
        if f.endswith("_metrics.csv"):
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
