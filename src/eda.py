"""
Exploratory Data Analysis (EDA) Module
SyntecxHub Machine Learning Internship Project - 1
Constraints addressed:
- Explore housing features
- Analyze correlation between features and House Price
- Generate and save professional visualization charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data_path=None, output_plots_dir=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_housing_data.csv')
    if output_plots_dir is None:
        output_plots_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'plots')
        
    os.makedirs(output_plots_dir, exist_ok=True)
    
    print(f"[*] Reading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Statistical Summary
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS - DESCRIPTIVE STATISTICS")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().round(2)
    print(stats)
    
    # 2. Correlation with Price
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS WITH TARGET (PRICE)")
    print("=" * 60)
    correlations = df[numeric_cols].corr()['Price'].sort_values(ascending=False)
    for col, corr in correlations.items():
        if col != 'Price':
            print(f"  * {col:<22}: {corr:+.4f}")
            
    # Set overall aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'figure.dpi': 150})
    
    # Plot 1: Price Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Price'], kde=True, color='#2563eb', bins=30, ax=ax, edgecolor='white', alpha=0.7)
    mean_val = df['Price'].mean()
    median_val = df['Price'].median()
    ax.axvline(mean_val, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Mean: ${mean_val:,.0f}')
    ax.axvline(median_val, color='#16a34a', linestyle='-', linewidth=1.5, label=f'Median: ${median_val:,.0f}')
    ax.set_title('House Price Distribution (Target Variable)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Price ($ USD)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    ax.legend()
    plt.tight_layout()
    p1 = os.path.join(output_plots_dir, 'price_distribution.png')
    fig.savefig(p1)
    plt.close()
    print(f"[*] Saved Price Distribution plot to: {p1}")
    
    # Plot 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    p2 = os.path.join(output_plots_dir, 'correlation_heatmap.png')
    fig.savefig(p2)
    plt.close()
    print(f"[*] Saved Correlation Heatmap to: {p2}")
    
    # Plot 3: Square Feet vs Price (Scatter with Regression Line)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='Square_Feet',
        y='Price',
        scatter_kws={'alpha': 0.4, 'color': '#2563eb', 's': 20},
        line_kws={'color': '#dc2626', 'linewidth': 2},
        ax=ax
    )
    ax.set_title('Living Area (Square Feet) vs House Price', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Square Feet', fontsize=11)
    ax.set_ylabel('Price ($ USD)', fontsize=11)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    plt.tight_layout()
    p3 = os.path.join(output_plots_dir, 'sqft_vs_price.png')
    fig.savefig(p3)
    plt.close()
    print(f"[*] Saved Sqft vs Price scatter plot to: {p3}")
    
    # Plot 4: Overall Quality vs Price Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x='Overall_Quality',
        y='Price',
        palette='Blues_r',
        ax=ax
    )
    ax.set_title('House Price by Overall Quality Score (1-10)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Overall Quality Score', fontsize=11)
    ax.set_ylabel('Price ($ USD)', fontsize=11)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    plt.tight_layout()
    p4 = os.path.join(output_plots_dir, 'quality_vs_price.png')
    fig.savefig(p4)
    plt.close()
    print(f"[*] Saved Quality vs Price boxplot to: {p4}")

    return correlations

if __name__ == '__main__':
    perform_eda()
