import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Set default style
plt.style.use('default')
sns.set_palette("husl")

def plot_missing_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a heatmap showing missing value patterns.

    Args:
        df: Input DataFrame
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create missing value matrix
    missing_matrix = df.isnull()

    # Only show columns with missing values
    cols_with_missing = missing_matrix.columns[missing_matrix.any()].tolist()

    if len(cols_with_missing) == 0:
        ax.text(0.5, 0.5, 'No Missing Values Found!',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Create heatmap
        sns.heatmap(missing_matrix[cols_with_missing],
                   yticklabels=False,
                   cbar_kws={'label': 'Missing Values'},
                   cmap='viridis',
                   ax=ax)

        ax.set_title('Missing Value Patterns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Rows', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def plot_missing_bar(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create a bar chart showing missing value percentages by column.

    Args:
        df: Input DataFrame
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    # Only show columns with missing values
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        ax.text(0.5, 0.5, 'No Missing Values Found!',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        bars = ax.bar(range(len(missing_pct)), missing_pct.values,
                     color='lightcoral', alpha=0.7)

        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Missing Percentage (%)', fontsize=12)
        ax.set_xticks(range(len(missing_pct)))
        ax.set_xticklabels(missing_pct.index, rotation=45, ha='right')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig

def plot_data_types(df: pd.DataFrame, figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Create a pie chart showing data type distribution.

    Args:
        df: Input DataFrame
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Count data types
    dtype_counts = df.dtypes.value_counts()

    # Simplify dtype names
    dtype_mapping = {
        'object': 'Text/Categorical',
        'int64': 'Integer',
        'float64': 'Float',
        'bool': 'Boolean',
        'datetime64[ns]': 'DateTime'
    }

    simplified_names = [dtype_mapping.get(str(dtype), str(dtype)) for dtype in dtype_counts.index]

    colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))

    wedges, texts, autotexts = ax.pie(dtype_counts.values,
                                     labels=simplified_names,
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)

    ax.set_title('Data Type Distribution', fontsize=14, fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    return fig

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a correlation heatmap with annotations.

    Args:
        corr_matrix: Correlation matrix
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Generate heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                fmt='.2f',
                ax=ax)

    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_outliers_boxplot(df: pd.DataFrame, columns: List[str], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create box plots for outlier visualization.

    Args:
        df: Input DataFrame
        columns: List of numeric columns to plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Numeric Columns Available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        return fig

    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]

            # Create box plot
            bp = ax.boxplot(df[col].dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)

            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Values')
            ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_distribution_grid(df: pd.DataFrame, numeric_cols: List[str], figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create a grid of distribution plots for numeric columns.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Numeric Columns Available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        return fig

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]

            # Plot histogram with KDE
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)

                # Add KDE if enough data points
                if len(data) > 5:
                    try:
                        sns.kdeplot(data=data, ax=ax, color='red', linewidth=2)
                    except:
                        pass

            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Distribution of Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_categorical_bars(df: pd.DataFrame, cat_cols: List[str], max_categories: int = 10, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create bar charts for categorical columns.

    Args:
        df: Input DataFrame
        cat_cols: List of categorical columns
        max_categories: Maximum number of categories to show per column
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not cat_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Categorical Columns Available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        return fig

    n_cols = min(2, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        if i < len(axes):
            ax = axes[i]

            # Get value counts and limit to top categories
            value_counts = df[col].value_counts().head(max_categories)

            if len(value_counts) > 0:
                bars = ax.bar(range(len(value_counts)), value_counts.values,
                             color='lightgreen', alpha=0.7)

                ax.set_title(f'{col}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values) * 0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)

            ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(cat_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Distribution of Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def format_percentage(value: float) -> str:
    """
    Format a percentage value for display.

    Args:
        value: Percentage value (0-100)

    Returns:
        Formatted percentage string
    """
    if value == 0:
        return "0%"
    elif value < 0.1:
        return "<0.1%"
    elif value < 1:
        return f"{value:.1f}%"
    else:
        return f"{value:.0f}%"

def create_quality_score_gauge(score: int, interpretation: str) -> go.Figure:
    """
    Create a gauge chart for quality score using Plotly.

    Args:
        score: Quality score (0-100)
        interpretation: Score interpretation

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Data Quality Score"},
        delta = {'reference': 75},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 75], 'color': "orange"},
                {'range': [75, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        font={'size': 16},
        height=400,
        annotations=[
            dict(
                x=0.5, y=0.1,
                text=f"<b>{interpretation}</b>",
                showarrow=False,
                font=dict(size=20)
            )
        ]
    )

    return fig

def plot_missing_pattern_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Create a matrix plot showing missing value patterns across rows.

    Args:
        df: Input DataFrame

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        ax.text(0.5, 0.5, 'No Missing Values Found!',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    # Create missing pattern matrix (sample first 1000 rows for performance)
    sample_size = min(1000, len(df))
    sample_df = df[missing_cols].sample(n=sample_size, random_state=42)

    # Convert to binary matrix (1 = missing, 0 = present)
    missing_matrix = sample_df.isnull().astype(int)

    # Plot matrix
    im = ax.imshow(missing_matrix.T, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')

    ax.set_title('Missing Value Patterns (Sample)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Features')
    ax.set_yticks(range(len(missing_cols)))
    ax.set_yticklabels(missing_cols)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Missing (1) / Present (0)')

    plt.tight_layout()
    return fig

def plot_feature_distributions(df: pd.DataFrame, columns: List[str], max_cols: int = 6) -> plt.Figure:
    """
    Create distribution plots for selected features.

    Args:
        df: Input DataFrame
        columns: List of columns to plot
        max_cols: Maximum number of columns to plot

    Returns:
        matplotlib Figure object
    """
    columns = columns[:max_cols]  # Limit number of plots

    if not columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Columns Selected',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        return fig

    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]

            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column - histogram with KDE
                data = df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.7, color='skyblue', density=True, edgecolor='black')

                    # Add KDE if enough data points
                    if len(data) > 5:
                        try:
                            sns.kdeplot(data=data, ax=ax, color='red', linewidth=2)
                        except:
                            pass

                    # Add vertical lines for mean and median
                    mean_val = data.mean()
                    median_val = data.median()
                    ax.axvline(mean_val, color='orange', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                    ax.legend()

            else:
                # Categorical column - bar chart
                value_counts = df[col].value_counts().head(10)  # Top 10 categories
                if len(value_counts) > 0:
                    bars = ax.bar(range(len(value_counts)), value_counts.values,
                                 color='lightcoral', alpha=0.7, edgecolor='black')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

                    # Add value labels on bars
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values) * 0.01,
                               f'{int(height)}', ha='center', va='bottom', fontsize=9)

            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_correlation_network(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> plt.Figure:
    """
    Create a network plot showing correlations above threshold.

    Args:
        corr_matrix: Correlation matrix
        threshold: Minimum correlation to show

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Find correlations above threshold
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corrs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    if not high_corrs:
        ax.text(0.5, 0.5, f'No correlations above {threshold} found',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        return fig

    # Create simple network visualization
    features = list(set([item['feature1'] for item in high_corrs] + [item['feature2'] for item in high_corrs]))

    # Position features in a circle
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    positions = {feature: (np.cos(angle), np.sin(angle)) for feature, angle in zip(features, angles)}

    # Draw connections
    for corr_info in high_corrs:
        f1, f2, corr = corr_info['feature1'], corr_info['feature2'], corr_info['correlation']
        x1, y1 = positions[f1]
        x2, y2 = positions[f2]

        # Color and width based on correlation strength
        color = 'red' if corr > 0 else 'blue'
        width = abs(corr) * 3
        alpha = min(abs(corr), 0.8)

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha)

    # Draw feature points
    for feature, (x, y) in positions.items():
        ax.scatter(x, y, s=200, c='lightgreen', edgecolor='black', zorder=3)
        ax.text(x + 0.1, y + 0.1, feature, fontsize=10, fontweight='bold')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Correlation Network (|r| ≥ {threshold})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    ax.text(0.02, 0.98, 'Red: Positive correlation\nBlue: Negative correlation\nThicker line: Stronger correlation',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig

def plot_pairwise_relationships(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None) -> plt.Figure:
    """
    Create scatter plot showing relationship between two variables.

    Args:
        df: Input DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        hue_col: Optional column for color coding

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if hue_col and hue_col in df.columns:
        # Colored by hue column
        unique_values = df[hue_col].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))

        for i, value in enumerate(unique_values):
            mask = df[hue_col] == value
            ax.scatter(df[mask][x_col], df[mask][y_col],
                      c=[colors[i]], label=str(value), alpha=0.6, s=50)

        ax.legend()
    else:
        # Simple scatter plot
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='blue')

    # Add trend line if both are numeric
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        # Calculate correlation
        correlation = df[x_col].corr(df[y_col])

        # Add trend line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2)

        # Add correlation to title
        ax.set_title(f'{y_col} vs {x_col}\nCorrelation: {correlation:.3f}',
                    fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig