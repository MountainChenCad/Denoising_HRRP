# visualization.py - Visualization tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Define color scheme for high-quality plotting
COLORS = {
    'noisy': '#7F7F7F',  # Gray
    'cgan': '#1F77B4',   # Blue
    'cae': '#2CA02C',    # Green
    'ae': '#D62728',     # Red
    'clean': '#000000',  # Black
    'background': '#F5F5F5',  # Light gray background
    'grid': '#E0E0E0',   # Grid line color
    'highlight': '#FF7F0E',  # Highlight color
    'text': '#333333'    # Text color
}


# Set high-quality plotting default parameters
def set_plot_style():
    """Set high-quality plotting parameters"""
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.facecolor'] = '#FAFAFA'
    plt.rcParams['figure.facecolor'] = '#FFFFFF'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'


def plot_single_hrrp(signal, title=None, ax=None, color='black', linestyle='-', linewidth=1.5, alpha=1.0,
                     xlabel='Range Bin', ylabel='Magnitude', grid=True, legend=False, label=None):
    """
    Plot a single HRRP signal

    Parameters:
        signal (numpy.ndarray): HRRP signal data
        title (str): Chart title
        ax (matplotlib.axes.Axes): Axes object to plot on, creates new one if None
        color (str): Line color
        linestyle (str): Line style
        linewidth (float): Line width
        alpha (float): Line transparency
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        grid (bool): Whether to display grid
        legend (bool): Whether to display legend
        label (str): Legend label

    Returns:
        matplotlib.axes.Axes: Plot axes object
    """
    # Ensure input is numpy array
    if not isinstance(signal, np.ndarray):
        if hasattr(signal, 'cpu') and hasattr(signal, 'numpy'):
            # PyTorch tensor
            signal = signal.cpu().numpy()
        else:
            signal = np.array(signal)

    # If batch data, use only the first sample
    if signal.ndim > 1 and signal.shape[0] == 1:
        signal = signal[0]

    # Create axes object (if needed)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Plot data
    ax.plot(signal, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)

    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set grid and legend
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)

    if legend and label:
        ax.legend()

    return ax


def compare_signals(clean, noisy, denoised, psnr=None, ssim=None, labels=None, title=None,
                    figsize=(15, 5), save_path=None, separate=True):
    """
    Compare clean, noisy and denoised signals

    Parameters:
        clean (numpy.ndarray): Clean signal
        noisy (numpy.ndarray): Noisy signal
        denoised (numpy.ndarray or dict): Denoised signal or dictionary of results from multiple methods
        psnr (float or dict): PSNR value (single value or dictionary for multiple methods)
        ssim (float or dict): SSIM value (single value or dictionary for multiple methods)
        labels (dict): Signal label dictionary, e.g., {'clean': 'Clean', 'noisy': 'Noisy'}
        title (str): Main title
        figsize (tuple): Figure size
        save_path (str): Save path, saves image if not None
        separate (bool): Whether to plot each signal separately

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # Handle labels
    if labels is None:
        labels = {'clean': 'Clean', 'noisy': 'Noisy'}

    # Handle multiple denoising results
    multi_denoised = isinstance(denoised, dict)

    # Determine subplot layout
    if separate:
        if multi_denoised:
            n_plots = 2 + len(denoised)  # Clean + Noisy + Each denoising method
        else:
            n_plots = 3  # Clean + Noisy + Denoised
    else:
        n_plots = 1  # All signals on one plot

    # Create figure and subplot layout
    fig = plt.figure(figsize=figsize)

    if separate:
        # Plot each signal separately
        # Plot clean signal
        ax1 = plt.subplot(1, n_plots, 1)
        plot_single_hrrp(clean, title=labels.get('clean', 'Clean'),
                         color=COLORS['clean'], ax=ax1)

        # Plot noisy signal
        ax2 = plt.subplot(1, n_plots, 2)
        noisy_title = labels.get('noisy', 'Noisy')
        if psnr is not None and not isinstance(psnr, dict):
            noisy_title += f" (PSNR: {psnr:.2f}dB)"
        plot_single_hrrp(noisy, title=noisy_title, color=COLORS['noisy'], ax=ax2)

        # Plot denoised signal(s)
        if multi_denoised:
            for i, (method, signal) in enumerate(denoised.items(), start=3):
                ax = plt.subplot(1, n_plots, i)
                method_title = labels.get(method, method)

                # Add PSNR/SSIM information (if provided)
                if isinstance(psnr, dict) and method in psnr:
                    method_title += f"\nPSNR: {psnr[method]:.2f}dB"
                if isinstance(ssim, dict) and method in ssim:
                    method_title += f", SSIM: {ssim[method]:.4f}"

                # Choose color
                color = COLORS.get(method.lower(), COLORS['highlight'])

                plot_single_hrrp(signal, title=method_title, color=color, ax=ax)
        else:
            ax3 = plt.subplot(1, n_plots, 3)
            denoised_title = labels.get('denoised', 'Denoised')

            # Add PSNR/SSIM information (if provided)
            if psnr is not None and not isinstance(psnr, dict):
                denoised_title += f"\nPSNR: {psnr:.2f}dB"
            if ssim is not None and not isinstance(ssim, dict):
                denoised_title += f", SSIM: {ssim:.4f}"

            plot_single_hrrp(denoised, title=denoised_title, color=COLORS['cgan'], ax=ax3)

    else:
        # Plot all signals on one plot
        ax = plt.subplot(1, 1, 1)

        # Plot clean signal
        plot_single_hrrp(clean, color=COLORS['clean'], label=labels.get('clean', 'Clean'), ax=ax, legend=True)

        # Plot noisy signal
        noisy_label = labels.get('noisy', 'Noisy')
        if psnr is not None and not isinstance(psnr, dict):
            noisy_label += f" (PSNR: {psnr:.2f}dB)"
        plot_single_hrrp(noisy, color=COLORS['noisy'], label=noisy_label, ax=ax, legend=True, alpha=0.7)

        # Plot denoised signal(s)
        if multi_denoised:
            for method, signal in denoised.items():
                method_label = labels.get(method, method)

                # Add PSNR/SSIM information (if provided)
                if isinstance(psnr, dict) and method in psnr:
                    method_label += f" (PSNR: {psnr[method]:.2f}dB"
                    if isinstance(ssim, dict) and method in ssim:
                        method_label += f", SSIM: {ssim[method]:.4f})"
                    else:
                        method_label += ")"

                # Choose color
                color = COLORS.get(method.lower(), COLORS['highlight'])

                plot_single_hrrp(signal, color=color, label=method_label, ax=ax, legend=True)
        else:
            denoised_label = labels.get('denoised', 'Denoised')

            # Add PSNR/SSIM information (if provided)
            if psnr is not None and not isinstance(psnr, dict):
                denoised_label += f" (PSNR: {psnr:.2f}dB"
                if ssim is not None and not isinstance(ssim, dict):
                    denoised_label += f", SSIM: {ssim:.4f})"
                else:
                    denoised_label += ")"

            plot_single_hrrp(denoised, color=COLORS['cgan'], label=denoised_label, ax=ax, legend=True)

        # Set legend
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)

    # Set overall title
    if title is not None:
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85)  # Leave space for title

    plt.tight_layout()

    # Save image
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_bar(metrics, metric_name, models=None, title=None, figsize=(10, 6),
                     higher_is_better=True, improvement=False, ref_value=None,
                     save_path=None, show_values=True, colors=None):
    """
    Plot bar chart of performance metrics

    Parameters:
        metrics (dict or list): Dictionary or list containing metrics for each model
        metric_name (str): Name of the metric to plot
        models (list): List of model names, if None it's extracted from metrics
        title (str): Chart title
        figsize (tuple): Figure size
        higher_is_better (bool): Whether higher values are better for this metric
        improvement (bool): Whether to show improvement relative to reference value
        ref_value (float): Reference value
        save_path (str): Save path
        show_values (bool): Whether to display values on bars
        colors (dict): Dictionary mapping models to colors

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # Handle input data format
    if isinstance(metrics, list):
        if models is None:
            raise ValueError("When metrics is a list, models must be provided")
        metrics_dict = {model: value for model, value in zip(models, metrics)}
    else:
        metrics_dict = metrics
        if models is None:
            models = list(metrics_dict.keys())

    # If no colors specified, use default colors
    if colors is None:
        colors = {model: COLORS.get(model.lower(), COLORS['highlight']) for model in models}

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    if improvement and ref_value is not None:
        values = [metrics_dict[model] - ref_value for model in models]
        if not higher_is_better:
            values = [-val for val in values]
    else:
        values = [metrics_dict[model] for model in models]

    # Draw bar chart
    bar_colors = [colors.get(model, COLORS['highlight']) for model in models]
    bars = ax.bar(models, values, color=bar_colors, edgecolor='black', linewidth=1)

    # Set title and labels
    if title is None:
        if improvement:
            title = f"{metric_name} Improvement"
        else:
            title = f"{metric_name} Comparison"

    ax.set_title(title)
    ax.set_xlabel('Models')

    if improvement:
        ax.set_ylabel(f"{metric_name} Improvement")
    else:
        ax.set_ylabel(metric_name)

    # Set grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels
    if show_values:
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + 0.01 * max(abs(max(values)), abs(min(values)))
            else:
                va = 'top'
                y_pos = height - 0.01 * max(abs(max(values)), abs(min(values)))

            if improvement:
                label = f"{height:.2f}"
                if height > 0:
                    label = "+" + label
            else:
                label = f"{height:.2f}"

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha='center', va=va, rotation=0)

    # Save chart
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_comparison(metrics_dict, psnr_levels, models=None, metric='psnr',
                            figsize=(12, 6), title=None, save_path=None, improvement=False):
    """
    Plot performance metrics comparison across different PSNR levels

    Parameters:
        metrics_dict (dict): Nested dictionary containing metrics for each model at each PSNR level
            Format: {psnr_level: {model: {metric: value}}}
        psnr_levels (list): List of PSNR levels
        models (list): List of model names, if None it's extracted automatically
        metric (str): Name of metric to plot, e.g., 'psnr', 'ssim', 'mse'
        figsize (tuple): Figure size
        title (str): Chart title
        save_path (str): Save path
        improvement (bool): Whether to show improvement relative to noise

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # If no models specified, extract all models from the first PSNR level
    if models is None:
        first_level = list(metrics_dict.keys())[0]
        models = [model for model in metrics_dict[first_level].keys()
                  if model != 'noisy' and not isinstance(metrics_dict[first_level][model], dict)]

    # Extract data
    data = {}
    ref_data = {}

    for model in models:
        data[model] = []

    for psnr in psnr_levels:
        # If noisy data exists as reference
        if 'noisy' in metrics_dict[psnr]:
            if metric in metrics_dict[psnr]['noisy']:
                ref_data[psnr] = metrics_dict[psnr]['noisy'][metric]
            elif 'metrics' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy']['metrics']:
                ref_data[psnr] = metrics_dict[psnr]['noisy']['metrics'][metric]

        # Collect data for each model
        for model in models:
            if model in metrics_dict[psnr]:
                if metric in metrics_dict[psnr][model]:
                    data[model].append(metrics_dict[psnr][model][metric])
                elif 'metrics' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['metrics']:
                    data[model].append(metrics_dict[psnr][model]['metrics'][metric])
                elif 'averages' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['averages']:
                    data[model].append(metrics_dict[psnr][model]['averages'][metric])
                else:
                    data[model].append(None)
            else:
                data[model].append(None)

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    # Draw grouped bar chart
    x = np.arange(len(psnr_levels))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        # Determine position
        pos = x + width * (i - len(models) / 2 + 0.5)

        # Get data, handle missing values
        values = []
        for j, val in enumerate(data[model]):
            if val is None:
                values.append(0)
            elif improvement and psnr_levels[j] in ref_data:
                if metric in ['psnr', 'ssim']:  # Higher is better
                    values.append(val - ref_data[psnr_levels[j]])
                else:  # 'mse' etc. - lower is better
                    values.append(ref_data[psnr_levels[j]] - val)
            else:
                values.append(val)

        # Draw bar chart
        color = COLORS.get(model.lower(), COLORS['highlight'])
        bars = ax.bar(pos, values, width=width, label=model, color=color, edgecolor='black', linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + 0.01 * max([max(data[m]) for m in models if data[m]])
            else:
                va = 'top'
                y_pos = height - 0.01 * max([max(data[m]) for m in models if data[m]])

            if improvement and height > 0:
                label = f"+{height:.2f}"
            else:
                label = f"{height:.2f}"

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha='center', va=va, rotation=90, fontsize=8)

    # Set chart properties
    if title is None:
        if improvement:
            title = f"{metric.upper()} Improvement Across Different Noise Levels"
        else:
            title = f"{metric.upper()} Comparison Across Different Noise Levels"

    ax.set_title(title)
    ax.set_xlabel('Input Noise Level (PSNR in dB)')

    if improvement:
        ax.set_ylabel(f"{metric.upper()} Improvement")
    else:
        ax.set_ylabel(metric.upper())

    ax.set_xticks(x)
    ax.set_xticklabels([f"{psnr}dB" for psnr in psnr_levels])
    ax.legend(loc='best', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save chart
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_table(metrics_dict, psnr_levels, models, metrics=['psnr', 'ssim', 'mse'],
                         save_path=None, show_improvement=True):
    """
    Create summary table containing all evaluation metrics

    Parameters:
        metrics_dict (dict): Nested dictionary containing metrics for models at each PSNR level
        psnr_levels (list): List of PSNR levels
        models (list): List of model names
        metrics (list): List of metrics to include
        save_path (str): Save path (CSV format)
        show_improvement (bool): Whether to show improvement relative to noise

    Returns:
        pandas.DataFrame: Summary table
    """
    # Create table data
    data = []

    for psnr in psnr_levels:
        # Add noisy baseline row
        if 'noisy' in metrics_dict[psnr]:
            noisy_row = {'PSNR Level': f"{psnr}dB", 'Model': 'Noisy'}

            for metric in metrics:
                if metric in metrics_dict[psnr]['noisy']:
                    noisy_row[metric.upper()] = metrics_dict[psnr]['noisy'][metric]
                elif 'metrics' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy']['metrics']:
                    noisy_row[metric.upper()] = metrics_dict[psnr]['noisy']['metrics'][metric]
                elif 'averages' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy']['averages']:
                    noisy_row[metric.upper()] = metrics_dict[psnr]['noisy']['averages'][metric]

            data.append(noisy_row)

        # Add rows for each model
        for model in models:
            if model in metrics_dict[psnr]:
                model_row = {'PSNR Level': f"{psnr}dB", 'Model': model}

                for metric in metrics:
                    # Direct metrics
                    model_value = None

                    if metric in metrics_dict[psnr][model]:
                        model_value = metrics_dict[psnr][model][metric]
                    elif 'metrics' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['metrics']:
                        model_value = metrics_dict[psnr][model]['metrics'][metric]
                    elif 'averages' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['averages']:
                        model_value = metrics_dict[psnr][model]['averages'][metric]

                    if model_value is not None:
                        model_row[metric.upper()] = model_value

                        # Add improvement information
                        if show_improvement and 'noisy' in metrics_dict[psnr]:
                            noisy_value = None

                            if metric in metrics_dict[psnr]['noisy']:
                                noisy_value = metrics_dict[psnr]['noisy'][metric]
                            elif 'metrics' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy'][
                                'metrics']:
                                noisy_value = metrics_dict[psnr]['noisy']['metrics'][metric]
                            elif 'averages' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy'][
                                'averages']:
                                noisy_value = metrics_dict[psnr]['noisy']['averages'][metric]

                            if noisy_value is not None:
                                if metric in ['psnr', 'ssim']:  # Higher is better
                                    improvement = model_value - noisy_value
                                    if improvement > 0:
                                        model_row[f"{metric.upper()} Imp."] = f"+{improvement:.2f}"
                                    else:
                                        model_row[f"{metric.upper()} Imp."] = f"{improvement:.2f}"
                                else:  # 'mse' etc. - lower is better
                                    improvement = noisy_value - model_value
                                    if improvement > 0:
                                        model_row[f"{metric.upper()} Imp."] = f"+{improvement:.6f}"
                                    else:
                                        model_row[f"{metric.upper()} Imp."] = f"{improvement:.6f}"

                data.append(model_row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def plot_heatmap(data_matrix, row_labels, col_labels, title=None, cmap='viridis',
                 annot=True, fmt='.2f', figsize=(10, 8), save_path=None):
    """
    Plot heatmap

    Parameters:
        data_matrix (numpy.ndarray): Data matrix to plot
        row_labels (list): Row labels
        col_labels (list): Column labels
        title (str): Chart title
        cmap (str): Color map
        annot (bool): Whether to display values in cells
        fmt (str): Value format
        figsize (tuple): Figure size
        save_path (str): Save path

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # Create chart
    plt.figure(figsize=figsize)

    # Create heatmap
    ax = sns.heatmap(data_matrix, annot=annot, fmt=fmt, cmap=cmap,
                     xticklabels=col_labels, yticklabels=row_labels)

    # Set title and labels
    if title:
        plt.title(title)

    plt.tight_layout()

    # Save chart
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def create_grid_visualization(clean_samples, noisy_samples, denoised_dict, metrics_dict=None,
                              num_samples=5, figsize=(15, 12), save_path=None):
    """
    Create grid visualization showing multiple samples and denoising results from multiple methods

    Parameters:
        clean_samples (list): List of clean samples
        noisy_samples (list): List of noisy samples
        denoised_dict (dict): Dictionary containing denoising results for each method {method: [results]}
        metrics_dict (dict): Optional, dictionary containing metrics for each method {method: [[metrics_per_sample]]}
        num_samples (int): Number of samples to display
        figsize (tuple): Figure size
        save_path (str): Save path

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # Limit sample count
    n_samples = min(len(clean_samples), len(noisy_samples), num_samples)

    # Calculate method count
    methods = list(denoised_dict.keys())
    n_methods = len(methods)

    # Create grid
    fig = plt.figure(figsize=figsize)

    # Each row is a sample, each column is a method (plus original and noisy)
    n_cols = n_methods + 2  # Clean + Noisy + Each method
    grid = gridspec.GridSpec(n_samples, n_cols, figure=fig)

    # Draw grid
    for i in range(n_samples):
        # Plot clean signal
        ax_clean = fig.add_subplot(grid[i, 0])
        plot_single_hrrp(clean_samples[i], title="Clean" if i == 0 else None,
                         color=COLORS['clean'], ax=ax_clean)

        # Plot noisy signal
        ax_noisy = fig.add_subplot(grid[i, 1])
        plot_single_hrrp(noisy_samples[i], title="Noisy" if i == 0 else None,
                         color=COLORS['noisy'], ax=ax_noisy)

        # Plot denoising results from each method
        for j, method in enumerate(methods):
            ax = fig.add_subplot(grid[i, j + 2])
            if i < len(denoised_dict[method]):
                # Choose color
                color = COLORS.get(method.lower(), COLORS['highlight'])

                # Set title (first row only)
                title = None
                if i == 0:
                    title = method

                # Plot denoised result
                plot_single_hrrp(denoised_dict[method][i], title=title, color=color, ax=ax)

                # Add metric information (if provided)
                if metrics_dict and method in metrics_dict and i < len(metrics_dict[method]):
                    metric_info = ""

                    if 'psnr' in metrics_dict[method][i]:
                        metric_info += f"PSNR: {metrics_dict[method][i]['psnr']:.2f}dB\n"

                    if 'ssim' in metrics_dict[method][i]:
                        metric_info += f"SSIM: {metrics_dict[method][i]['ssim']:.4f}"

                    if metric_info:
                        ax.set_title(metric_info, fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save chart
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_classifier_accuracy(clean_acc, noisy_acc, denoised_acc_dict, figsize=(12, 6),
                             title="Classification Accuracy Comparison", save_path=None):
    """
    Plot classification accuracy comparison

    Parameters:
        clean_acc (float): Accuracy on clean data
        noisy_acc (float): Accuracy on noisy data
        denoised_acc_dict (dict): Dictionary of accuracies for each denoising method
        figsize (tuple): Figure size
        title (str): Chart title
        save_path (str): Path to save the figure

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    set_plot_style()

    # Extract model names and accuracies
    models = list(denoised_acc_dict.keys())
    denoised_accs = [denoised_acc_dict[model] for model in models]

    # Create x positions for bars
    x = np.arange(len(models) + 2)  # +2 for clean and noisy

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar colors
    bar_colors = [COLORS['clean'], COLORS['noisy']]
    bar_colors.extend([COLORS.get(model.lower(), COLORS['highlight']) for model in models])

    # Create bar labels
    bar_labels = ['Clean Data', 'Noisy Data'] + models

    # Create bar values
    bar_values = [clean_acc, noisy_acc] + denoised_accs

    # Create bars
    bars = ax.bar(x, bar_values, color=bar_colors, edgecolor='black', linewidth=1)

    # Add labels and title
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')

    # Set y-axis to start from 0
    ax.set_ylim(0, 105)  # Max 105% to leave room for text

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig