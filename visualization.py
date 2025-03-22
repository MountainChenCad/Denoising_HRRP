# visualization.py - 可视化工具
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# 定义高质量绘图风格的配色方案
COLORS = {
    'noisy': '#7F7F7F',  # 灰色
    'cgan': '#1F77B4',  # 蓝色
    'cae': '#2CA02C',  # 绿色
    'ae': '#D62728',  # 红色
    'clean': '#000000',  # 黑色
    'background': '#F5F5F5',  # 淡灰色背景
    'grid': '#E0E0E0',  # 网格线颜色
    'highlight': '#FF7F0E',  # 高亮颜色
    'text': '#333333'  # 文本颜色
}


# 设置高质量绘图默认参数
def set_plot_style():
    """设置高质量绘图参数"""
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
    绘制单个HRRP信号

    参数:
        signal (numpy.ndarray): HRRP信号数据
        title (str): 图表标题
        ax (matplotlib.axes.Axes): 要绘制的轴对象，如果为None则创建新的
        color (str): 线条颜色
        linestyle (str): 线条样式
        linewidth (float): 线条宽度
        alpha (float): 线条透明度
        xlabel (str): x轴标签
        ylabel (str): y轴标签
        grid (bool): 是否显示网格
        legend (bool): 是否显示图例
        label (str): 图例标签

    返回:
        matplotlib.axes.Axes: 绘图轴对象
    """
    # 确保输入是numpy数组
    if not isinstance(signal, np.ndarray):
        if hasattr(signal, 'cpu') and hasattr(signal, 'numpy'):
            # PyTorch张量
            signal = signal.cpu().numpy()
        else:
            signal = np.array(signal)

    # 如果是批量数据，只使用第一个样本
    if signal.ndim > 1 and signal.shape[0] == 1:
        signal = signal[0]

    # 创建轴对象（如果需要）
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # 绘制数据
    ax.plot(signal, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)

    # 设置标题和标签
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 设置网格和图例
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)

    if legend and label:
        ax.legend()

    return ax


def compare_signals(clean, noisy, denoised, psnr=None, ssim=None, labels=None, title=None,
                    figsize=(15, 5), save_path=None, separate=True):
    """
    比较干净、噪声和去噪信号

    参数:
        clean (numpy.ndarray): 干净信号
        noisy (numpy.ndarray): 噪声信号
        denoised (numpy.ndarray or dict): 去噪信号或多种方法去噪结果的字典
        psnr (float or dict): PSNR值（单个值或多种方法的字典）
        ssim (float or dict): SSIM值（单个值或多种方法的字典）
        labels (dict): 信号标签字典，如 {'clean': '干净', 'noisy': '噪声'}
        title (str): 总标题
        figsize (tuple): 图形大小
        save_path (str): 保存路径，如果不为None则保存图像
        separate (bool): 是否单独绘制每个信号

    返回:
        matplotlib.figure.Figure: 图形对象
    """
    set_plot_style()

    # 处理标签
    if labels is None:
        labels = {'clean': 'Clean', 'noisy': 'Noisy'}

    # 处理多个去噪结果情况
    multi_denoised = isinstance(denoised, dict)

    # 确定子图布局
    if separate:
        if multi_denoised:
            n_plots = 2 + len(denoised)  # 干净 + 噪声 + 每种去噪方法
        else:
            n_plots = 3  # 干净 + 噪声 + 去噪
    else:
        n_plots = 1  # 所有信号在一个图上

    # 创建图形和子图布局
    fig = plt.figure(figsize=figsize)

    if separate:
        # 分开绘制每个信号
        # 绘制干净信号
        ax1 = plt.subplot(1, n_plots, 1)
        plot_single_hrrp(clean, title=labels.get('clean', 'Clean'),
                         color=COLORS['clean'], ax=ax1)

        # 绘制噪声信号
        ax2 = plt.subplot(1, n_plots, 2)
        noisy_title = labels.get('noisy', 'Noisy')
        if psnr is not None and not isinstance(psnr, dict):
            noisy_title += f" (PSNR: {psnr:.2f}dB)"
        plot_single_hrrp(noisy, title=noisy_title, color=COLORS['noisy'], ax=ax2)

        # 绘制去噪信号
        if multi_denoised:
            for i, (method, signal) in enumerate(denoised.items(), start=3):
                ax = plt.subplot(1, n_plots, i)
                method_title = labels.get(method, method)

                # 添加PSNR/SSIM信息（如果提供）
                if isinstance(psnr, dict) and method in psnr:
                    method_title += f"\nPSNR: {psnr[method]:.2f}dB"
                if isinstance(ssim, dict) and method in ssim:
                    method_title += f", SSIM: {ssim[method]:.4f}"

                # 选择颜色
                color = COLORS.get(method.lower(), COLORS['highlight'])

                plot_single_hrrp(signal, title=method_title, color=color, ax=ax)
        else:
            ax3 = plt.subplot(1, n_plots, 3)
            denoised_title = labels.get('denoised', 'Denoised')

            # 添加PSNR/SSIM信息（如果提供）
            if psnr is not None and not isinstance(psnr, dict):
                denoised_title += f"\nPSNR: {psnr:.2f}dB"
            if ssim is not None and not isinstance(ssim, dict):
                denoised_title += f", SSIM: {ssim:.4f}"

            plot_single_hrrp(denoised, title=denoised_title, color=COLORS['cgan'], ax=ax3)

    else:
        # 在一个图上绘制所有信号
        ax = plt.subplot(1, 1, 1)

        # 绘制干净信号
        plot_single_hrrp(clean, color=COLORS['clean'], label=labels.get('clean', 'Clean'), ax=ax, legend=True)

        # 绘制噪声信号
        noisy_label = labels.get('noisy', 'Noisy')
        if psnr is not None and not isinstance(psnr, dict):
            noisy_label += f" (PSNR: {psnr:.2f}dB)"
        plot_single_hrrp(noisy, color=COLORS['noisy'], label=noisy_label, ax=ax, legend=True, alpha=0.7)

        # 绘制去噪信号
        if multi_denoised:
            for method, signal in denoised.items():
                method_label = labels.get(method, method)

                # 添加PSNR/SSIM信息（如果提供）
                if isinstance(psnr, dict) and method in psnr:
                    method_label += f" (PSNR: {psnr[method]:.2f}dB"
                    if isinstance(ssim, dict) and method in ssim:
                        method_label += f", SSIM: {ssim[method]:.4f})"
                    else:
                        method_label += ")"

                # 选择颜色
                color = COLORS.get(method.lower(), COLORS['highlight'])

                plot_single_hrrp(signal, color=color, label=method_label, ax=ax, legend=True)
        else:
            denoised_label = labels.get('denoised', 'Denoised')

            # 添加PSNR/SSIM信息（如果提供）
            if psnr is not None and not isinstance(psnr, dict):
                denoised_label += f" (PSNR: {psnr:.2f}dB"
                if ssim is not None and not isinstance(ssim, dict):
                    denoised_label += f", SSIM: {ssim:.4f})"
                else:
                    denoised_label += ")"

            plot_single_hrrp(denoised, color=COLORS['cgan'], label=denoised_label, ax=ax, legend=True)

        # 设置图例
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)

    # 设置整体标题
    if title is not None:
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85)  # 为标题留出空间

    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_bar(metrics, metric_name, models=None, title=None, figsize=(10, 6),
                     higher_is_better=True, improvement=False, ref_value=None,
                     save_path=None, show_values=True, colors=None):
    """
    绘制性能指标的条形图

    参数:
        metrics (dict or list): 包含各模型指标的字典或列表
        metric_name (str): 要绘制的指标名称
        models (list): 模型名称列表，如果为None则从metrics中获取
        title (str): 图表标题
        figsize (tuple): 图形大小
        higher_is_better (bool): 指标是否越高越好
        improvement (bool): 是否显示相对于参考值的改进
        ref_value (float): 参考值
        save_path (str): 保存路径
        show_values (bool): 是否在条形上显示数值
        colors (dict): 模型对应的颜色字典

    返回:
        matplotlib.figure.Figure: 图形对象
    """
    set_plot_style()

    # 处理输入数据格式
    if isinstance(metrics, list):
        if models is None:
            raise ValueError("When metrics is a list, models must be provided")
        metrics_dict = {model: value for model, value in zip(models, metrics)}
    else:
        metrics_dict = metrics
        if models is None:
            models = list(metrics_dict.keys())

    # 如果没有指定颜色，使用默认颜色
    if colors is None:
        colors = {model: COLORS.get(model.lower(), COLORS['highlight']) for model in models}

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    if improvement and ref_value is not None:
        values = [metrics_dict[model] - ref_value for model in models]
        if not higher_is_better:
            values = [-val for val in values]
    else:
        values = [metrics_dict[model] for model in models]

    # 绘制条形图
    bar_colors = [colors.get(model, COLORS['highlight']) for model in models]
    bars = ax.bar(models, values, color=bar_colors, edgecolor='black', linewidth=1)

    # 设置标题和标签
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

    # 设置网格
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加数值标签
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

    # 保存图表
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_comparison(metrics_dict, psnr_levels, models=None, metric='psnr',
                            figsize=(12, 6), title=None, save_path=None, improvement=False):
    """
    绘制不同PSNR级别下各模型的性能指标对比

    参数:
        metrics_dict (dict): 嵌套字典，包含各PSNR级别下各模型的指标
            格式: {psnr_level: {model: {metric: value}}}
        psnr_levels (list): PSNR级别列表
        models (list): 模型名称列表，如果为None则自动提取
        metric (str): 要绘制的指标名称，如'psnr', 'ssim', 'mse'
        figsize (tuple): 图形大小
        title (str): 图表标题
        save_path (str): 保存路径
        improvement (bool): 是否显示相对于噪声的改进

    返回:
        matplotlib.figure.Figure: 图形对象
    """
    set_plot_style()

    # 如果没有指定模型，从第一个PSNR级别中获取所有模型
    if models is None:
        first_level = list(metrics_dict.keys())[0]
        models = [model for model in metrics_dict[first_level].keys()
                  if model != 'noisy' and not isinstance(metrics_dict[first_level][model], dict)]

    # 提取数据
    data = {}
    ref_data = {}

    for model in models:
        data[model] = []

    for psnr in psnr_levels:
        # 如果有噪声数据作为参考
        if 'noisy' in metrics_dict[psnr]:
            if metric in metrics_dict[psnr]['noisy']:
                ref_data[psnr] = metrics_dict[psnr]['noisy'][metric]
            elif 'metrics' in metrics_dict[psnr]['noisy'] and metric in metrics_dict[psnr]['noisy']['metrics']:
                ref_data[psnr] = metrics_dict[psnr]['noisy']['metrics'][metric]

        # 收集各模型数据
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

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制条形图分组
    x = np.arange(len(psnr_levels))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        # 确定位置
        pos = x + width * (i - len(models) / 2 + 0.5)

        # 获取数据，处理缺失值
        values = []
        for j, val in enumerate(data[model]):
            if val is None:
                values.append(0)
            elif improvement and psnr_levels[j] in ref_data:
                if metric in ['psnr', 'ssim']:  # 越高越好
                    values.append(val - ref_data[psnr_levels[j]])
                else:  # 'mse' 等越低越好
                    values.append(ref_data[psnr_levels[j]] - val)
            else:
                values.append(val)

        # 绘制条形图
        color = COLORS.get(model.lower(), COLORS['highlight'])
        bars = ax.bar(pos, values, width=width, label=model, color=color, edgecolor='black', linewidth=1)

        # 添加数值标签
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

    # 设置图表属性
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

    # 保存图表
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_table(metrics_dict, psnr_levels, models, metrics=['psnr', 'ssim', 'mse'],
                         save_path=None, show_improvement=True):
    """
    创建包含所有评估指标的汇总表格

    参数:
        metrics_dict (dict): 嵌套字典，包含各PSNR级别下各模型的指标
        psnr_levels (list): PSNR级别列表
        models (list): 模型名称列表
        metrics (list): 要包含的指标列表
        save_path (str): 保存路径（CSV格式）
        show_improvement (bool): 是否显示相对于噪声的改进

    返回:
        pandas.DataFrame: 汇总表格
    """
    # 创建表格数据
    data = []

    for psnr in psnr_levels:
        # 添加噪声基准行
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

        # 添加各模型行
        for model in models:
            if model in metrics_dict[psnr]:
                model_row = {'PSNR Level': f"{psnr}dB", 'Model': model}

                for metric in metrics:
                    # 直接指标
                    model_value = None

                    if metric in metrics_dict[psnr][model]:
                        model_value = metrics_dict[psnr][model][metric]
                    elif 'metrics' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['metrics']:
                        model_value = metrics_dict[psnr][model]['metrics'][metric]
                    elif 'averages' in metrics_dict[psnr][model] and metric in metrics_dict[psnr][model]['averages']:
                        model_value = metrics_dict[psnr][model]['averages'][metric]

                    if model_value is not None:
                        model_row[metric.upper()] = model_value

                        # 添加改进信息
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
                                if metric in ['psnr', 'ssim']:  # 越高越好
                                    improvement = model_value - noisy_value
                                    if improvement > 0:
                                        model_row[f"{metric.upper()} Imp."] = f"+{improvement:.2f}"
                                    else:
                                        model_row[f"{metric.upper()} Imp."] = f"{improvement:.2f}"
                                else:  # 'mse' 等越低越好
                                    improvement = noisy_value - model_value
                                    if improvement > 0:
                                        model_row[f"{metric.upper()} Imp."] = f"+{improvement:.6f}"
                                    else:
                                        model_row[f"{metric.upper()} Imp."] = f"{improvement:.6f}"

                data.append(model_row)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存到CSV
    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def plot_heatmap(data_matrix, row_labels, col_labels, title=None, cmap='viridis',
                 annot=True, fmt='.2f', figsize=(10, 8), save_path=None):
    """
    绘制热图

    参数:
        data_matrix (numpy.ndarray): 要绘制的数据矩阵
        row_labels (list): 行标签
        col_labels (list): 列标签
        title (str): 图表标题
        cmap (str): 颜色映射
        annot (bool): 是否在单元格中显示数值
        fmt (str): 数值格式
        figsize (tuple): 图形大小
        save_path (str): 保存路径

    返回:
        matplotlib.figure.Figure: 图形对象
    """
    set_plot_style()

    # 创建图表
    plt.figure(figsize=figsize)

    # 创建热图
    ax = sns.heatmap(data_matrix, annot=annot, fmt=fmt, cmap=cmap,
                     xticklabels=col_labels, yticklabels=row_labels)

    # 设置标题和标签
    if title:
        plt.title(title)

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def create_grid_visualization(clean_samples, noisy_samples, denoised_dict, metrics_dict=None,
                              num_samples=5, figsize=(15, 12), save_path=None):
    """
    创建网格可视化，展示多个样本和多种方法的去噪结果

    参数:
        clean_samples (list): 干净样本列表
        noisy_samples (list): 噪声样本列表
        denoised_dict (dict): 包含各方法去噪结果的字典 {method: [results]}
        metrics_dict (dict): 可选，包含各方法指标的字典 {method: [[metrics_per_sample]]}
        num_samples (int): 要显示的样本数量
        figsize (tuple): 图形大小
        save_path (str): 保存路径

    返回:
        matplotlib.figure.Figure: 图形对象
    """
    set_plot_style()

    # 限制样本数量
    n_samples = min(len(clean_samples), len(noisy_samples), num_samples)

    # 计算方法数量
    methods = list(denoised_dict.keys())
    n_methods = len(methods)

    # 创建网格
    fig = plt.figure(figsize=figsize)

    # 每行为一个样本，每列为一个方法（加上原始和噪声）
    n_cols = n_methods + 2  # 清洁 + 噪声 + 各种方法
    grid = gridspec.GridSpec(n_samples, n_cols, figure=fig)

    # 绘制网格
    for i in range(n_samples):
        # 绘制干净信号
        ax_clean = fig.add_subplot(grid[i, 0])
        plot_single_hrrp(clean_samples[i], title="Clean" if i == 0 else None,
                         color=COLORS['clean'], ax=ax_clean)

        # 绘制噪声信号
        ax_noisy = fig.add_subplot(grid[i, 1])
        plot_single_hrrp(noisy_samples[i], title="Noisy" if i == 0 else None,
                         color=COLORS['noisy'], ax=ax_noisy)

        # 绘制各方法的去噪结果
        for j, method in enumerate(methods):
            ax = fig.add_subplot(grid[i, j + 2])
            if i < len(denoised_dict[method]):
                # 选择颜色
                color = COLORS.get(method.lower(), COLORS['highlight'])

                # 设置标题（仅第一行）
                title = None
                if i == 0:
                    title = method

                # 绘制去噪结果
                plot_single_hrrp(denoised_dict[method][i], title=title, color=color, ax=ax)

                # 添加指标信息（如果提供）
                if metrics_dict and method in metrics_dict and i < len(metrics_dict[method]):
                    metric_info = ""

                    if 'psnr' in metrics_dict[method][i]:
                        metric_info += f"PSNR: {metrics_dict[method][i]['psnr']:.2f}dB\n"

                    if 'ssim' in metrics_dict[method][i]:
                        metric_info += f"SSIM: {metrics_dict[method][i]['ssim']:.4f}"

                    if metric_info:
                        ax.set_title(metric_info, fontsize=8)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig