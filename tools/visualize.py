import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read_frame_counts(filename):
    """读取帧数文件"""
    frame_counts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                # 提取帧数
                parts = line.strip().split(':')
                frame_count = int(parts[1].strip())
                frame_counts.append(frame_count)
    return frame_counts

def create_bins(frame_counts, bin_size=3):
    """将帧数分组到区间（以 bin_size 为间隔）"""
    if not frame_counts:
        return {}, []

    min_frames = min(frame_counts)
    max_frames = max(frame_counts)

    # 创建区间
    bins = defaultdict(int)
    bin_edges = list(range((min_frames // bin_size) * bin_size,
                           max_frames + bin_size + 1, bin_size))

    # 统计每个区间的视频数量
    for count in frame_counts:
        bin_start = (count // bin_size) * bin_size
        bins[bin_start] += 1

    return bins, bin_edges

def visualize(bins, frame_counts, bin_size=3):
    """创建美观的柱状图"""
    if not bins:
        print("没有数据可视化")
        return

    # 计算平均数和中位数
    mean_frames = np.mean(frame_counts)
    median_frames = np.median(frame_counts)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 准备数据
    sorted_bins = sorted(bins.items())
    bin_labels = [f"{start}-{start+bin_size-1}" for start, _ in sorted_bins]
    bin_starts = [start for start, _ in sorted_bins]
    counts = [count for _, count in sorted_bins]

    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制柱状图
    bars = ax.bar(range(len(bin_labels)), counts,
                   color='steelblue', edgecolor='navy', linewidth=1.2, alpha=0.8)

    # 在柱子上方显示数值
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 找到平均数和中位数对应的x位置
    def find_x_position(value, bin_starts, bin_size):
        """找到值在图表上的x位置"""
        for i, start in enumerate(bin_starts):
            if start <= value < start + bin_size:
                # 在区间内插值
                offset = (value - start) / bin_size
                return i + offset
        # 如果不在任何区间内，返回最近的位置
        if value < bin_starts[0]:
            return 0
        return len(bin_starts) - 1

    mean_x = find_x_position(mean_frames, bin_starts, bin_size)
    median_x = find_x_position(median_frames, bin_starts, bin_size)

    # 绘制平均数线
    mean_line = ax.axvline(x=mean_x, color='#FF6B6B', linestyle='--',
                           linewidth=2.5, alpha=0.9, label=f'Mean: {mean_frames:.2f}', zorder=5)

    # 绘制中位数线
    median_line = ax.axvline(x=median_x, color='#4ECDC4', linestyle='--',
                             linewidth=2.5, alpha=0.9, label=f'Median: {median_frames:.2f}', zorder=5)

    # 在线的上方添加文本标注
    y_max = ax.get_ylim()[1]
    text_y = y_max * 0.95

    # 平均数标注
    ax.text(mean_x, text_y, f'Mean\n{mean_frames:.2f}',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF6B6B',
                     edgecolor='darkred', alpha=0.85, linewidth=2),
            color='white', zorder=6)

    # 中位数标注
    ax.text(median_x, text_y * 0.82, f'Median\n{median_frames:.2f}',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#4ECDC4',
                     edgecolor='darkcyan', alpha=0.85, linewidth=2),
            color='white', zorder=6)

    # 设置标题和标签
    ax.set_xlabel('Frame Count Range', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Videos', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Video Frame Counts (3-frame intervals)',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置 x 轴刻度
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')

    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # 添加图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='gray', fancybox=True, shadow=True)

    # 优化布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('result.png', dpi=300, bbox_inches='tight')
    print("图表已保存到 result.png")

    # 显示统计信息
    total_videos = sum(counts)
    print(f"\n统计信息：")
    print(f"总视频数：{total_videos}")
    print(f"平均帧数：{mean_frames:.2f}")
    print(f"中位数帧数：{median_frames:.2f}")
    print(f"帧数区间：{bin_labels[0]} 到 {bin_labels[-1]}")
    print(f"最多视频的区间：{bin_labels[counts.index(max(counts))]} ({max(counts)} 个视频)")

def main():
    # 读取帧数数据
    frame_counts = read_frame_counts('frames.txt')
    print(f"成功读取 {len(frame_counts)} 个视频的帧数信息")

    # 创建区间并统计
    bins, _ = create_bins(frame_counts, bin_size=3)

    # 可视化
    visualize(bins, frame_counts, bin_size=3)

if __name__ == '__main__':
    main()
