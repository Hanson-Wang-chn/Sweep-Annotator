#!/usr/bin/env python3
"""
工具脚本：富化任务primitives描述

功能：
- 读取导出后的LeRobot数据集（如 data/sweep2E_dualarm_v1_annotated）
- 转换Sweep格式的primitives为更具描述性的文本格式
- 将形如 "<Sweep> <Box> <0.379, 0.737, 0.464, 0.942> <to> <Position> <0.214, 0.750>" 的primitives
  转换为 "Sweep red LEGO blocks in Box <0.379, 0.737, 0.464, 0.942> to <0.214, 0.750>"
- 保持其他类型的primitives不变（Clear, Refine等）
- 保存修改后的数据集到新的目录，保持LeRobot 2.0格式
"""

import os
import re
import json
import shutil
from pathlib import Path


# ========== 配置参数 ==========

# 输入数据集路径（已标注的数据集）
INPUT_DATASET_PATH = "/Users/wanghaisheng/Downloads/sweep2cross_lerobot21_prim"

# 输出数据集路径（富化后的数据集）
OUTPUT_DATASET_PATH = "/Users/wanghaisheng/Downloads/sweep2cross_lerobot21_prim_enriched"

# 用于Sweep primitives的颜色描述
LEGO_COLOR = "red"


# ========== 核心函数 ==========

def parse_sweep_primitive(primitive_str):
    """
    解析Sweep primitive字符串

    参数:
        primitive_str: 原始primitive字符串，如 "<Sweep> <Box> <0.379, 0.737, 0.464, 0.942> <to> <Position> <0.214, 0.750>"

    返回:
        如果是Sweep primitive，返回字典包含shape和coordinates；否则返回None

    正则表达式解释:
        - ^<Sweep>\s*: 匹配开头的"<Sweep>"标签及后续空格
        - <(Box|Triangle)>\s*: 捕获形状类型（Box或Triangle），形成捕获组1
        - <([0-9., ]+)>: 捕获形状坐标，形成捕获组2（包含数字、逗号、点、空格）
        - \s*<to>\s*: 匹配中间的"<to>"标签及周围空格
        - <Position>\s*: 匹配"<Position>"标签及后续空格
        - <([0-9., ]+)>: 捕获目标位置坐标，形成捕获组3
        - $: 匹配字符串结尾
    """
    # 正则表达式：匹配Sweep primitive格式
    # 格式: <Sweep> <Box/Triangle> <coords> <to> <Position> <target_coords>
    pattern = r'^<Sweep>\s*<(Box|Triangle)>\s*<([0-9., ]+)>\s*<to>\s*<Position>\s*<([0-9., ]+)>$'

    match = re.match(pattern, primitive_str.strip())
    if match:
        shape = match.group(1)  # "Box" 或 "Triangle"
        coords = match.group(2).strip()  # 形状坐标
        target = match.group(3).strip()  # 目标位置

        return {
            'shape': shape,
            'coords': coords,
            'target': target
        }

    return None


def enrich_primitive(primitive_str, color):
    """
    富化primitive描述

    参数:
        primitive_str: 原始primitive字符串
        color: LEGO积木的颜色描述

    返回:
        富化后的primitive字符串

    转换规则:
        - Sweep primitives: "<Sweep> <Box> <coords> <to> <Position> <target>"
          转换为: "Sweep {color} LEGO blocks in Box <coords> to <target>"
        - 其他primitives: 保持不变
    """
    parsed = parse_sweep_primitive(primitive_str)

    if parsed:
        # 构建富化后的描述
        # 格式: "Sweep {color} LEGO blocks in {Shape} <coords> to <target>"
        enriched = f"Sweep {color} LEGO blocks in {parsed['shape']} <{parsed['coords']}> to <{parsed['target']}>"
        return enriched

    # 如果不是Sweep primitive或解析失败，返回原始字符串
    return primitive_str


def process_dataset(input_path, output_path, color):
    """
    处理整个数据集

    参数:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        color: LEGO积木颜色

    流程:
        1. 复制整个数据集到输出路径
        2. 读取并修改 meta/episodes.jsonl 中的 tasks 字段
        3. 读取并修改 meta/tasks.jsonl 中的 task 字段
        4. 写回修改后的文件
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 确保输入路径存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入数据集路径不存在: {input_path}")

    print(f"开始处理数据集...")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"LEGO颜色: {color}")
    print()

    # 步骤1: 复制整个数据集
    print("步骤 1/3: 复制数据集到输出路径...")
    if output_path.exists():
        print(f"警告: 输出路径已存在，将被删除: {output_path}")
        shutil.rmtree(output_path)

    shutil.copytree(input_path, output_path)
    print("✓ 数据集复制完成\n")

    # 步骤2: 处理 meta/episodes.jsonl
    print("步骤 2/3: 处理 meta/episodes.jsonl...")
    episodes_path = output_path / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl 文件不存在: {episodes_path}")

    # 读取所有episodes
    episodes = []
    with open(episodes_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    print(f"  找到 {len(episodes)} 个episodes")

    # 修改每个episode的tasks字段
    modified_count = 0
    for episode in episodes:
        if 'tasks' in episode and isinstance(episode['tasks'], list):
            new_tasks = []
            for task in episode['tasks']:
                enriched_task = enrich_primitive(task, color)
                new_tasks.append(enriched_task)
                if enriched_task != task:
                    modified_count += 1
            episode['tasks'] = new_tasks

    # 写回episodes.jsonl
    with open(episodes_path, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')

    print(f"✓ 修改了 {modified_count} 个episode tasks\n")

    # 步骤3: 处理 meta/tasks.jsonl
    print("步骤 3/3: 处理 meta/tasks.jsonl...")
    tasks_path = output_path / "meta" / "tasks.jsonl"

    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl 文件不存在: {tasks_path}")

    # 读取所有tasks
    tasks = []
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f"  找到 {len(tasks)} 个task定义")

    # 修改每个task的task字段
    modified_count = 0
    for task_obj in tasks:
        if 'task' in task_obj:
            original_task = task_obj['task']
            enriched_task = enrich_primitive(original_task, color)
            task_obj['task'] = enriched_task
            if enriched_task != original_task:
                modified_count += 1

    # 写回tasks.jsonl
    with open(tasks_path, 'w', encoding='utf-8') as f:
        for task_obj in tasks:
            f.write(json.dumps(task_obj, ensure_ascii=False) + '\n')

    print(f"✓ 修改了 {modified_count} 个task定义\n")

    print("=" * 60)
    print("处理完成！")
    print(f"富化后的数据集已保存到: {output_path}")
    print("=" * 60)


def verify_format(output_path):
    """
    验证输出数据集格式是否正确

    检查项:
        - meta/episodes.jsonl 是否存在且格式正确
        - meta/tasks.jsonl 是否存在且格式正确
        - 数据目录结构是否完整
    """
    output_path = Path(output_path)

    print("\n验证数据集格式...")

    # 检查必要文件
    required_files = [
        output_path / "meta" / "episodes.jsonl",
        output_path / "meta" / "tasks.jsonl",
        output_path / "meta" / "info.json",
    ]

    for file_path in required_files:
        if not file_path.exists():
            print(f"✗ 缺少文件: {file_path}")
            return False
        print(f"✓ 文件存在: {file_path.name}")

    # 检查episodes.jsonl格式
    episodes_path = output_path / "meta" / "episodes.jsonl"
    try:
        with open(episodes_path, 'r', encoding='utf-8') as f:
            episode_count = 0
            for line in f:
                if line.strip():
                    episode = json.loads(line)
                    # 验证必需字段
                    assert 'episode_index' in episode
                    assert 'tasks' in episode
                    assert 'length' in episode
                    episode_count += 1
        print(f"✓ episodes.jsonl 格式正确 ({episode_count} episodes)")
    except Exception as e:
        print(f"✗ episodes.jsonl 格式错误: {e}")
        return False

    # 检查tasks.jsonl格式
    tasks_path = output_path / "meta" / "tasks.jsonl"
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            task_count = 0
            for line in f:
                if line.strip():
                    task_obj = json.loads(line)
                    # 验证必需字段
                    assert 'task_index' in task_obj
                    assert 'task' in task_obj
                    task_count += 1
        print(f"✓ tasks.jsonl 格式正确 ({task_count} tasks)")
    except Exception as e:
        print(f"✗ tasks.jsonl 格式错误: {e}")
        return False

    # 检查data和videos目录
    data_path = output_path / "data"
    videos_path = output_path / "videos"

    if data_path.exists():
        print(f"✓ data 目录存在")
    else:
        print(f"✗ data 目录不存在")
        return False

    if videos_path.exists():
        print(f"✓ videos 目录存在")
    else:
        print(f"✗ videos 目录不存在")
        return False

    print("\n✓ 数据集格式验证通过！")
    return True


def main():
    """
    主函数
    """
    try:
        # 处理数据集
        process_dataset(INPUT_DATASET_PATH, OUTPUT_DATASET_PATH, LEGO_COLOR)

        # 验证格式
        verify_format(OUTPUT_DATASET_PATH)

        print("\n" + "=" * 60)
        print("富化完成示例:")
        print("=" * 60)
        print("原始: <Sweep> <Box> <0.379, 0.737, 0.464, 0.942> <to> <Position> <0.214, 0.750>")
        print(f"富化后: Sweep {LEGO_COLOR} LEGO blocks in Box <0.379, 0.737, 0.464, 0.942> to <0.214, 0.750>")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
