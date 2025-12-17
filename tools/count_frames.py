import cv2
import os
from pathlib import Path

def count_frames(video_path):
    """统计视频帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1

    # 获取帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    videos_dir = Path('videos')
    output_file = 'frames.txt'

    # 获取所有视频文件
    video_files = sorted(videos_dir.glob('*.mp4'))

    print(f"找到 {len(video_files)} 个视频文件")
    print("开始统计帧数...")

    results = []

    for i, video_path in enumerate(video_files, 1):
        frame_count = count_frames(str(video_path))
        results.append(f"{video_path.name}: {frame_count}")

        if i % 20 == 0:
            print(f"已处理 {i}/{len(video_files)} 个视频")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"完成！结果已保存到 {output_file}")

if __name__ == '__main__':
    main()
