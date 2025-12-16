# Build a new version of Sweep-Annotator

## 任务要求

- 删除 Gradio 前端中的“Overlap Frames”模块；删除后端中与之相关的逻辑，且确保不要影响其他功能。
- 原来代码中Primitives中的坐标都归一化到[0, 1]，我希望归一化到[0, 1000]，有效数字是一样的，但用整数而不是浮点数来表示。Primitives的其他格式不变。请仔细检查代码，在所有相关位置（包括但不限于calibration、annotation、历史记录保存、数据集导出等）完成这一修改。
- 依次添加下面两个功能：
  - Primitives可视化工具：在Gradio前端中增加一个输入框（位置处于“Reset Points”“Undo Last Point”一行之后，“Segment List”之前），允许用户输入任意格式正确的一条Primitive（坐标归一化到[0, 1000]），然后在把该Primitive可视化出来。可视化的效果需要和标注的时候一样（可视化展示在标注时点击的窗口中），直接把相关的点、先、矩形框等展示出来。
  - Snapshot功能：在Gradio前端中的“Episode”“Frame”一行，增加一个“Snapshot”按钮（位置处于“Mark Start Frame”“Mark End Frame & Save Segment”一行的中间位置），保存当前视频帧到当前加载的数据集`data/<dataset name>/`目录下的`snapshot/`文件目录中（文件名为`snapshot-<dataset name>-<episode index>-<frame index>.png`）。如果有calibration或加载了calibration，则保存校准后的图片；否则保存数据集原始图片。
- 确保代码中的其他功能没有被修改，且新代码可以正确运行。
- 完善好代码后，更新 README.md 文档（该版本标记为 2.0）。
