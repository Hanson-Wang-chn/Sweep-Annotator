# Add "Masked" Export Feature

## 任务要求

- 首先请阅读 README.md，了解项目内容。
- 请按照下面的要求完善代码：我希望在导出数据集时（即点击Export时），能选择导出模式，“Original”或“Masked”。
  - 对于“Original”，导出逻辑和现有的代码完全一样。
  - 当选择“Masked”时，我希望能增加导出可视化的内容。具体来说，当前页面中的“Visualize”按钮，可以把一个primitive“画”到image上。在导出时，会把原始数据集分隔为许多小片段，每个片段都对应一组视频（三个视角，分别由多帧图片组合而成）和一条primitive，以及若干其他信息。我希望能用和“Visualize”一样的效果，把每一条primitive“画”到该primitive对应的main视角视频中的每一帧上（其他两个视角正常导出即可）。**注意，画好的primitive必须实际渲染在视频帧上，也就是当我用普通播放器打开导出的视频观看时，必须能看到primitive的可视化效果；并不是简单调用一下“Visualize”相关的函数就可以的。**
- 不要修改与上述内容无关的文件，确保代码可以正常运行。
- 修改完代码后，请完善 README.md，记为 Version 2.2。