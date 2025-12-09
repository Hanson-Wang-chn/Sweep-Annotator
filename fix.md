# Fix the export format of Sweep-Annotator

## 总体要求

- 修改 Sweep-Annotator，使其导出数据集的格式，完全符合 LeRobot 2.1 的要求。
- 依照新修改，完善 README.md。

## 具体要求

- 不要修改除了 export 格式之外的内容，包括前端页面、标注逻辑等。
- **必须确保修改后的代码兼容原来的 calibration.json 和 annotations.json** （见`/data/sweep2D_dualarm_v1/calibration.json`和`/data/sweep2D_dualarm_v1/annotations.json`）。
- 正确的 LeRobot 2.1 格式，请完全参考 `/data/sweep2D_dualarm_v1/`。这里面是完全正确的格式，可以用于 pi0 和 pi0.5 的微调。Sweep-Annotator导出的数据集文件，每一个文件（包括但不限于`episodes_stats.jsonl`, `episodes.jsonl`, `info.json`, `tasks.jsonl`）都必须符合 LeRobot 2.1 的要求，不得有更多或更少的字段。
- 现在的 Sweep-Annotator 导出后的数据集格式参考`/data/sweep2D_dualarm_v1_primitives_200/`，**这是错误的**，格式不符合要求（尤其是meta中的各个字段）。
- 不要对`data/`文件目录下的文件做任何修改。
