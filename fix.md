# Fix contents in the dataset exported from Sweep-Annotator

## 总体要求

- 修改 Sweep-Annotator，使其导出数据集中的`meta/episodes.jsonl`中的"tasks"字段中包括完整的primitives，比如`["<Sweep> <Box> <0.549, 0.446, 0.737, 0.549> <to> <Position> <0.893, 0.549>"]`。对应地，`meta/tasks.jsonl`中应当包含每一个不同的Task（每一个primitive几乎不可能相同，这些视为不同的Task）。
- 依照新修改，完善 README.md。

## 具体要求

- 不要修改除了 export 数据集内容之外的部分，包括前端页面、标注逻辑等。
- **必须确保修改后的代码兼容原来的 calibration.json 和 annotations.json** （见`data/sweep2E_dualarm_v1/calibration.json`和`data/sweep2E_dualarm_v1/annotations.json`）。
- **必须确保修改后的代码export后输出正确的 LeRobot 2.1 格式。** `data/sweep2E_dualarm_v1/`和`data/sweep2E_dualarm_v1_primitives_200/`里面都是完全正确的格式，可以用于 pi0 和 pi0.5 的微调。Sweep-Annotator导出的数据集文件，每一个文件（包括但不限于`episodes_stats.jsonl`, `episodes.jsonl`, `info.json`, `tasks.jsonl`）都必须符合 LeRobot 2.1 的要求，不得有更多或更少的字段。现有的输出格式是正确的，保持不变即可。
- 不要对`data/`文件目录下的文件做任何修改。