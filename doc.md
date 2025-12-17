# Write `tools/enrich_task_primitives.py`

## 任务要求

- 首先请阅读 README.md，了解 Sweep-Annotator 项目。`data/sweep2E_dualarm_v1`是原始数据集，`data/sweep2E_dualarm_v1_annotated`是标注后导出的数据集。
- 我希望在`tools/enrich_task_primitives.py`中实现，根据一个导出后数据集的路径（以`data/sweep2E_dualarm_v1_annotated`为例），修改`path_to_dataset/meta/episodes.jsonl`中的`tasks`字段，和`path_to_dataset/meta/tasks.jsonl`中的`task`字段，然后把修改后的完整数据集保存到指定文件目录中（原来的数据集保持不变）。修改方式为，把形如`<Sweep> <Box> <0.379, 0.737, 0.464, 0.942> <to> <Position> <0.214, 0.750>`这样的Sweep格式的primitives，转化为 `Sweep {color} LEGO blocks in Box <0.379, 0.737, 0.464, 0.942> to <0.214, 0.750>`（{color}取red），忽略其他形式的primitives。需要指定的路径、颜色等，放在代码里即可，无需命令行参数。确
- 确保代码能正常运行，且转化后的数据集必须仍然为 LeRobot 2.0 格式，与`data/sweep2E_dualarm_v1`、`data/sweep2E_dualarm_v1_annotated`一致。
- 代码中请加入充分的中文注释，尤其需要解释清楚正则表达式（如有）等，方便后续人工完善task primitives描述。
- 注意不要修改项目中 `tools/enrich_task_primitives.py` 之外的任何文件。
