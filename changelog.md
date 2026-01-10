# Changelog

本仓库基于上游 triplane-edit / GAN3D 相关实现进行课程复现与工程化整理。
**以下改动以“可复现/可跑通”为目标，默认不改变论文算法与训练/推理逻辑**（除非特别说明）。

---

## [Unreleased]

### Added
- `scripts/run_demo.py`：提供命令行方式跑 demo（避免只能 Jupyter 手动点）。
- `scripts/env_check.py`：环境自检脚本（Python/CUDA/编译工具/torch.cuda 可用性等）。


### Fixed
- Windows 下 CUDA extensions 动态加载问题：`bias_act_plugin / upfirdn2d_plugin` 编译成功但 `import` 找不到的情况（修复加载/导入流程）。
- `face_parsing_pytorch` 子模块在 Windows 下的导入问题（其内部使用 `from resnet import ...` 的相对导入写法）：通过脚本注入 `sys.path` / `PYTHONPATH` 解决。
- 修复 Swin/注意力模块中常量的 device 不一致问题：原实现使用 `torch.tensor(...)` 构造常量导致其默认在 CPU，
与 CUDA 张量一起参与 `torch.clamp` 等运算时报错；
现改为使用 `math.log(...)`（Python float）或确保常量与输入张量在同一 device，从而避免 CPU/CUDA 混用。

---


### Notes
- 通过 `run_demo.py` 可直接生成编辑结果图（示例：eyes/hair/mouth/nose 等）。
- 已在本机 CUDA 12.8 + PyTorch CUDA 环境跑通（不同显卡只要满足 CUDA 可用与编译工具齐全，原则上可复现）。
