# triplane_edit — Reproduction Guide (Windows)



## 0) 你将得到什么

完成本文档后，你应该能做到：

- 一次性装好依赖并通过环境检查
- 自动编译 PyTorch CUDA 扩展（`bias_act`, `upfirdn2d` 等）
- 运行 demo 脚本并在 `out/` 看到生成图
- （可选）在 Jupyter 里也能顺利 `import` 并运行 pipeline

---

## 1) 前置要求

- Windows 10/11
- NVIDIA 显卡 + 驱动（RTX 30/40/50 系都可以，只要 PyTorch 能识别 CUDA）
- CUDA Toolkit（本仓库在 **CUDA 12.8** 环境测试通过）
- Visual Studio 2022 Build Tools（用于编译 CUDA extensions）
  - 需要包含：**MSVC v143**、**Windows 10/11 SDK**、**C++ CMake tools for Windows**
- Miniconda/Anaconda
- Python **3.10**（本项目在 `python=3.10.x` 环境测试通过）

> 检查：进入 Python 后 `import torch; torch.cuda.is_available()` 须返回 **True**。

---

## 2) 克隆仓库（含子模块）

```bash
git clone --recursive https://github.com/keke0712/triplane_edit.git
cd triplane_edit
```

---

## 3) 创建 Conda 环境（Python 3.10）

```bash
conda create -n triplane_edit310 python=3.10 -y
conda activate triplane_edit310
```

安装依赖（按仓库提供的 requirements / instructions）：

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```


---

## 4) 用 VS “x64 Native Tools” 终端运行（非常重要）

**编译 CUDA 扩展**时，建议在：

> **x64 Native Tools Command Prompt for VS 2022**

里执行后续命令（能自动带上 `cl.exe` / MSVC 环境）。

然后进入仓库目录并激活 conda：

```bat
cd /d C:\path\to\triplane_edit
conda activate triplane_edit310
```

---

## 5) 准备 checkpoints（必须）

在仓库根目录创建并放入：

```
checkpoints/
  ffhqrebalanced512-128.pkl
  encoder_FFHQ.pt
  encoder_FFHQ_finetuned.pt
  afa_FFHQ.pt
  79999_iter.pth
  encoder_FFHQ.pt
  model_ir_se50.pth
```

检查：

```bat
dir checkpoints
```

> 说明：这些权重/模型文件通常来自论文作者提供的下载链接或 release。请按仓库说明/上游项目给出的路径与文件名放置（文件名必须一致）。

---

## 6) 一键环境检查

```bash
python -m scripts.env_check
```

理想输出应包含：

- `torch.cuda.is_available: True`
- `Quick import check: OK`（或至少能正确定位到缺失项）

---

## 7) 跑 demo（会自动编译扩展）

示例：`79.png -> 40.png` 编辑 `eyes`：

```bash
python scripts/run_demo.py --input_base_dir example --src 79.png --dst 40.png --label eyes --outdir out
```

实例：`79.png -> 40.png`编辑`eyes,brows`：

```bash
python scripts/run_demo.py --input_base_dir example --src 79.png --dst 40.png --label eyes,brows --outdir out
```

如果你要启用 runtime optimization：

```bash
python scripts/run_demo.py --input_base_dir example --src 40.png --dst 68.png --label hair --runtime_optim --outdir out
```

输出结果默认会写到 `out/`（或项目内部指定的输出目录）。

---



---

## 8) 常见问题排查（Troubleshooting）

子模块目录为空 / 缺失
```bash
git submodule update --init --recursive
```

### A) `from resnet import Resnet18` 找不到（face_parsing_pytorch 内部相对 import）
本仓库的 `scripts/run_demo.py` 已做 `sys.path` 注入处理。
如果你手动 import 仍报错，在当前终端临时设置：

```bat
set PYTHONPATH=%CD%;%CD%\face_parsing_pytorch;%PYTHONPATH%
```

### B) CUDA 扩展编译失败（找不到 `cl.exe` / `nvcc`）
- 请用 **x64 Native Tools Command Prompt for VS 2022**
- 确认 `where cl`、`where nvcc` 都能找到
- 确认 CUDA Toolkit 已安装

### C) `Expected all tensors to be on the same device`（CPU/CUDA 混用）
某些常量若用 `torch.tensor(...)` 构造会默认落在 CPU，参与 CUDA 计算会报错。
本仓库已修复该问题（将常量改为 Python float 或确保跟输入在同一 device）。

---


## License

本仓库遵循原作者许可证：Apache-2.0（见 LICENSE/ 说明）。
