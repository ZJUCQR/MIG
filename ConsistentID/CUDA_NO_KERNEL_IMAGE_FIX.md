# `no kernel image is available for execution on the device` 简易解决方案

这个报错通常**不是代码逻辑 bug**，而是：

- 你当前服务器上的 **GPU 架构**
- 和你当前环境里的 **PyTorch / CUDA 二进制版本**

**不匹配**。

最常见的情况有两种：

1. **GPU 太新**，但你环境里的 `torch==2.0.0` / `xformers==0.0.19` 太老
2. **GPU 太老**，当前 PyTorch CUDA wheel 根本不支持这张卡

---

## 最省事的判断方法

先执行这两个命令：

```bash
nvidia-smi
```

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
```

---

# 解决方案

## 方案 A：最常见、最推荐

### 适用情况

如果你的 GPU 是这些比较新的卡：

- RTX 3090 / 4090
- A5000 / A6000
- L40 / L40S
- A100 / H100
- V100（有时也建议这样做）

那最简单的方法就是：

> **重装一个更新一点的 PyTorch + xformers 组合**

### 操作步骤

建议新建一个干净环境，不要在旧环境里来回改。

```bash
conda create -n consistentid_fix python=3.8.10 -y
conda activate consistentid_fix
cd /path/to/ConsistentID
pip install -r requirements.txt
pip uninstall -y torch torchvision torchaudio xformers
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23.post1
pip install onnxruntime-gpu==1.16.1
```

然后重新测试：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model /path/to/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE \
  --consistentid_ckpt /path/to/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin \
  --image_encoder_path /path/to/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --face_parsing_path /path/to/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth \
  --input_image /path/to/ConsistentID/examples/albert_einstein.jpg \
  --prompt "A man wearing a santa hat" \
  --output_path /path/to/ConsistentID/outputs_sd15/test.png
```

---

## 方案 B：如果你的 GPU 很老

### 常见老卡例子

比如：

- Tesla K80
- K40
- M40
- 更老的一些 Tesla / Quadro 卡

这种情况下，最简单的结论通常是：

> **不要继续折腾当前环境，直接换一台更新的 GPU 服务器。**

因为这个项目本身就比较重，而且当前依赖栈对老卡不友好。

### 最省事的做法

换到这些卡之一：

- V100
- 3090
- 4090
- A100
- L40

然后重新按最简运行指南建环境和下载模型。

---

## 方案 C：只想快速确认是不是环境问题

你可以开一下同步报错模式：

```bash
export CUDA_LAUNCH_BLOCKING=1
```

然后再运行一次命令。

如果还是同样的 `no kernel image is available for execution on the device`，那基本就可以确定：

- **不是 prompt 的问题**
- **不是模型权重的问题**
- **是 GPU / PyTorch / CUDA 环境兼容性问题**

---

# 我建议你怎么做

## 如果你只是想最快跑起来

直接按下面选：

### 你的 GPU 比较新
就用 **方案 A**：

- 新建环境
- 换成 `torch==2.1.2`
- 换成 `xformers==0.0.23.post1`
- 重新跑

### 你的 GPU 比较老
就用 **方案 B**：

- 直接换服务器 / 换 GPU

---

# 一句话总结

这个报错的最简理解就是：

> **你现在这台服务器的 GPU，和当前环境里的 PyTorch CUDA 编译版本不兼容。**

最省事的解决方式：

1. **新 GPU** → 重建环境，升级 `torch` 和 `xformers`
2. **老 GPU** → 直接换卡，不要继续硬调
