# DUSt3R Linux 评测指南（适配本项目）

适用目录：`/path/to/mvdream_diffusers/dust3r`

这份指南是给你当前这套流程用的：

- `generate_prompt.py` 生成 prompt JSON
- `run_mvdream.py` 生成多视角图片到 `mvdream_result/<id>/`
- `run_wan.py` 生成多视角图片到 `wan_result/<id>/`
- 用 `DUSt3R` 对每个 `<id>` 文件夹里的多视角图片做几何一致性检查

这份文档只保留必要步骤：
- 创建环境
- 安装依赖
- 下载权重
- 跑 DUSt3R Demo
- 如何用它评你生成的多视角图片

---

## 1. 先说清楚：DUSt3R 在你这里怎么用

对你当前的任务，DUSt3R 更适合做：

- 多视角图片的几何一致性检查
- 看同一组图能不能重建出稳定点云
- 看视角之间是否自洽
- 看相机姿态估计是否稳定

你现在最实用的用法不是官方 `visloc.py`。

因为 `visloc.py` 主要是给有真值位姿 / 真值 3D 的定位 benchmark 用的，
而你现在手里主要是模型生成的多视角图片。

所以对于你现在的脚本，推荐流程是：

1. 先生成多视角图片
2. 取某个 `id` 的图片文件夹
3. 用 DUSt3R 跑重建
4. 看点云、相机位姿、深度/置信度是否稳定

---

## 2. 你的输入数据长什么样

你当前脚本输出的结果目录是：

### MVDream

```text
mvdream_result/
  mv_1/
    view_0.jpg
    view_1.jpg
    view_2.jpg
    view_3.jpg
```

### Wan

```text
wan_result/
  mv_1/
    view_0.png
    view_1.png
    view_2.png
    view_3.png
```

也就是说：

- 每个 `id` 一个子目录
- 子目录里是同一主体的多视角图片
- 这正好适合拿来喂给 DUSt3R 做一组多图重建

---

## 3. 从 0 开始：创建 DUSt3R 环境

先假设你的项目目录是：

```bash
REPO=/path/to/mvdream_diffusers
DUST3R=$REPO/dust3r
```

进入目录：

```bash
cd "$DUST3R"
```

如果你是 git clone 下来的源码，建议先做：

```bash
git submodule update --init --recursive
```

然后创建 conda 环境：

```bash
eval "$(conda shell.bash hook)"
conda create -n dust3r_eval python=3.11 cmake=3.14.0 -y
conda activate dust3r_eval
```

安装 PyTorch。

### CUDA 12.1 示例

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### CUDA 11.8 示例

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

然后安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_optional.txt
```

---

## 4. 下载 DUSt3R 权重

在 `dust3r` 目录里执行：

```bash
cd "$DUST3R"
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

下载后你应该有：

```text
dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

---

## 5. 最小环境检查

执行：

```bash
cd "$DUST3R"
eval "$(conda shell.bash hook)"
conda activate dust3r_eval

python - <<'PY'
import torch
import cv2
import roma
import gradio
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device =', torch.cuda.get_device_name(0))
print('opencv =', cv2.__version__)
print('imports ok')
PY
```

如果没有报错，就说明环境基本可以用了。

---

## 6. 最简单可用方式：直接启动 DUSt3R Demo

这个方式最适合你现在的评测需求。

执行：

```bash
cd "$DUST3R"
eval "$(conda shell.bash hook)"
conda activate dust3r_eval

python demo.py \
  --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --image_size 512 \
  --device cuda \
  --server_name 0.0.0.0 \
  --server_port 7860
```

然后在浏览器打开：

```text
http://<你的服务器IP>:7860
```

---

## 7. 你实际怎么评自己的结果

以 MVDream 为例，你先生成结果：

```bash
cd /path/to/mvdream_diffusers
python run_mvdream.py --prompt-file multi_view_prompts.json
```

然后会得到：

```text
mvdream_result/
  mv_1/
    view_0.jpg
    view_1.jpg
    view_2.jpg
    view_3.jpg
```

这时在 DUSt3R demo 页面里：

1. 打开 `mvdream_result/mv_1/`
2. 上传这 4 张图
3. 运行重建
4. 看输出结果

Wan 也是同样流程：

```text
wan_result/mv_1/
```

上传该目录下的 4 张图即可。

---

## 8. 评测时主要看什么

对你现在这个项目，DUSt3R 主要看以下几点：

### 1) 点云是否稳定

如果这组多视角图几何一致，通常会看到：

- 点云主体比较集中
- 不同视角能对到同一结构
- 不会严重撕裂、漂移、重影

### 2) 相机位姿是否合理

DUSt3R 会估计每张图的相机姿态。
如果几何一致性较好，通常：

- 相机分布更稳定
- 多个视角围绕主体排列更自然

### 3) 深度/置信度是否连续

如果模型生成的多视角彼此冲突，通常会看到：

- 深度图很乱
- 置信度低
- 主体区域断裂

所以你可以把 DUSt3R 当作：

- **多视角几何一致性诊断工具**
- 而不是“单一分数评测器”

---

## 9. DUSt3R 底层是怎么工作的

你不一定需要改源码，但建议知道最关键的 3 件事：

1. DUSt3R 底层输入不是单张图，而是**图像对**
2. 多张图时会先做配对，再跑推理，再做全局对齐
3. 图多于 2 张时，会走全局优化流程

对应源码位置：

- 图像配对：`dust3r/dust3r/image_pairs.py`
- 推理入口：`dust3r/dust3r/inference.py`
- 全局对齐：`dust3r/dust3r/cloud_opt/__init__.py`
- Demo 多图重建：`dust3r/dust3r/demo.py:135-166`

从 demo 代码可以看出，它的核心流程就是：

```python
pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=1, verbose=not silent)
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=0.01)
```

所以它天然适合拿来检查一组多视角图能不能形成稳定 3D 结构。

---

## 10. 什么时候才需要 `visloc.py`

只有在你有下面这些真值时，才建议用 `visloc.py`：

- GT 相机位姿
- GT 3D 点 / 地图
- benchmark 规定的数据组织方式

否则：

- `visloc.py` 不是你当前阶段最合适的工具
- 对你当前的 MVDream / Wan 结果，直接用 demo 做重建检查更实用

---

## 11. 一条最短实用流程

如果你现在只想尽快把完整流程跑起来，按这个顺序就行：

### 第一步：生成 prompt

```bash
cd /path/to/mvdream_diffusers
python generate_prompt.py
```

### 第二步：生成多视角图

MVDream：

```bash
python run_mvdream.py --prompt-file multi_view_prompts.json
```

Wan：

```bash
python run_wan.py --prompt-file multi_view_prompts.json
```

### 第三步：启动 DUSt3R

```bash
cd /path/to/mvdream_diffusers/dust3r
python demo.py \
  --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --image_size 512 \
  --device cuda \
  --server_name 0.0.0.0 \
  --server_port 7860
```

### 第四步：上传某个 id 的 4 张图检查几何一致性

例如：

- `mvdream_result/mv_1/view_0.jpg` ~ `view_3.jpg`
- 或 `wan_result/mv_1/view_0.png` ~ `view_3.png`

---

## 12. 你现在最需要记住的结论

1. **你当前最适合用 DUSt3R Demo 做多视角几何一致性检查。**
2. **输入就是某个 `id` 子目录里的多张视角图。**
3. **对你的当前流程，重点看点云、相机位姿、深度和置信度是否稳定。**
