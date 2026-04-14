# ConsistentID 评测运行指南

本文把 **从建环境 → 下载权重 → 准备参考图片 → 跑 inference 生成图片 → 用生成图片做评测** 的流程整理成一份可直接照做的说明。

> 先说结论：
>
> - 这个仓库 **不是只有一个模型**。
> - 当前代码里至少有 **两个主模型版本**：
>   1. **SD1.5 版本**：`ConsistentID-v1.bin`，对应 `infer.py` / `pipline_StableDiffusion_ConsistentID.py`
>   2. **SDXL 版本**：`ConsistentID_SDXL-v1.bin`，对应 `infer_SDXL.py` / `pipline_StableDiffusionXL_ConsistentID.py`
> - 另外还有两个任务型变体：
>   - Inpaint：`demo/inpaint_demo.py`
>   - ControlNet Inpaint：`demo/controlnet_demo.py`
> - **如果你的目标是先把 evaluation 跑通，建议优先用 SD1.5 版本。**
> - 当前仓库里 **没有一个完整公开的一键评测/打分 CLI**。仓库里有：
>   - `evaluation/EvaluationIMGs_stars_prompts.csv`：评测 prompt 表
>   - `evaluation/eval_function.py`：一些辅助函数
>   - `evaluation/style_template.py`：风格模板
>   - 但**没有**完整的 `run_eval.py` / `eval.sh` 这类官方端到端评分脚本
>
> 所以目前最实际的流程是：
>
> 1. 配环境
> 2. 下载权重
> 3. 准备参考图片
> 4. 生成评测图片
> 5. 用生成图片做本地指标统计（例如 FaceID cosine similarity）
>
> 下面就是这个流程的具体步骤。

---

## 1. 仓库里有哪些模型

### 1.1 主模型

#### A. SD1.5 版（推荐先跑这个）
- 主权重：`ConsistentID-v1.bin`
- 推理入口：`infer.py`
- pipeline：`pipline_StableDiffusion_ConsistentID.py`
- base model 默认建议下载到本地，例如：`./pretrained_models/Realistic_Vision_V6.0_B1_noVAE`

#### B. SDXL 版
- 主权重：`ConsistentID_SDXL-v1.bin`
- 推理入口：`infer_SDXL.py`
- pipeline：`pipline_StableDiffusionXL_ConsistentID.py`

### 1.2 辅助模型 / 依赖模型

无论你跑哪个主模型，当前代码还会用到这些辅助模型：

- **基础扩散模型**
  - SD1.5 默认：`SG161222/Realistic_Vision_V6.0_B1_noVAE`
- **CLIP image encoder**
  - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- **Face parsing 模型**
  - `face_parsing.pth`
- **InsightFace 人脸特征模型**
  - `buffalo_l`

### 1.3 LLaVA 目前不是必需的

代码里虽然提到了 LLaVA，但当前推理路径中 `get_prepare_llva_caption()` 实际返回的是一个固定的人脸描述句子，**并没有真的加载 LLaVA 13B**。

所以：

- **当前跑 inference / evaluation，不需要额外下载 LLaVA。**

---

## 2. 当前仓库里“evaluation”到底包含什么

### 已包含

- `evaluation/EvaluationIMGs_stars_prompts.csv`
  - 一张评测 prompt 表，至少包含 `Image_Name` 和 `Prompt`
- `evaluation/eval_function.py`
  - 一些辅助函数 / 编码器定义
- `evaluation/style_template.py`
  - 风格 prompt 模板

### 未包含

- 一个可以直接运行的官方评分主脚本
- 完整 benchmark 参考图像集

### 这意味着什么

当前仓库里，**最可靠能跑起来的 evaluation 流程**是：

1. 准备参考人物图像
2. 读取 `evaluation/EvaluationIMGs_stars_prompts.csv`
3. 批量生成结果图
4. 再对这些结果图做本地评测（例如 FaceID 相似度）

> `examples/` 目录里只有少量 demo 图，并不包含完整 benchmark 的所有人物图。
> 如果你要完整复现 prompt CSV 对应的评测，需要你自己准备对应的参考图像，或者下载 FGID 数据集后整理出来。

---

## 3. 环境准备

推荐直接用 conda 新建环境。

```bash
conda create -n consistentid python=3.8.10 -y
conda activate consistentid
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 可选：给 InsightFace 启用 GPU

`requirements.txt` 里有 `onnxruntime==1.16.1`，这足够跑，但如果你希望 `insightface` 尽量走 GPU，可以额外安装：

```bash
pip install onnxruntime-gpu==1.16.1
```

如果不装也通常能跑，只是 `FaceAnalysis` 可能回退到 CPU。

---

## 4. 下载权重

### 4.1 你至少需要哪些东西

如果你先跑 **SD1.5 版 evaluation**，建议准备以下四项：

1. base model
   - `SG161222/Realistic_Vision_V6.0_B1_noVAE`
2. ConsistentID 主权重
   - `ConsistentID-v1.bin`
3. CLIP image encoder
   - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
4. face parsing 权重
   - `face_parsing.pth`

另外：

5. InsightFace 的 `buffalo_l`
   - 一般会在第一次运行时自动下载

### 4.2 推荐目录结构

建议把模型都下载到一个统一目录，比如：

```text
E:/VBench/ConsistentID/
├── pretrained_models/
│   ├── Realistic_Vision_V6.0_B1_noVAE/
│   ├── CLIP-ViT-H-14-laion2B-s32B-b79K/
│   └── JackAILab_ConsistentID/
│       ├── ConsistentID-v1.bin
│       ├── ConsistentID_SDXL-v1.bin
│       └── face_parsing.pth
```

### 4.3 推荐下载方式：提前下载到本地目录

#### 下载 base model（SD1.5）

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='SG161222/Realistic_Vision_V6.0_B1_noVAE', local_dir='E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE'))"
```

#### 下载 CLIP image encoder

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K'))"
```

#### 下载 ConsistentID SD1.5 主权重

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='JackAILab/ConsistentID', filename='ConsistentID-v1.bin', repo_type='model', local_dir='E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID'))"
```

#### 下载 face parsing 权重

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='JackAILab/ConsistentID', filename='face_parsing.pth', repo_type='model', local_dir='E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID'))"
```

#### 如果你以后想跑 SDXL 版，也可以顺手下载

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='JackAILab/ConsistentID', filename='ConsistentID_SDXL-v1.bin', repo_type='model', local_dir='E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID'))"
```

### 4.4 关于 `convert_weights.py`

README 里提到先运行：

```bash
python evaluation/convert_weights.py
```

但这个脚本的用途是：

- 读取当前目录下的 `pytorch_model.bin`
- 裁剪出更小的 `ConsistentID-v1.bin`

因此：

- **如果你已经从 Hugging Face 下载到了 `ConsistentID-v1.bin`，可以跳过这一步。**
- **只有你手头是训练生成的大 checkpoint（`pytorch_model.bin`）时，才需要跑 `convert_weights.py`。**

---

## 5. 下载完模型之后，如何准备参考图片

这是 evaluation 最关键的一步。

### 5.1 先建一个参考图片目录

建议在仓库根目录下创建：

```text
E:/VBench/ConsistentID/ref_images/
```

### 5.2 图片怎么命名

`evaluation/EvaluationIMGs_stars_prompts.csv` 里的 `Image_Name` 会像这样：

```text
albert_einstein.png
andrew_ng.png
barack_obama.png
...
```

所以最简单的做法是：

- 让 `ref_images/` 里的图片文件名尽量与 CSV 中的 `Image_Name` 对应

例如：

```text
ref_images/
├── albert_einstein.png
├── andrew_ng.png
├── barack_obama.png
└── ...
```

### 5.3 支持哪些格式

当前更新后的推理脚本支持这些常见格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.bmp`

也就是说：

- CSV 里写 `albert_einstein.png`
- 你本地放 `albert_einstein.jpg`

脚本也会尝试按同名 stem 自动匹配。

### 5.4 图片应该满足什么条件

建议参考图满足以下要求：

- 图中最好只有一个主要人物
- 脸部清晰、不要太小
- 尽量是正脸或接近正脸
- 不要严重遮挡
- 尽量不要多人合影

因为当前评测流程依赖：

- InsightFace 提取 FaceID
- face parsing 提取脸部区域

如果参考图里检测不到稳定人脸，生成和评分都会受影响。

### 5.5 先用 demo 图做 smoke test

如果你只是先确认环境是否可用，可以先用仓库里的：

- `examples/albert_einstein.jpg`
- `examples/scarlett_johansson.jpg`

然后再换成你自己的 `ref_images/`。

---

## 6. 单张图推理：先确认环境没问题

更新后的 `infer.py` 不需要你手改源码，直接传参数即可。

### 6.1 查看脚本参数

```bash
python infer.py --help
```

### 6.2 跑一个单张图 smoke test（SD1.5）

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_image "E:/VBench/ConsistentID/examples/albert_einstein.jpg" ^
  --prompt "A man wearing a santa hat" ^
  --output_path "E:/VBench/ConsistentID/outputs_sd15/albert_einstein_santa.png"
```

### 6.3 如果你想保留原来 demo 里的“电影风模板”

加上：

```bash
--use_prompt_template
```

例如：

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_image "E:/VBench/ConsistentID/examples/albert_einstein.jpg" ^
  --prompt "A man wearing a santa hat" ^
  --use_prompt_template ^
  --output_path "E:/VBench/ConsistentID/outputs_sd15/albert_einstein_santa.png"
```

### 6.4 预期结果

运行成功后，你应该得到：

- 一张生成图片，例如 `outputs_sd15/albert_einstein_santa.png`

如果这一步能跑通，说明：

- 环境基本正确
- 权重路径正确
- GPU / torch / diffusers / insightface / face parsing 基本可用

---

## 7. 用更新后的 `infer.py` 跑 evaluation 批量生成

这是当前最推荐的评测图片生成方式。

### 7.1 先看一下 CSV 的作用

`evaluation/EvaluationIMGs_stars_prompts.csv` 每行大致像这样：

```text
Image_Name,Prompt
albert_einstein.png,a man wearing a red hat
albert_einstein.png,a man wearing a santa hat
...
```

这意味着：

- `Image_Name` 指向参考人物
- `Prompt` 是你要让模型生成的目标描述

### 7.2 批量运行命令

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_dir "E:/VBench/ConsistentID/ref_images" ^
  --prompt_csv "E:/VBench/ConsistentID/evaluation/EvaluationIMGs_stars_prompts.csv" ^
  --output_dir "E:/VBench/ConsistentID/eval_outputs_sd15"
```

### 7.3 先做小规模测试

建议先只跑前几条，确认目录映射没问题：

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_dir "E:/VBench/ConsistentID/ref_images" ^
  --prompt_csv "E:/VBench/ConsistentID/evaluation/EvaluationIMGs_stars_prompts.csv" ^
  --output_dir "E:/VBench/ConsistentID/eval_outputs_sd15" ^
  --limit 5
```

### 7.4 输出目录会是什么样子

更新后的 `infer.py` 会按 subject 分目录保存：

```text
E:/VBench/ConsistentID/eval_outputs_sd15/
├── albert_einstein/
│   ├── 0001__a_man_wearing_a_red_hat.png
│   ├── 0002__a_man_wearing_a_santa_hat.png
│   └── ...
├── andrew_ng/
└── ...
```

这样后续做评分更方便。

---

## 8. 如果你想测试 SDXL 版

更新后的 `infer_SDXL.py` 已经修复了原来脚本里的 CLI 问题，不再依赖未定义变量。

### 8.1 查看参数

```bash
python infer_SDXL.py --help
```

### 8.2 单张图运行示例

```bash
python infer_SDXL.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/sdxl-base" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID_SDXL-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_image "E:/VBench/ConsistentID/examples/scarlett_johansson.jpg" ^
  --prompt "A woman wearing a santa hat" ^
  --output_path "E:/VBench/ConsistentID/outputs_sdxl/scarlett_santa.png"
```

### 8.3 说明

- `infer_SDXL.py` 现在支持直接传 `--input_image`
- 也支持通过 `--star_name` + `--input_dir` 自动查找图片
- 目前这份指南仍然建议你把 **SD1.5 作为 evaluation 主路径**
- SDXL 更适合做对照测试或额外实验

---

## 9. 生成图片之后，如何进行评测

### 9.1 先说明真实情况

当前仓库里没有公开的官方一键打分脚本，所以你有两个选择：

#### 选择 A：做你自己的本地指标评测（推荐先这样）
例如：
- FaceID cosine similarity
- 人工主观检查
- 你自己的 CLIP / identity consistency 指标

#### 选择 B：如果你有作者未公开的 benchmark scorer，再把生成结果接进去

### 9.2 这里给出一个可直接运行的本地评测方法：FaceID cosine similarity

这个方法不是论文官方 FGIS 打分脚本，但它非常实用，而且和当前项目本身的 FaceID 机制是一致的。

把下面代码保存为 `score_faceid.py`：

```python
from pathlib import Path
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis

ref_dir = Path(r"E:/VBench/ConsistentID/ref_images")
gen_dir = Path(r"E:/VBench/ConsistentID/eval_outputs_sd15")
csv_path = Path(r"E:/VBench/ConsistentID/evaluation/EvaluationIMGs_stars_prompts.csv")
out_csv = Path(r"E:/VBench/ConsistentID/faceid_scores.csv")

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding


subjects = set()
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        subjects.add(Path(row["Image_Name"]).stem)

rows = []
all_scores = []

for subject in sorted(subjects):
    ref_path = ref_dir / f"{subject}.png"
    if not ref_path.exists():
        alt_jpg = ref_dir / f"{subject}.jpg"
        alt_jpeg = ref_dir / f"{subject}.jpeg"
        if alt_jpg.exists():
            ref_path = alt_jpg
        elif alt_jpeg.exists():
            ref_path = alt_jpeg

    if not ref_path.exists():
        print(f"[skip] reference image missing: {subject}")
        continue

    ref_emb = get_embedding(ref_path)
    if ref_emb is None:
        print(f"[skip] no face detected in reference image: {ref_path}")
        continue

    subject_dir = gen_dir / subject
    if not subject_dir.exists():
        print(f"[skip] generated directory missing: {subject_dir}")
        continue

    for gen_path in sorted(subject_dir.glob("*.png")):
        gen_emb = get_embedding(gen_path)
        if gen_emb is None:
            print(f"[skip] no face detected in generated image: {gen_path}")
            continue

        score = float(np.dot(ref_emb, gen_emb))
        rows.append((subject, gen_path.name, score))
        all_scores.append(score)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "generated_image", "faceid_cosine_similarity"])
    writer.writerows(rows)

if len(all_scores) == 0:
    print("No valid scores were produced.")
else:
    print(f"Saved scores to: {out_csv}")
    print(f"Average FaceID cosine similarity: {sum(all_scores) / len(all_scores):.6f}")
```

### 9.3 运行评测

```bash
python score_faceid.py
```

### 9.4 输出结果

你会得到：

- `faceid_scores.csv`
- 终端里会打印平均 FaceID cosine similarity

这个结果可以作为一个**本地可复现的 identity consistency 指标**。

---

## 10. 推荐的最小可运行流程

如果你只想先确认整套流程能跑通，推荐这样做：

### 第一步：建环境

```bash
conda create -n consistentid python=3.8.10 -y
conda activate consistentid
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 第二步：下载 4 个必要权重

- base model
- `ConsistentID-v1.bin`
- CLIP image encoder
- `face_parsing.pth`

### 第三步：准备参考图片

创建：

```text
E:/VBench/ConsistentID/ref_images/
```

然后把参考图按 subject 名字放进去。

### 第四步：跑一次单图 smoke test

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_image "E:/VBench/ConsistentID/examples/albert_einstein.jpg" ^
  --prompt "A man wearing a santa hat" ^
  --output_path "E:/VBench/ConsistentID/outputs_sd15/albert_einstein_santa.png"
```

### 第五步：跑 evaluation 批量生成

```bash
python infer.py ^
  --base_model "E:/VBench/ConsistentID/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" ^
  --consistentid_ckpt "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" ^
  --image_encoder_path "E:/VBench/ConsistentID/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" ^
  --face_parsing_path "E:/VBench/ConsistentID/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" ^
  --input_dir "E:/VBench/ConsistentID/ref_images" ^
  --prompt_csv "E:/VBench/ConsistentID/evaluation/EvaluationIMGs_stars_prompts.csv" ^
  --output_dir "E:/VBench/ConsistentID/eval_outputs_sd15"
```

### 第六步：本地评分

```bash
python score_faceid.py
```

---

## 11. 常见问题

### Q1. 当前代码只有一个模型吗？
不是。

当前代码里至少有：

- **SD1.5 主模型**：`ConsistentID-v1.bin`
- **SDXL 主模型**：`ConsistentID_SDXL-v1.bin`
- **Inpaint 版**：基于同一套 ConsistentID 思路的变体
- **ControlNet Inpaint 版**：同上

如果只是为了稳定地先跑 evaluation，优先使用 **SD1.5 主模型**。

### Q2. 为什么 README 说有 evaluation code，但我没找到一键脚本？
因为仓库里公开出来的更像是：

- 评测 prompt 文件
- 一些辅助函数
- demo / inference 代码

但没有整理成一个可直接执行的官方评分入口。

### Q3. 我一定要下载 LLaVA 吗？
不需要。
当前代码路径里它没有真的被加载，推理时使用的是固定的人脸描述字符串。

### Q4. 我一定要先跑 `convert_weights.py` 吗？
不一定。

- 如果你已经有 `ConsistentID-v1.bin`，不用跑
- 只有你拿到的是 `pytorch_model.bin` 原始训练 checkpoint，才需要转换

### Q5. `infer.py` 现在还能做单图 demo 吗？
可以。

- 不传 `--prompt_csv` 就是单图模式
- 传了 `--prompt_csv` 就是批量 evaluation 模式

### Q6. Evaluation 时 prompt 会不会自动被改写？
默认不会。

- 更新后的 `infer.py` 默认直接使用你提供的 prompt
- 只有你显式加上 `--use_prompt_template`，才会套用原来 demo 里的电影风 prompt 模板

这样更适合做可复现 evaluation。

---

## 12. 我建议你实际采用的方案

如果你的目标是“先把 evaluation 跑起来”，最稳妥的选择是：

1. **先用 SD1.5 版**，不要先碰 SDXL
2. **先跑单图**，确认环境与权重没有问题
3. **再准备 `ref_images/`**
4. **再跑 CSV 批量生成**
5. **最后做 FaceID cosine similarity 评测**

这样你可以最快得到一套可运行、可复现、可解释的本地 evaluation 流程。
