# ConsistentID 最简运行指南（Linux 服务器）

如果你现在的目标只是：

- 在 Linux 服务器上把 ConsistentID 跑起来
- 下载模型
- 准备图片
- 生成结果图

那就按下面做。

> 建议先跑 **SD1.5 版本**，也就是 `infer.py`。
> 这是当前最稳的路径。

---

## 1. 进入项目目录

```bash
export PROJECT_ROOT=/path/to/ConsistentID
cd "$PROJECT_ROOT"
```

建议先开一个 `tmux`：

```bash
tmux new -s consistentid
```

---

## 2. 创建环境

```bash
conda create -n consistentid python=3.8.10 -y
conda activate consistentid
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install onnxruntime-gpu==1.16.1
```

---

## 3. 创建目录

```bash
cd "$PROJECT_ROOT"
mkdir -p pretrained_models/Realistic_Vision_V6.0_B1_noVAE
mkdir -p pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K
mkdir -p pretrained_models/JackAILab_ConsistentID
mkdir -p ref_images
mkdir -p outputs_sd15
mkdir -p eval_outputs_sd15
```

---

## 4. 下载模型

直接运行这段：

```bash
cd "$PROJECT_ROOT"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download

root = "pretrained_models"

snapshot_download(
    repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    local_dir=f"{root}/Realistic_Vision_V6.0_B1_noVAE"
)

snapshot_download(
    repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    local_dir=f"{root}/CLIP-ViT-H-14-laion2B-s32B-b79K"
)

for name in ["ConsistentID-v1.bin", "face_parsing.pth"]:
    hf_hub_download(
        repo_id="JackAILab/ConsistentID",
        filename=name,
        repo_type="model",
        local_dir=f"{root}/JackAILab_ConsistentID"
    )

print("model download done")
PY
```

下载完成后，你至少会有：

```text
pretrained_models/
├── Realistic_Vision_V6.0_B1_noVAE/
├── CLIP-ViT-H-14-laion2B-s32B-b79K/
└── JackAILab_ConsistentID/
    ├── ConsistentID-v1.bin
    └── face_parsing.pth
```

---

## 5. 先跑一个单图测试

最简单的方法：直接用仓库自带图片。

```bash
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model "$PROJECT_ROOT/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" \
  --consistentid_ckpt "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" \
  --image_encoder_path "$PROJECT_ROOT/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --face_parsing_path "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" \
  --input_image "$PROJECT_ROOT/examples/albert_einstein.jpg" \
  --prompt "A man wearing a santa hat" \
  --output_path "$PROJECT_ROOT/outputs_sd15/test.png"
```

如果成功，输出图在：

```bash
$PROJECT_ROOT/outputs_sd15/test.png
```

---

## 6. 如果你想跑 batch evaluation

### 6.1 准备参考图

把你的参考图放到：

```bash
$PROJECT_ROOT/ref_images/
```

文件名尽量和 `evaluation/EvaluationIMGs_stars_prompts.csv` 里的 `Image_Name` 对应，例如：

```text
ref_images/
├── albert_einstein.png
├── andrew_ng.png
├── barack_obama.png
└── ...
```

如果图片在你本地电脑上，可以上传：

```bash
scp -r ./ref_images your_user@your_server:$PROJECT_ROOT/
```

---

### 6.2 先试跑前 5 条

```bash
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model "$PROJECT_ROOT/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" \
  --consistentid_ckpt "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" \
  --image_encoder_path "$PROJECT_ROOT/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --face_parsing_path "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" \
  --input_dir "$PROJECT_ROOT/ref_images" \
  --prompt_csv "$PROJECT_ROOT/evaluation/EvaluationIMGs_stars_prompts.csv" \
  --output_dir "$PROJECT_ROOT/eval_outputs_sd15" \
  --limit 5
```

---

### 6.3 没问题后跑全量

```bash
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model "$PROJECT_ROOT/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" \
  --consistentid_ckpt "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" \
  --image_encoder_path "$PROJECT_ROOT/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --face_parsing_path "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" \
  --input_dir "$PROJECT_ROOT/ref_images" \
  --prompt_csv "$PROJECT_ROOT/evaluation/EvaluationIMGs_stars_prompts.csv" \
  --output_dir "$PROJECT_ROOT/eval_outputs_sd15"
```

输出目录在：

```bash
$PROJECT_ROOT/eval_outputs_sd15/
```

结果大概会长这样：

```text
eval_outputs_sd15/
├── albert_einstein/
│   ├── 0001__a_man_wearing_a_red_hat.png
│   ├── 0002__a_man_wearing_a_santa_hat.png
│   └── ...
├── andrew_ng/
└── ...
```

---

## 7. 你只需要记住这 3 步

### A. 装环境

```bash
conda create -n consistentid python=3.8.10 -y
conda activate consistentid
pip install -r requirements.txt
pip install onnxruntime-gpu==1.16.1
```

### B. 下载模型

用第 4 步那段 `python - <<'PY' ... PY`

### C. 运行

#### 单图测试

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model "$PROJECT_ROOT/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" \
  --consistentid_ckpt "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" \
  --image_encoder_path "$PROJECT_ROOT/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --face_parsing_path "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" \
  --input_image "$PROJECT_ROOT/examples/albert_einstein.jpg" \
  --prompt "A man wearing a santa hat" \
  --output_path "$PROJECT_ROOT/outputs_sd15/test.png"
```

#### 批量 evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base_model "$PROJECT_ROOT/pretrained_models/Realistic_Vision_V6.0_B1_noVAE" \
  --consistentid_ckpt "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin" \
  --image_encoder_path "$PROJECT_ROOT/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --face_parsing_path "$PROJECT_ROOT/pretrained_models/JackAILab_ConsistentID/face_parsing.pth" \
  --input_dir "$PROJECT_ROOT/ref_images" \
  --prompt_csv "$PROJECT_ROOT/evaluation/EvaluationIMGs_stars_prompts.csv" \
  --output_dir "$PROJECT_ROOT/eval_outputs_sd15"
```

---

## 8. 说明

- 现在最推荐你先用 **SD1.5**，也就是 `infer.py`
- 这个仓库 **没有官方一键打分脚本**
- 所以你现在先把它理解成：
  - **evaluation = 按 CSV 批量生成 benchmark 图片**

如果你后面还想加“评分”，再另外做 FaceID/CLIP 指标就行。
