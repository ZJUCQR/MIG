# Multi-Image Evaluation

这个文档对应 `run_infer.py` 生成的文生组图结果评测脚本:

`python evaluation/evaluate_multi_image.py`

## 1. 适用范围

当前任务是“纯文生组图”:
- 输入是 `run_infer.py` 输出的多张生成图
- 没有真实图
- 没有外部参考图

因此论文里的指标分成两类:

严格成立:
- `CLIP-T`

代理版本:
- `CLIP-I anchor`
- `DINO anchor`
- `FaceSim anchor`
- `FGIS anchor`

这里的 `anchor` 表示:
- 每个 prompt 组按文件名数字排序
- 默认取第 1 张图作为锚点图
- 其余图片都与这张锚点图比较

这不是论文原始定义，因为论文原定义里的 `CLIP-I / DINO / FaceSim / FGIS` 都需要真实图或参考图。

本仓库这版实现里:
- `FID` 不实现

## 2. 指标说明

`clip_t_sub`
- 每张生成图和对应 `sub_prompt` 的 CLIP 余弦相似度
- 如果该图片没有对应 `sub_prompt`，回退到 `global_prompt`，再回退到整条 `prompt`

`clip_t_global`
- 每张生成图和该组 `global_prompt` 的 CLIP 余弦相似度

`clip_i_anchor`
- 第 1 张图和组内其余图片的 CLIP 图像 embedding 余弦相似度

`dino_anchor`
- 第 1 张图和组内其余图片的 DINO embedding 余弦相似度

`facesim_anchor`
- 使用 `FaceNet` 比较第 1 张图和组内其余图片的人脸 embedding 相似度
- 人脸检测失败的图片会跳过，不记为 0 分

`fgis_anchor`
- 先做人脸检测，再用 `BiSeNet` 提取细粒度脸部区域
- 再对脸部区域做 DINO embedding 相似度
- 分割失败或检测失败会跳过，不记为 0 分

## 3. 依赖

脚本依赖:
- `transformers`
- `timm`
- `facenet-pytorch`
- `torch`
- `Pillow`

已经在 [requirements.txt](/E:/VBench/ConsistentID/requirements.txt) 里补充了:

```txt
facenet-pytorch==2.5.3
```

如果你要跑 `fgis_anchor`，还需要:
- `face_parsing.pth`

默认位置:

```text
./pretrained_models/JackAILab_ConsistentID/face_parsing.pth
```

## 4. DINO 权重准备

现在脚本默认支持自动下载官方 DINO 权重。

默认行为:
- 如果你没有传 `--dino-checkpoint`
- 且 `--dino-model` 是 `vit_small_patch16_224` 或 `vit_base_patch16_224`
- 脚本会自动下载官方 checkpoint
- 默认缓存目录是 `~/.cache/torch/hub/checkpoints`

你也可以显式指定缓存目录:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,dino_anchor \
  --dino-model vit_small_patch16_224 \
  --dino-cache-dir /path/to/dino_cache
```

如果你想手动下载，再传给脚本，也可以。

官方 DINO ViT-S/16:

```bash
mkdir -p /path/to/dino_cache
cd /path/to/dino_cache
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
```

官方 DINO ViT-B/16:

```bash
mkdir -p /path/to/dino_cache
cd /path/to/dino_cache
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```

手动下载后，运行时这样指定:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,dino_anchor,fgis_anchor \
  --dino-model vit_small_patch16_224 \
  --dino-checkpoint /path/to/dino_cache/dino_deitsmall16_pretrain.pth \
  --face-parsing-path ./pretrained_models/JackAILab_ConsistentID/face_parsing.pth
```

## 5. 输入目录约定

脚本默认读取:

```text
evaluation/multi_image_prompts.json
multi_image_results/
  cn_001/
    1.png
    2.png
    ...
  cn_002/
    1.png
    2.png
    ...
```

这和 `run_infer.py` 的默认输出结构一致。

## 6. 基本用法

只跑严格成立的文本对齐指标:

```bash
python evaluation/evaluate_multi_image.py \
  --prompt-file evaluation/multi_image_prompts.json \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t
```

跑完整的当前实现:

```bash
python evaluation/evaluate_multi_image.py \
  --prompt-file evaluation/multi_image_prompts.json \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,clip_i_anchor,dino_anchor,facesim_anchor,fgis_anchor \
  --face-parsing-path ./pretrained_models/JackAILab_ConsistentID/face_parsing.pth
```

如果让脚本自动下载 DINO:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,dino_anchor \
  --dino-model vit_small_patch16_224
```

如果 DINO 使用本地 checkpoint:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,dino_anchor \
  --dino-model vit_small_patch16_224 \
  --dino-checkpoint /path/to/dino_checkpoint.pth
```

如果只想评测部分指标:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,facesim_anchor
```

## 7. 输出文件

脚本会写 3 个文件到 `--output-dir`:

`summary.json`
- 全局配置
- 有效 group / image 数量
- 各指标全局均值和有效样本数
- `facesim_valid_ratio` 和 `fgis_valid_ratio`

`per_group.csv`
- 每个 prompt 组一行
- 包含:
  - `status`
  - `expected_images`
  - `found_images`
  - 各指标组均值
  - `facesim_valid_ratio`
  - `fgis_valid_ratio`

`per_image.csv`
- 每张图片一行
- 包含:
  - `prompt_id`
  - `image_index`
  - `is_anchor`
  - `sub_prompt_used`
  - `clip_t_sub`
  - `clip_t_global`
  - `clip_i_anchor`
  - `dino_anchor`
  - `facesim_anchor`
  - `fgis_anchor`
  - `face_detected`
  - `facial_region_ready`

## 8. 常见情况

`facesim_valid_ratio` 很低
- 说明 FaceNet/MTCNN 没有稳定检测到人脸
- 艺术风格图、黑白图、涂鸦图会明显降低覆盖率

`fgis_valid_ratio` 很低
- 说明人脸检测或脸部分割失败较多
- 这通常比 `facesim_anchor` 更严格

某个组只有 1 张图
- 仍然会计算 `clip_t_sub` 和 `clip_t_global`
- `clip_i_anchor / dino_anchor / facesim_anchor / fgis_anchor` 会为空

组目录不存在
- `per_group.csv` 里该组会记为 `missing`

## 9. 常见报错

如果你看到这类报错:

```text
RuntimeError: Unknown model (vit_small_patch16_224.dino)
```

原因是:
- 你的 `timm` 版本不认识 `vit_small_patch16_224.dino` 这个预训练标签
- 这不是 `run_infer.py` 的问题，是评测脚本里 DINO 模型名和当前 `timm` 版本不兼容

优先解决方案:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,dino_anchor,fgis_anchor \
  --dino-model vit_small_patch16_224 \
  --face-parsing-path ./pretrained_models/JackAILab_ConsistentID/face_parsing.pth
```

这里要注意:
- `--dino-model` 写架构名 `vit_small_patch16_224`
- 不传 `--dino-checkpoint` 时，脚本会自动下载官方 DINO ViT-S/16 权重
- 如果你想手动指定本地权重，也可以再传 `--dino-checkpoint`
- 不要把 `--dino-model` 继续写成 `vit_small_patch16_224.dino`

如果你暂时没有 DINO checkpoint，可以先去掉依赖 DINO 的指标:

```bash
python evaluation/evaluate_multi_image.py \
  --image-root multi_image_results \
  --output-dir multi_image_eval \
  --metrics clip_t,clip_i_anchor,facesim_anchor
```

这时会跳过:
- `dino_anchor`
- `fgis_anchor`

如果你就是想继续用 `vit_small_patch16_224.dino` 这个名字，那就需要升级到支持这个标签的 `timm` 版本。

## 10. 实现说明

当前评测设计遵循这几个原则:
- 没有参考图时，不伪造“论文原始指标”
- 所有代理版本都显式带 `anchor` 后缀
- 人脸检测失败和分割失败会跳过，不硬记 0 分

如果后面你补了真实参考图，我可以再把这套脚本扩展成:
- 标准 `CLIP-I`
- 标准 `DINO`
- 标准 `FaceSim`
- 标准 `FGIS`
- `FID`
