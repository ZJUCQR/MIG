# WAN 2.7 Image Pro 评测简明教程

本文档说明如何用当前仓库的 benchmark 来评测 `wan2.7-image-pro` 的文生图效果。

这里的 benchmark 指两部分：

1. Attend-and-Excite 使用的 prompt 集合
2. 仓库现有的两个评测脚本
   - `metrics/compute_clip_similarity.py`
   - `metrics/blip_captioning_and_clip_similarity.py`

当前仓库已经适配了 `run_inference.py`，可以直接调用 DashScope 的 `wan2.7-image-pro` 批量生成图片，并按本项目指标脚本要求的目录结构保存结果。

## 一、现在的评测思路

你现在不是在复现 Attend-and-Excite 本身，而是在复用它的 benchmark：

1. 用官方 prompt 列表生成图片
2. 把图片保存成仓库指标脚本可直接读取的格式
3. 用现有指标脚本计算结果

也就是说：

- 生成模型换成 `wan2.7-image-pro`
- prompt 和指标沿用 Attend-and-Excite 项目

## 二、`run_inference.py` 现在做了什么

当前 `run_inference.py` 已经改成面向 benchmark 评测，而不是原来那套多图 JSON 结构。

它现在支持：

1. 从 prompt 文件读取评测文本
   - 支持 `.txt`，每行一个 prompt
   - 也支持简单 `.json` 数组

2. 调用 DashScope 的 `wan2.7-image-pro` 生成图片

3. 按 benchmark 需要的目录结构保存结果
   - 形式是 `outputs_root/<prompt>/0.png, 1.png, 2.png...`

4. 支持断点续跑
   - 如果某个 prompt 目录里已经有足够数量的图片，会自动跳过

5. 支持分批请求
   - 如果单次 API 调用不适合请求太多张图，可以通过 `--api-batch-size` 分批补齐

6. 输出 `_manifest.jsonl`
   - 记录每次请求的 task id、文件名、耗时等信息，方便排查问题

7. 兼容当前的 `a.e_prompts.txt`
   - 这个文件虽然扩展名是 `.txt`，但内容实际上是 JSON 对象
   - `run_inference.py` 现在会自动展开其中所有类别下的 prompt

## 三、准备 prompt 文件

建议使用 `README.md` 里提供的官方 prompt 文件：

- `a.e_prompts.txt`

把它放在项目根目录，例如：

```text
/path/to/Attend-and-Excite/a.e_prompts.txt
```

`run_inference.py` 默认会优先查找：

1. `./a.e_prompts.txt`
2. `./evaluation/a.e_prompts.txt`
3. `./evaluation/prompts.txt`

如果都没有，就需要显式传 `--prompt-file`。

你当前仓库里的 [a.e_prompts.txt](/E:/VBench/Attend-and-Excite/a.e_prompts.txt) 不是“每行一个 prompt”的纯文本，而是一个 JSON 对象，包含 3 个子集：

- `animals`：66 条
- `animals_objects`：144 条
- `objects`：66 条

总计 276 条 prompt。

`run_inference.py` 现在会把这 3 个列表按顺序展开，作为完整评测集处理。

## 四、Linux 环境准备

建议环境：

- Linux
- Python 3.8+
- Conda 或 venv
- 可以访问 DashScope API

如果你还要跑仓库自带指标，建议准备 CUDA 环境，因为 CLIP/BLIP 在 GPU 上会更稳。

进入项目目录：

```bash
cd /path/to/Attend-and-Excite
```

## 五、安装依赖

如果你沿用仓库环境：

```bash
conda env create -f environment/environment.yaml
conda activate ldm
pip install -r environment/requirements.txt
pip install lavis
```

如果你只想跑 `wan2.7-image-pro` 生成，至少需要保证这些包可用：

- `dashscope`
- `requests`

如果没有，可以补装：

```bash
pip install dashscope requests
```

## 六、配置 DashScope API Key

运行前设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

如果没有这个变量，`run_inference.py` 不会执行。

## 七、生成 benchmark 图片

最常用的命令如下：

```bash
python run_inference.py \
  --prompt-file ./a.e_prompts.txt \
  --output-root ./outputs_wan_2_7_image_pro \
  --model-name wan2.7-image-pro \
  --image-size 2K \
  --num-images-per-prompt 4 \
  --api-batch-size 4
```

参数说明：

- `--prompt-file`
  - prompt 文件路径，推荐用官方 `a.e_prompts.txt`

- `--output-root`
  - 生成结果根目录

- `--model-name`
  - 默认为 `wan2.7-image-pro`

- `--image-size`
  - 传给 DashScope 的图片尺寸，例如 `2K`

- `--num-images-per-prompt`
  - 每个 prompt 生成多少张图
  - 当前默认值是 `4`

- `--api-batch-size`
  - 每次 API 请求生成多少张图
  - 如果目标总数更大，脚本会自动循环补齐
  - 当前默认值也是 `4`

生成后的目录结构应类似：

```text
outputs_wan_2_7_image_pro/
|-- a cat and a dog/
|   |-- 0.png
|   |-- 1.png
|   |-- 2.png
|   |-- 3.png
|-- a horse and a dog/
|   |-- 0.png
|   |-- 1.png
|   |-- 2.png
|   |-- 3.png
|-- _manifest.jsonl
```

这已经兼容本项目的指标脚本。

## 八、当前评测具体是怎么跑的

你现在这套评测，按当前代码默认设置，具体流程是这样的：

### 1. prompt 集合

使用 [a.e_prompts.txt](/E:/VBench/Attend-and-Excite/a.e_prompts.txt) 里的全部 276 条 prompt。

这 276 条 prompt 分成三类：

- `animals`：两个动物，如 `a cat and a dog`
- `animals_objects`：动物和物体，或者动物带属性，如 `a elephant with a crown`
- `objects`：两个物体或属性组合，如 `a pink crown and a purple bow`

### 2. 每个 prompt 生成多少张图

当前 `run_inference.py` 的默认设置是：

- `DEFAULT_NUM_IMAGES_PER_PROMPT = 4`
- `DEFAULT_API_BATCH_SIZE = 4`

所以默认情况下：

- 每个 prompt 生成 4 张图片
- 一次 API 调用就请求 4 张

如果某个模型接口一次不能稳定返回这么多张，可以把 `--api-batch-size` 调小，例如设成 `1` 或 `2`，脚本会循环请求，直到该 prompt 的图片数量补满 `--num-images-per-prompt`。

### 3. `run_inference.py` 如何对待每个 prompt

对每个 prompt，脚本的处理逻辑是：

1. 用 prompt 原文作为目录名
   - 例如 `outputs_wan_2_7_image_pro/a cat and a dog/`

2. 检查目录里已经有多少张图
   - 如果已经大于等于目标数量，默认直接跳过
   - 这就是断点续跑机制

3. 如果图片不够，就继续调用 DashScope
   - 每次请求 `batch_size = min(api_batch_size, 剩余张数)`

4. 下载返回的图片并按序编号保存
   - 文件名是 `0.png`、`1.png`、`2.png`、`3.png` ...

5. 把本次请求信息写入 `_manifest.jsonl`
   - 包括 prompt、task_id、返回张数、下载文件、耗时、模型名、尺寸

6. 如果传了 `--overwrite`
   - 会先清空该 prompt 目录下已有图片，再重新生成

因此，当前代码对每个 prompt 的策略不是“只请求一次”，而是“保证最终目录里达到指定图片数”。

### 4. 最终输出结构

按默认设置，最终会得到类似目录：

```text
outputs_wan_2_7_image_pro/
|-- a cat and a dog/
|   |-- 0.png
|   |-- 1.png
|   |-- 2.png
|   |-- 3.png
|-- a elephant with a crown/
|   |-- 0.png
|   |-- 1.png
|   |-- 2.png
|   |-- 3.png
|-- ...
|-- _manifest.jsonl
```

按默认值估算，完整跑完 276 条 prompt，会生成：

- `276 x 4 = 1104` 张图片

## 九、当前指标是如何计算的

当前项目里有两个指标脚本。

### 1. CLIP 图文相似度指标

命令：

```bash
python metrics/compute_clip_similarity.py \
  --output_path ./outputs_wan_2_7_image_pro \
  --metrics_save_path ./metrics_results_wan
```

这个脚本会对每个 prompt 目录下的所有图片做评测。

它对每个 prompt 做三类分数：

1. `full_text`
   - 图片和完整 prompt 的 CLIP 相似度

2. `first_half`
   - 如果 prompt 包含 ` and `，就按 `and` 切成两半
   - 如果 prompt 包含 ` with `，就按 `with` 切成两半
   - 计算图片和前半句的 CLIP 相似度

3. `second_half`
   - 计算图片和后半句的 CLIP 相似度

然后它会聚合出两个总指标：

- `full_text_aggregation`
  - 所有图片对完整 prompt 的平均相似度

- `min_first_second_aggregation`
  - 对每张图，取 `first_half` 和 `second_half` 中较小的那个值
  - 再对全体图片求平均

第二个指标更严格，因为它要求图像同时兼顾 prompt 的两个部分，而不是只贴近其中一半。

### 2. BLIP Caption + CLIP 指标

命令：

```bash
python metrics/blip_captioning_and_clip_similarity.py \
  --output_path ./outputs_wan_2_7_image_pro \
  --metrics_save_path ./metrics_results_wan
```

这个脚本的流程是：

1. 对每张图用 BLIP 生成 caption
2. 用 CLIP 计算该 caption 与原 prompt 的相似度
3. 对所有图片求：
   - `average_similarity`
   - `std_similarity`

它更偏向“图片内容能否被描述模型正确读出来”。

## 十、常用运行方式

只跑前 10 条 prompt：

```bash
python run_inference.py \
  --prompt-file ./a.e_prompts.txt \
  --output-root ./outputs_wan_2_7_image_pro \
  --limit 10
```

从第 50 条 prompt 开始跑：

```bash
python run_inference.py \
  --prompt-file ./a.e_prompts.txt \
  --output-root ./outputs_wan_2_7_image_pro \
  --start-index 50
```

覆盖重跑：

```bash
python run_inference.py \
  --prompt-file ./a.e_prompts.txt \
  --output-root ./outputs_wan_2_7_image_pro \
  --overwrite
```

如果输出目录里已经有足够数量的图片，默认会跳过，适合断点续跑。

## 十一、运行 CLIP 指标

图片生成完成后，直接跑项目原有脚本：

```bash
python metrics/compute_clip_similarity.py \
  --output_path ./outputs_wan_2_7_image_pro \
  --metrics_save_path ./metrics_results_wan
```

这个脚本会输出：

1. `full_text_aggregation`
   - 图像与完整 prompt 的平均 CLIP 相似度

2. `min_first_second_aggregation`
   - prompt 按 `and` 或 `with` 拆成两部分后，分别算相似度
   - 每张图取较小值，再整体平均

输出文件：

```text
metrics_results_wan/
|-- clip_raw_metrics.json
|-- clip_aggregated_metrics.json
```

## 十二、运行 BLIP + CLIP 指标

如果你还想评估 caption-level 语义一致性，运行：

```bash
python metrics/blip_captioning_and_clip_similarity.py \
  --output_path ./outputs_wan_2_7_image_pro \
  --metrics_save_path ./metrics_results_wan
```

输出文件：

```text
metrics_results_wan/
|-- blip_raw_metrics.json
|-- blip_aggregated_metrics.json
```

## 十三、建议的评测设置

如果你是要做模型对比，建议固定以下条件：

1. 使用同一份 `a.e_prompts.txt`
2. 每个 prompt 生成相同数量的图片
3. 所有模型使用相同评测脚本
4. 各模型分别写入不同的 `output_root`

例如：

- `./outputs_wan_2_7_image_pro`
- `./outputs_other_model`

然后分别对这些目录跑同样的指标。

## 十四、已知限制

1. 这个 benchmark 最初是为单图 T2I 设计的。
   - 你现在是在用它评估 WAN 的文生图结果，本质上是“复用 prompt 和指标”。

2. `compute_clip_similarity.py` 只适合包含 `and` 或 `with` 的 prompt。

3. 当前指标脚本不会读取 `_manifest.jsonl`。
   - 它只看图片目录。

4. `wan2.7-image-pro` 是否支持完全可控的 seed，不在当前脚本里处理。
   - 所以这套流程更适合做 benchmark 打分，不适合严格随机种子对齐实验。

5. DashScope 的正确模型名是 `wan2.7-image-pro`。
   - 如果传成 `wan-2.7-image-pro`，接口会报 `Model not exist`

6. 当前默认并不是论文 README 里举例的 65 张每 prompt。
   - 你现在改造后的 `run_inference.py` 默认是每个 prompt 4 张图
   - 如果你想拉高评测稳定性，可以把 `--num-images-per-prompt` 改大

## 十五、最短流程

如果你只想尽快跑通：

1. 准备 `a.e_prompts.txt`
2. `export DASHSCOPE_API_KEY=...`
3. 跑 `run_inference.py` 生成图片
4. 跑 `metrics/compute_clip_similarity.py`
5. 需要的话再跑 `metrics/blip_captioning_and_clip_similarity.py`

对应命令：

```bash
python run_inference.py --prompt-file ./a.e_prompts.txt --output-root ./outputs_wan_2_7_image_pro
python metrics/compute_clip_similarity.py --output_path ./outputs_wan_2_7_image_pro --metrics_save_path ./metrics_results_wan
python metrics/blip_captioning_and_clip_similarity.py --output_path ./outputs_wan_2_7_image_pro --metrics_save_path ./metrics_results_wan
```
