# Temporal 评测详细说明

本文档说明如何在 `E:/VBench/Temporal` 中完成：
1. 生成多图 prompt
2. 调用 Wan 生成多张图片
3. 使用新的 `temporal_sequence` 模式进行评测

---

## 1. 当前整体流程

当前 `Temporal` 项目的逻辑分为三步：

### 第一步：从 benchmark prompt 生成多图 prompt
脚本：
- `E:/VBench/Temporal/change_prompt_to_multi_image.py`

输入：
- `E:/VBench/Temporal/prompts/meta_info/*.json`

输出：
- `E:/VBench/Temporal/temporal_prompts.json`

这个输出文件中，每条样本都包含：
- `id`
- `dimension`
- `prompt`
- `global_prompt`
- `num_images`
- `sub_prompts`
- `auxiliary_info`

---

### 第二步：调用 Wan 生成多张图片
脚本：
- `E:/VBench/Temporal/run_inference_wan.py`

输入：
- `E:/VBench/Temporal/temporal_prompts.json`

输出：
- `E:/VBench/Temporal/temporal_results/<id>/<index>.png`

例如：
```text
E:/VBench/Temporal/temporal_results/
├── tem_1/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
├── tem_2/
│   ├── 1.png
│   ├── 2.png
│   └── ...
```

这些图片现在就被视为“关键帧序列”。

---

### 第三步：评测这些关键帧序列
脚本：
- `E:/VBench/Temporal/evaluate.py`

模式：
- `temporal_sequence`

输入：
- `E:/VBench/Temporal/temporal_results/`
- `E:/VBench/Temporal/temporal_prompts.json`

输出：
- `evaluation_results/*.json`

评测时，程序会把 `temporal_results/<id>/1.png,2.png,...` 当作一段有序关键帧序列，而不是视频。

---

# 2. 需要的模型与依赖

## 2.1 需要哪些模型

当前评测逻辑依赖以下模型：

### 对于 `Dynamic_Attribute` 和 `Dynamic_Spatial_Relationship`
需要：
- `lmms-lab/LLaVA-Video-7B-Qwen2`

### 对于 `Motion_Order_Understanding` 和 `Complex_Plot`
需要：
- `lmms-lab/LLaVA-Video-7B-Qwen2`
- `Qwen/Qwen2.5-7B-Instruct`

对应代码位置：
- `E:/VBench/Temporal/vbench2/utils.py:313-329`

---

## 2.2 会自动下载吗？

**会。**

在评测时，`init_submodules(...)` 会自动检查模型是否存在；如果不存在，会尝试通过：
- `huggingface-cli download`

自动下载模型。

也就是说，如果你环境正确，第一次运行评测时会自动下载。

---

## 2.3 模型默认下载到哪里

模型缓存目录在：
- `VBENCH2_CACHE_DIR` 环境变量指定的位置
- 如果你没有设置这个环境变量，则默认下载到：

```text
~/.cache/vbench2
```

在 Windows 下通常会对应到类似：
```text
C:/Users/<你的用户名>/.cache/vbench2
```

对应代码位置：
- `E:/VBench/Temporal/vbench2/utils.py:37-39`

---

## 2.4 建议手动提前下载吗？

**建议。**

因为：
- 首次评测下载模型会比较慢
- 可能受网络、Hugging Face 访问、CLI 配置影响
- 提前下载更稳定

---

# 3. 如何手动下载评测模型

## 3.1 安装 Hugging Face CLI
在虚拟环境中执行：

```bash
E:/VBench/Temporal/.venv/Scripts/python.exe -m pip install "huggingface_hub[cli]"
```

---

## 3.2 下载 LLaVA-Video-7B-Qwen2

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "C:/Users/admin/.cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
```

---

## 3.3 下载 Qwen2.5-7B-Instruct

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "C:/Users/admin/.cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

---

## 3.4 如果你想把缓存目录改到别处
例如改到：

```text
E:/model_cache/vbench2
```

### PowerShell
```powershell
$env:VBENCH2_CACHE_DIR="E:/model_cache/vbench2"
```

### cmd
```cmd
set VBENCH2_CACHE_DIR=E:/model_cache/vbench2
```

之后再运行评测，模型就会下载/读取到这个目录。

---

# 4. 环境准备

## 4.1 激活虚拟环境

```bash
E:/VBench/Temporal/.venv/Scripts/activate
```

如果不激活，也可以直接调用：

```bash
E:/VBench/Temporal/.venv/Scripts/python.exe
```

---

## 4.2 设置 DashScope API Key

### PowerShell
```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"
```

### cmd
```cmd
set DASHSCOPE_API_KEY=你的API_KEY
```

---

## 4.3 可选：设置模型缓存目录

### PowerShell
```powershell
$env:VBENCH2_CACHE_DIR="C:/Users/admin/.cache/vbench2"
```

### cmd
```cmd
set VBENCH2_CACHE_DIR=C:/Users/admin/.cache/vbench2
```

---

# 5. 生成多图 prompt

如果你还没生成 `temporal_prompts.json`，先运行：

```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/change_prompt_to_multi_image.py"
```

输出文件：

```text
E:/VBench/Temporal/temporal_prompts.json
```

---

# 6. 运行 Wan 推理

```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/run_inference_wan.py"
```

它会读取：
- `temporal_prompts.json`

并生成：
- `temporal_results/<id>/1.png ...`

---

# 7. 运行评测

## 7.1 评测全部四个维度

```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results"
```

---

## 7.2 参数解释

### `--videos_path`
虽然名字还是 `videos_path`，但在 `temporal_sequence` 模式下，它实际上表示：

```text
图片序列根目录
E:/VBench/Temporal/temporal_results
```

程序会到这里找：
- `tem_1/`
- `tem_2/`
- ...

每个目录里的：
- `1.png`
- `2.png`
- `3.png`

会被当作关键帧。

---

### `--mode temporal_sequence`
这是关键参数。

表示：
- 不再从 `.mp4` 中均匀抽帧
- 而是直接读取 `temporal_results/<id>/` 里的有序图片序列

---

### `--temporal_prompt_file`
传入：

```text
E:/VBench/Temporal/temporal_prompts.json
```

作用：
- 建立 `id -> dimension`
- 建立 `id -> prompt`
- 建立 `id -> auxiliary_info`

评测时会根据这里的 `id` 去找：
- `temporal_results/<id>/`

---

### `--dimension`
当前支持：
- `Dynamic_Attribute`
- `Dynamic_Spatial_Relationship`
- `Motion_Order_Understanding`
- `Complex_Plot`

你可以只评其中一个，也可以一起评。

---

## 7.3 只评某个单一维度

### 只评 `Dynamic_Attribute`
```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Dynamic_Attribute \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results_dynamic_attribute"
```

### 只评 `Dynamic_Spatial_Relationship`
```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Dynamic_Spatial_Relationship \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results_spatial"
```

### 只评 `Motion_Order_Understanding`
```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Motion_Order_Understanding \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results_motion"
```

### 只评 `Complex_Plot`
```bash
E:/VBench/Temporal/.venv/Scripts/python.exe "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Complex_Plot \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results_plot"
```

---

# 8. 评测内部现在是怎么工作的

## 8.1 新模式的核心变化
旧模式：
- 输入视频 `.mp4`
- 从视频里均匀抽样关键帧

新模式：
- 输入图片文件夹 `temporal_results/<id>/`
- 直接把 `1.png, 2.png, ...` 当作关键帧序列

也就是说：
**你生成的这些图片本身就是 evaluator 要看的关键帧。**

---

## 8.2 当前四个 evaluator 的含义

### Dynamic_Attribute
检查：
- 开始状态是否正确
- 结束状态是否正确
- 属性变化是否发生

### Dynamic_Spatial_Relationship
检查：
- 起始空间关系是否正确
- 结束空间关系是否正确

### Motion_Order_Understanding
检查：
- 图像序列中的动作顺序是否正确

### Complex_Plot
检查：
- 图像序列是否覆盖关键剧情节点

---

# 9. 输出结果

评测完成后，会在你指定的 `output_path` 下生成结果文件，例如：

```text
E:/VBench/Temporal/evaluation_results/results_YYYY-MM-DD-HH-MM-SS_eval_results.json
E:/VBench/Temporal/evaluation_results/results_YYYY-MM-DD-HH-MM-SS_full_info.json
```

通常：
- `*_eval_results.json`：最终维度得分与明细
- `*_full_info.json`：本次实际用于评测的样本映射信息

---

# 10. 常见问题排查

## 10.1 找不到 `temporal_prompts.json`
确认文件存在：

```text
E:/VBench/Temporal/temporal_prompts.json
```

---

## 10.2 找不到某个 `tem_x` 文件夹
检查：

```text
E:/VBench/Temporal/temporal_results/tem_x/
```

是否存在，并且里面有：
- `1.png`
- `2.png`
- ...

---

## 10.3 图片命名顺序不对
当前评测按数字排序：
- `1.png`
- `2.png`
- `3.png`

不要用：
- `img1.png`
- `frame_a.png`
- `001_final.png`

最稳妥就是纯数字文件名。

---

## 10.4 模型下载失败
请先确认：

### 是否安装了 Hugging Face CLI
```bash
E:/VBench/Temporal/.venv/Scripts/python.exe -m pip install "huggingface_hub[cli]"
```

### 是否能访问 Hugging Face
如果不能稳定访问，建议手动下载。

---

## 10.5 显存/加载失败
LLaVA-Video-7B-Qwen2 和 Qwen2.5-7B-Instruct 都比较大。
如果显存不足，可能会在评测时失败。

建议：
- 先只跑一个维度测试
- 确认模型能正常加载后，再跑全部

---

# 11. 推荐完整执行顺序

```bash
# 1. 激活虚拟环境
E:/VBench/Temporal/.venv/Scripts/activate

# 2. 设置 API Key
# PowerShell:
$env:DASHSCOPE_API_KEY="你的API_KEY"

# 3. 可选：设置模型缓存目录
$env:VBENCH2_CACHE_DIR="C:/Users/admin/.cache/vbench2"

# 4. 生成多图 prompt
python "E:/VBench/Temporal/change_prompt_to_multi_image.py"

# 5. 调用 Wan 生成图片
python "E:/VBench/Temporal/run_inference_wan.py"

# 6. 跑评测
python "E:/VBench/Temporal/evaluate.py" \
  --videos_path "E:/VBench/Temporal/temporal_results" \
  --dimension Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot \
  --mode temporal_sequence \
  --temporal_prompt_file "E:/VBench/Temporal/temporal_prompts.json" \
  --output_path "E:/VBench/Temporal/evaluation_results"
```

---

# 12. 总结

你现在的评测逻辑已经不再是：
- 从视频中抽样关键帧

而是：
- 直接把 `run_inference_wan.py` 生成的多张图片当作关键帧序列来评测

也就是说，现在 Temporal 的 evaluator 本质上已经是：

**关键帧序列评测器**

而不是纯视频评测器。
