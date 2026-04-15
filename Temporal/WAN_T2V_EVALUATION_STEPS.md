# Wan T2V：从生成到评测

这份文档对应：
- `E:/VBench/Temporal/run_inference_wan_t2v.py`
- `E:/VBench/Temporal/evaluate.py`

目标是让 **Wan text-to-video** 走和当前 Hunyuan 一样的 benchmark pipeline。

---

## 1. 使用哪份 prompt

Wan T2V 现在使用：

```text
E:/VBench/Temporal/prompts/VBench2_full_text_info.json
```

每条样本会读取：
- `caption`：作为视频生成 prompt
- `dimension`：决定输出目录
- `auxiliary_info`：作为后续评测标准参考

也就是说：
- Wan 多图模型：你之前用的是 `temporal_prompts.json`
- Wan T2V 视频模型：现在用的是 `VBench2_full_text_info.json`

---

## 2. 先准备环境

建议在新环境里运行：

```bash
python3 -m venv /path/to/Temporal/.venv
source /path/to/Temporal/.venv/bin/activate
```

安装基础依赖：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
```

然后安装 Wan / diffsynth 相关依赖。

> 这一部分依赖以你本地实际可用的 Wan 环境为准；关键是要确保下面这几个 import 能成功：

```python
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
```

如果你已经在另一套环境里跑通过 Wan T2V，可以直接复用那套环境。

---

## 3. 运行逻辑是什么

`run_inference_wan_t2v.py` 现在会：

1. 读取 `prompts/VBench2_full_text_info.json`
2. 用 `caption` 作为 prompt
3. 按 `dimension` 分目录输出
4. 默认每条生成 **3 个视频**
5. 文件名直接命名成：

```text
<prompt前180字符>-0.mp4
<prompt前180字符>-1.mp4
<prompt前180字符>-2.mp4
```

这样做的原因是：
- 当前 `evaluate.py --mode vbench_standard` 的逻辑写死每条 prompt 默认找 3 个视频
- 所以 Wan T2V 输出必须和这个规则一致，才能直接评测

---

## 4. 先做 smoke test

先只生成 1 条：

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

如果你只想先试某个维度，比如 `Dynamic_Attribute`：

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --dimensions Dynamic_Attribute \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

---

## 5. 批量生成全部视频

直接跑：

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

如果中断后继续跑：

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --variants-per-prompt 3 \
  --skip-existing \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

---

## 6. 常用参数

### 指定维度

```bash
--dimensions Dynamic_Attribute Motion_Order_Understanding
```

### 只生成前几条做测试

```bash
--limit 2
```

### 控制随机种子

```bash
--base-seed 123
```

### 改模型

```bash
--model-id Wan-AI/Wan2.1-T2V-1.3B
```

### 改保存视频参数

```bash
--fps 15 --quality 5
```

### 关闭 tiled

```bash
--no-tiled
```

---

## 7. 输出目录结构

生成完成后目录类似：

```text
/path/to/Temporal/wan_t2v_videos/
├── Dynamic_Attribute/
│   ├── <prompt前180字符>-0.mp4
│   ├── <prompt前180字符>-1.mp4
│   ├── <prompt前180字符>-2.mp4
│   └── ...
├── Dynamic_Spatial_Relationship/
├── Motion_Order_Understanding/
└── Complex_Plot/
```

---

## 8. 生成之后怎么评测

直接用现有的：

```text
E:/VBench/Temporal/evaluate.py
```

### 8.1 先测一个维度

例如先测 `Dynamic_Attribute`：

```bash
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/wan_t2v_videos/Dynamic_Attribute" \
  --dimension Dynamic_Attribute \
  --mode vbench_standard \
  --output_path "/path/to/Temporal/evaluation_results_wan_t2v/Dynamic_Attribute"
```

### 8.2 再跑全部 4 个维度

```bash
for dim in Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot; do
  python "/path/to/Temporal/evaluate.py" \
    --videos_path "/path/to/Temporal/wan_t2v_videos/${dim}" \
    --dimension "${dim}" \
    --mode vbench_standard \
    --output_path "/path/to/Temporal/evaluation_results_wan_t2v/${dim}"
done
```

---

## 9. 评测结果看哪里

每个维度跑完后，结果会在对应输出目录里：

```text
results_YYYY-MM-DD-HH-MM-SS_eval_results.json
```

最后比较这 4 个维度：
- `Dynamic_Attribute`
- `Dynamic_Spatial_Relationship`
- `Motion_Order_Understanding`
- `Complex_Plot`

总分就是：

```text
(Complex Plot + Dynamic Attribute + Dynamic Spatial Relationship + Motion Order Understanding) / 4
```

---

## 10. 最短执行顺序

### Step 1：生成 1 条测试

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

### Step 2：先测一个维度

```bash
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/wan_t2v_videos/Dynamic_Attribute" \
  --dimension Dynamic_Attribute \
  --mode vbench_standard \
  --output_path "/path/to/Temporal/evaluation_results_wan_t2v/Dynamic_Attribute"
```

### Step 3：批量生成全部

```bash
python "/path/to/Temporal/run_inference_wan_t2v.py" \
  --variants-per-prompt 3 \
  --skip-existing \
  --output-root "/path/to/Temporal/wan_t2v_videos"
```

### Step 4：批量评测全部

```bash
for dim in Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot; do
  python "/path/to/Temporal/evaluate.py" \
    --videos_path "/path/to/Temporal/wan_t2v_videos/${dim}" \
    --dimension "${dim}" \
    --mode vbench_standard \
    --output_path "/path/to/Temporal/evaluation_results_wan_t2v/${dim}"
done
```

---

## 11. 一句话总结

你现在的 Wan T2V pipeline 已经改成和 Hunyuan 一样的思路了：
- 用 `VBench2_full_text_info.json` 的 `caption` 生成视频
- 每条默认生成 3 个视频
- 文件名直接适配 `vbench_standard`
- 生成完就能直接接 `evaluate.py`
