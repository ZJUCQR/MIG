# HunyuanVideo：从下载模型到生成视频到评测

这份文档完全按照你当前 `E:/VBench/Temporal` 里的**现有逻辑**来写。

你的目标是：
- Wan 多图已经生成并评测完
- 现在用 **HunyuanVideo** 生成视频
- 然后用 `E:/VBench/Temporal/evaluate.py` 评测视频
- 最后和 Wan 的四个维度结果做对比

---

## 1. 当前使用哪份 prompt

HunyuanVideo 不用：

```text
E:/VBench/Temporal/temporal_prompts.json
```

因为那份是给多图模型改写过的 prompt。

HunyuanVideo 现在用：

```text
E:/VBench/Temporal/prompts/VBench2_full_text_info.json
```

这份文件里现在已经有 97 条数据，每条都带有：
- `caption`
- `dimension`
- `auxiliary_info`

也就是说：
- **Hunyuan 生成视频时** 用 `caption`
- **评测标准** 仍然沿用相同的 `auxiliary_info`

---

## 2. 从新环境开始

Linux 示例：

```bash
python3 -m venv /path/to/Temporal/.venv
source /path/to/Temporal/.venv/bin/activate
```

---

## 3. 安装 Temporal 评测依赖

这是给 `E:/VBench/Temporal/evaluate.py` 用的：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install decord pillow tqdm requests pyyaml scenedetect gdown
python -m pip install transformers sentencepiece accelerate "huggingface_hub[cli]"
python -m pip install -e "/path/to/Temporal/vbench2/third_party/LLaVA_NeXT"
```

如果报：

```text
No module named 'llava'
```

通常就是最后一条没有装好。

---

## 4. 下载 Temporal 评测模型

这是**评测**用的，不是 Hunyuan 生成视频用的：

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"

huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

---

## 5. 下载 HunyuanVideo-1.5

先进入你准备放 Hunyuan 的目录，然后执行：

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
pip install -U "huggingface_hub[cli]"
pip install modelscope
```

下载主模型：

```bash
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts
```

下载文本编码器：

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
```

如果后面还需要 vision encoder，再补：

```bash
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/vision_encoder/siglip --token <your_hf_token>
```

---

## 6. 先生成 1 条视频做 smoke test

你现在已经有：

- `E:/VBench/Temporal/run_inference_hunyuan.py`

它会：
- 读取 `prompts/VBench2_full_text_info.json`
- 用 `caption` 作为视频 prompt
- 默认每条生成 **3 个视频**
- 输出文件名直接按当前 `vbench_standard` 评测要求命名：
  - `...-0.mp4`
  - `...-1.mp4`
  - `...-2.mp4`

所以先做最小测试：

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

如果只想先测试某个维度，比如 `Dynamic_Attribute`：

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --dimensions Dynamic_Attribute \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

---

## 7. 批量生成全部 97 条视频

直接运行：

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

如果中途断了，重跑时建议加：

```bash
--skip-existing
```

例如：

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --variants-per-prompt 3 \
  --skip-existing \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

---

## 8. 生成后的视频目录结构

生成完成后，目录大致是：

```text
/path/to/Temporal/hunyuan_videos/
├── Dynamic_Attribute/
│   ├── <prompt前180字符>-0.mp4
│   ├── <prompt前180字符>-1.mp4
│   ├── <prompt前180字符>-2.mp4
│   └── ...
├── Dynamic_Spatial_Relationship/
├── Motion_Order_Understanding/
└── Complex_Plot/
```

这和当前 `evaluate.py` / `vbench_standard` 的匹配逻辑是兼容的。

当前代码里写死了每个 prompt 默认找 3 个视频，见：
- `E:/VBench/Temporal/vbench2/__init__.py:76`
- `E:/VBench/Temporal/vbench2/__init__.py:77`

所以你现在默认每条生成 3 个视频，是为了**直接适配现有评测逻辑**。

---

## 9. 生成视频之后怎么评测

现在不需要改代码，直接评测即可。

### 9.1 先测一个维度
例如先测 `Dynamic_Attribute`：

```bash
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/hunyuan_videos/Dynamic_Attribute" \
  --dimension Dynamic_Attribute \
  --mode vbench_standard \
  --output_path "/path/to/Temporal/evaluation_results_hunyuan/Dynamic_Attribute"
```

---

### 9.2 再跑全部 4 个维度

```bash
for dim in Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot; do
  python "/path/to/Temporal/evaluate.py" \
    --videos_path "/path/to/Temporal/hunyuan_videos/${dim}" \
    --dimension "${dim}" \
    --mode vbench_standard \
    --output_path "/path/to/Temporal/evaluation_results_hunyuan/${dim}"
done
```

---

## 10. 评测结果怎么看

每个维度跑完后，会在对应输出目录下生成：

```text
results_YYYY-MM-DD-HH-MM-SS_eval_results.json
```

你最终比较这 4 个维度：
- `Dynamic_Attribute`
- `Dynamic_Spatial_Relationship`
- `Motion_Order_Understanding`
- `Complex_Plot`

总分就是：

```text
(Complex Plot + Dynamic Attribute + Dynamic Spatial Relationship + Motion Order Understanding) / 4
```

---

## 11. 最短执行顺序

### Step 1：建环境

```bash
python3 -m venv /path/to/Temporal/.venv
source /path/to/Temporal/.venv/bin/activate
```

### Step 2：装 Temporal 评测依赖

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install decord pillow tqdm requests pyyaml scenedetect gdown
python -m pip install transformers sentencepiece accelerate "huggingface_hub[cli]"
python -m pip install -e "/path/to/Temporal/vbench2/third_party/LLaVA_NeXT"
```

### Step 3：下 Temporal 评测模型

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

### Step 4：下 Hunyuan 并装依赖

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
pip install -U "huggingface_hub[cli]"
pip install modelscope
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
```

### Step 5：先生成 1 条做 smoke test

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --limit 1 \
  --variants-per-prompt 3 \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

### Step 6：批量生成 97 条

```bash
python "/path/to/Temporal/run_inference_hunyuan.py" \
  --hunyuan-root "/path/to/HunyuanVideo-1.5" \
  --model-path "/path/to/HunyuanVideo-1.5/ckpts" \
  --variants-per-prompt 3 \
  --skip-existing \
  --output-root "/path/to/Temporal/hunyuan_videos"
```

### Step 7：先测一个维度

```bash
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/hunyuan_videos/Dynamic_Attribute" \
  --dimension Dynamic_Attribute \
  --mode vbench_standard \
  --output_path "/path/to/Temporal/evaluation_results_hunyuan/Dynamic_Attribute"
```

### Step 8：再测全部维度

```bash
for dim in Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot; do
  python "/path/to/Temporal/evaluate.py" \
    --videos_path "/path/to/Temporal/hunyuan_videos/${dim}" \
    --dimension "${dim}" \
    --mode vbench_standard \
    --output_path "/path/to/Temporal/evaluation_results_hunyuan/${dim}"
done
```

---

## 12. 最后一句话总结

你现在这套逻辑已经是通的：
- `VBench2_full_text_info.json` 提供视频 prompt + auxiliary_info
- `run_inference_hunyuan.py` 负责生成 97 条、每条 3 个视频
- `evaluate.py` 直接按 `vbench_standard` 评测

所以你现在只需要：
**下载 Hunyuan → 跑 `run_inference_hunyuan.py` → 跑 `evaluate.py`。**
