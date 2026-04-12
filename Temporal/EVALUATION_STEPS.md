# HunyuanVideo：从下载模型到生成视频再到评测

你的当前情况：
- `E:/VBench/Temporal/temporal_prompts.json` 是给 **Wan 多图生成** 用的
- 你想评测 **HunyuanVideo** 这种视频模型
- 你希望 **沿用同样的 `auxiliary_info`** 作为评测标准
- 但 **prompt 不强行共用**：多图模型用多图 prompt，视频模型用视频 prompt

这是合理的，我建议就这样做。

---

## 1. 先说清楚现在的数据关系

你现在有两份关键数据：

### 1.1 多图 prompt 文件

```text
E:/VBench/Temporal/temporal_prompts.json
```

这份文件里已经有：
- `id`
- `dimension`
- `prompt`
- `auxiliary_info`

它更适合 Wan 这种多图模型。

---

### 1.2 原始 benchmark 文本信息

```text
E:/VBench/Temporal/prompts/VBench2_full_text_info.json
```

这份文件原本是从 benchmark prompt 来的。现在我已经帮你把 **同样的 `auxiliary_info` 加回去了**，一共 97 条，和当前 benchmark 对齐。

也就是说，现在它已经包含：
- `caption`
- `dimension`
- `auxiliary_info`

你可以把它理解成：
- `caption`：更接近原始 benchmark / 更适合视频模型的文本描述
- `auxiliary_info`：统一评测标准

---

## 2. 推荐的最终使用方式

### Wan2.1 T2I
用：

```text
E:/VBench/Temporal/temporal_prompts.json
```

因为它里面的 prompt 已经改成多图风格了。

---

### HunyuanVideo
建议用：

```text
E:/VBench/Temporal/prompts/VBench2_full_text_info.json
```

因为里面的 `caption` 更像视频生成 prompt。

同时继续沿用：
- 相同 `dimension`
- 相同 `auxiliary_info`

这样就达到了你想要的效果：
- **image 模型** 用 image prompt
- **video 模型** 用 video prompt
- **评测标准** 用同一套 `auxiliary_info`

---

## 3. 能不能直接 `pip install VBench`？

**不建议。**

原因：
- 你现在评测用的是本地改过的 `E:/VBench/Temporal`
- 这个仓库只保留了 4 个维度
- 你已经改过多图评测逻辑
- 当前 evaluator 依赖本地 `vbench2` 和本地 `llava`

所以请直接使用你本地这个仓库：

```text
E:/VBench/Temporal/evaluate.py
```

---

## 4. 从新环境开始

Linux 示例：

```bash
python3 -m venv /path/to/Temporal/.venv
source /path/to/Temporal/.venv/bin/activate
```

---

## 5. 安装评测依赖

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

通常就是上面最后一条没装。

---

## 6. 下载 Temporal 评测用模型

这是 **评测** 用的，不是 Hunyuan 生成视频用的。

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"

huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

---

## 7. 下载 HunyuanVideo-1.5

推荐用官方 `HunyuanVideo-1.5`。

先装下载工具：

```bash
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

官方还提到 vision encoder：

```bash
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/vision_encoder/siglip --token <your_hf_token>
```

如果你现在先跑 **T2V**，可以先优先保证主模型和 text encoder 可用。

---

## 8. 安装 HunyuanVideo-1.5 代码

官方 README 里的基本依赖方式是：

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

如果你只想最小先跑通 T2V，可以先不折腾 FlashAttention / SageAttention / sparse attention 这些加速库。

---

## 9. HunyuanVideo 用什么 prompt

你现在不要直接用 `temporal_prompts.json` 里的多图 prompt 去生成视频。

### 推荐做法
Hunyuan 这边使用：

```text
E:/VBench/Temporal/prompts/VBench2_full_text_info.json
```

其中：
- 用 `caption` 作为 **video prompt**
- 用同条目的 `auxiliary_info` 作为后续评测参考标准

也就是：

- Wan：用 `temporal_prompts.json -> prompt`
- Hunyuan：用 `VBench2_full_text_info.json -> caption`
- 两边评测时都遵循相同的 `auxiliary_info`

这正是你想要的“prompt 各用各的，auxiliary_info 同步”。

---

## 10. HunyuanVideo 最小生成方式

官方仓库的生成入口是 `generate.py`。

最核心命令形式：

```bash
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH
```

对于 **T2V**：
- `--image_path none`
- `--model_path ./ckpts`

官方常用参数还包括：
- `--num_inference_steps 50`
- `--video_length 121`
- `--rewrite false`
- `--offloading true/false`

---

## 11. 给你的最小 Hunyuan T2V 示例

假设你已经进入 HunyuanVideo-1.5 仓库目录：

```bash
PROMPT="$(python - <<'PY'
import json
from pathlib import Path
p = Path('/path/to/Temporal/prompts/VBench2_full_text_info.json')
data = json.loads(p.read_text(encoding='utf-8'))
first_key = next(iter(data))
print(data[first_key]['caption'])
PY
)"

torchrun --nproc_per_node=1 generate.py \
  --prompt "$PROMPT" \
  --image_path none \
  --resolution 720p \
  --aspect_ratio 16:9 \
  --num_inference_steps 50 \
  --video_length 121 \
  --rewrite false \
  --output_path ./outputs/sample.mp4 \
  --model_path ./ckpts
```

建议你先只生成 1 条，确认链路能跑通。

---

## 12. 怎么批量生成 97 条视频

你最终需要做的是：

1. 读取 `E:/VBench/Temporal/prompts/VBench2_full_text_info.json`
2. 对每一条：
   - 取 `caption` 作为 Hunyuan 的视频 prompt
   - 保留它的 `dimension`
   - 保留它的 `auxiliary_info`
3. 调 Hunyuan `generate.py` 生成视频
4. 把输出视频整理到按维度划分的目录里

例如：

```text
/path/to/hunyuan_videos/
├── Dynamic_Attribute/
├── Dynamic_Spatial_Relationship/
├── Motion_Order_Understanding/
└── Complex_Plot/
```

---

## 13. 评测前最重要的注意点

当前 `Temporal` 的 **视频评测** 走的是 `vbench_standard`。

所以它默认按这个规则找视频：

```text
<prompt前180字符>-0.mp4
<prompt前180字符>-1.mp4
<prompt前180字符>-2.mp4
```

也就是说：

### 如果你直接用 Hunyuan 自己生成的输出文件名
通常 **不能直接评测**。

你需要二选一：

### 方案 A：重命名视频
把 Hunyuan 生成结果重命名成 benchmark 需要的名字。

### 方案 B：改评测代码
把视频评测逻辑改成像你多图模式那样，支持：
- `id -> video_path`
- `id -> prompt`
- `id -> auxiliary_info`

**我更推荐方案 B**，因为这样和你现在的多图 pipeline 更统一。

---

## 14. 当前最短执行顺序

### A. 配好 Temporal 评测环境

```bash
python3 -m venv /path/to/Temporal/.venv
source /path/to/Temporal/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install decord pillow tqdm requests pyyaml scenedetect gdown
python -m pip install transformers sentencepiece accelerate "huggingface_hub[cli]"
python -m pip install -e "/path/to/Temporal/vbench2/third_party/LLaVA_NeXT"
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

### B. 配好 Hunyuan 环境并下载模型

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

### C. 先生成 1 条视频做 smoke test
- prompt 来自 `VBench2_full_text_info.json` 的 `caption`
- rewrite 关闭
- 输出 1 个 mp4

### D. 批量生成 97 条
- 用 `caption` 生成
- 用同样的 `auxiliary_info` 做评测标准

### E. 再接 Temporal 的视频评测
- 如果文件名不匹配，先重命名或者改代码

---

## 15. 最后结论

你现在最合理的做法就是：

- **Wan**：`temporal_prompts.json` 里的多图 prompt
- **Hunyuan**：`VBench2_full_text_info.json` 里的 `caption`
- **两边统一评测标准**：同样的 `auxiliary_info`

这就是你想要的：
- prompt 各用各的
- auxiliary_info 保持同步

如果你愿意，我下一步可以直接继续帮你做两个中的一个：

1. **写一个 `run_inference_hunyuan.py`**
   - 读取 `E:/VBench/Temporal/prompts/VBench2_full_text_info.json`
   - 用 `caption` 批量生成 97 条视频
   - 同时保留 `dimension/auxiliary_info`

2. **改 `Temporal` 的视频评测代码**
   - 不再依赖长文件名
   - 改成按 `id` 对齐视频和评测信息
