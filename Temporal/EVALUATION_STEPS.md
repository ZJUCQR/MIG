# Temporal 评测详细说明（Linux 服务器版）

你现在已经完成了：
- `change_prompt_to_multi_image.py`
- `run_inference_wan.py`

也就是说，你已经有了：

```text
/path/to/Temporal/temporal_prompts.json
/path/to/Temporal/temporal_results/<id>/1.png, 2.png, ...
```

你当前卡在 **运行评测之前的环境配置**。这份文档专门面向 **Linux 服务器**，详细说明如何把评测环境配置好，然后开始评测。

> 下面命令中的 `/path/to/Temporal` 请替换成你服务器上的实际项目路径。

---

# 1. 你现在的评测输入是什么

评测时需要两类输入：

## 1.1 prompt 元数据
文件：

```text
/path/to/Temporal/temporal_prompts.json
```

作用：
- 提供每条样本的 `id`
- 提供每条样本的 `dimension`
- 提供改写后的 `prompt`
- 提供对应 `auxiliary_info`

---

## 1.2 图片序列结果
目录：

```text
/path/to/Temporal/temporal_results/
```

结构例如：

```text
temporal_results/
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

评测时，程序会把这些图片当作“有序关键帧序列”。

---

# 2. 评测依赖什么环境

当前评测依赖两部分：

## 2.1 Python 运行环境
建议使用独立虚拟环境，例如：

```text
/path/to/Temporal/.venv
```

如果你还没有虚拟环境，可以创建：

```bash
python3 -m venv /path/to/Temporal/.venv
```

---

## 2.2 模型与推理依赖
当前四个 retained evaluator 依赖：

### 对于 `Dynamic_Attribute` 和 `Dynamic_Spatial_Relationship`
需要：
- `lmms-lab/LLaVA-Video-7B-Qwen2`

### 对于 `Motion_Order_Understanding` 和 `Complex_Plot`
需要：
- `lmms-lab/LLaVA-Video-7B-Qwen2`
- `Qwen/Qwen2.5-7B-Instruct`

模型加载逻辑在：
- `vbench2/utils.py`

---

# 3. 第一步：激活虚拟环境

```bash
source /path/to/Temporal/.venv/bin/activate
```

激活后你应该看到 shell 前面出现类似：

```text
(.venv)
```

如果你不想激活环境，也可以后续一直使用：

```bash
/path/to/Temporal/.venv/bin/python
```

---

# 4. 第二步：安装评测依赖

建议在虚拟环境中安装以下依赖。

## 4.1 基础依赖

```bash
/path/to/Temporal/.venv/bin/python -m pip install --upgrade pip
/path/to/Temporal/.venv/bin/python -m pip install torch torchvision torchaudio
/path/to/Temporal/.venv/bin/python -m pip install decord pillow tqdm requests pyyaml scenedetect gdown
/path/to/Temporal/.venv/bin/python -m pip install "huggingface_hub[cli]"
/path/to/Temporal/.venv/bin/python -m pip install transformers sentencepiece accelerate
```

如果你已经装过部分依赖，可以跳过重复步骤。

---

## 4.2 可选依赖（按报错再补）

```bash
/path/to/Temporal/.venv/bin/python -m pip install mmcv==2.2.0
/path/to/Temporal/.venv/bin/python -m pip install retinaface_pytorch==0.0.8 --no-deps
```

说明：
- `mmcv` 在 Linux 上通常比 Windows 好装，但仍可能依赖 CUDA / PyTorch 版本匹配
- 如果你只是先测试链路，可以先不装，缺了再补

---

# 5. 第三步：设置环境变量

你至少需要设置：
- `DASHSCOPE_API_KEY`

可选设置：
- `VBENCH2_CACHE_DIR`

建议在当前 shell 中直接 export：

## 5.1 设置 `DASHSCOPE_API_KEY`

```bash
export DASHSCOPE_API_KEY="你的API_KEY"
```

检查是否设置成功：

```bash
echo $DASHSCOPE_API_KEY
```

---

## 5.2 设置模型缓存目录（推荐）

推荐你显式指定模型缓存目录，例如：

```text
/path/to/model_cache/vbench2
```

设置方式：

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"
```

检查：

```bash
echo $VBENCH2_CACHE_DIR
```

如果不设置，默认缓存目录是：

```text
~/.cache/vbench2
```

也就是通常：

```text
/home/<your_user>/.cache/vbench2
```

---

# 6. 第四步：下载评测模型

## 是否必须手动下载？

**不是必须，但强烈建议手动下载。**

原因：
- 自动下载可能因为网络问题失败
- 第一次运行评测时会卡很久
- 手动下载便于确认缓存位置和模型完整性

---

## 6.1 先安装 Hugging Face CLI

如果上一步已经执行过，可以跳过；否则执行：

```bash
/path/to/Temporal/.venv/bin/python -m pip install "huggingface_hub[cli]"
```

测试 CLI：

```bash
huggingface-cli --help
```

如果提示找不到命令，可以直接用：

```bash
/path/to/Temporal/.venv/bin/python -m huggingface_hub.commands.huggingface_cli --help
```

---

## 6.2 下载 `LLaVA-Video-7B-Qwen2`

如果你设置了：

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"
```

那么建议下载到：

```text
/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2
```

命令：

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
```

---

## 6.3 下载 `Qwen/Qwen2.5-7B-Instruct`

建议下载到：

```text
/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct
```

命令：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

---

## 6.4 如果服务器不能直接访问 Hugging Face
如果你所在服务器不能稳定访问 Hugging Face，可以考虑：

1. 在本地或另一台有网络的机器上下载模型；
2. 用 `scp`/`rsync` 上传到服务器；
3. 保持目录结构不变，例如：

```text
/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2
/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct
```

上传后，再设置：

```bash
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"
```

即可让评测直接读取。

---

# 7. 第五步：检查模型目录是否正确

下载完成后，至少应该看到类似：

```text
/path/to/model_cache/vbench2/
├── lmms-lab/
│   └── LLaVA-Video-7B-Qwen2/
└── Qwen/
    └── Qwen2.5-7B-Instruct/
```

如果你没设置 `VBENCH2_CACHE_DIR`，则去：

```text
~/.cache/vbench2
```

里找。

---

# 8. 第六步：从你当前状态开始运行评测

你现在已经完成了 Wan 推理，所以可以直接执行评测：

```bash
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/temporal_results" \
  --dimension Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot \
  --mode temporal_sequence \
  --temporal_prompt_file "/path/to/Temporal/temporal_prompts.json" \
  --output_path "/path/to/Temporal/evaluation_results"
```

---

# 9. 评测参数详细解释

## `--videos_path`
虽然参数名还是 `videos_path`，但在 `temporal_sequence` 模式下，它实际表示：

```text
/path/to/Temporal/temporal_results
```

即：
- `tem_1/`
- `tem_2/`
- ...

这些子目录中存放的 `1.png, 2.png, 3.png...` 会被当作关键帧序列。

---

## `--mode temporal_sequence`
必须加这个参数。

含义：
- 不再读取 `.mp4`
- 不再从视频里抽帧
- 直接把图片目录视为有序时序帧序列

---

## `--temporal_prompt_file`
这里传：

```text
/path/to/Temporal/temporal_prompts.json
```

作用：
- 用 `id` 定位样本
- 读取该样本所属的 `dimension`
- 读取该样本的改写后 `prompt`
- 读取该样本的 `auxiliary_info`

---

## `--dimension`
当前支持：
- `Dynamic_Attribute`
- `Dynamic_Spatial_Relationship`
- `Motion_Order_Understanding`
- `Complex_Plot`

你可以只测一个维度，也可以全部一起测。

---

# 10. 单维度测试（强烈推荐先做）

在真正跑全部之前，建议你先跑一个最小测试，确认环境没问题。

## 10.1 先只跑 `Dynamic_Attribute`

```bash
/path/to/Temporal/.venv/bin/python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/temporal_results" \
  --dimension Dynamic_Attribute \
  --mode temporal_sequence \
  --temporal_prompt_file "/path/to/Temporal/temporal_prompts.json" \
  --output_path "/path/to/Temporal/evaluation_results_debug"
```

如果这个能正常跑，再跑全部维度。

---

# 11. 评测输出会保存到哪里

例如你传：

```text
--output_path /path/to/Temporal/evaluation_results
```

则最终会生成：

```text
/path/to/Temporal/evaluation_results/results_YYYY-MM-DD-HH-MM-SS_eval_results.json
/path/to/Temporal/evaluation_results/results_YYYY-MM-DD-HH-MM-SS_full_info.json
```

其中：
- `*_eval_results.json`：维度分数和每条样本结果
- `*_full_info.json`：本次评测实际用到的样本映射信息

---

# 12. 如果自动下载模型失败怎么办

## 12.1 先确认 CLI 是否能工作

```bash
huggingface-cli --help
```

## 12.2 先确认缓存目录是否存在
例如：

```bash
ls /path/to/model_cache/vbench2
```

## 12.3 手动重新下载模型

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"
```

## 12.4 如果公司/学校服务器封网
可以在本地或中转机下载后上传：

```bash
rsync -av /local/model_cache/vbench2/ user@server:/path/to/model_cache/vbench2/
```

---

# 13. 常见问题排查

## 13.1 找不到 `temporal_prompts.json`
确认：

```bash
ls /path/to/Temporal/temporal_prompts.json
```

---

## 13.2 找不到某个 `tem_x` 文件夹
检查：

```bash
ls /path/to/Temporal/temporal_results/tem_x
```

并确认里面至少有：
- `1.png`
- `2.png`

---

## 13.3 图片命名不规范
当前图片序列按数字顺序排序：
- `1.png`
- `2.png`
- `3.png`

不要混用：
- `img1.png`
- `frame_1.png`
- `001_final.png`

---

## 13.4 显存不足
LLaVA-Video-7B-Qwen2 与 Qwen2.5-7B-Instruct 体积较大。

建议：
- 先跑单一维度测试
- 不要一上来就跑全部
- 必要时只跑前几个样本做 smoke test

---

## 13.5 CUDA / PyTorch 版本不兼容
如果模型加载时报 CUDA 或 torch 错误，先检查：

```bash
/path/to/Temporal/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 是 `False`，则说明当前 PyTorch 没有正确识别 GPU。

---

# 14. 推荐最稳妥的执行顺序

```bash
# 1. 激活环境
source /path/to/Temporal/.venv/bin/activate

# 2. 设置环境变量
export DASHSCOPE_API_KEY="你的API_KEY"
export VBENCH2_CACHE_DIR="/path/to/model_cache/vbench2"

# 3. 安装依赖
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install decord pillow tqdm requests pyyaml scenedetect gdown
python -m pip install "huggingface_hub[cli]"
python -m pip install transformers sentencepiece accelerate

# 4. 手动下载评测模型
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --repo-type model --local-dir "/path/to/model_cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --repo-type model --local-dir "/path/to/model_cache/vbench2/Qwen/Qwen2.5-7B-Instruct"

# 5. 先测试一个维度
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/temporal_results" \
  --dimension Dynamic_Attribute \
  --mode temporal_sequence \
  --temporal_prompt_file "/path/to/Temporal/temporal_prompts.json" \
  --output_path "/path/to/Temporal/evaluation_results_debug"

# 6. 再跑全部维度
python "/path/to/Temporal/evaluate.py" \
  --videos_path "/path/to/Temporal/temporal_results" \
  --dimension Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot \
  --mode temporal_sequence \
  --temporal_prompt_file "/path/to/Temporal/temporal_prompts.json" \
  --output_path "/path/to/Temporal/evaluation_results"
```

---

# 15. 总结

你现在的情况不是“评测逻辑没写好”，而是：
- **生成已经完成**
- **评测模式已经改好了**
- **你现在卡在 Linux 服务器上的评测环境准备这一步**

所以最关键的是：
1. 创建/激活虚拟环境
2. 安装依赖
3. 设置 `DASHSCOPE_API_KEY`
4. 设置 `VBENCH2_CACHE_DIR`
5. 手动下载 LLaVA / Qwen 模型
6. 先跑一个维度测试
7. 再跑完整评测

如果你在服务器上执行某一步报错，建议直接把报错贴出来，再继续定位。