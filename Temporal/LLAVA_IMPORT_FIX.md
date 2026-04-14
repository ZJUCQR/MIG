# LLaVA 导入冲突修复说明

如果你在运行：

```bash
for dim in Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Order_Understanding Complex_Plot; do
  CUDA_VISIBLE_DEVICES=1,2,3 python "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/evaluate.py" \
    --videos_path "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/wan_t2v_videos/${dim}" \
    --dimension "${dim}" \
    --mode vbench_standard \
    --output_path "./evaluation_results_wan_t2v/${dim}"
done
```

时报错：

```text
NotImplementedError: UnImplemented dimension Complex_Plot!, cannot import name 'LlavaLlamaForCausalLM' from 'llava.model'
```

这通常不是评测逻辑的问题，而是 **`llava` 包导入冲突**。

---

## 1. 问题本质

你当前实际加载到的 `llava` 很可能不是：

```text
Temporal/vbench2/third_party/LLaVA_NeXT
```

而是别的路径下已经存在的一份 `llava`，例如你报错里出现的：

```text
/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/src/llava/llava/model/__init__.py
```

这会导致：
- `Temporal/vbench2/complex_plot.py` 试图导入 `llava.model.builder`
- 但实际导入的是另一套 `llava`
- 那套 `llava` 里缺少当前 evaluator 需要的模型注册
- 最终报：
  - `cannot import name 'LlavaLlamaForCausalLM' from 'llava.model'`

---

## 2. 先确认当前到底加载的是哪个 llava

执行：

```bash
python - <<'PY'
import llava
import llava.model
print(llava.__file__)
print(llava.model.__file__)
PY
```

### 正常情况下
你希望看到的路径应该指向：

```text
.../Temporal/vbench2/third_party/LLaVA_NeXT/...
```

### 如果输出是别的路径
比如：

```text
.../temporal/src/llava/...
```

那就是导入冲突。

---

## 3. 卸载错误的 llava

先清理环境里已经装过的 `llava`：

```bash
pip uninstall -y llava
```

如果你之前装过其他 LLaVA 变体，也建议一起清理。

---

## 4. 安装当前 Temporal 仓库自带的 LLaVA_NeXT

执行：

```bash
pip install -e "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/vbench2/third_party/LLaVA_NeXT"
```

这一步的目的就是让当前 `Temporal` evaluator 使用它自己绑定的那份 `llava`。

---

## 5. 检查是否有 PYTHONPATH 干扰

先看当前环境变量：

```bash
echo $PYTHONPATH
```

如果里面出现这类路径：

```text
/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/src
/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/src/llava
```

那它们很可能会抢占导入。

临时清掉：

```bash
unset PYTHONPATH
```

然后重新运行检查。

---

## 6. 做最小导入测试

先测试最关键的一步：

```bash
python - <<'PY'
from llava.model.builder import load_pretrained_model
print("llava ok")
PY
```

如果这一步成功，说明最核心的导入链已经正常了。

---

## 7. 如果还报错，再看真正失败的底层 import

执行：

```bash
python - <<'PY'
import llava.model
PY
```

因为 `llava/model/__init__.py` 里会打印真实失败的子模块导入错误，能进一步判断是：
- 版本不对
- 依赖缺失
- transformers / torch 版本不兼容
- 还是路径冲突没清干净

---

## 8. 推荐你直接执行的修复顺序

```bash
# 1. 看当前加载的是哪个 llava
python - <<'PY'
import llava
import llava.model
print(llava.__file__)
print(llava.model.__file__)
PY

# 2. 卸载错误版本
pip uninstall -y llava

# 3. 清理可能干扰的 PYTHONPATH
unset PYTHONPATH

# 4. 安装 Temporal 自带的 LLaVA_NeXT
pip install -e "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/vbench2/third_party/LLaVA_NeXT"

# 5. 测试关键导入
python - <<'PY'
from llava.model.builder import load_pretrained_model
print("llava ok")
PY
```

---

## 9. 修好后再重新跑评测

先建议只跑一个维度测试，例如：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/evaluate.py" \
  --videos_path "/mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/Temporal/wan_t2v_videos/Complex_Plot" \
  --dimension Complex_Plot \
  --mode vbench_standard \
  --output_path "./evaluation_results_wan_t2v/Complex_Plot"
```

如果 `Complex_Plot` 能过，再跑整套四个维度。

---

## 10. 一句话结论

这次报错的根因大概率是：

**评测时加载到了错误路径下的 `llava`，而不是 `Temporal/vbench2/third_party/LLaVA_NeXT` 这一份。**
