# MVDream `flash-attention / xformers` 报错解决

适用报错：

```text
CUDA error (third_party/flash-attention/csrc/flash_attn/src/fmha_fwd_launch_template.h:89): no kernel image is available for execution on the device
```

这个报错的最简单解决方案是：

**不要再走 `xformers/flash-attention`，直接把 `mv_unet.py` 里的 attention 改成普通 PyTorch attention。**

相关位置：
- `mv_unet.py:176-188`
- `mv_unet.py:215-228`

---

## 一步修复

在服务器项目目录里执行下面这段命令：

```bash
cd /mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/mvdream_diffusers

python - <<'PY'
from pathlib import Path
import re

p = Path("mv_unet.py")
text = p.read_text()

pattern = r"""    def _attention\(self, q, k, v\):\n(?:        .*\n)+?    def forward\(self, x, context=None\):"""
replacement = """    def _attention(self, q, k, v):
        attn = torch.bmm(q.float(), k.float().transpose(1, 2)) * (self.dim_head ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        return torch.bmm(attn, v.float()).to(dtype=q.dtype)

    def forward(self, x, context=None):"""

new_text, n = re.subn(pattern, replacement, text)
if n != 1:
    raise SystemExit("patch failed: _attention block not found uniquely")

p.write_text(new_text)
print("patched mv_unet.py")
PY
```

---

## 检查是否修好

执行：

```bash
grep -n "memory_efficient_attention" mv_unet.py
```

正常情况下：

- **不应该再有输出**

如果还有输出，说明补丁没有成功打上。

---

## 重新运行

```bash
CUDA_VISIBLE_DEVICES=3 python run_mvdream.py "a cute owl"
```

---

## 如果想先更快验证一次

原脚本会连续生成 5 次。
如果你只想先验证能不能跑通，可以把生成次数改成 1 次。

执行：

```bash
python - <<'PY'
from pathlib import Path
p = Path("run_mvdream.py")
text = p.read_text()
text = text.replace("for i in range(5):", "for i in range(1):")
text = text.replace(
    'image = pipe(args.prompt, guidance_scale=5, num_inference_steps=30, elevation=0)',
    'image = pipe(args.prompt, guidance_scale=5, num_inference_steps=20, elevation=0)'
)
p.write_text(text)
print("patched run_mvdream.py")
PY
```

然后再运行：

```bash
CUDA_VISIBLE_DEVICES=3 python run_mvdream.py "a cute owl"
```

---

## 跑通后会得到什么

当前目录会生成：

```text
test_mvdream_0.jpg
```

如果没有把 `range(5)` 改成 `range(1)`，则会生成：

```text
test_mvdream_0.jpg
test_mvdream_1.jpg
test_mvdream_2.jpg
test_mvdream_3.jpg
test_mvdream_4.jpg
```

每张图里是 **4 个视角拼成的 2x2 网格图**。

---

## 备注

下面这句 warning 不是主问题，可以先忽略：

```text
Keyword arguments {'trust_remote_code': True} are not expected by MVDreamPipeline and will be ignored.
```
