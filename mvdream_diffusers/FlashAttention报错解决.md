# MVDream `flash-attention / xformers` 报错解决

适用报错：

```text
CUDA error (third_party/flash-attention/csrc/flash_attn/src/fmha_fwd_launch_template.h:89): no kernel image is available for execution on the device
```

这个报错的最简单解决方案是：

**彻底不要走 `xformers / flash-attention`，把 `mv_unet.py` 里的 attention 改成普通 PyTorch attention。**

---

## 一步修复

在你的 Linux 服务器项目目录里执行：

```bash
cd /mnt/workspace/cv_multimodal/qirui/multi-image-generation/temporal/MIG/mvdream_diffusers

python - <<'PY'
from pathlib import Path
import re

p = Path("mv_unet.py")
s = p.read_text()

s = s.replace(
    "# require xformers!\nimport xformers\nimport xformers.ops\n",
    "# use standard PyTorch attention for broader GPU compatibility\n"
)

pattern = r"    def _attention\(self, q, k, v\):\n[\s\S]*?    def forward\(self, x, context=None\):"
replacement = """    def _attention(self, q, k, v):
        attention_scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * (self.dim_head ** -0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        return torch.bmm(attention_probs, v.float()).to(dtype=q.dtype)

    def forward(self, x, context=None):"""

s, n = re.subn(pattern, replacement, s, count=1)
if n != 1:
    raise SystemExit("patch failed: _attention block not found")

p.write_text(s)
print("patched mv_unet.py")
PY
```

---

## 立刻检查是否真的修好

执行：

```bash
grep -n "xformers\|memory_efficient_attention" mv_unet.py
```

正常情况下：

- **不应该有任何输出**

如果还有输出，说明服务器上的 `mv_unet.py` 还没真正改成功，所以还会继续报 flash-attention 的错。

---

## 重新运行

```bash
CUDA_VISIBLE_DEVICES=3 python run_mvdream.py "a cute owl"
```

---

## 如果你想先快速验证能不能跑通

原脚本会连续生成 5 次。可以先改成只生成 1 次：

```bash
python - <<'PY'
from pathlib import Path
p = Path("run_mvdream.py")
s = p.read_text()
s = s.replace("for i in range(5):", "for i in range(1):")
p.write_text(s)
print("patched run_mvdream.py")
PY
```

然后再运行：

```bash
CUDA_VISIBLE_DEVICES=3 python run_mvdream.py "a cute owl"
```

---

## 跑通后会生成什么

如果你把 `range(5)` 改成了 `range(1)`，当前目录会生成：

```text
test_mvdream_0.jpg
```

如果没有改，默认会生成：

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
