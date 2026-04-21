# XSkill 代码总览与分析

> **论文**: *XSkill: Continual Learning from Experience and Skills in Multimodal Agents*
> **作者**: Guanyu Jiang, Zhaochen Su, Xiaoye Qu, Yi R. Fung
> **发表**: ICML 2026 | [arXiv:2603.12056](https://arxiv.org/abs/2603.12056) | [项目主页](https://xskill-agent.github.io/xskill_page/) | [GitHub](https://github.com/XSkill-Agent/XSkill)

---

## 1. 论文核心思想

### 1.1 解决的问题

多模态智能体（Multimodal Agent）在使用工具进行复杂推理时，存在两个核心瓶颈：
- **工具使用效率低**：Agent 不能从过去的交互中学习，每次任务都"从零开始"
- **编排不灵活**：缺乏结构化的任务级指导，导致工具调用混乱

### 1.2 核心创新

XSkill 提出了一个**双流（Dual-Stream）持续学习框架**，无需参数更新，通过从 Agent 轨迹中自动提取两种互补知识：

| 知识类型 | 级别 | 存储格式 | 作用 |
|---------|------|---------|------|
| **Skill（技能）** | 任务级（Task-level） | Markdown 文档 | 提供结构化工作流和工具模板，指导规划 |
| **Experience（经验）** | 动作级（Action-level） | JSON 条目 | 提供上下文相关的战术洞察，指导工具选择 |

### 1.3 形式化定义

- **Skill**: `k = (M, W, P)` — 元数据 M（名称/描述/版本）、工作流序列 W、可复用工具模板 P
- **Experience**: `e = (c, a, v_e)` — 触发条件 c、推荐动作 a、语义嵌入向量 v_e（用于检索）

整个问题被形式化为**部分可观测马尔可夫决策过程（POMDP）**，目标是构建外部知识库 `KB = (K, E)` 最大化正确回答概率。

---

## 2. 系统架构概览

XSkill 使用**两个独立的 MLLM 实例**：

| 模型 | 角色 | 说明 |
|------|------|------|
| `MLLM_exec` | **执行模型** | 负责工具使用推理，可以是轻量模型 |
| `MLLM_kb` | **知识管理模型** | 负责知识提取、整合、适配，通常使用更强的模型 |

这种分离设计支持**跨模型知识迁移**——由一个模型积累的知识可以被另一个模型使用。

### 两阶段流程

```
Phase I（积累阶段）             Phase II（推理阶段）
┌────────────────────┐     ┌──────────────────────┐
│ 训练数据 → 多路Rollout  │     │ 测试任务 → 子任务分解     │
│    ↓                │     │    ↓                  │
│ Rollout Summary     │     │ 知识检索（Experience+Skill）│
│    ↓                │     │    ↓                  │
│ Cross-Rollout Critique│   │ 视觉上下文适配            │
│    ↓                │     │    ↓                  │
│ 分层整合 → 知识库更新   │     │ 注入System Prompt → 执行 │
└────────────────────┘     └──────────────────────┘
```

---

## 3. Phase I：知识积累（Accumulation）

### 3.1 Rollout Summary（轨迹摘要）

对每个训练任务，执行模型进行 N 次独立 rollout，生成轨迹集合。知识管理模型执行**视觉锚定的轨迹摘要**：

```
输入: 轨迹集合 R_i, 任务图片 I_i, 查询 q_i, 真实答案 y_i*, 适配后的技能 K_adapted
输出: 轨迹摘要 S_Ri（包含关键决策点、工具使用模式、失败原因）
       技能片段 ΔK_i
```

### 3.2 Cross-Rollout Critique（跨轨迹批判）

对成功和失败的轨迹进行**对比分析**，提取泛化知识：

```
输入: 轨迹摘要 S_Ri, 真实答案 y_i*, 已使用的经验 E_ret
输出: 结构化经验更新 ΔE_i = {op_1, op_2, ...}
       每个操作为 (add, e) 或 (modify, e_id, e')
```

### 3.3 分层整合

- **Experience 整合**: 当经验条目数超过阈值 `experience-max-items`（默认120）时，进行精炼合并
- **Skill 整合**: 将新的技能片段合并到全局 Skill 文档，限制最大长度 `skill-max-length`（默认1000词）

---

## 4. Phase II：知识检索与推理（Inference）

### 4.1 子任务分解（Subtask Decomposition）

将测试查询分解为细粒度子任务，用于多路检索：

```
测试任务 T_test → [子任务_1, 子任务_2, ..., 子任务_k]
```

### 4.2 经验检索（Experience Retrieval）

1. 对每个子任务生成查询嵌入
2. 通过**语义相似度**从 Experience Bank 中检索 Top-K 相关经验
3. 支持**查询重写**（`--experience-retrieval-rewrite`）优化检索质量

### 4.3 技能适配（Skill Adaptation）

将全局 Skill 文档适配到当前视觉上下文：
- 根据任务图像和查询选择相关的工作流片段
- 生成适配后的技能注入到 System Prompt

### 4.4 知识注入与执行

最终将检索到的经验和适配后的技能注入 Agent 的 System Prompt，执行模型据此进行推理。

---

## 5. 代码结构详解

### 5.1 目录结构

```
XSkill/
├── eval/                          # 核心代码目录
│   ├── infer_api.py               # 主入口：推理引擎
│   ├── infer_api_utils.py         # 推理管线工具函数
│   ├── configs/
│   │   └── tool_configs.yaml      # 工具运行时配置
│   ├── engine/                    # API调用、工具分发、上下文管理
│   ├── exskill/                   # ⭐ 核心模块：经验与技能学习
│   │   ├── experience_critique.py # Cross-Rollout Critique 实现
│   │   ├── experience_manager.py  # 经验库管理（增删改查）
│   │   ├── experience_retriever.py# 经验检索（嵌入+相似度）
│   │   ├── skill_builder.py       # 技能文档构建与整合
│   │   ├── trajectory_summary.py  # 轨迹摘要生成
│   │   └── multimodal_analysis.py # 多模态分析（视觉锚定）
│   ├── tools/                     # 工具实现
│   │   ├── code_interpreter.py    # 代码执行器
│   │   ├── web_search.py          # 网页搜索（SerpAPI）
│   │   ├── image_search.py        # 图片搜索
│   │   ├── visit.py               # 网页内容获取（Jina）
│   │   └── zoom.py                # 图像裁剪/缩放
│   ├── prompts/                   # Prompt 模板
│   ├── search/                    # 搜索树和节点结构
│   └── utils/                     # 共享工具函数
├── memory_bank/                   # 运行时生成的知识库
│   └── test/
│       ├── experiences.json       # 经验库（JSON格式）
│       └── SKILL.md               # 技能文档（Markdown格式）
├── output/                        # 推理输出
├── logs/                          # 运行日志
├── requirements.txt               # 依赖配置
└── 2603.12056v2.pdf               # 论文PDF
```

### 5.2 核心模块说明

#### `eval/exskill/` — 经验与技能学习核心

| 文件 | 功能 | 对应论文部分 |
|------|------|------------|
| `trajectory_summary.py` | 视觉锚定的轨迹摘要生成 | §2.2.1 Rollout Summary |
| `experience_critique.py` | 跨轨迹对比批判，提取经验更新 | §2.2.2 Cross-Rollout Critique |
| `experience_manager.py` | 经验库的 CRUD 操作和整合 | §2.2 Experience 管理 |
| `experience_retriever.py` | 基于嵌入的经验语义检索 | §2.3 Phase II 检索 |
| `skill_builder.py` | Skill 文档构建、合并与精炼 | §2.2 Skill 整合 |
| `multimodal_analysis.py` | 视觉上下文分析与锚定 | 贯穿全文 |

#### `eval/tools/` — 工具集

| 工具 | 功能 | 依赖 |
|------|------|------|
| `code_interpreter.py` | 在沙箱中执行 Python 代码 | 本地 Python 环境 |
| `web_search.py` | 调用搜索引擎查询 | SerpAPI |
| `image_search.py` | 反向图片搜索 | SerpAPI + ImgBB |
| `visit.py` | 获取并解析网页内容 | Jina API / trafilatura |
| `zoom.py` | 对图像进行裁剪和缩放操作 | PIL/OpenCV |

#### `eval/engine/` — 推理引擎

负责：
- OpenAI 兼容的 API 调用管理（支持轮询 fallback）
- 工具调度（Function Calling）
- 多轮对话上下文管理
- 图像管理（最多支持100张图像）

---

## 6. 运行脚本详解

### 6.1 Phase I 积累阶段 (`run_exskill_train.sh`)

```bash
bash eval/run_exskill_train.sh
```

**关键参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| `--rollouts-per-sample` | 2 | 每个样本执行2次独立 rollout |
| `--experience-online-generate` | 启用 | 在线生成经验 |
| `--experience-library-update` | 启用 | 更新经验库 |
| `--experience-refine` | 启用 | 精炼经验 |
| `--experience-max-ops` | 3 | 每次最多3个经验更新操作 |
| `--experience-max-items` | 120 | 经验库最大条目数 |
| `--skill-refine` | 启用 | 精炼技能文档 |
| `--skill-max-length` | 1000 | 技能文档最大词数 |

**与 Inference 的区别**: Train 脚本额外启用了 `--experience-online-generate`、`--experience-library-update`、`--experience-refine`、`--skill-refine`，形成持续学习循环。

### 6.2 Phase II 推理阶段 (`run_exskill_inference.sh`)

```bash
bash eval/run_exskill_inference.sh
```

**关键参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| `--skill-enable` | 启用 | 使用技能库 |
| `--skill-inference` | 启用 | 推理时适配技能 |
| `--experience-enable` | 启用 | 使用经验库 |
| `--experience-retrieval` | 启用 | 检索经验 |
| `--experience-retrieval-top-k` | 3 | 检索前3条最相关经验 |
| `--experience-retrieval-decomposition` | 启用 | 子任务分解检索 |
| `--experience-retrieval-rewrite` | 启用 | 查询重写优化检索 |

**注意**: Inference 脚本注释掉了 `--experience-online-generate` 等参数，即纯推理模式不更新知识库。

---

## 7. 配置系统

### 7.1 三类模型配置

XSkill 需要配置三类模型（均需兼容 OpenAI API）：

| 模型角色 | 环境变量 | 用途 |
|---------|---------|------|
| **Reasoning Model** | `REASONING_MODEL_NAME`, `REASONING_API_KEY`, `REASONING_END_POINT` | 主 Agent（MLLM_exec） |
| **Verifier Model** | `VERIFIER_MODEL_NAME`, `VERIFIER_API_KEY`, `VERIFIER_END_POINT` | LLM-as-Judge 评分 |
| **Experience Model** | `EXPERIENCE_MODEL_NAME`, `EXPERIENCE_API_KEY`, `EXPERIENCE_END_POINT` | 知识管理（MLLM_kb） |

另外需要一个 **Embedding Model**（如 `text-embedding-3-small`）用于经验检索。

### 7.2 外部工具 API

| API Key | 用途 |
|---------|------|
| `SERPAPI_KEY` | Web Search / Image Search |
| `JINA_API_KEY` | 网页内容获取 |
| `imgbb_api_key` | 图片上传托管（ImgBB） |

### 7.3 工具配置 (`tool_configs.yaml`)

```yaml
code_interpreter:
  work_dir: "/tmp/code_interpreter"    # 代码执行工作目录
  output_timeout: 30                   # 执行超时(秒)

web_search:
  max_results: 10                      # 最大搜索结果数
  timeout: 10

visit:
  max_content_length: 150000           # 页面最大提取字符数
  timeout: 120

image_search:
  imgbb_api_key: ""                    # ImgBB API密钥
  max_results: 5
  search_image_max_pixels: 1000000     # 搜索图片最大像素
  search_image_quality: 85             # JPEG压缩质量
```

---

## 8. 依赖技术栈

| 类别 | 库 | 用途 |
|------|-----|------|
| **LLM API** | `openai` (2.1.0) | 调用 OpenAI 兼容的 API |
| **多模态** | `qwen-vl-utils` (0.0.14) | Qwen-VL 图像处理工具 |
| **图像处理** | `pillow`, `opencv-python`, `scikit-image` | 图像裁剪/缩放/分析 |
| **深度学习** | `torch` (2.8.0), `torchvision` (0.23.0) | 向量化等操作 |
| **数据处理** | `numpy`, `pandas`, `scipy` | 数据分析和计算 |
| **网页解析** | `trafilatura` (2.0.0) | 网页正文提取 |
| **配置** | `pyyaml` (6.0.2) | YAML 配置解析 |
| **科学计算** | `sympy` (1.13.3) | 符号数学计算 |

---

## 9. 数据格式

### 输入数据 (JSON)

```json
[
  {
    "doc_id": "sample_001",
    "problem": "What is shown in <image>? Describe the object in detail.",
    "images": ["relative/path/to/image.jpg"],
    "solution": "A red bicycle parked against a wall."
  }
]
```

- `<image>` 占位符会按顺序替换为 `images` 中的图片
- `solution` 可选，无真实答案的样本得分为 0.0

### 知识库格式

**Experience Bank** (`experiences.json`):
```json
[
  {
    "id": 1,
    "condition": "当图像中的文字过小或模糊时",
    "action": "先使用 zoom 工具裁剪放大相关区域，再进行 OCR",
    "embedding": [0.12, -0.34, ...]
  }
]
```

**Skill Library** (`SKILL.md`):
```markdown
## Skill: Image Text Recognition
**Description**: 从图像中识别和提取文字信息
**Workflow**:
1. 分析图像质量和文字区域
2. 若文字较小，使用 zoom 裁剪放大
3. 调用 code_interpreter 执行 OCR
**Tool Templates**:
- zoom: crop_region(x1, y1, x2, y2)
- code: pytesseract.image_to_string(img)
```

---

## 10. 实验结果摘要

### 评测基准

| 基准 | 领域 | 评测内容 |
|------|------|---------|
| **VisualToolBench** | 视觉工具使用 | 多模态工具调用推理 |
| **TIR-Bench** | 文本图像推理 | 文本+图像组合推理 |
| **MMSearch-Plus** | 多模态搜索 | 涉及搜索的多模态问答 |
| **AgentVista** | 综合智能体 | 综合多模态推理 |
| **MMBrowseComp** | 浏览理解 | 多模态浏览+理解 |

### 性能提升

- 在四个骨干模型上，XSkill 的 **Average@4 提升 2.58~6.71 分**
- 在挑战性设置上，最高提升 **11.13 分**
- 展现出**零样本跨任务迁移**能力

### 消融实验关键发现

- **Experience 和 Skill 互补**：单独使用各有优势，组合效果最好
- **Experience** 主要提升工具使用的鲁棒性（减少错误的工具调用）
- **Skill** 主要提升规划质量和工具编排
- 子任务分解和查询重写对检索质量有显著提升

---

## 11. 快速上手指南

### Step 1: 安装

```bash
git clone https://github.com/XSkill-Agent/XSkill.git
cd XSkill
pip install -r requirements.txt  # Python 3.11 推荐
```

### Step 2: 配置 API

编辑 `eval/run_exskill_train.sh` 或 `eval/run_exskill_inference.sh`，填入：
- Reasoning/Verifier/Experience 模型的 API Key 和 Endpoint
- SerpAPI Key、Jina API Key
- ImgBB API Key（在 `tool_configs.yaml` 中）

### Step 3: 准备数据

将基准数据放置在 `benchmark/` 目录下，格式参考第9节。

### Step 4: 运行积累

```bash
bash eval/run_exskill_train.sh
```

这会在 `memory_bank/` 下生成 `experiences.json` 和 `SKILL.md`。

### Step 5: 运行推理

```bash
bash eval/run_exskill_inference.sh
```

结果输出到 `output/` 目录。

---

## 12. 设计亮点与学习要点

1. **无参数更新的持续学习**: 通过外部知识库实现 Agent 能力提升，不需要微调模型
2. **双流互补设计**: Skill（宏观规划）和 Experience（微观战术）分层互补
3. **视觉锚定**: 知识的提取和检索都基于视觉上下文，而非纯文本
4. **跨模型迁移**: 知识管理和执行解耦，支持知识在不同模型间迁移
5. **自动精炼**: 知识库通过持续使用自动精炼，避免冗余膨胀
6. **对比学习**: Cross-Rollout Critique 通过对比成功/失败轨迹提取因果知识
