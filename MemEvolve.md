# MemEvolve 项目代码概览

## 项目简介

**MemEvolve: Meta-Evolution of Agent Memory Systems** 是一个自动化演进 LLM Agent 记忆系统的框架。传统的自改进记忆系统在固定的记忆架构下运行,仅更新记忆内容。MemEvolve 提出了 **双重演进(Dual-Evolution)** 过程,同时演进记忆内容和记忆架构本身,使记忆系统能够自适应地改进。

项目基于 **Flash-Searcher**(DAG 并行执行 Agent 框架)构建,并集成了 **mini-swe-agent** 用于自动代码修复。

---

## 顶层目录结构

```
MemEvolve/
├── README.md                    # 项目说明文档
├── LICENSE                      # Apache 2.0 许可证
├── assets/                      # 图片资源
└── Flash-Searcher-main/         # 核心代码目录
    ├── MemEvolve/               # 记忆系统演进框架(核心)
    ├── FlashOAgents/            # Agent 执行框架
    ├── EvolveLab/               # 统一记忆系统库(13+ 种记忆实现)
    ├── mini-swe-agent/          # 轻量级 SWE Agent(自动代码修复)
    ├── evolve_cli.py            # 命令行入口
    ├── base_agent.py            # Agent 类定义
    ├── eval_utils.py            # 评估工具
    ├── run_flash_searcher_*.py  # 各数据集运行脚本
    └── data/                    # 数据集目录
```

---

## 三大核心模块及关系

```
                    evolve_cli.py (命令行入口)
                         |
            +------------+------------+
            |                         |
       MemEvolve/                EvolveLab/
    (演进引擎)                (记忆系统库)
       |    |                     |
       |    +--- 生成/修改 ------>|
       |                          |
       +--- 调用 FlashOAgents/ ---|--- 被 Agent 使用
            (Agent 执行框架)
                  |
             base_agent.py
            (Agent 类封装)
```

**数据流**: 演进引擎(MemEvolve) 分析 Agent 执行轨迹 -> 生成新记忆系统代码 -> 写入 EvolveLab -> 验证 -> Agent 使用新记忆系统执行任务 -> 收集新轨迹 -> 循环

---

## 模块一: MemEvolve/ (记忆系统演进框架)

负责自动化地分析、生成、创建和验证新的记忆系统。

### 文件关系

```
MemEvolve/
├── config.py                  # 全局常量与配置
├── core/
│   ├── auto_evolver.py        # 外层循环: 多轮锦标赛式演进
│   └── memory_evolver.py      # 内层循环: 单轮四阶段流水线
├── phases/
│   ├── phase_analyzer.py      # 阶段1: 轨迹分析
│   ├── phase_generator.py     # 阶段2: LLM 生成新系统配置
│   ├── memory_creator.py      # 阶段3: 将配置写为实际代码
│   └── phase_validator.py     # 阶段4: 静态检查 + 仿真测试
├── validators/
│   └── swe_agent_validator.py # 自动代码修复(调用 mini-swe-agent)
├── utils/
│   ├── trajectory_tools.py    # 轨迹查看/分析工具
│   └── run_provider.py        # 调用外部运行脚本
└── prompts/
    ├── analysis_prompt.yaml   # 分析阶段 Prompt 模板
    └── generation_prompt.yaml # 生成阶段 Prompt 模板
```

### 核心流程

| 组件 | 职责 |
|------|------|
| `AutoEvolver` | 外层循环,管理多轮演进,实现锦标赛选拔(含 Pareto 多目标优化) |
| `MemoryEvolver` | 内层循环,编排四个阶段的顺序执行,管理持久化状态 |
| `PhaseAnalyzer` | 使用 LLM Agent + 轨迹工具分析执行日志,识别记忆系统瓶颈 |
| `PhaseGenerator` | 基于分析报告,调用 LLM 生成新记忆系统的代码和配置 |
| `MemorySystemCreator` | 将生成的配置写为 Python 文件,更新枚举和映射注册 |
| `PhaseValidator` | 静态 AST 检查 + 隔离环境仿真测试,失败时调用 SWE Agent 自动修复 |
| `SWEAgentValidator` | 封装 mini-swe-agent,根据错误报告自动修复记忆系统代码 |
| `TrajectoryFeedbackAggregator` | 计算多维度评估指标(准确率、token 用量、步数等) |

### 调用关系

```
AutoEvolver (多轮锦标赛)
  └── MemoryEvolver (每轮生成 N 个候选系统)
        ├── PhaseAnalyzer       -> 使用 AnalysisAgent + TrajectoryViewerTool
        ├── PhaseGenerator      -> 调用 OpenAI API
        ├── MemorySystemCreator -> 写文件到 EvolveLab/
        └── PhaseValidator      -> 静态检查 + 仿真测试
              └── SWEAgentValidator (失败时自动修复)
```

---

## 模块二: FlashOAgents/ (Agent 执行框架)

基于 smolagents 改造的 ReAct 式多步 Agent 框架,支持 DAG 并行执行。

### 文件关系

```
FlashOAgents/
├── agents.py             # 核心: MultiStepAgent / ToolCallingAgent
├── models.py             # LLM 抽象层: OpenAIServerModel
├── memory.py             # Agent 内部记忆(步骤记录)
├── tools.py              # Tool 基类 + FinalAnswerTool
├── agent_types.py        # Agent I/O 类型封装
├── search_tools.py       # WebSearchTool / CrawlPageTool
├── mm_tools.py           # 多模态工具(图片/文档/音频)
├── mm_tools_utils.py     # 文档转 Markdown 引擎
├── monitoring.py         # Rich 日志系统
├── utils.py              # 通用工具(错误类/JSON解析/图片编码)
└── prompts/              # Agent Prompt 模板(YAML)
```

### 分层架构

```
+--------------------------------------------------+
|          agents.py (编排层)                        |
|   MultiStepAgent -> ToolCallingAgent              |
|   plan -> act -> observe 循环                     |
+--------------------------------------------------+
|          models.py (模型层)                        |
|   OpenAIServerModel (重试/token计数)              |
+--------------------------------------------------+
|          memory.py (记忆层)                        |
|   ActionStep / PlanningStep / SummaryStep         |
+--------------------------------------------------+
|     tools.py / search_tools.py / mm_tools.py      |
|              (工具实现层)                          |
+--------------------------------------------------+
|  agent_types.py / utils.py / monitoring.py        |
|              (基础设施层)                          |
+--------------------------------------------------+
```

### 关键文件说明

| 文件 | 说明 |
|------|------|
| `agents.py` | 核心 Agent 循环,集成外部记忆系统(EvolveLab),支持规划/总结/工具调用 |
| `models.py` | 封装 OpenAI API,含重试逻辑和 o3/o4 模型特殊处理 |
| `memory.py` | 定义 Agent 执行历史的数据结构,可序列化为 LLM 消息格式 |
| `search_tools.py` | 网页搜索(Serper API)和网页抓取(Jina/crawl4ai) |
| `mm_tools.py` | 图片(GPT-4o)、文档(多格式)、音频(Whisper)处理工具 |
| `mm_tools_utils.py` | 支持 PDF/DOCX/XLSX/PPTX/HTML 等格式转 Markdown |
| `monitoring.py` | 基于 Rich 的彩色日志和 Agent 树可视化 |

---

## 模块三: EvolveLab/ (统一记忆系统库)

提供统一的记忆接口和 13 种记忆系统实现(11 个 baseline + 2 个演进生成)。

### 文件关系

```
EvolveLab/
├── __init__.py                    # 公共 API 导出
├── memory_types.py                # 核心类型定义(枚举/数据类)
├── base_memory.py                 # 抽象基类 BaseMemoryProvider
├── config.py                      # 所有 Provider 的配置中心
└── providers/
    ├── base_provider_template.py  # Provider 编写模板/示例
    ├── agent_kb_provider.py       # Agent-KB
    ├── skillweaver_provider.py    # SkillWeaver
    ├── mobilee_provider.py        # Mobile-Agent-E
    ├── expel_provider.py          # ExpeL
    ├── voyager_memory_provider.py # Voyager
    ├── dilu_memory_provider.py    # DILU
    ├── generative_memory_provider.py    # Generative Agents
    ├── memp_memory_provider.py          # MEMP
    ├── dynamic_cheatsheet_provider.py   # Dynamic Cheatsheet
    ├── agent_workflow_memory_provider.py # Agent Workflow Memory
    ├── evolver_memory_provider.py       # Evolver
    ├── lightweight_memory_provider.py   # [演进生成] 轻量级双记忆系统
    └── cerebra_fusion_memory_provider.py # [演进生成] CerebraFusion 记忆系统
```

### 核心接口

所有 Provider 必须继承 `BaseMemoryProvider` 并实现三个方法:

| 方法 | 说明 |
|------|------|
| `initialize()` | 加载数据、建立索引 |
| `provide_memory(request)` | 根据查询和阶段(BEGIN/IN)返回相关记忆 |
| `take_in_memory(trajectory)` | 从执行轨迹中提取并存储新记忆 |

Provider 通过 `MemoryType` 枚举 + `PROVIDER_MAPPING` 字典实现动态加载,无需全部静态导入。

---

## 辅助模块

### base_agent.py (Agent 类封装)

| 类 | 说明 |
|------|------|
| `BaseAgent` | 基础封装: 运行 Agent + 捕获轨迹 |
| `SearchAgent` | 搜索 Agent: WebSearch + CrawlPage + 记忆 |
| `MMSearchAgent` | 多模态 Agent: 增加图片/文档/音频工具 |
| `AnalysisAgent` | 分析 Agent: 使用轨迹查看工具(用于演进分析阶段) |
| `ReviewAgent` | 评审 Agent: 无工具,纯推理 |

### evolve_cli.py (命令行入口)

| 命令 | 说明 |
|------|------|
| `analyze` | 分析轨迹日志 |
| `generate` | 生成新记忆系统配置 |
| `create` | 创建记忆系统代码文件 |
| `validate` | 验证记忆系统 |
| `auto-evolve` | 自动多轮演进(锦标赛) |
| `delete` | 删除记忆系统 |
| `list` | 列出所有已注册记忆系统 |
| `status` | 查看演进状态 |
| `run-all` | 顺序执行全部阶段 |

### eval_utils.py (评估工具)

提供 `TaskTimer`、`TokenCounter`、结果保存、统一报告生成等评估辅助功能。

### mini-swe-agent/ (轻量级 SWE Agent)

独立子项目,提供自动化代码修复能力。支持多种 LLM 后端(Anthropic/LiteLLM/OpenRouter 等)和执行环境(Local/Docker/Singularity)。在 MemEvolve 中被 `SWEAgentValidator` 调用,用于自动修复验证失败的记忆系统代码。

---

## 整体数据流

```
1. 初始 Provider 在任务上执行
        |
        v
2. 收集执行轨迹 (JSON logs)
        |
        v
3. PhaseAnalyzer 分析轨迹 -> 识别记忆系统瓶颈
        |
        v
4. PhaseGenerator 用 LLM 生成新记忆系统配置
        |
        v
5. MemorySystemCreator 写入代码到 EvolveLab/providers/
        |
        v
6. PhaseValidator 验证 (静态检查 + 隔离仿真)
        |    |
        |    v (失败)
        |  SWEAgentValidator 自动修复 -> 重试
        |
        v (通过)
7. 新 Provider + 旧 Provider 在相同任务上评估
        |
        v
8. 锦标赛选拔优胜者 -> 作为下一轮基线 -> 回到步骤1
```
