# Enterprise AI Platform

> 面向企业内部知识库的 AI 应用项目：支持文档入库、检索增强问答与受约束的 Agent 工具调用。

**项目定位：本地运行的全栈 MVP，尚未达到生产环境部署要求。** 项目包含 FastAPI 服务、PostgreSQL/pgvector 检索、Redis/RQ 异步入库与 Web 界面；通过 OpenAI 兼容接口对接 LM Studio 或 vLLM。

## 项目概述

- **应用场景**：针对企业资料分散、关键词检索覆盖不足，以及模型回答缺少可核查依据等问题，提供基于内部文档的问答服务。
- **实现方案**：文档解析与切块 → 向量化入库 → 向量/BM25 召回 → CrossEncoder 精排 → 基于检索片段生成回答并展示来源。
- **已实现功能**：用户注册与登录、文档上传与处理状态查询、知识库搜索、普通对话与 RAG 对话、回答来源查看及历史会话管理；Web 界面位于 `/ui/`。
- **适用范围**：当前版本用于本地开发与功能验证；限流、监控、压测及 Worker 任务恢复等生产能力尚未完成。

## 技术架构与实现

| 方向 | 具体实现 |
| --- | --- |
| RAG 检索 | `bge-m3` + pgvector HNSW 向量索引；jieba/BM25 关键词召回；RRF 对照模式和 CrossEncoder 精排模式；回答附检索来源。 |
| Agent 约束 | LangGraph 编排 OpenAI 兼容 Function Calling；RAG Skill 要求先调用 `rag_search`；`sql_query` 仅允许预定义只读模板，数学工具仅解析 AST 白名单表达式。工具由项目内部注册表管理，当前未接入 MCP。 |
| 数据隔离 | JWT 认证；文档、检索、会话及 Agent 工具使用服务端身份限定用户范围；Embedding 与用户隔离的搜索结果分别使用 Redis 缓存。当前隔离发生在应用查询层，并非 PostgreSQL RLS。 |
| 异步处理 | 上传返回 `202 Accepted`，RQ Worker 负责解析、切块和入库；状态由 `pending → processing → ready / failed`，仅 `ready` 文档参与检索。 |
| 可测试性 | FastAPI 路由、业务服务与数据访问分层；Embedder/LLM/Reranker 提供协议与测试替身，常规自动化测试无需加载真实模型或 GPU；配有 pytest、Ruff 与 GitHub Actions 工作流。 |

核心链路：

```text
浏览器 /ui/ → FastAPI（JWT / API / 会话）
                  ├─ 上传 → PostgreSQL 文档记录 → Redis/RQ → Worker → pgvector Chunk
                  └─ 搜索/聊天 → 向量 + BM25 → 融合或精排 → RAG Skill / Agent
                                                              └─ 模型回答 + 来源
```

主要代码分别位于 [`app/api/routes/`](app/api/routes/)（API）、[`app/services/`](app/services/)（检索与 Agent）、[`app/workers/document_ingest.py`](app/workers/document_ingest.py)（异步入库）和 [`frontend/`](frontend/)（界面）。

## 检索策略与评测结果

历史检索实验使用 **3 篇文档 / 53 个 Chunk**、**20 道题（15 道可答、5 道库外）**、`top_k=5` 和相同生成模型 `ornith-1.0-9b`。以下指标来自仓库保留的历史评测报告，**未经当前版本重新验证，不代表大规模真实企业语料的表现**：

| 策略 | hit@5 | hit@1 | MRR | 库外题正确拒答 |
| --- | ---: | ---: | ---: | ---: |
| 向量检索 | 86.7% | 60.0% | 0.722 | 100% |
| BM25 | 93.3% | 80.0% | 0.867 | 80% |
| RRF 混合 | 93.3% | 80.0% | 0.850 | 60% |
| **混合召回 + CrossEncoder 精排** | **100.0%** | **93.3%** | **0.947** | **100%** |

**实验分析**：RRF 融合在该题库上未优于 BM25。两路 top-20 候选的并集覆盖了全部 15 道可答题，表明主要问题在于候选排序及库外题门控。据此保留 RRF 作为对照，并采用 CrossEncoder 对合并候选进行精排。历史题库中的 hit@1 由向量检索基线的 60.0% 提升至 93.3%。评测题目、统计口径和逐题结果见 [`evals/questions.jsonl`](evals/questions.jsonl)、[向量报告](evals/report-vector.md)、[BM25 报告](evals/report-bm25.md)、[RRF 报告](evals/report-hybrid.md)和[精排报告](evals/report-hybrid_rerank.md)。

> **复现条件**：现有 [`evals/run_eval.py`](evals/run_eval.py) 编写于认证与会话 Skill 改造之前，未携带 JWT，也未建立 RAG Skill 会话，因此无法直接用于当前版本的同口径复测。历史报告的延迟采集方式不一致，本文不进行跨策略延迟比较。开展新的端到端评测前，应适配认证、知识库用户归属及当前聊天流程，并统一语料、配置和串行测量方法。

## 本地运行（Windows PowerShell）

运行环境需安装 **Python 3.11+、Docker Compose**；问答功能还需接入支持 OpenAI 兼容 Chat Completions / Function Calling 的模型服务。首次下载 Embedding 与精排模型需要网络连接和相应磁盘空间。无 CUDA 环境时可将 `EMBEDDING_DEVICE` 设置为 `cpu`，但推理速度会降低。

```powershell
# 在项目根目录创建环境并安装服务/开发/模型依赖
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"

# 复制示例配置，编辑 .env 中的密码、JWT Secret、模型地址/名称；按机器修改 HF_HOME
Copy-Item .env.example .env

# 启动 PostgreSQL 与 Redis，并执行数据库迁移
docker compose up -d
.\.venv\Scripts\python.exe -m alembic upgrade head
```

在项目根目录分别使用两个 PowerShell 窗口运行以下命令；队列名称应与 `.env` 中的 `INGEST_QUEUE_NAME` 一致：

```powershell
# 窗口一：Windows 使用 SpawnWorker；保持运行以处理上传任务
.\.venv\Scripts\rq.exe worker --url redis://127.0.0.1:6379/0 --worker-class rq.worker.SpawnWorker document-ingest

# 窗口二：API 与前端
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Web 界面地址为 <http://127.0.0.1:8000/ui/>，API 文档地址为 <http://127.0.0.1:8000/docs>。验证 RAG 功能时，应先注册并登录，上传 [`test-data/chinese-rag-corpus/`](test-data/chinese-rag-corpus/) 中的**虚构测试资料**，等待文档状态变为 `ready`，再创建会话、选择「RAG 检索」并检查回答来源。普通对话模式不检索文档。解析 `.doc` 文件另需安装 LibreOffice；验证基本流程可使用 `.md` 文件。界面功能说明见 [`frontend/README.md`](frontend/README.md)。

## 测试与已知限制

```powershell
# 使用独立测试库：若尚未创建，在 compose 默认凭据下执行一次
docker compose exec -T db psql -U app -d e_ai_database -c "CREATE DATABASE e_ai_database_test;"

.\.venv\Scripts\python.exe -m ruff check app tests scripts evals
.\.venv\Scripts\python.exe -m ruff format --check app tests scripts evals
.\.venv\Scripts\python.exe -m pytest -q
```

运行测试前，应确认 `.env` 中的 `TEST_DATABASE_URL` 指向已创建的测试数据库。单元与接口测试使用模型替身；真实模型问答依赖另行启动的模型服务。当前尚未实现 Redis 限流、结构化链路监控、完整 LLM Gateway、生产压测以及 Worker 中断后的自动恢复与重试。前端令牌存储于 `localStorage`，不适合作为生产环境的最终方案。Agent 历史题库记录为 19/20，但固定语料下的 **RAG → 计算多步端到端流程尚未验收**，因此不列为已验证能力。
