# ESP32 RAG 评测（baseline vs RAG）

用同一套题库，对比「裸大模型（不检索）」和「本项目 RAG」在硬事实问题上的
准确率、引用率与耗时，用来量化 RAG 的价值、并指导优化。

## 目录

```
eval/
├── questions.yaml          # 题库模板（可自由增删改）
├── reports/                # 运行生成的报告（已 gitignore）
└── README.md               # 本文档

scripts/eval/
├── run_eval.py             # 评测脚本
└── __init__.py
```

## 题库格式（`questions.yaml`）

每题为一条 YAML 记录：

```yaml
- id: gpio_s3                     # 唯一标识
  category: pin_count             # 分类
  question: ESP32-S3 一共有多少个 GPIO 引脚？
  expected: "45 个 GPIO"          # 标准答案（裁判据此判对错，请对照语料核对）
  keywords: ["45"]                # 关键词兜底（裁判失败时使用）
  source_hint: esp32-s3 datasheet # 期望命中来源（人工核对用）
  verify: true                    # 可选：建议再核对一次
```

分类：`pin_count / cpu / memory / wireless / peripheral / variant / boot / register / errata`。

> `expected` 是事实基准，不是唯一表述；裁判只看事实是否一致。
> 建议持续补充「errata / 寄存器地址 / 引脚陷阱 / 变体差异 / 私域语料」类问题。

---

## 两种测试方法

### 方式 A：本地 `.venv`

```powershell
# 全量
.\.venv\Scripts\python.exe scripts\eval\run_eval.py

# 常用：限量 / 指定分类 / 指定题 / 不做 LLM 裁判
.\.venv\Scripts\python.exe scripts\eval\run_eval.py --limit 5
.\.venv\Scripts\python.exe scripts\eval\run_eval.py --category register
.\.venv\Scripts\python.exe scripts\eval\run_eval.py --ids gpio_s3,reg_gpio_in,bt_s2
.\.venv\Scripts\python.exe scripts\eval\run_eval.py --no-judge
```

前提：
- `.env` 配好 `LLM_API_KEY / LLM_BASE_URL / LLM_MODEL`（脚本会自动加载 `.env`）
- 已有向量库（`.chroma_esp32_all/`，RAG 检索需要）

### 方式 B：Docker

`eval/` 已通过 Dockerfile `COPY eval/ ./eval/` 打进镜像，`scripts/eval/` 也在镜像内。
所以重建镜像后，直接在运行中的容器里跑即可（容器自带 `LLM_*` 环境变量和挂载的数据卷）。

```powershell
# 1) 先重建镜像（本轮新增了 scripts/eval、rag_answer、eval/，必须重建一次）
docker compose up -d --build

# 2) 在容器里评测
docker exec -w /app esp-rag python scripts/eval/run_eval.py --limit 5
docker exec -w /app esp-rag python scripts/eval/run_eval.py --category register
docker exec -w /app esp-rag python scripts/eval/run_eval.py --ids gpio_s3,bt_s2
```

> 不想重建镜像时，可用一次性 `docker run` 挂载最新代码（示例）：
> ```powershell
> docker run --rm -w /app --env-file .env `
>   -v "${PWD}/scripts:/app/scripts:ro" -v "${PWD}/eval:/app/eval:ro" `
>   -v "${PWD}/config.yaml:/app/config.yaml:ro" -v "${PWD}/models:/app/models:ro" `
>   -v "${PWD}/source:/app/source:ro" -v "${PWD}/source_code:/app/source_code:ro" `
>   -v "${PWD}/.chroma_esp32_all:/app/.chroma_esp32_all" `
>   -v "${PWD}/.chroma_esp32_code:/app/.chroma_esp32_code" `
>   esp-rag:latest python scripts/eval/run_eval.py --limit 5
> ```

---

## 参数

| 参数 | 说明 |
|---|---|
| `--questions <path>` | 题库路径（默认 `eval/questions.yaml`） |
| `--out <dir>` | 报告输出目录（默认 `eval/reports`） |
| `--limit N` | 只跑前 N 条（0=全部） |
| `--category <c>` | 只跑某分类 |
| `--ids a,b,c` | 只跑指定 id |
| `--top_k N` | RAG 检索 chunk 数（默认 8） |
| `--no-judge` | 不做 LLM 裁判，仅关键词兜底（更快） |

## 输出

- `eval/reports/report_<时间戳>.md` — 可读报告（汇总表 + 逐题明细 + 答案与裁判理由）
- `eval/reports/report_<时间戳>.json` — 机器可读结果

指标：baseline/RAG 的**准确率**、**引用率([Source:])**、**平均耗时**，以及
`improved_ids`（RAG 对而 baseline 错）和 `worse_ids`（反之）。

## 解读建议

- 热门通用事实，baseline 常也能答对 → 差异主要体现在**引用率/可审计性**。
- 真正拉开差距的是**硬题**（寄存器地址、errata、引脚陷阱、变体细微差异）和
  **私域/最新语料**——这些 baseline 易错，RAG 有据可查。
- 先小批量（`--limit`/`--ids`）迭代题库，再全量跑出对比基线。
