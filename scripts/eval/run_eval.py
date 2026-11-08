#!/usr/bin/env python3
"""
ESP32 RAG 评测脚本 —— 对比「裸大模型」与「本项目 RAG」。

对题库里的每个问题，分别用两种方式回答：
  1. baseline : 只用 LLM 自身知识（不检索）
  2. rag      : 本项目的 RAG 流程（改写 → 检索 docs+code → 生成）

再让 LLM 裁判依据「标准答案」判断对错，并统计：
  - 准确率（baseline vs rag）
  - 引用率（答案是否带 [Source: ...]）
  - 平均耗时
  - 每条问题的明细

用法：
    python scripts/eval/run_eval.py                     # 跑全部
    python scripts/eval/run_eval.py --limit 5           # 只跑前 5 条
    python scripts/eval/run_eval.py --category wireless # 只跑某分类
    python scripts/eval/run_eval.py --ids gpio_s3,bt_s3 # 只跑指定 id
    python scripts/eval/run_eval.py --no-judge          # 不做 LLM 裁判（仅关键词兜底）
    python scripts/eval/run_eval.py --questions eval/questions.yaml --out eval/reports

两种测试方法（详见 eval/README.md）：
    方式 A（本地 .venv）:
        .\\.venv\\Scripts\\python.exe scripts\\eval\\run_eval.py --limit 5
    方式 B（Docker，需先 docker compose up -d --build 重建镜像）:
        docker exec -w /app esp-rag python scripts/eval/run_eval.py --limit 5

输出：
    eval/reports/report_<时间戳>.md   —— 可读报告
    eval/reports/report_<时间戳>.json —— 机器可读结果

前提：需在 .env 配好 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL；RAG 需要已构建的向量库。
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── pysqlite3 patch (MUST be before any chromadb import) ────────────
try:
    import pysqlite3 as _pysqlite3
    sys.modules['sqlite3'] = _pysqlite3
except ImportError:
    pass

SKILL_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = SKILL_DIR / 'scripts'
sys.path.insert(0, str(SKILL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault('RAG_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_all'))
os.environ.setdefault('RAG_DOCS_DIR', str(SKILL_DIR / 'source'))
os.environ.setdefault('RAG_CODE_CHROMA_DIR', str(SKILL_DIR / '.chroma_esp32_code'))
os.environ.setdefault('RAG_CODE_DOCS_DIR', str(SKILL_DIR / 'source_code'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# Silence chromadb's noisy telemetry warnings (harmless posthog mismatch).
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')

# Load .env for LLM config.
try:
    from dotenv import load_dotenv
    load_dotenv(SKILL_DIR / '.env')
except Exception:
    pass

logging.getLogger('chromadb').setLevel(logging.ERROR)
for _n in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
           'chromadb.telemetry', 'chromadb.rate_limiting']:
    logging.getLogger(_n).setLevel(logging.ERROR)

import yaml  # noqa: E402

from scripts.api.llm import chat_completion, LLMConfig  # noqa: E402
from scripts.api.chat import rag_answer  # noqa: E402

# Quiet the per-stage chat INFO logs (must be after the import, which sets INFO).
logging.getLogger('scripts.api.chat').setLevel(logging.WARNING)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

BASELINE_SYSTEM = (
    "你是一位资深嵌入式工程师。请仅凭你自己已有的知识回答用户问题，"
    "不要检索、不要声称查阅了文档。回答要简洁准确；涉及具体数字、型号、"
    "寄存器、外设数量等事实时，给出你认为正确的值。"
)

JUDGE_SYSTEM = (
    "你是严格的技术评测裁判。根据【问题】和【标准答案】，判断【待评判回答】是否正确。\n"
    "只输出一个 JSON 对象，不要任何多余文字：\n"
    '{"correct": true 或 false, "reason": "简短理由"}\n'
    "判定规则：\n"
    "- 回答中的关键事实（数字、型号、内核、功能、寄存器等）与标准答案一致 → true\n"
    "- 关键数字/事实错误、自相矛盾、把不同芯片混淆、或答非所问 → false\n"
    "- 回避问题、未给出确定答案、或只说“需查阅文档”而未回答 → false\n"
)


def _load_questions(path: Path, category=None, ids=None):
    data = yaml.safe_load(open(path, encoding='utf-8')) or []
    if category:
        data = [q for q in data if q.get('category') == category]
    if ids:
        want = {i.strip() for i in ids.split(',') if i.strip()}
        data = [q for q in data if q.get('id') in want]
    return data


def _keyword_match(expected: str, keywords, answer: str) -> bool:
    if not answer:
        return False
    a = answer.lower()
    kws = [str(k).lower() for k in (keywords or []) if str(k).strip()]
    if kws:
        return all(k in a for k in kws)
    return bool(expected) and expected.lower() in a


def _has_citation(answer: str) -> bool:
    return bool(answer) and bool(re.search(r'\[Source:', answer))


def _parse_json_obj(text: str):
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _judge(question: str, expected: str, answer: str, cfg: LLMConfig):
    """Return (correct: bool, reason: str, source: 'llm'|'keyword')."""
    # Empty answer is always wrong.
    if not answer or not answer.strip():
        return False, "空回答", "keyword"
    try:
        prompt = (f"问题：{question}\n标准答案：{expected}\n"
                  f"待评判回答：{answer}")
        out = await chat_completion(JUDGE_SYSTEM, prompt, config=cfg)
        obj = _parse_json_obj(out)
        if obj is not None and isinstance(obj.get("correct"), bool):
            return obj["correct"], str(obj.get("reason", ""))[:200], "llm"
    except Exception as e:
        return _keyword_match(expected, [], answer), f"judge error: {e}", "keyword"
    # Fallback to keyword matching (LLM judge returned unparsable output).
    return _keyword_match(expected, [], answer), "judge unparsable", "keyword"


async def _eval_one(q: dict, args, base_cfg: LLMConfig, judge_cfg: LLMConfig) -> dict:
    qid = q.get('id', '?')
    question = q.get('question', '')
    expected = str(q.get('expected', ''))
    keywords = q.get('keywords') or []

    # ── baseline: raw LLM ──
    b_ans, b_lat, b_err = "", 0.0, None
    t0 = time.time()
    try:
        b_ans = await chat_completion(BASELINE_SYSTEM, question, config=base_cfg)
        b_lat = time.time() - t0
    except Exception as e:
        b_err = str(e)

    # ── rag ──
    r_ans, r_src, r_lat, r_err = "", [], 0.0, None
    t0 = time.time()
    try:
        r_ans, r_src = await rag_answer(question, top_k=args.top_k, mode="auto")
        r_lat = time.time() - t0
    except Exception as e:
        r_err = str(e)

    # ── judge ──
    if args.no_judge:
        b_corr = _keyword_match(expected, keywords, b_ans)
        r_corr = _keyword_match(expected, keywords, r_ans)
        b_reason = r_reason = "keyword"
        b_src = r_src_ = "keyword"
    else:
        b_corr, b_reason, b_src = await _judge(question, expected, b_ans, judge_cfg)
        r_corr, r_reason, r_src_ = await _judge(question, expected, r_ans, judge_cfg)

    return {
        "id": qid,
        "category": q.get('category', ''),
        "question": question,
        "expected": expected,
        "baseline": {
            "answer": b_ans,
            "correct": bool(b_corr),
            "judge": b_src,
            "reason": b_reason,
            "has_citation": _has_citation(b_ans),
            "latency_s": round(b_lat, 2),
            "error": b_err,
        },
        "rag": {
            "answer": r_ans,
            "correct": bool(r_corr),
            "judge": r_src_,
            "reason": r_reason,
            "has_citation": _has_citation(r_ans),
            "latency_s": round(r_lat, 2),
            "num_sources": len(r_src),
            "top_doc_titles": sorted({s.doc_title for s in r_src if getattr(s, 'doc_title', '')})[:5],
            "error": r_err,
        },
    }


def _write_report(rows, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(rows)

    def acc(key):
        return sum(1 for r in rows if r[key]["correct"])

    def cit(key):
        return sum(1 for r in rows if r[key]["has_citation"])

    def avg_lat(key):
        vals = [r[key]["latency_s"] for r in rows if not r[key]["error"]]
        return sum(vals) / len(vals) if vals else 0.0

    b_acc, r_acc = acc("baseline"), acc("rag")
    b_cit, r_cit = cit("baseline"), cit("rag")

    summary = {
        "total": n,
        "baseline_accuracy": f"{b_acc}/{n}",
        "rag_accuracy": f"{r_acc}/{n}",
        "baseline_citation_rate": f"{b_cit}/{n}",
        "rag_citation_rate": f"{r_cit}/{n}",
        "baseline_avg_latency_s": round(avg_lat("baseline"), 2),
        "rag_avg_latency_s": round(avg_lat("rag"), 2),
        "improved_ids": [r["id"] for r in rows if r["rag"]["correct"] and not r["baseline"]["correct"]],
        "worse_ids": [r["id"] for r in rows if r["baseline"]["correct"] and not r["rag"]["correct"]],
    }

    # JSON
    json_path = out_dir / f"report_{ts}.json"
    json_path.write_text(json.dumps({"summary": summary, "results": rows},
                                    ensure_ascii=False, indent=2), encoding='utf-8')

    # Markdown
    md = []
    md.append(f"# ESP32 RAG 评测报告 ({ts})\n")
    md.append(f"- 题目数：**{n}**   top_k={args.top_k}   judge={'off' if args.no_judge else 'on'}\n")
    md.append("## 汇总\n")
    md.append("| 指标 | 裸大模型 (baseline) | 本项目 RAG |")
    md.append("|---|---|---|")
    md.append(f"| 准确率 | {b_acc}/{n} | {r_acc}/{n} |")
    md.append(f"| 引用率([Source:]) | {b_cit}/{n} | {r_cit}/{n} |")
    md.append(f"| 平均耗时 | {avg_lat('baseline'):.2f}s | {avg_lat('rag'):.2f}s |")
    md.append("")
    if summary["improved_ids"]:
        md.append(f"- ✅ RAG 答对而 baseline 答错：`{', '.join(summary['improved_ids'])}`")
    if summary["worse_ids"]:
        md.append(f"- ⚠️ baseline 答对而 RAG 答错：`{', '.join(summary['worse_ids'])}`")
    md.append("")
    md.append("## 逐题明细\n")
    md.append("| id | 分类 | 问题 | baseline | RAG | 来源数 |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        b = "✅" if r["baseline"]["correct"] else "❌"
        g = "✅" if r["rag"]["correct"] else "❌"
        if r["baseline"]["error"]:
            b += "(err)"
        if r["rag"]["error"]:
            g += "(err)"
        md.append(f"| {r['id']} | {r['category']} | {r['question']} | {b} | {g} | {r['rag'].get('num_sources','')} |")
    md.append("")

    # Per-question detail (answers + reasons) for manual inspection.
    md.append("## 答案与裁判理由\n")
    for r in rows:
        md.append(f"### {r['id']} — {r['question']}")
        md.append(f"- 标准答案：{r['expected']}")
        md.append(f"- **baseline**（{'✅' if r['baseline']['correct'] else '❌'}, {r['baseline']['judge']}）：{r['baseline']['answer'][:400]}")
        md.append(f"  - 理由：{r['baseline']['reason']}")
        md.append(f"- **RAG**（{'✅' if r['rag']['correct'] else '❌'}, {r['rag']['judge']}, 来源{r['rag'].get('num_sources',0)}条）：{r['rag']['answer'][:400]}")
        md.append(f"  - 理由：{r['rag']['reason']}")
        md.append("")

    md_path = out_dir / f"report_{ts}.md"
    md_path.write_text("\n".join(md), encoding='utf-8')
    return md_path, json_path, summary


async def main():
    ap = argparse.ArgumentParser(description="ESP32 RAG baseline-vs-RAG evaluation")
    ap.add_argument("--questions", default=str(SKILL_DIR / "eval" / "questions.yaml"))
    ap.add_argument("--out", default=str(SKILL_DIR / "eval" / "reports"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--category", default=None)
    ap.add_argument("--ids", default=None, help="逗号分隔的 id 列表")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="不做 LLM 裁判，仅关键词兜底")
    args = ap.parse_args()

    cfg = LLMConfig()
    if not cfg.is_configured:
        print("ERROR: LLM 未配置，请在 .env 设置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")
        sys.exit(1)

    questions = _load_questions(Path(args.questions), args.category, args.ids)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        print("没有匹配的题目。")
        sys.exit(1)

    base_cfg = LLMConfig()
    judge_cfg = LLMConfig()
    judge_cfg.temperature = 0.0
    judge_cfg.max_tokens = 200

    print(f"评测开始：{len(questions)} 题，model={cfg.model}，"
          f"judge={'off' if args.no_judge else 'on'}，top_k={args.top_k}\n")

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.get('id')}: {q.get('question')}")
        row = await _eval_one(q, args, base_cfg, judge_cfg)
        rows.append(row)
        b = "OK " if row["baseline"]["correct"] else "X  "
        g = "OK " if row["rag"]["correct"] else "X  "
        print(f"    baseline={b} ({row['baseline']['latency_s']}s) | "
              f"RAG={g} ({row['rag']['latency_s']}s, {row['rag'].get('num_sources',0)} src)\n")

    md_path, json_path, summary = _write_report(rows, args, Path(args.out))
    print("=" * 60)
    print(f"汇总：baseline 准确率 {summary['baseline_accuracy']}，"
          f"RAG 准确率 {summary['rag_accuracy']}")
    print(f"      引用率 baseline {summary['baseline_citation_rate']} vs RAG {summary['rag_citation_rate']}")
    print(f"报告：{md_path}")
    print(f"      {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
