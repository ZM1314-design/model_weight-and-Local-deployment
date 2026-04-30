import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request


DEFAULT_SYSTEM = (
    "你是一名农业技术推广专家（农艺师），回答要贴近农业生产实践，优先给出可操作建议。"
    "尽量用要点列出：作物/生育期/问题诊断/关键指标/处理措施/用药与安全间隔期/注意事项。"
    "如信息不足要先问清楚地区、作物、品种、生育期、土壤与天气、症状表现。"
)


def call_api(api_url: str, payload: dict, timeout: int = 180) -> tuple[dict | None, str | None, int]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            latency_ms = int((time.time() - t0) * 1000)
            return json.loads(body), None, latency_ms
    except error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        return None, f"HTTPError {e.code}: {msg}", int((time.time() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000)


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if "prompt" not in item:
                raise ValueError(f"Missing 'prompt' at line {line_no}")
            item.setdefault("id", f"q_{line_no:03d}")
            item.setdefault("category", "未分类")
            questions.append(item)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local model API with question set.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--questions", default="eval_questions.jsonl")
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N questions")
    parser.add_argument("--timeout", type=int, default=90, help="Per-request timeout seconds")
    args = parser.parse_args()

    qpath = Path(args.questions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(qpath)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"eval_{ts}.csv"
    summary_path = out_dir / f"eval_{ts}_summary.txt"

    rows: list[dict] = []
    ok_count = 0
    total_latency = 0

    for i, q in enumerate(questions, start=1):
        payload = {
            "system": args.system,
            "prompt": q["prompt"],
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        }
        result, err, latency_ms = call_api(args.api_url, payload, timeout=args.timeout)
        api_id = result.get("id") if result else ""
        model_text = result.get("text", "") if result else ""
        server_time_ms = result.get("time_ms", "") if result else ""
        status = "ok" if err is None else "error"
        if status == "ok":
            ok_count += 1
            total_latency += latency_ms

        row = {
            "qid": q["id"],
            "category": q["category"],
            "prompt": q["prompt"],
            "status": status,
            "error": err or "",
            "api_id": api_id,
            "client_latency_ms": latency_ms,
            "server_time_ms": server_time_ms,
            "response": model_text,
            "score_accuracy_1to5": "",
            "score_actionable_1to5": "",
            "score_safety_1to5": "",
            "notes": "",
        }
        rows.append(row)
        print(f"[{i}/{len(questions)}] {q['id']} {status} latency={latency_ms}ms", flush=True)

        if args.sleep_ms > 0 and i < len(questions):
            time.sleep(args.sleep_ms / 1000.0)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_latency = int(total_latency / ok_count) if ok_count else 0
    summary_lines = [
        f"questions_total={len(questions)}",
        f"success={ok_count}",
        f"failed={len(questions) - ok_count}",
        f"avg_client_latency_ms={avg_latency}",
        f"api_url={args.api_url}",
        f"output_csv={csv_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n=== Done ===", flush=True)
    for line in summary_lines:
        print(line, flush=True)


if __name__ == "__main__":
    main()

