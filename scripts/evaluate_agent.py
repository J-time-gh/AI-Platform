"""评估 Agent 的工具调用和回答。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc

        required = {"id", "category", "question", "expected_tools"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} 缺少字段：{sorted(missing)}")

        cases.append(case)

    return cases


def tools_match(case: dict[str, Any], actual_tools: list[str]) -> bool:
    expected_tools = case["expected_tools"]
    forbidden_tools = set(case.get("must_not_call_tools", []))

    if any(tool in forbidden_tools for tool in actual_tools):
        return False

    return actual_tools == expected_tools


def arguments_match(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> bool:
    expected_arguments = case.get("expected_arguments")
    if expected_arguments is None:
        return True

    actual_arguments = [call["arguments"] for call in tool_calls]
    return actual_arguments[: len(expected_arguments)] == expected_arguments


def answer_match(case: dict[str, Any], answer: str) -> bool:
    return all(text in answer for text in case.get("expected_answer_contains", []))


def evaluate_case(
    client: httpx.Client,
    base_url: str,
    case: dict[str, Any],
    model: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": case["question"]}
    if model:
        payload["model"] = model

    response = client.post(f"{base_url}/agent", json=payload)
    result: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "http_status": response.status_code,
        "passed": False,
    }

    if response.status_code != 200:
        result["error"] = response.text
        return result

    body = response.json()
    tool_calls = body["tool_calls"]
    actual_tools = [call["tool_name"] for call in tool_calls]
    min_calls = case.get("min_tool_calls", 0)
    max_calls = case.get("max_tool_calls", sys.maxsize)

    result.update(
        {
            "answer": body["answer"],
            "actual_tools": actual_tools,
            "tool_calls": tool_calls,
            "tool_match": tools_match(case, actual_tools),
            "arguments_match": arguments_match(case, tool_calls),
            "answer_match": answer_match(case, body["answer"]),
            "call_count_match": min_calls <= len(tool_calls) <= max_calls,
        },
    )
    result["passed"] = all(
        [
            result["tool_match"],
            result["arguments_match"],
            result["answer_match"],
            result["call_count_match"],
        ],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/agent_questions.jsonl"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/agent_report.json"),
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results: list[dict[str, Any]] = []

    with httpx.Client(timeout=60.0) as client:
        for case in cases:
            result = evaluate_case(client, args.base_url, case, args.model)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {result['id']}")

    category_totals = Counter(result["category"] for result in results)
    category_passed = Counter(result["category"] for result in results if result["passed"])
    passed_count = sum(result["passed"] for result in results)

    report = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results) if results else 0.0,
        "by_category": {
            category: {
                "total": total,
                "passed": category_passed[category],
                "pass_rate": category_passed[category] / total,
            }
            for category, total in sorted(category_totals.items())
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n总计：{report['passed']}/{report['total']}")
    print(f"报告：{args.output}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
