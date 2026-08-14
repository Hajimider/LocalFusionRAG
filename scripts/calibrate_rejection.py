"""在校准集上选择 CrossEncoder 拒答阈值，并在独立测试子集报告效果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def top_score(case: dict) -> float:
    scores = [
        source.get("rerank_score")
        for source in case.get("retrieved_sources", [])
        if source.get("rerank_score") is not None
    ]
    return max(scores, default=float("-inf"))


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    calibration, test = [], []
    groups = {True: [], False: []}
    for case in cases:
        groups[bool(case.get("expected_sources"))].append(case)
    for group in groups.values():
        for index, case in enumerate(sorted(group, key=lambda item: str(item.get("id", "")))):
            (calibration if index % 2 == 0 else test).append(case)
    return calibration, test


def metrics(cases: list[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for case in cases:
        expected_answerable = bool(case.get("expected_sources"))
        predicted_answerable = top_score(case) >= threshold
        if expected_answerable and predicted_answerable:
            tp += 1
        elif expected_answerable:
            fn += 1
        elif predicted_answerable:
            fp += 1
        else:
            tn += 1
    answerable_recall = tp / (tp + fn) if tp + fn else 0.0
    rejection_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "count": len(cases),
        "accuracy": round((tp + tn) / len(cases), 4) if cases else 0.0,
        "balanced_accuracy": round((answerable_recall + rejection_recall) / 2, 4),
        "answerable_recall": round(answerable_recall, 4),
        "rejection_recall": round(rejection_recall, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def choose_threshold(cases: list[dict]) -> tuple[float, dict]:
    values = sorted({top_score(case) for case in cases if top_score(case) != float("-inf")})
    candidates = [0.0, *values]
    scored = [(metrics(cases, threshold), threshold) for threshold in candidates]
    best_metrics, best_threshold = max(
        scored,
        key=lambda item: (
            item[0]["balanced_accuracy"],
            item[0]["rejection_recall"],
            item[0]["answerable_recall"],
            -item[1],
        ),
    )
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="校准低相关拒答阈值")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rejection_calibration.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = report["retrieval_ablation"]["hybrid_rerank"]["cases"]
    calibration, test = split_cases(cases)
    threshold, calibration_metrics = choose_threshold(calibration)
    result = {
        "source_report": args.report.as_posix(),
        "selection_rule": "校准集 balanced_accuracy 最大；并列时优先拒答召回率",
        "threshold": round(threshold, 6),
        "calibration": calibration_metrics,
        "test": metrics(test, threshold),
        "calibration_ids": [case.get("id") for case in calibration],
        "test_ids": [case.get("id") for case in test],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
