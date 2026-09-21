"""
run_evaluation.py — CLI entry point to run SATQuery reproducible evaluation benchmark.

Generates:
  - outputs/evaluation/evaluation_report.json
  - outputs/evaluation/EVALUATION_REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.benchmark_runner import SATQueryBenchmarkRunner
from src.evaluation.report_generator import generate_evaluation_reports


def main():
    print("=============================================================")
    print("   SATQuery Reproducible Benchmark Evaluation Engine")
    print("=============================================================")

    runner = SATQueryBenchmarkRunner(output_dir="outputs/evaluation")
    report_data = runner.run_all_evaluations()

    json_path, md_path = generate_evaluation_reports(report_data, output_dir="outputs/evaluation")

    summary = report_data["overall_summary"]
    print("\nEVALUATION SUMMARY:")
    print(f"  Workflows Evaluated:          {summary['total_workflows_evaluated']}")
    print(f"  Successful Workflows:         {summary['successful_workflows']}")
    print(f"  Workflow Execution Success:  {summary['workflow_execution_success_rate'] * 100:.1f}%")
    print(f"  Query Routing Success Rate:   {summary['routing_success_rate'] * 100:.1f}%")
    print(f"  Verifier Acceptance Rate:     {summary['verification_valid_case_acceptance_rate'] * 100:.1f}%")
    print(f"  Verifier Rejection Rate:      {summary['verification_invalid_case_rejection_rate'] * 100:.1f}%")
    print(f"  Total Runtime:                {summary['total_evaluation_runtime_seconds']} s")

    print("\nREPORTS GENERATED:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print("=============================================================")


if __name__ == "__main__":
    main()
