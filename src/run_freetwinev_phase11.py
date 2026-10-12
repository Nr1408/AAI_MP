from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "freetwinev"
REPORTS = ROOT / "reports"


EXPECTED_PHASE6 = {
    ("60s", "Test_MAE"): 0.2271,
    ("60s", "Test_RMSE"): 0.8256,
    ("60s", "Test_R2"): 0.8060,
    ("300s", "Test_MAE"): 1.9302,
    ("300s", "Test_RMSE"): 4.9649,
    ("300s", "Test_R2"): 0.5836,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def metric_check(name: str, actual: float, expected: float, tol: float = 5e-4) -> dict[str, object]:
    return {
        "status": "PASS" if abs(actual - expected) <= tol else "FAIL",
        "expected": expected,
        "actual": actual,
    }


def audit_status(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    status = data.get("status")
    overall = data.get("overall_status")
    if status == "PASS" or overall == "PASS":
        return "PASS"
    return str(status or overall or "UNKNOWN")


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    final_report = REPORTS / "freetwinev_phase11_final_report.md"
    slide_source = REPORTS / "freetwinev_phase11_presentation.md"

    for label, path in {
        "artifact_exists_freetwinev_phase11_final_report.md": final_report,
        "artifact_exists_freetwinev_phase11_presentation.md": slide_source,
    }.items():
        checks[label] = {
            "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size if path.exists() else 0,
        }

    phase6_rows = read_csv_rows(RESULTS / "phase6_results.csv")
    by_horizon = {row["Horizon"]: row for row in phase6_rows}
    for (horizon, metric), expected in EXPECTED_PHASE6.items():
        actual = float(by_horizon[horizon][metric])
        checks[f"phase6_{horizon}_{metric}"] = metric_check(
            f"phase6_{horizon}_{metric}", actual, expected
        )

    audit_files = {
        "phase5": RESULTS / "phase5_audit.json",
        "phase6": RESULTS / "phase6_audit.json",
        "phase7": RESULTS / "phase7_audit.json",
        "phase8": RESULTS / "phase8_audit.json",
        "phase9": RESULTS / "phase9_audit.json",
        "phase10": RESULTS / "phase10_dashboard_audit.json",
    }
    audit_notes: dict[str, object] = {}
    for phase, path in audit_files.items():
        observed = audit_status(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        checks[f"{phase}_audit_pass"] = {
            "status": "PASS" if observed == "PASS" else "FAIL",
            "observed": observed,
            "top_level_status": raw.get("status"),
            "overall_status": raw.get("overall_status"),
        }
        audit_notes[phase] = {
            "top_level_status": raw.get("status"),
            "overall_status": raw.get("overall_status"),
        }

    frozen_files = [
        RESULTS / "phase5_audit.json",
        RESULTS / "phase5_results.csv",
        RESULTS / "phase6_audit.json",
        RESULTS / "phase6_results.csv",
        RESULTS / "phase6_results_per_run.csv",
        RESULTS / "phase7_audit.json",
        RESULTS / "phase8_audit.json",
        RESULTS / "phase8_thresholds.json",
        RESULTS / "phase9_audit.json",
        RESULTS / "phase9_recommendation_config.json",
        RESULTS / "phase10_dashboard_audit.json",
        RESULTS / "preprocessed" / "feature_manifest.json",
        RESULTS / "preprocessed" / "preprocessing_config.json",
        RESULTS / "preprocessed" / "split_manifest.csv",
    ]

    failed = [name for name, check in checks.items() if check["status"] != "PASS"]
    report = {
        "phase": "Phase 11",
        "status": "PASS" if not failed else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failed_checks": failed,
        "checks": checks,
        "audit_status_note": (
            "Phase 8 and Phase 9 use top-level status COMPLETE with overall_status PASS. "
            "This audit treats overall_status PASS as the substantive prior-phase audit result."
        ),
        "prior_audit_statuses": audit_notes,
        "frozen_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in frozen_files if path.exists()
        },
        "phase11_artifacts": {
            "report": str(final_report.relative_to(ROOT)),
            "presentation": str(slide_source.relative_to(ROOT)),
        },
        "methodology_changes": False,
        "model_retraining": False,
        "test_tuning": False,
        "new_predictions": False,
    }

    (RESULTS / "phase11_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failed_checks": failed}, indent=2))


if __name__ == "__main__":
    main()

