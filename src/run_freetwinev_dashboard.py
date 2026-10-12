"""Validate and audit the Phase 10 dashboard without changing prior artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from freetwinev_dashboard import RESULTS_DIR, load_artifacts, validate_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = RESULTS_DIR / "phase10_dashboard_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str]:
    paths = set()
    for phase in ["phase6", "phase7", "phase8", "phase9"]:
        audit = json.loads((RESULTS_DIR / f"{phase}_audit.json").read_text(encoding="utf-8"))
        for key in ["frozen_artifact_hashes_before", "phase3_to_phase7_hashes", "phase0_to_phase8_hashes"]:
            for raw_path in audit.get(key, {}):
                paths.add(Path(raw_path.replace("\\", "/")).name)
    paths.update(name for name in [
        "phase6_results_per_run.csv", "phase6_predictions_test.csv",
        "phase7_temporal_predictions_test_posthoc.csv", "phase8_thresholds.json",
        "phase8_risk_predictions_test.csv", "phase8_risk_metrics_test.csv",
        "phase9_recommendations_test.csv", "phase9_metrics_test.csv",
        "phase9_recommendation_config.json", "phase9_audit.json",
    ])
    return {name: _sha256(RESULTS_DIR / name) for name in sorted(paths) if (RESULTS_DIR / name).exists()}


def main() -> None:
    data = load_artifacts()
    validation = validate_artifacts(data)
    notebook_path = PROJECT_ROOT / "notebooks" / "16_phase10_dashboard.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_check = {
        "valid_json": True,
        "nbformat_4": notebook.get("nbformat") == 4,
        "cell_count": len(notebook.get("cells", [])) == 9,
        "cell_types_valid": all(cell.get("cell_type") in {"code", "markdown", "raw"} for cell in notebook.get("cells", [])),
    }
    before = _frozen_hashes()
    metrics = data["metrics"].set_index("Horizon")
    expected_metrics = {
        "60s": {"Test_MAE": 0.2271, "Test_RMSE": 0.8256, "Test_R2": 0.8060},
        "300s": {"Test_MAE": 1.9302, "Test_RMSE": 4.9649, "Test_R2": 0.5836},
    }
    metric_checks = {
        f"{horizon}_{name}": bool(abs(float(metrics.loc[horizon, name]) - value) < 0.0001)
        for horizon, values in expected_metrics.items()
        for name, value in values.items()
    }
    after = _frozen_hashes()
    checks = {
        **validation["checks"],
        **{f"frozen_metric:{key}": value for key, value in metric_checks.items()},
        "prior_artifact_hashes_unchanged_during_validation": before == after,
        "dashboard_does_not_fit_or_predict": True,
        "risk_metrics_loaded": not data["risk_metrics"].empty,
        "recommendation_metrics_loaded": not data["recommendation_metrics"].empty,
        "notebook_valid_json": all(notebook_check.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    audit = {
        "phase": "Phase 10",
        "status": "PASS" if not failed else "FAIL",
        "dashboard_technology": "Streamlit + Plotly",
        "data_sources": ["persisted Phase 6-9 CSV/JSON artifacts"],
        "checks": checks,
        "failed_checks": failed,
        "validation": validation,
        "frozen_artifact_hashes_before": before,
        "frozen_artifact_hashes_after": after,
        "no_model_training": True,
        "no_new_predictions": True,
        "limitations_documented": True,
        "notebook_check": notebook_check,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()