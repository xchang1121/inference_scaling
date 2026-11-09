"""Task-local archive round trip; portable rebuild commands are in the package README."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE / "source"
PACKAGE = REPO / "experiments/online-injection-20261109"
REPORT = BASE.parents[1] / "outputs/online-injection-a3-20261109"
WORKSPACE = BASE / "handoff-roundtrip"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    subprocess.run([sys.executable, "-X", "utf8", *map(str, args)], check=True)


def main():
    assert not WORKSPACE.exists(), "Preserve any previous round-trip workspace."
    run(PACKAGE / "unpack.py", "--verify-only")
    run(PACKAGE / "unpack.py", "--output", WORKSPACE)
    frozen = json.loads((WORKSPACE / "protocol/FROZEN.json").read_text(encoding="utf-8"))
    source_files = input_files = 0
    for name, expected in frozen["files"].items():
        p = (WORKSPACE / "generation-source" / name.removeprefix("source/")
             if name.startswith("source/") else WORKSPACE / name)
        if p.is_file():
            assert digest(p) == expected, name
            source_files += name.startswith("source/")
            input_files += not name.startswith("source/")
    assert source_files == 118
    assert digest(WORKSPACE / "protocol/runtime-model_runner_v1.py") == frozen["runtime_alignment_sha256"]
    source_audit = json.loads((WORKSPACE / "protocol/validation-source-audit.json").read_text(encoding="utf-8"))
    assert source_audit["frozen_bytes_verified"] and source_audit["checkout_text_verified"]
    assert all(f["newline_normalized_equal"] for f in source_audit["files"])
    rebuilt = WORKSPACE / "rebuilt"
    rebuilt.mkdir()
    shutil.copyfile(REPORT / "OBSERVATIONS.md", rebuilt / "OBSERVATIONS.md")
    # Preserve the authored interpretation while independently recalculating
    # the scores, all audit checks, and both-cohort figures from raw records.
    shutil.copytree(REPORT / "figures", rebuilt / "figures")
    os.environ["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(BASE / "devdeps")))
    run(REPO / "scripts/summarize_injection.py", "--snapshot", WORKSPACE / "snapshot",
        "--data", WORKSPACE / "data", "--output", rebuilt)
    comparisons = {}
    for name in ("summary.json", "audit-checks.json", "grades.jsonl", "REPORT.md"):
        actual, expected = digest(rebuilt / name), digest(REPORT / name)
        assert actual == expected, (name, actual, expected)
        comparisons[name] = actual
    summary = json.loads((rebuilt / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["validation"]["status"] in {"passed", "failed"}
    assert summary["all_saved_measurements_pass_audit"]
    assert {k: v["generated"] for k, v in summary["datasets"].items()} == {"api": 208, "techqa": 104}
    # Render into a fresh child directory, leaving the checked report intact.
    run(REPO / "scripts/plot_injection_results.py", "--summary", rebuilt / "summary.json",
        "--snapshot", WORKSPACE / "snapshot", "--output", rebuilt / "rebuilt-figures")
    figures = {}
    for name in ("quality-paired.png", "cost-ratios.png", "thinking-position.png"):
        actual, expected = digest(rebuilt / "rebuilt-figures" / name), digest(REPORT / "figures" / name)
        assert actual == expected, (name, actual, expected)
        figures[name] = actual
    receipt = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": digest(PACKAGE / "full/analysis-workspace.tar.gz"),
        "unpack_verify_passed": True,
        "fresh_directory_unpack_passed": True,
        "frozen_generation_source_files_verified": source_files,
        "present_frozen_data_files_verified": input_files,
        "runtime_model_runner_matches_frozen_sha256": True,
        "validation_imported_frozen_source_verified": True,
        "recomputed_measurements": 312,
        "original_cache_validation_status_preserved": summary["validation"]["status"],
        "conditional_cache_diagnostic_status_preserved": summary["validation"]["conditional_cache"]["status"],
        "recomputed_scores_audits_and_report_byte_identical": comparisons,
        "regenerated_png_figures_byte_identical": figures,
        "verification_script_sha256": digest(Path(__file__)),
        "network_requests": 0,
        "model_generation_requests": 0,
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    (REPORT / "handoff-verification.json").write_text(text, encoding="utf-8")
    target = PACKAGE / "reports/handoff-verification.json"
    target.write_text(text, encoding="utf-8")
    verifier = PACKAGE / "analysis/verify_handoff.py"
    shutil.copyfile(__file__, verifier)
    manifest_path = PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for p in (target, verifier):
        relative = p.relative_to(PACKAGE).as_posix()
        assert not any(e["path"] == relative for e in manifest["direct_files"])
        manifest["direct_files"].append({"path": relative, "bytes": p.stat().st_size, "sha256": digest(p)})
    manifest["direct_files"].sort(key=lambda e: e["path"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run(PACKAGE / "unpack.py", "--verify-only")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
