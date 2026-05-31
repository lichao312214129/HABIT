"""Run all habitat train → predict script pairs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAIRS: list[tuple[str, str]] = [
    ("habitat_one_step_raw_concat_train.py", "habitat_one_step_raw_concat_predict.py"),
    ("habitat_one_step_voxel_radiomics_train.py", "habitat_one_step_voxel_radiomics_predict.py"),
    ("habitat_two_step_raw_concat_train.py", "habitat_two_step_raw_concat_predict.py"),
    ("habitat_two_step_voxel_radiomics_train.py", "habitat_two_step_voxel_radiomics_predict.py"),
    ("habitat_two_step_supervoxel_radiomics_train.py", "habitat_two_step_supervoxel_radiomics_predict.py"),
    ("habitat_direct_pooling_raw_concat_train.py", "habitat_direct_pooling_raw_concat_predict.py"),
    ("habitat_direct_pooling_voxel_radiomics_train.py", "habitat_direct_pooling_voxel_radiomics_predict.py"),
]


def _run(script_name: str, timeout_sec: int) -> tuple[int, str]:
    script = ROOT / "tests" / "habitat" / script_name
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out[-3000:] if len(out) > 3000 else out
    return proc.returncode, tail


def main() -> int:
    results: list[dict] = []
    for train_script, predict_script in PAIRS:
        print(f"\n=== TRAIN {train_script} ===", flush=True)
        train_code, train_tail = _run(train_script, timeout_sec=900)
        print(f"train exit={train_code}", flush=True)
        predict_code = -1
        predict_tail = ""
        if train_code == 0:
            print(f"=== PREDICT {predict_script} ===", flush=True)
            predict_code, predict_tail = _run(predict_script, timeout_sec=600)
            print(f"predict exit={predict_code}", flush=True)
        else:
            print(f"skip predict (train failed): {predict_script}", flush=True)
        results.append(
            {
                "train": train_script,
                "predict": predict_script,
                "train_code": train_code,
                "predict_code": predict_code,
                "train_tail": train_tail,
                "predict_tail": predict_tail,
            }
        )

    report = ROOT / "tests" / "_habitat_train_predict_report.txt"
    lines = ["Habitat train → predict report", "=" * 72, ""]
    ok = fail = 0
    for row in results:
        train_ok = row["train_code"] == 0
        pred_ok = row["predict_code"] == 0
        if train_ok and pred_ok:
            ok += 1
            status = "OK"
        else:
            fail += 1
            status = "FAIL"
        lines.append(f"[{status}] {row['train']} -> {row['predict']}")
        lines.append(f"  train={row['train_code']} predict={row['predict_code']}")
        lines.append("  --- train tail ---")
        lines.append(row["train_tail"])
        lines.append("  --- predict tail ---")
        lines.append(row["predict_tail"])
        lines.append("")
    lines.append(f"SUMMARY: {ok} ok, {fail} fail (of {len(results)} pairs)")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {report}")
    print(lines[-1])
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
