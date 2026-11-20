#!/usr/bin/env python3
"""Run the whole pipeline from the frozen inputs and compare every output table with reference/.

  python scripts/run_all.py               all eight stages, then the comparison
  python scripts/run_all.py --check-only  only the input and script integrity check

The run starts by checking every input file and script against SOURCE_INPUT_MANIFEST.json, writes all
outputs to a new generated/ folder, and ends with "SUCCESS <n> exact CSV matches" when each output table
is byte-identical to its copy in reference/. Reference files are used only for this comparison, never
as inputs. The log of each stage is written to verification/.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS, OUT, SCRIPTS = ROOT / "inputs", ROOT / "generated", ROOT / "scripts"

# Output folders in stage order; each one has a matching folder in reference/
OUTPUT_FOLDERS = ["climate", "scarcity_index", "hindcast", "forecasts", "decision", "extensions", "tables"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages():
    """(description, command) of the eight stages."""
    py = sys.executable
    gen = ["--generated-dir", str(OUT)]
    inp = ["--input-dir", str(INPUTS)]
    return [
        (
            "1 Climate predictors and FAO-56 crop water requirements",
            [py, SCRIPTS / "01_climate.py", *inp, "--output-dir", OUT / "climate"],
        ),
        (
            "2 Relative water scarcity index (RAWSI)",
            [
                py,
                SCRIPTS / "02_scarcity_index.py",
                *inp,
                "--output-dir",
                OUT / "scarcity_index",
                "--climate-file",
                OUT / "climate" / "pm_seasonal_fao56_16districts_3crops_2015_2025.csv",
            ],
        ),
        (
            "3 Hindcast yield models and SHAP (descriptive)",
            [py, SCRIPTS / "03_hindcast_models.py", *inp, "--output-dir", OUT / "hindcast"],
        ),
        (
            "4 Decision-time yield forecasts",
            [py, SCRIPTS / "04_forecasts.py", *inp, "--output-dir", OUT / "forecasts"],
        ),
        (
            "5 Decision value of the forecasts and robust planning",
            [py, SCRIPTS / "05_decision_value.py", *inp, *gen, "--output-dir", OUT / "decision"],
        ),
        (
            "6 District allocation model and scenarios",
            [py, SCRIPTS / "06_allocation_scenarios.py", *inp, *gen, "--output-dir", OUT / "extensions"],
        ),
        ("7 Tables", [py, SCRIPTS / "07_tables.py", *gen, "--output-dir", OUT / "tables"]),
        ("8 Figures", [py, SCRIPTS / "08_figures.py", *inp, *gen, "--output-dir", OUT / "figures"]),
    ]


def check_inputs():
    manifest = json.loads((ROOT / "SOURCE_INPUT_MANIFEST.json").read_text())
    for rel, expected in manifest.items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError("Input or script differs from SOURCE_INPUT_MANIFEST.json: " + rel)
    print(f"Input and script integrity: {len(manifest)} files matched", flush=True)


def run_stages():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    (ROOT / "verification").mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"  # single-threaded linear algebra, so results do not depend on the machine
    receipts = []
    for i, (name, cmd) in enumerate(stages(), 1):
        print("RUN", name, flush=True)
        proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, text=True, capture_output=True)
        (ROOT / "verification" / f"step_{i:02d}.stdout.txt").write_text(proc.stdout)
        (ROOT / "verification" / f"step_{i:02d}.stderr.txt").write_text(proc.stderr)
        receipts.append(dict(step=name, exit_code=proc.returncode))
        if proc.returncode:
            raise RuntimeError(f"Stage {i} failed:\n" + proc.stderr[-1500:])
    return receipts


def compare_with_reference():
    rows = []
    for folder in OUTPUT_FOLDERS:
        for ref in sorted((ROOT / "reference" / folder).glob("*.csv")):
            gen = OUT / folder / ref.name
            if not gen.exists():
                raise FileNotFoundError("Output missing: " + str(gen))
            matched = sha256(ref) == sha256(gen)
            rows.append(dict(folder=folder, file=ref.name, sha_match=matched))
            if not matched:
                raise AssertionError("Output differs from reference: " + str(gen))
    return rows


if __name__ == "__main__":
    check_inputs()
    if "--check-only" in sys.argv:
        sys.exit(0)
    receipts = run_stages()
    rows = compare_with_reference()
    import pandas as pd

    pd.DataFrame(rows).to_csv(ROOT / "verification" / "OUTPUT_CERTIFICATE.csv", index=False)
    meta = dict(python=platform.python_version(), steps=receipts, num_exact_csv=len(rows))
    (ROOT / "verification" / "REPLAY_RECEIPT.json").write_text(json.dumps(meta, indent=2))
    print("SUCCESS", len(rows), "exact CSV matches", flush=True)
