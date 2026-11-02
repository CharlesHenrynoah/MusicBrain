#!/usr/bin/env python3
"""MusicBrain — Z-score then L2 normalisation of 32-D pattern vectors.

Does not modify musical-event JSON, event matrices, or original vectors32 sources
beyond reading them. Writes dedicated normalised artefacts + a verification report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12


def zscore_transform(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Column-wise z-score. Constant dims (σ≈0) → z=0."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    constant_dims = np.where(sigma < EPS)[0].tolist()

    Z = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        if sigma[j] < EPS:
            Z[:, j] = 0.0
        else:
            Z[:, j] = (X[:, j] - mu[j]) / sigma[j]
    return Z, mu, sigma, constant_dims


def l2_normalize(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Row-wise L2 normalisation. All-zero rows stay zero."""
    norms = np.linalg.norm(Z, axis=1)
    V = np.empty_like(Z, dtype=np.float64)
    zero_rows: list[int] = []
    for i in range(Z.shape[0]):
        if norms[i] < EPS:
            V[i] = 0.0
            zero_rows.append(i)
        else:
            V[i] = Z[i] / norms[i]
    return V, norms, zero_rows


def describe(arr: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(arr.shape),
        "mean_per_dim": arr.mean(axis=0).tolist(),
        "std_per_dim": arr.std(axis=0, ddof=0).tolist(),
        "global_mean": float(arr.mean()),
        "global_std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def verify(
    X: np.ndarray,
    Z: np.ndarray,
    V: np.ndarray,
    norms_after: np.ndarray,
    zero_rows: list[int],
    index: list[dict[str, Any]],
    feature_names: list[str],
) -> list[str]:
    anomalies: list[str] = []

    if X.shape[0] != 305:
        anomalies.append(f"Expected 305 patterns, got {X.shape[0]}")
    if X.shape[1] != 32 or Z.shape[1] != 32 or V.shape[1] != 32:
        anomalies.append(f"Expected 32 dims, got X={X.shape}, Z={Z.shape}, V={V.shape}")
    if len(index) != X.shape[0]:
        anomalies.append(f"Index length {len(index)} != n_patterns {X.shape[0]}")
    if len(feature_names) != 32:
        anomalies.append(f"feature_names length {len(feature_names)} != 32")

    for name, arr in ("original", X), ("zscore", Z), ("normalized", V):
        if np.isnan(arr).any():
            anomalies.append(f"{name}: NaN detected")
        if np.isinf(arr).any():
            anomalies.append(f"{name}: Inf detected")

    for i in range(V.shape[0]):
        n = float(np.linalg.norm(V[i]))
        if i in zero_rows:
            if n > EPS:
                anomalies.append(f"Row {i}: marked zero but L2={n}")
        else:
            if abs(n - 1.0) > 1e-6:
                anomalies.append(f"Row {i}: L2={n} (expected 1)")

    # Correspondence: pattern_id uniqueness within (id, instrument)
    keys = [(row["pattern_id"], row["instrument"]) for row in index]
    if len(keys) != len(set(keys)):
        anomalies.append("Duplicate (pattern_id, instrument) in index")

    return anomalies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", type=Path, default=Path("data/vectors32.npy"))
    parser.add_argument("--input-meta", type=Path, default=Path("data/vectors32_meta.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/normalized"))
    args = parser.parse_args()

    X = np.load(args.input_npy).astype(np.float64)
    meta = json.loads(args.input_meta.read_text())
    feature_names: list[str] = meta["feature_names"]
    index: list[dict[str, Any]] = meta["index"]

    # Keep a pristine copy of originals for export (do not mutate source file).
    X_original = X.copy()

    Z, mu, sigma, constant_dims = zscore_transform(X_original)
    V, norms_before_l2, zero_rows = l2_normalize(Z)
    norms_after = np.linalg.norm(V, axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": args.out_dir / "patterns_32d_original.npy",
        "zscore": args.out_dir / "patterns_32d_zscore.npy",
        "normalized": args.out_dir / "patterns_32d_normalized.npy",
        "scaler": args.out_dir / "scaler_params.json",
        "report": args.out_dir / "normalization_report.json",
        "index": args.out_dir / "pattern_index.json",
    }

    np.save(paths["original"], X_original)
    np.save(paths["zscore"], Z)
    np.save(paths["normalized"], V)

    scaler = {
        "n_patterns": int(X_original.shape[0]),
        "n_dims": int(X_original.shape[1]),
        "feature_names": feature_names,
        "mean": mu.tolist(),
        "std": sigma.tolist(),
        "ddof": 0,
        "eps": EPS,
        "constant_dims": constant_dims,
        "constant_feature_names": [feature_names[j] for j in constant_dims],
        "zero_l2_rows_after_zscore": zero_rows,
        "transform": [
            "z_ij = 0 if sigma_j < eps else (x_ij - mu_j) / sigma_j",
            "V_i = 0 if ||Z_i||_2 < eps else Z_i / ||Z_i||_2",
        ],
        "index": index,
        "source_vectors": str(args.input_npy),
        "source_meta": str(args.input_meta),
    }
    paths["scaler"].write_text(json.dumps(scaler, ensure_ascii=False, indent=2) + "\n")
    paths["index"].write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    anomalies = verify(
        X_original, Z, V, norms_after, zero_rows, index, feature_names
    )

    # Stats before / after z-score
    stats_before = describe(X_original)
    stats_after_z = describe(Z)

    # L2 norms of raw and zscored (before unit normalisation) vs after
    l2_raw = np.linalg.norm(X_original, axis=1)
    l2_z = norms_before_l2
    l2_n = norms_after

    def norm_summary(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }

    report = {
        "checks": {
            "n_patterns": int(X_original.shape[0]),
            "n_dims": int(X_original.shape[1]),
            "has_nan": bool(
                np.isnan(X_original).any() or np.isnan(Z).any() or np.isnan(V).any()
            ),
            "has_inf": bool(
                np.isinf(X_original).any() or np.isinf(Z).any() or np.isinf(V).any()
            ),
            "l2_unit_ok": len(anomalies) == 0
            or all(not a.startswith("Row") for a in anomalies),
            "anomalies": anomalies,
            "passed": len(anomalies) == 0,
        },
        "constant_dimensions": {
            "indices": constant_dims,
            "names": [feature_names[j] for j in constant_dims],
            "values": [float(mu[j]) for j in constant_dims],
        },
        "zero_vectors_after_zscore": {
            "count": len(zero_rows),
            "rows": zero_rows,
            "patterns": [index[i] for i in zero_rows],
        },
        "stats_before_zscore": stats_before,
        "stats_after_zscore": stats_after_z,
        "l2_norms": {
            "original": norm_summary(l2_raw),
            "after_zscore_before_l2": norm_summary(l2_z),
            "after_l2_normalization": norm_summary(l2_n),
            "non_zero_after_l2_dev_from_1_max": float(
                np.max(np.abs(l2_n[l2_n > EPS] - 1.0)) if np.any(l2_n > EPS) else 0.0
            ),
        },
        "files": {k: str(v.resolve()) for k, v in paths.items()},
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # Console report
    print("=" * 64)
    print("MusicBrain — Normalisation 32D")
    print("=" * 64)
    print(f"Patterns : {X_original.shape[0]}  |  Dims : {X_original.shape[1]}")
    print(f"Checks   : {'PASSED' if report['checks']['passed'] else 'FAILED'}")
    if anomalies:
        for a in anomalies:
            print(f"  ! {a}")

    print("\nDimensions constantes (σ≈0) :")
    if constant_dims:
        for j in constant_dims:
            print(f"  D{j+1:02d} {feature_names[j]} = {mu[j]:.6f}")
    else:
        print("  aucune")

    print("\nStats globales avant Z-score :")
    print(f"  mean={stats_before['global_mean']:.6f}  std={stats_before['global_std']:.6f}")
    print(f"  min={stats_before['min']:.6f}  max={stats_before['max']:.6f}")
    print("Stats globales après Z-score :")
    print(f"  mean={stats_after_z['global_mean']:.6e}  std={stats_after_z['global_std']:.6f}")
    print(f"  min={stats_after_z['min']:.6f}  max={stats_after_z['max']:.6f}")

    print("\nNormes L2 :")
    for label, summary in report["l2_norms"].items():
        if label.endswith("_max"):
            continue
        if isinstance(summary, dict):
            print(
                f"  {label:28s} "
                f"min={summary['min']:.6f} max={summary['max']:.6f} "
                f"mean={summary['mean']:.6f}"
            )
    print(
        "  max |L2-1| (non-nuls)       "
        f"{report['l2_norms']['non_zero_after_l2_dev_from_1_max']:.2e}"
    )

    print(f"\nVecteurs nuls après Z-score : {len(zero_rows)}")
    print("\nFichiers :")
    for k, p in paths.items():
        print(f"  {k:12s} {p}")


if __name__ == "__main__":
    main()
