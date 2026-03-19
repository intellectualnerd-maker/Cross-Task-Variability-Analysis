import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging

def outlier_quickcheck(
    df: pd.DataFrame,
    audit_log_path: Union[str, Path] = "notebooks/reports/results/audit_log.md",
    save_flags: bool = True,
    flags_basename: str = "numeric_outlier_flags"
) -> pd.DataFrame:
    """
    Step 9 — Outlier quickcheck for numeric columns.

    Flags potential outliers in each numeric column based on 1st and 99th percentiles 
    and 5*IQR rule. Does not remove any data, only flags for audit purposes.
    """

    # Resolve paths
    audit_log_path = Path(audit_log_path)
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().isoformat(sep=" ", timespec="seconds") + " UTC"
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n# Step 9 — Outlier Quickcheck\nGenerated: {ts}\n\n")

    outlier_flags = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty:
            continue

        q01, q99 = s.quantile([0.01, 0.99])
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q01 - 5 * iqr
        upper_bound = q99 + 5 * iqr

        min_val, max_val = s.min(), s.max()
        flag = False

        if min_val < lower_bound or max_val > upper_bound:
            flag = True
            outlier_flags[col] = {
                "min": float(min_val),
                "q01": float(q01),
                "q1": float(q1),
                "median": float(s.median()),
                "q3": float(q3),
                "q99": float(q99),
                "max": float(max_val),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "possible_outlier": True
            }
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n- Possible outliers detected\n")
                f.write(f"- Min: {min_val}, Q01: {q01}, Q1: {q1}, Median: {s.median()}, Q3: {q3}, Q99: {q99}, Max: {max_val}\n")
                f.write(f"- Lower bound: {lower_bound}, Upper bound: {upper_bound}\n\n")
        else:
            outlier_flags[col] = {
                "min": float(min_val),
                "q01": float(q01),
                "q1": float(q1),
                "median": float(s.median()),
                "q3": float(q3),
                "q99": float(q99),
                "max": float(max_val),
                "possible_outlier": False
            }
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n- No extreme outliers detected\n\n")

    # Save the flags as JSON for documentation
    if save_flags and outlier_flags:
        out_dir = audit_log_path.parent
        json_path = out_dir / f"{flags_basename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(outlier_flags, f, indent=4)
        logging.info(f"Outlier flags saved to {json_path}")

    logging.info(f" Flags computed for {len(outlier_flags)} numeric column(s).")
    return df
