import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
from data_loader import data_loader

# Load your dataframe
df = data_loader()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def dtype_audit(
    df: pd.DataFrame,
    audit_report_path: Union[str, Path] = "notebooks/reports/results/column_audit.json",
    audit_df: Optional[pd.DataFrame] = None,
    audit_log_path: Union[str, Path] = "notebooks/reports/results/audit_log.md",
    save_updated_audit: bool = True,
    updated_audit_basename: str = "column_audit.fixed"
) -> pd.DataFrame:
    """
    Step 7 — Non-destructive type fixes driven by schema audit.
    """

    # --- Resolve project root ---
    project_root = Path.cwd()  # Use current working directory as project root
    audit_report_path = project_root / audit_report_path
    audit_log_path = project_root / audit_log_path

    # --- Local JSON serializer ---
    def _json_default(obj):
        """Safe JSON serializer for NumPy / pandas types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return str(obj)

    # --- Load audit report ---
    if audit_df is None:
        if not audit_report_path.exists():
            raise FileNotFoundError(f"Audit report not found at: {audit_report_path}")

        if audit_report_path.suffix.lower() == ".json":
            with open(audit_report_path, "r") as f:
                records = json.load(f)
            audit_df = pd.DataFrame(records)
        elif audit_report_path.suffix.lower() in (".xls", ".xlsx"):
            audit_df = pd.read_excel(audit_report_path)
        elif audit_report_path.suffix.lower() == ".csv":
            audit_df = pd.read_csv(audit_report_path)
        else:
            raise ValueError(f"Unsupported audit file type: {audit_report_path.suffix}")

    # --- Ensure expected columns exist ---
    if "column_name" not in audit_df.columns:
        raise KeyError("Audit report must contain 'column_name' column.")

    # --- Prepare logging file ---
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat(sep=" ", timespec="seconds") + " UTC"
    header = f"# Step 7 — Type Fix Log\n\nGenerated: {ts}\n\n"
    with open(audit_log_path, "w", encoding="utf-8") as f:
        f.write(header)

    conversions_applied = []
    updated_rows = []

    # --- Iterate over audit entries ---
    for idx, row in audit_df.iterrows():
        col = row.get("column_name")
        if pd.isna(col):
            continue
        col = str(col)

        if col not in df.columns:
            logging.warning(f"Column '{col}' listed in audit but not present in DataFrame. Skipping.")
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n- Status: not in dataframe, skipped\n\n")
            continue

        dtype_expected = row.get("dtype_expected") or row.get("expected") or None
        if isinstance(dtype_expected, str):
            dtype_expected = dtype_expected.strip().lower()

        if not dtype_expected:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                dtype_expected = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(s):
                dtype_expected = "datetime"
            else:
                dtype_expected = None

        if dtype_expected not in ("numeric", "datetime"):
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n- Expected: {dtype_expected}\n- Action: none\n\n")
            continue

        orig_col = f"{col}_original"
        if orig_col not in df.columns:
            df[orig_col] = df[col].copy()

        missing_before = int(df[col].isna().sum())

        try:
            if dtype_expected == "numeric":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    conv_action = "pd.to_numeric(errors='coerce')"
                else:
                    conv_action = "already_numeric (skipped)"
            elif dtype_expected == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    conv_action = "pd.to_datetime(errors='coerce')"
                else:
                    conv_action = "already_datetime (skipped)"
            else:
                conv_action = "no_action"

            missing_after = int(df[col].isna().sum())

            if pd.api.types.is_numeric_dtype(df[col]):
                new_detected = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                new_detected = "datetime"
            else:
                new_detected = str(df[col].dtype)

            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n")
                f.write(f"- Audit expected: {dtype_expected}\n")
                f.write(f"- Conversion applied: {conv_action}\n")
                f.write(f"- Missing before: {missing_before}\n")
                f.write(f"- Missing after: {missing_after}\n")
                f.write(f"- New detected dtype: {new_detected}\n\n")

            conversions_applied.append({
                "column": col,
                "expected": dtype_expected,
                "action": conv_action,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "new_detected": new_detected
            })

            updated_rows.append((idx, {
                "num_missing": int(missing_after),
                "dtype_detected": new_detected
            }))

        except Exception as e:
            logging.exception(f"Failed converting column '{col}': {e}")
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"### Column: {col}\n- ERROR during conversion: {e}\n\n")
            continue

    # --- Save updated audit if needed ---
    if save_updated_audit and updated_rows:
        for (idx, upd) in updated_rows:
            for k, v in upd.items():
                audit_df.at[idx, k] = v

        out_dir = audit_log_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{updated_audit_basename}.csv"
        json_path = out_dir / f"{updated_audit_basename}.json"
        excel_path = out_dir / f"{updated_audit_basename}.xlsx"

        try:
            audit_df.to_csv(csv_path, index=False)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(audit_df.to_dict(orient="records"), f, indent=4, default=_json_default)
            audit_df.to_excel(excel_path, index=False)
            logging.info(f"Saved updated audit to {csv_path}, {json_path}, and {excel_path}")
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"\nSaved updated audit: {csv_path.name}, {json_path.name}, {excel_path.name}\n")
        except Exception as e:
            logging.exception(f"Failed to save updated audit files: {e}")
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(f"\nFailed to save updated audit: {e}\n")

    logging.info(f"Step 7 complete. Conversions applied to {len(conversions_applied)} column(s).")
    return df



print(dtype_audit(df))



