import pandas as pd

def schema_audit(df, expected_schema=None, missing_threshold=0.3, unique_threshold=0.9):
    """
    Audits the schema of a DataFrame and suggests actions for each column.

    Parameters:
    - df: pandas DataFrame
    - expected_schema: optional dict {column_name: expected_dtype}
    - missing_threshold: float, proportion of missing values above which flag column
    - unique_threshold: float, proportion of unique values above which flag column

    Returns:
    - audit_df: pandas DataFrame with schema audit results
    """
    results = []

    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        pct_missing = n_missing / n_total if n_total > 0 else 0
        n_unique = series.nunique(dropna=True)
        pct_unique = n_unique / n_total if n_total > 0 else 0

        # Detect dtype
        detected_dtype = pd.api.types.infer_dtype(series, skipna=True)
        expected_dtype = expected_schema[col] if expected_schema and col in expected_schema else detected_dtype

        # Decide action
        if n_missing == n_total:  
            action = "drop_column"   # Entirely missing
        elif pct_missing > missing_threshold:  
            action = "review_manually"  # Too many missing
        elif pct_unique >= unique_threshold and n_unique > 20:  
            action = "review_manually"  # Likely ID column or too many uniques
        elif detected_dtype != expected_dtype:  
            action = "review_manually"  # Mismatched type
        else:
            action = "keep"  # Safe column

        results.append({
            "column_name": col,
            "dtype_detected": detected_dtype,
            "dtype_expected": expected_dtype,
            "num_missing": n_missing,
            "pct_missing": round(pct_missing, 3),
            "n_unique": n_unique,
            "pct_unique": round(pct_unique, 3),
            "sample_values": series.dropna().unique()[:5].tolist(),
            "suggested_action": action
        })

    audit_df = pd.DataFrame(results)
    return audit_df



def check_primary_key(
    df: pd.DataFrame,
    candidates: List[str] = None,
    combo_candidates: List[List[str]] = None,
    save_path: Union[str, Path] = "docs/duplicates_samples.csv"
) -> Dict[str, str]:
    """
    Check which column(s) can act as primary key(s).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    candidates : list of str, optional
        Candidate columns to test as primary keys.
    combo_candidates : list of list of str, optional
        Candidate column combinations to test.
    save_path : str or Path, optional
        Path to save duplicate samples for combos.

    Returns
    -------
    dict
        Mapping of valid primary keys (single or combo).
    """
    primary_keys = {}
    n_rows = len(df)

    # Check single-column candidates
    if candidates:
        for col in candidates:
            if col not in df.columns:
                logging.warning(f"Candidate column '{col}' not found in DataFrame.")
                continue

            try:
                n_unique = df[col].nunique()
                n_null = df[col].isnull().sum()
                logging.info(f"Column: {col} | Unique: {n_unique} | Nulls: {n_null}")

                if n_unique == n_rows and n_null == 0:
                    primary_keys[col] = "single"
            except Exception as e:
                logging.error(f"Error checking column '{col}': {e}", exc_info=True)

    # Check multi-column combos
    if combo_candidates:
        for combo in combo_candidates:
            missing = [c for c in combo if c not in df.columns]
            if missing:
                logging.warning(f"Combo {combo} skipped. Missing columns: {missing}")
                continue

            try:
                combo_df = df[combo]
                duplicate_check = combo_df[combo_df.duplicated()]

                if duplicate_check.empty:
                    logging.info(f"Combination {combo} is unique.")
                    primary_keys[tuple(combo)] = "combo"
                else:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    duplicate_check.to_csv(save_path, index=False)
                    logging.warning(
                        f"Combination {combo} has duplicates. Sample saved to {save_path}"
                    )
            except Exception as e:
                logging.error(f"Error checking combo {combo}: {e}", exc_info=True)

    if not primary_keys:
        logging.info("No valid primary keys found.")

    return primary_keys

def duplicates_audit(df:pd.DataFrame,save_path:str='docs/duplicate_rows.csv'):
    """
    It counts and perform the audit of duplicate rows and saves them in the project-root.
    """
    # Count the number of duplicate rows:
    num_duplicates = df.duplicated().sum()
    print('The number of duplcated rows are {num_duplicates}')

    # If there are no duplicates:
    if num_duplicates == 0:
        print('There no duplicate rows')

    # Collection of duplicate rows:
    duplicated_rows = df[df.duplicated()]
    # Ensure the directory for the save path exists (creates folders if missing)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save only the first 20 duplicate rows to a CSV file for reference
    duplicated_rows.head(20).to_csv(save_path, index=False)

    print(f" Saved first 20 duplicates to {save_path}")
    duplicates_fraction=num_duplicates/len(df)
    # Duplicates audit action stem:
    if duplicates_fraction <= 0.05:
        print('Dropping duplicate rows')
        df_clean=num_duplicates.drop_duplicates() # Drops duplicated rows
        print('Duplicates dropped')
    else:
        print('Duplicates excess error. Manual inspection required.')
    return df
