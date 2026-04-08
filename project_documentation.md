# Project Documentation: Cross-Task Variability Analysis

This document tracks all changes made to the project, user questions, and detailed technical explanations provided during the development process.

---

## 📅 Log: 2026-03-25

### 📝 Change 1: Path Handling Refactoring
**File**: [`feature_engineering.py`](file:///d:/Project%20backup/project-root/Cross-task%20validation/src/preprocessing/feature_engineering.py)

#### Description
Replaced hardcoded absolute paths (e.g., `D:/project-root/...`) with dynamic relative paths to ensure the script works on any machine.

#### Technical Details
- **Dynamic Root Detection**: Used `os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))` to find the project root relative to the script's location (`src/preprocessing/`).
- **Constant Naming**: Renamed path variables to `UPPER_CASE` (e.g., `RESULTS_DIR`) to comply with PEP 8 standards.
- **Portability**: The script now calculates its own dependencies, meaning it can be run from any folder or IDE without manual configuration.

---

### 📝 Change 2: Efficiency Improvement
**File**: [`feature_engineering.py`](file:///d:/Project%20backup/project-root/Cross-task%20validation/src/preprocessing/feature_engineering.py)

#### Description
Optimized how the engineered features are combined to significantly improve processing speed.

#### Technical Details
- **List Collection**: Replaced the iterative `pd.concat([engineered, stats])` inside the loop with a list-based approach (`engineered_list.append(stats)`).
- **Single Concatenation**: Used a single `pd.concat(engineered_list, axis=1)` call outside the loop. This is much faster because it avoids re-copying the entire DataFrame in every iteration.
- **Robustness**: Added an `if engineered_list:` check to handle cases where no base features are found gracefully.

---

### 📝 Change 3: Style Improvements & PEP 8 Compliance
**File**: [`feature_engineering.py`](file:///d:/Project%20backup/project-root/Cross-task%20validation/src/preprocessing/feature_engineering.py)

#### Description
Improved code readability and followed official Python style guidelines (PEP 8).

#### Technical Details
- **Naming Consistency**: Renamed `Label_Col` to `LABEL_COL` to reflect its status as a constant.
- **Formatting Fixes**: Removed extra spaces in function arguments (e.g., `axis= 1` to `axis=1`).

---

## 📅 Log: 2026-04-06

### 📝 Bug Analysis: Feature Engineering Script
**File**: [`feature_engineering.py`](file:///d:/Project%20backup/project-root/Cross-task%20validation/src/preprocessing/feature_engineering.py)

#### Description
Conducted a thorough bug check of the feature engineering pipeline and identified a critical data loss issue along with several code quality improvements.

#### Identified Issues
- **CRITICAL: Subject ID Loss**: 
  - **Problem**: The script groups features using a numbering pattern (e.g., `air_time1`). Columns that don't match this (like the **`ID` column**) are silently dropped.
  - **Impact**: The results cannot be linked back to specific subjects, rendering the output file nearly useless for downstream analysis.
- **Redundant Constants**: 
  - **Problem**: Duplicate definitions for `LABEL_COL` and `Label_Col`.
  - **Impact**: Inconsistent naming and potential confusion for future maintainers.
- **Performance Inefficiency**: 
  - **Problem**: `pd.to_numeric` is called iteratively inside a loop.
  - **Impact**: Unnecessary overhead that slows down processing for larger datasets.
- **Calculation Safety**: 
  - **Problem**: Potential division by zero or distortion in Coefficient of Variation (CV) calculation if the mean is negative and matches `-EPS`.
