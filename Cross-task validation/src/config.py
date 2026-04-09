"""
Centralized configuration for Cross-Task Variability Analysis.
All paths are relative to project root - no hardcoded absolute paths.
"""
import os
from datetime import datetime

# Dynamically find project root (Cross-task validation/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# DARWIN dataset - user must place data.csv here
DARWIN_DATASET = os.path.join(RAW_DATA_DIR, 'DARWIN_DATASET', 'data.csv')

# Run versioning - creates new directory per pipeline execution
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_ID = f"run_{RUN_TIMESTAMP}"

# Results directories (versioned)
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, RUN_ID)
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
CLEANED_DATA_DIR = os.path.join(RESULTS_DIR, 'cleaned')

# Output files (all within versioned run directory)
ENGINEERED_FEATURES_CSV = os.path.join(RESULTS_DIR, 'engineered_features.csv')
CLEANED_FEATURES_CSV = os.path.join(CLEANED_DATA_DIR, 'cleaned_features.csv')
SELECTED_FEATURES_CSV = os.path.join(RESULTS_DIR, 'selected_features.csv')
EXPERIMENT_RESULTS_CSV = os.path.join(METRICS_DIR, 'baseline_vs_variability_results.csv')

# Symlink to latest run (for convenience)
LATEST_RUN_LINK = os.path.join(RESULTS_BASE_DIR, 'latest')

# Labels
LABEL_COL = 'class'
LABEL_MAP = {'H': 0, 'P': 1}  # Healthy: 0, Patient (AD): 1

# Numerical stability
EPS = 1e-9

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5

# Feature selection
TOP_N_FEATURES = 25
CORRELATION_THRESHOLD = 0.9


def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        DATA_DIR, RAW_DATA_DIR,
        RESULTS_BASE_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR, CLEANED_DATA_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create/update 'latest' symlink pointing to current run
    try:
        if os.path.islink(LATEST_RUN_LINK):
            os.unlink(LATEST_RUN_LINK)
        elif os.path.exists(LATEST_RUN_LINK):
            os.remove(LATEST_RUN_LINK)
        os.symlink(RESULTS_DIR, LATEST_RUN_LINK)
    except OSError:
        # Symlinks may fail on Windows - skip silently
        pass
    
    print(f"Run ID: {RUN_ID}")
    print(f"Results will be saved to: {RESULTS_DIR}")


def check_dataset_exists():
    """Check if DARWIN dataset is present and provide guidance if not."""
    if not os.path.exists(DARWIN_DATASET):
        print(f"ERROR: DARWIN dataset not found at {DARWIN_DATASET}")
        print(f"Please place data.csv in: {os.path.dirname(DARWIN_DATASET)}")
        return False
    return True


def get_previous_runs():
    """List all previous pipeline runs."""
    if not os.path.exists(RESULTS_BASE_DIR):
        return []
    runs = [d for d in os.listdir(RESULTS_BASE_DIR) 
            if d.startswith('run_') and os.path.isdir(os.path.join(RESULTS_BASE_DIR, d))]
    return sorted(runs)