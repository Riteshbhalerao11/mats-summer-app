# Codebase Restructuring Summary

## Changes Made

### Directory Structure

Successfully reorganized the codebase from a flat structure to an organized hierarchy:

**New Structure:**
```
Summer app/
├── src/                          # All Python source code
│   ├── dataset/                  # Dataset generation modules
│   ├── steering/                 # Model steering modules
│   ├── probes/                   # Probe training modules
│   ├── evaluation/               # Judging and visualization
│   └── prompts/                  # Prompt templates
├── notebooks/                    # Jupyter notebooks
├── data/                         # Input data files
│   ├── datasets/                 # Parquet and CSV files
│   └── refusal_dataset/          # JSON refusal data
├── outputs/                      # All generated outputs
│   ├── judge_outputs/
│   ├── steered-outs/
│   ├── plots/
│   ├── probe_checkpoints/
│   ├── refusal_svs/
│   └── runs/
├── crosscoder_learning/          # (unchanged)
├── eval_data/                    # (unchanged)
├── run.py                        # Helper script for running modules
└── README.md                     # Project documentation
```

### Files Moved

**To `src/dataset/`:**
- `create_dataset.py`
- `generate_conv_data.py`
- `generate_user_descriptions.py`

**To `src/steering/`:**
- `steer_model.py`
- `steer_utils.py`
- `train_test_utils.py`

**To `src/probes/`:**
- `probes.py`

**To `src/evaluation/`:**
- `judge_outputs.py`
- `visualize_judgments.py`

**To `src/prompts/`:**
- `judge_prompts.py`
- `prompts_no_refusal.py`
- `prompts_refusal.py`

**To `notebooks/`:**
- All `.ipynb` files (6 files)

**To `data/datasets/`:**
- `*.parquet` files
- `*.csv` files

**To `data/`:**
- `refusal_dataset/` directory

**To `outputs/`:**
- `judge_outputs/`
- `steered-outs/`
- `plots/`
- `probe_checkpoints/`
- `refusal_svs/`
- `runs/`

### Code Updates

**Import Paths:**
- Updated `generate_conv_data.py` to use relative imports for prompts modules

**Directory References:**
- Updated `judge_outputs.py` to use `outputs/steered-outs/` and `outputs/judge_outputs/`
- Updated `visualize_judgments.py` to use `outputs/judge_outputs/`
- Updated `steer_model.py` default paths to `outputs/probe_checkpoints/` and `outputs/refusal_svs/`
- Updated `steer_utils.py` default paths to `outputs/probe_checkpoints/` and `outputs/refusal_svs/`

### New Files Created

**Module Initialization:**
- `src/__init__.py`
- `src/dataset/__init__.py`
- `src/steering/__init__.py`
- `src/probes/__init__.py`
- `src/evaluation/__init__.py`
- `src/prompts/__init__.py`

**Documentation:**
- `README.md` - Comprehensive project documentation with structure, usage, and examples
- `RESTRUCTURE_SUMMARY.md` - This file

**Helper Script:**
- `run.py` - Simplifies running modules from project root

**Updated Configuration:**
- `.gitignore` - Enhanced with Python, IDE, and output patterns

## How to Use the New Structure

### Running Modules

**Option 1: Using the helper script (recommended)**
```bash
python run.py dataset.generate_conv_data --help
python run.py evaluation.judge_outputs --model llama --vec-type control
```

**Option 2: Using Python module syntax**
```bash
python -m src.dataset.generate_conv_data --help
python -m src.evaluation.judge_outputs --model llama --vec-type control
```

**Option 3: Direct execution (from project root)**
```bash
python src/dataset/generate_conv_data.py --help
python src/evaluation/judge_outputs.py --model llama --vec-type control
```

### Importing Modules

When working in notebooks or other scripts:
```python
# Import from src modules
from src.dataset import generate_conv_data
from src.steering import steer_utils
from src.evaluation import judge_prompts
from src.prompts import prompts_refusal
```

## Backward Compatibility

All command-line arguments remain the same. The only changes are:

1. **File locations** - Execute scripts from project root or use the helper
2. **Output paths** - Now under `outputs/` directory (configurable via CLI args)
3. **Default probe/refusal paths** - Point to `outputs/` subdirectories

## Benefits

✅ **Cleaner root directory** - Only essential files at top level  
✅ **Logical organization** - Related code grouped by functionality  
✅ **Better discoverability** - Clear structure for new developers  
✅ **Separated concerns** - Code, notebooks, data, and outputs in distinct locations  
✅ **Maintained flexibility** - All paths remain configurable via CLI arguments  
✅ **No breaking changes** - All functionality preserved, same CLI interface
