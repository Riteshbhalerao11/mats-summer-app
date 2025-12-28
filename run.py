#!/usr/bin/env python3
"""
Helper script to run modules from src/ directory.
Allows running modules as: python run.py dataset.generate_conv_data --args

Usage:
    python run.py dataset.create_dataset --help
    python run.py steering.steer_model --model llama --vec-type control ...
    python run.py evaluation.judge_outputs --model llama --vec-type control ...
"""
import sys
import importlib.util
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python run.py <module_path> [args...]")
    print("\nAvailable modules:")
    print("  dataset.create_dataset")
    print("  dataset.generate_conv_data")
    print("  dataset.generate_user_descriptions")
    print("  steering.steer_model")
    print("  probes.probes")
    print("  evaluation.judge_outputs")
    print("  evaluation.visualize_judgments")
    sys.exit(1)

module_path = sys.argv[1]
module_args = sys.argv[2:]

# Convert module path to file path
parts = module_path.split('.')
file_path = Path("src") / Path(*parts[:-1]) / f"{parts[-1]}.py"

if not file_path.exists():
    print(f"Error: Module file not found: {file_path}")
    sys.exit(1)

# Load and run the module
spec = importlib.util.spec_from_file_location("__main__", file_path)
module = importlib.util.module_from_spec(spec)

# Replace sys.argv with module args
sys.argv = [str(file_path)] + module_args

# Execute the module
spec.loader.exec_module(module)
