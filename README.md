# MATS Summer App - AI Safety Research

Research project for AI alignment and safety, focusing on trust mechanisms, refusal behaviors, and steering vectors in language models.

## Project Structure

```
Summer app/
├── src/                          # Core source code
│   ├── dataset/                  # Dataset generation & handling
│   │   ├── create_dataset.py     # Convert JSON conversations to Parquet
│   │   ├── generate_conv_data.py # Generate synthetic conversations
│   │   └── generate_user_descriptions.py  # Generate user description prompts
│   ├── steering/                 # Steering vectors & model steering
│   │   ├── steer_model.py        # Apply steering vectors to models
│   │   └── steer_utils.py        # Steering utilities and helpers
│   ├── probes/                   # Probe training & analysis
│   │   └── probes.py             # Linear probe implementation
│   ├── evaluation/               # Judging & evaluation
│   │   ├── judge_outputs.py      # Evaluate model outputs with LLM judges
│   │   └── visualize_judgments.py # Visualization tools for judgments
│   └── prompts/                  # Prompt templates
│       ├── judge_prompts.py      # Judge prompt templates
│       ├── prompts_no_refusal.py # Prompts without refusal scenarios
│       └── prompts_refusal.py    # Prompts with refusal scenarios
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── collect_activations.ipynb # Collect model activations
│   ├── train_probe.ipynb         # Train linear probes
│   ├── probe_analysis.ipynb      # Analyze probe performance
│   ├── plotting.ipynb            # General plotting utilities
│   ├── plotting_judge.ipynb      # Plot judgment results
│   └── utils.ipynb               # Utility functions
├── data/                         # Data files
│   ├── datasets/                 # Parquet and CSV files
│   └── refusal_dataset/          # Refusal training/test data (JSON)
├── outputs/                      # Generated outputs
│   ├── judge_outputs/            # LLM judgment results
│   ├── steered-outs/             # Steered model outputs
│   ├── plots/                    # Generated plots and visualizations
│   ├── probe_checkpoints/        # Saved probe models
│   ├── refusal_svs/              # Refusal steering vectors
│   └── runs/                     # Training run logs
├── crosscoder_learning/          # Crosscoder experiments (separate module)
├── eval_data/                    # Evaluation datasets
├── .env                          # Environment variables (API keys, etc.)
└── .gitignore                    # Git ignore rules
```

## Quick Start

The project includes a helper script (`run.py`) to easily run modules from anywhere:

```bash
# Run modules without cd'ing into src/
python run.py dataset.generate_conv_data --scenario information_sharing --trust-level high_trust
python run.py evaluation.judge_outputs --model llama --vec-type control --dataset harmful
```

Alternatively, run modules directly from the project root:

```bash
python -m src.dataset.generate_conv_data --help
python -m src.evaluation.judge_outputs --help
```

## Usage

### Dataset Generation

```bash
# Generate synthetic conversations
python src/dataset/generate_conv_data.py --scenario-type <type> --trust-level <level>

# Generate user descriptions
python src/dataset/generate_user_descriptions.py

# Create Parquet dataset from JSON
python src/dataset/create_dataset.py --prompt-category <category>
```

### Model Steering

**Using the bash script (recommended for batch runs):**
```bash
# Edit run_steering.sh to configure your parameters
./run_steering.sh
```

**Direct command:**
```bash
# Apply steering vectors to model
python src/steering/steer_model.py --vec-type <type> --layer-id <layer> --strength <value>
```

### Evaluation

```bash
# Judge model outputs
python src/evaluation/judge_outputs.py --vec-type <type> --layer-id <layer>

# Visualize judgments
python src/evaluation/visualize_judgments.py
```

### Probes

```bash
# Train probes (use notebooks/train_probe.ipynb)
# Analyze probe performance (use notebooks/probe_analysis.ipynb)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- See individual module dependencies

## Environment Setup

1. Copy `.env.example` to `.env` (if available)
2. Add your API keys and configuration
3. Install dependencies: `pip install -r requirements.txt` (if available)

## Notes

- All outputs are saved to the `outputs/` directory
- Data files should be placed in `data/` subdirectories
- Notebooks are in `notebooks/` for interactive analysis
