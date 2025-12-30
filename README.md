# Does Your LLM Trust You?

Accompanying code for [this research project.](https://docs.google.com/document/d/1WL7svZFSTwD44l2e8L0EF8kXBfPXAqdd2a4JV7hFifs/edit?usp=sharing) 

## Project Structure

```
Summer app/
├── src/                       # Core source code
│   ├── dataset/               # Dataset generation & handling
│   ├── steering/              # Steering vectors & model steering
│   ├── probes/                # Linear probe implementation
│   ├── evaluation/            # LLM judging & visualization
│   └── prompts/               # Prompt templates
│
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── collect_activations.ipynb
│   ├── train_probe.ipynb
│   ├── probe_analysis.ipynb
│   └── plotting*.ipynb
│
├── data/                      # Input data
│   ├── datasets/              # Parquet and CSV files
│   └── refusal_dataset/       # Refusal training/test data (JSON)
│
├── outputs/                   # Generated outputs
│   ├── judge_outputs/         # LLM judgment results
│   ├── steered-outs/          # Steered model outputs
│   ├── plots/                 # Visualizations
│   ├── probe_checkpoints/     # Saved probe models
│   └── refusal_svs/           # Steering vectors
│
└── crosscoder_learning/       # Crosscoder experiments
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

# Use random control vectors (same magnitude as base vectors)
python src/steering/steer_model.py --vec-type <type> --layer-id <layer> --strength <value> --random-vector
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
## Notes

- All outputs are saved to the `outputs/` directory
- Data files should be placed in `data/` subdirectories
- Notebooks are in `notebooks/` for interactive analysis
