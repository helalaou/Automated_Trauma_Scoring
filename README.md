# Glasgow Coma Scale VLM Evaluation

This code evaluates various Vision Language Models (VLMs) on their ability to assess Glasgow Coma Scale motor responses from clinical images.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key. You can either:

   **Option A: Use a .env file (recommended)**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```
   
   **Option B: Set as environment variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Configuration

All settings are configured in `config.yaml`:

- **Models**: Configure which VLMs to evaluate and their parameters
- **Prompts**: Jinja2 templates for both zero-shot and few-shot evaluation
- **Dataset**: Paths and category mappings for the trauma dataset
- **Evaluation**: Output directory and result saving options

### Key Configuration Sections

- `models`: List of models to evaluate with their parameters (temperature, max_tokens, etc.)
- `prompting.zero_shot_template`: Path to zero-shot Jinja2 template
- `prompting.few_shot_template`: Path to few-shot Jinja2 template
- `few_shot.examples_per_category`: Number of example images to include per category
- `dataset.categories`: Mapping of dataset folders to GCS motor scores

## Usage

### Quick start

```bash
# 1) Install deps into a local virtualenv (.venv)
./install

# 2) Run evaluation (defaults to --mode both and config.yaml)
./run

# Examples
# 6-bin standard (model predicts 1..6 directly)
./run --mode zero_shot --task 6bin
./run --mode few_shot --task 6bin

# 3-bin simplified (model predicts 1..3 directly using dedicated 3-bin prompts)
./run --mode zero_shot --task 3bin
./run --mode few_shot --task 3bin

# Run both tasks (6-bin and 3-bin) for both modes
./run --mode both --task both
```

Run the evaluation script directly (alternative):

```bash
# Evaluate both zero-shot and few-shot (6-bin by default)
python evaluate_gcs.py

# Evaluate only zero-shot 3-bin
python evaluate_gcs.py --mode zero_shot --task 3bin

# Evaluate only few-shot
python evaluate_gcs.py --mode few_shot

# Use custom config file
python evaluate_gcs.py --config custom_config.yaml --mode both --task both
```

## Results

Results are saved in the `results/` directory:

- `results_zero_shot_TIMESTAMP.json`: Individual zero-shot evaluation results
- `summary_zero_shot_TIMESTAMP.json`: Zero-shot summary metrics
- `results_few_shot_TIMESTAMP.json`: Individual few-shot evaluation results
- `summary_few_shot_TIMESTAMP.json`: Few-shot summary metrics
- `plots/`: Directory containing visualization plots

Each result includes:
- Predicted GCS motor score
- Confidence level
- Reasoning
- Accuracy metrics per model and category

### Visualization Plots

The evaluation automatically generates several visualization plots (if `generate_plots: true` in config):

1. **Overall Accuracy** - Bar charts comparing accuracy and confidence across models
2. **Per-Category Accuracy** - Heatmap and grouped bar charts showing accuracy for each GCS category
3. **Confusion Matrices** - One per model showing predicted vs expected scores
4. **Confidence Distributions** - Histograms of confidence scores for each model
5. **Zero-Shot vs Few-Shot Comparison** - Side-by-side comparison when both modes are evaluated

All plots are saved as high-resolution PNG files (300 DPI) in `results/plots/`.

### Plotting Existing Results

You can also plot results from previous evaluations using the standalone plotting script:

```bash
# Plot a single summary file
python plot_results.py --summary results/summary_zero_shot_TIMESTAMP.json

# Compare zero-shot vs few-shot
python plot_results.py --zero-summary results/summary_zero_shot_TIMESTAMP.json --few-summary results/summary_few_shot_TIMESTAMP.json
```

## 6-bin vs 3-bin: What’s the difference?

- 6-bin task (`--task 6bin`):
  - The model is prompted with the full 6-level Glasgow motor response scale.
  - The model returns `gcs_motor_score` in 1..6.
  - We also compute a derived 3-bin score from that 6-bin prediction for convenience:
    - 1–3 → Bin 1 (Unresponsive/Abnormal)
    - 4–5 → Bin 2 (Withdraws/Localizes)
    - 6 → Bin 3 (Obeys Commands/Normal)

- 3-bin task (`--task 3bin`):
  - The model is prompted with a dedicated 3-bin prompt and returns `gcs_motor_bin` in 1..3 directly.
  - This is NOT converted from 6-bin; it’s the model’s native 3-bin decision.

Console output rules:
- In 6-bin runs, you will see only the 6-bin result.
- In 3-bin runs, you will see only the 3-bin result.
- This avoids confusion (e.g., seeing 6-bin values while running a 3-bin prompt).

Summary files include both systems’ metrics so you can compare:
- 6-bin: accuracy per model and per category
- 3-bin: accuracy per model and per 3-bin group

How summaries are labeled (to avoid confusion):
- When `--task 6bin` is used:
  - 6-bin Accuracy: reported normally.
  - 3-bin Accuracy (derived from 6-bin): shown as a mapping from the 6-bin predictions (1–3→1, 4–5→2, 6→3).
- When `--task 3bin` is used:
  - 6-bin Accuracy: N/A (not evaluated for 3-bin prompt).
  - 3-bin Accuracy (direct): reported from the model’s `gcs_motor_bin` output.

The saved summary JSON also records the run `task` so you can tell whether 3-bin metrics are direct or derived when analyzing results offline.

## Dataset Structure

The trauma dataset should be organized as:
```
trauma_dataset/
  ├── 1_no_response/
  ├── 2_extension/
  ├── 3_abnormal_flexion/
  ├── 4_normal_flexion/
  ├── 5_localizing/
  └── 6_obeys_commands/
```

Each folder contains images (PNG, JPG) representing that GCS motor score category.

