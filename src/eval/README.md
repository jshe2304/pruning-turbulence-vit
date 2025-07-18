# Model Evaluation

This directory contains tools for evaluating trained Vision Transformer models.

## Usage

### Running Analysis

To run model evaluation, use the main script with a TOML configuration file:

```bash
python analyze_model.py config/eval.toml
```

**Note:** The script always performs RMSE, ACC, and SPECTRA analyses - no need to enable/disable them in the config.

### Configuration

The evaluation script uses TOML configuration files that follow the repository's standard format. See `config/eval.toml` for an example.

#### Configuration Sections:

- **`[model]`**: Model architecture parameters and checkpoint path
- **`[train_dataset]`**: Training dataset configuration (for compatibility)
- **`[test_dataset]`**: Test dataset configuration
- **`[evaluation]`**: Analysis parameters
- **`[evaluation.climatology]`**: Climatology baseline configuration (optional)

#### Key Parameters:

- `checkpoint_path`: Path to the trained model checkpoint
- `data_dir`: Directory containing the dataset
- `frame_ranges`: List of frame ranges to use for evaluation
- `target_offset`: Temporal offset for target frames
- `spectra_leadtimes`: Lead times for spectral analysis
- `batch_size`: Batch size for evaluation
- `analysis_length`: Length of analysis sequences
- `num_ensembles`: Number of ensemble runs

### API

The evaluation script can also be used programmatically:

```python
from analyze_model import run_analysis_from_config

# Run analysis from config file
results = run_analysis_from_config("config/eval.toml")

# Or load config manually and run analysis
import toml
from analyze_model import run_short_analysis

config = toml.load("config/eval.toml")
results = run_short_analysis(config)
```

### Output

The script always returns a dictionary containing RMSE, accuracy, and spectral analysis results, along with any other requested analysis types. 