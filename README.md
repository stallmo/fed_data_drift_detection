# Federated Data Drift Detection

A framework for detecting data drift in federated learning environments using fuzzy clustering and the Davies-Bouldin index.

## Overview

This repository implements a federated data drift detection system that monitors changes in data distributions across multiple clients in a federated learning setup. The approach uses fuzzy c-means clustering in combination with the Davies-Bouldin index to detect when data distributions have drifted beyond acceptable thresholds.

## Features

- **Federated Clustering**: Implements federated fuzzy c-means clustering for distributed data
- **Data Drift Detection**: Detects both local and global data drift using Davies-Bouldin index monitoring
- **Synthetic Data Generation**: Flexible Gaussian data generators for simulating various drift scenarios
- **Multiple Drift Scenarios**: Supports different types of drift patterns:
  - `global_drift`: Global drift affecting all clients
  - `no_local_drift`: No drift scenario (for false positive testing)
  - `local_but_no_global_drift`: Local changes that don't affect global distribution
  - `global_drift_unused_generators`: Drift caused by disappearing data generators
- **Experiment Tracking**: Integration with MLflow for experiment tracking and result logging

## Repository Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_data_drift_experiments.py       # Main experiment runner
├── data_generators.py                  # Data generation classes
├── helpers/
│   └── generator_helpers.py            # Helper functions for data generation
├── 2D_experiment_inspector.ipynb       # Jupyter notebook for 2D experiment visualization
├── result_summary_analysis.ipynb       # Jupyter notebook for analyzing experiment results
└── result_summary/                     # Directory for experiment results
```

## How It Works

1. **Initial Setup**: Creates a federated learning environment with multiple clients, each having their own Gaussian data distributions
2. **Global Model**: Learns a global federated fuzzy c-means clustering model from initial client data
3. **Baseline Calculation**: Calculates the initial Davies-Bouldin index as a baseline for cluster quality
4. **Drift Monitoring**: At each time step:
   - Generates new data points (potentially from drifted distributions)
   - Recalculates the Davies-Bouldin index
   - Tests if the index falls outside acceptable threshold bounds
   - Flags drift when threshold is exceeded
5. **Results**: Records drift detection events and experiment parameters

## Requirements

- Python 3.x
- NumPy 1.19.5
- scikit-learn 0.24.2
- plotly 5.18.0
- pandas 1.1.5
- mlflow ~1.23.1
- Additional dependency: `cluster_library` (federated clustering implementation)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: This repository requires the `cluster_library` module which contains the federated clustering and cluster validation implementations. Make sure the `cluster_library` is available in the parent directory.

## Usage

### Running Experiments

The main experiment runner can be executed directly:

```bash
python run_data_drift_experiments.py
```

This will run a series of experiments with different parameter configurations and log results to MLflow.

### Custom Experiments

You can also use the `run_data_drift_experiments` function programmatically:

```python
from run_data_drift_experiments import run_data_drift_experiments

run_data_drift_experiments(
    run_mode='global_drift',
    acceptability_threshold=0.025,
    dim_data=2,
    n_repeats=20,
    n_time_steps=10,
    n_clients=10,
    save_results=True,
    log_mlflow=True
)
```

### Key Parameters

- `run_mode`: Type of drift scenario to simulate
- `acceptability_threshold`: Tolerance for Davies-Bouldin index deviation (default: 0.025)
- `dim_data`: Dimensionality of data points (default: 2)
- `n_clients`: Number of federated clients (default: 10)
- `n_repeats`: Number of experimental repeats (default: 20)
- `n_time_steps`: Number of time steps per experiment (default: 10)
- `ratio_drifted_clients`: Fraction of clients experiencing drift (default: 0.5)
- `ratio_new_data_distribution`: Proportion of data from new distributions (default: 0.1)

## Analysis

The repository includes Jupyter notebooks for analyzing experiment results:

- **2D_experiment_inspector.ipynb**: Visualize 2D experiments and drift patterns
- **result_summary_analysis.ipynb**: Comprehensive analysis of experiment results

## MLflow Integration

If MLflow logging is enabled, experiments are tracked at `http://localhost:8080`. Make sure an MLflow server is running before starting experiments with `log_mlflow=True`.

## Output

Experiment results are saved as pickle files in the `./results/` directory with the format:
```
{run_mode}_{timestamp}_results.pkl
```

Each result file contains:
- Experiment configuration parameters
- Initial cluster centers and Davies-Bouldin scores
- Per-timestep drift detection results
- Drift detection counters

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Citation

[Add citation information if applicable]
