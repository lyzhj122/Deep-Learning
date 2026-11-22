# Server Anomaly Detection with LSTM and GRU Autoencoders

This project compares LSTM and GRU autoencoders for unsupervised anomaly detection on the Server Machine Dataset (SMD). It uses 38 anonymized server KPI features to learn normal reconstruction patterns, then flags test observations whose reconstruction error exceeds a threshold calibrated from the training series.

The accompanying Jupyter notebook is:

- `SMD_Anomaly_Detection_With_LSTM and GRU Autoencoder.ipynb`

This project performs anomaly detection on already observed measurements. It does not forecast future incidents or KPI values.

## Dataset

The project includes the SMD files under `ServerMachineDataset/`:

```text
ServerMachineDataset/
  train/                  Normal or predominantly normal training KPI series
  test/                   Test KPI series
  test_label/             Point-level binary test labels
  interpretation_label/   Labeled anomalous dimensions for some test periods
  LICENSE
```

There are 28 machines in each of the `train`, `test`, `test_label`, and `interpretation_label` folders. Files are CSV-style text files without headers. The default notebook machine, `machine-1-1`, contains 28,479 observations in both its training and test series, with 38 KPI columns per observation.

For `machine-1-1`:

- Test observations: 28,479
- Labeled anomaly observations: 2,694 (9.46%)
- Continuous labeled anomaly periods: 8
- Training/test values are already normalized
- Test labels are used only for evaluation and visualization

The original SMD project is available at [OmniAnomaly/ServerMachineDataset](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset).

## Method

1. Load the training, test, and test-label files for one machine.
2. Remove constant KPI columns.
3. Split the training series chronologically:
   - 70% training
   - 15% validation
   - 15% calibration
4. Create overlapping windows of 60 observations.
5. Train an LSTM autoencoder and a GRU autoencoder separately.
6. Reconstruct each input window and compute mean squared reconstruction error at the final observation.
7. Set each model's threshold from the 99th percentile of calibration-window errors.
8. Flag test windows whose score is above the threshold.
9. Compare predictions with the official test labels using precision, recall, F1-score, and a confusion matrix.

The default machine has 30 retained KPI columns after removing 8 constant columns. The resulting data shapes are:

| Dataset | Rows/windows |
|---|---:|
| Training | 19,935 rows / 19,876 windows |
| Validation | 4,272 rows / 4,213 windows |
| Calibration | 4,272 rows / 4,213 windows |
| Test | 28,479 rows / 28,420 windows |

Test predictions are aligned to their window-ending observation.

## Environment

The saved notebook metadata uses:

- Python 3.11.17
- Jupyter Notebook / VS Code notebook support
- TensorFlow

The notebook imports:

- NumPy
- pandas
- Matplotlib
- IPython
- scikit-learn
- TensorFlow/Keras

The recorded run used an Apple M5 system with Metal-enabled TensorFlow device acceleration. CPU execution is also supported; GPU acceleration is not required.

## Setup

Run the following from the project directory.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Python 3.11 is recommended because it matches the environment recorded in the notebook.

### 2. Install dependencies

```bash
python -m pip install numpy pandas matplotlib ipython scikit-learn jupyter tensorflow
```

For Jupyter kernel registration:

```bash
python -m ipykernel install --user --name smd-anomaly-lstm-gru --display-name "SMD LSTM/GRU"
```

Open the notebook in Jupyter or VS Code and select the `SMD LSTM/GRU` kernel.

```bash
jupyter lab
```

### 3. Apple Silicon acceleration

TensorFlow Metal support can improve training speed on compatible Apple Silicon systems. If TensorFlow does not discover the Metal device, CPU execution remains sufficient; run the notebook under the verified environment and ensure compatible TensorFlow and Metal plugin versions are installed.

Do not use a generic TensorFlow package intended for a different Python, OS, or hardware architecture.

## Run the Notebook

Keep the project directory as the working directory so `ServerMachineDataset/` resolves correctly.

In Jupyter or VS Code:

1. Open `SMD_Anomaly_Detection_With_LSTM and GRU Autoencoder.ipynb`.
2. Select the `.venv` or `SMD LSTM/GRU` kernel.
3. Run all cells.

The default configuration is:

| Parameter | Default |
|---|---:|
| Machine | `machine-1-1` |
| Window size | 60 |
| RNN units | 64 |
| Latent size | 16 |
| Maximum epochs | 20 |
| Training batch | 32 |
| Prediction batch | 128 |
| Learning rate | 0.001 |
| Early-stopping patience | 5 |
| Threshold percentile | 99 |
| Random seed | 42 |

To test another machine, change `MACHINE` in the notebook configuration. The corresponding train, test, and test-label files must exist.

## Saved Run Results

The notebook's saved output for `machine-1-1` with the fixed configuration is summarized below.

| Model | Epochs | Best validation MSE | Parameters | Threshold | Alerts |
|---|---:|---:|---:|---:|---:|
| LSTM | 20 | 0.000313 | 48,046 | 0.002573 | 2,809 |
| GRU | 20 | 0.000274 | 37,166 | 0.002577 | 1,815 |

Point-level test metrics:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| LSTM | 0.5198 | 0.5419 | 0.5306 |
| GRU | 0.4782 | 0.3222 | 0.3850 |

In this run, LSTM achieved the higher test F1-score. This result applies to one machine and the fixed threshold; it does not establish general superiority across SMD.

## Notebook Outputs

The notebook includes:

- Normal and anomaly observation counts
- Three example test KPIs with labeled anomaly periods
- Chronological train/validation/calibration/test row counts
- LSTM and GRU training/validation loss curves
- Calibration-based anomaly thresholds
- Test anomaly-score and threshold plots
- Actual-versus-reconstructed KPI plots
- Precision, recall, F1-score, and confusion matrices
- A final LSTM-versus-GRU metric chart

## Hyperparameter Tuning

The notebook configuration can be adjusted:

- `WINDOW_SIZE`: number of observations in each sequence
- `UNITS`: recurrent hidden-state size
- `LATENT`: bottleneck dimension
- `EPOCHS`: maximum training epochs
- `TRAIN_BATCH`: training batch size
- `PREDICT_BATCH`: prediction batch size
- `LEARNING_RATE`: Adam learning rate
- `PATIENCE`: early-stopping patience
- `THRESHOLD_PERCENTILE`: calibration percentile
- `MACHINE`: SMD machine file
- `SEED`: random seed

Changing the threshold changes the precision/recall trade-off. A higher threshold may miss anomalies, while a lower threshold produces more false alarms.

## Known Notebook Note

The final summary cell refers to a parameter named `WINDOW`, but the configuration defines `WINDOW_SIZE`. If that cell is run as-is, it raises a `NameError`.

This README documents the issue without modifying the notebook or source code.

## Limitations

- The official SMD training data is not fully labeled and may contain unusual behavior.
- Test labels are used for evaluation only, not threshold selection or training.
- The reconstruction score is point-level, aligned to the end of each 60-observation window.
- Evaluation covers only `machine-1-1` in the default configuration.
- Results should not be interpreted as future incident forecasting or verified outage prediction.
- SMD KPI names are anonymized and no timestamps are provided.
- Interpretation-label behavior should be validated separately because it describes anomalous dimensions rather than every test anomaly observation.

## Repository Structure

```text
.
├── SMD_Anomaly_Detection_With_LSTM and GRU Autoencoder.ipynb
├── README.md
└── ServerMachineDataset/
    ├── train/
    ├── test/
    ├── test_label/
    ├── interpretation_label/
    └── LICENSE
```

## License

The license for the bundled SMD dataset is provided in `ServerMachineDataset/LICENSE` and uses the MIT License.
