# AVLite LSTM Predictor Plugin

An LSTM-based trajectory prediction plugin for AVLite that integrates through the `PredictionStrategy` interface.

The predictor observes each tracked agent's recent `(x, y)` positions, converts them into relative displacements, runs an LSTM model, and converts the predicted displacements back into future absolute trajectories.

## Overview

### Prediction Pipeline

```text
Agent Positions
    ↓
History Buffer
    ↓
TrajectoryPreprocessor
    ↓
LSTM Model
    ↓
Trajectory Postprocessing
    ↓
perception_model.trajectories
```

### Model I/O

Input:

```text
[num_agents, obs_len - 1, 2]
```

Output:

```text
[num_agents, pred_len, 2]
```

where each feature corresponds to an `(x, y)` displacement.

---

## Repository Structure

```text
basic_predictor/
├── __init__.py
├── predictor.py
├── settings.py
├── lstm.py
├── trajectory_preprocessor.py
└── chekpoint/
    ├── argv2.pth
    └── openv2v.pth
```

### File Description

| File                         | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| `predictor.py`               | AVLite prediction strategy implementation   |
| `lstm.py`                    | LSTM network definition                     |
| `trajectory_preprocessor.py` | Trajectory preprocessing and postprocessing |
| `settings.py`                | Runtime and model configuration             |
| `*.pth`                      | Pretrained model checkpoints                |

---

## Installation

### Dependencies

```bash
pip install torch
```

This plugin depends on:

* AVLite
* PyTorch

The plugin is independent of `avlite/extensions/e10_basic_predictor`, but relies on AVLite's public prediction interfaces.

---

## Plugin Registration

Register the plugin in:

```yaml
configs/c40_execution.yaml
```

```yaml
community_plugins:
  e10_basic_predictor_plugin: /absolute/path/to/e10_basic_predictor_plugin
```

Ensure the active execution profile uses:

```yaml
perception: PerceptionPipeline
```

---

## Predictor Selection

Select the predictor in:

```yaml
configs/c10_perception.yaml
```

```yaml
prediction_strategy: PluginBasicLSTMPredictor
```

Alternatively, select **PluginBasicLSTMPredictor** from the AVLite UI.

### Automatic Registration

AVLite automatically registers any class that inherits from:

```python
from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
```

This plugin provides:

```python
class PluginBasicLSTMPredictor(PredictionStrategy):
```

which is registered under:

```text
PluginBasicLSTMPredictor
```

---

## Configuration

Edit `settings.py` to configure the predictor.

Example:

```python
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = _PLUGIN_DIR / "chekpoint"

class ExtensionSettings:
    exclude = ["exclude", "filepath"]
    filepath = str(_PLUGIN_DIR / "config" / "e10_basic_predictor_plugin.yaml")
    checkpoint = str(CHECKPOINT_DIR / "openv2v.pth")

    sensor_fps = 30.0
    model_fps = 30.0
    obs_len = 20
    pred_len = 30
    device = "cuda:0"
```

---

## Available Settings

### Checkpoint Selection

Select the pretrained checkpoint file stored under `chekpoint/`:

```python
checkpoint = str(_PLUGIN_DIR / "chekpoint" / "openv2v.pth")
```

or

```python
checkpoint = str(_PLUGIN_DIR / "chekpoint" / "argv2.pth")
```

Choose a checkpoint that matches your data domain. Prediction quality typically degrades when the deployment domain differs significantly from the training dataset.

### Pretrained Model Details

* `argv2.pth` — trained on the Argoverse 2 dataset.
  - Argoverse 2 is an autonomous driving dataset with multi-agent trajectories, high-definition maps, and full 360° sensor coverage.
  - It includes diverse urban driving scenarios, lane geometry, and long horizon trajectory forecasting.

* `openv2v.pth` — trained on the OpenV2V dataset.
  - OpenV2V focuses on cooperative vehicle-to-vehicle perception and trajectory prediction.
  - It contains multi-agent scenes shared over V2V communication, emphasizing collaboration between connected vehicles.

Prediction quality is best when the deployed scene is similar to the training dataset.

### `device`

Inference device:

```python
device = "cpu"
```

or

```python
device = "cuda:0"
```

GPU inference is recommended when available.

### `sensor_fps`

Rate at which AVLite receives perception updates.

Example:

```python
sensor_fps = 30.0
```

### `model_fps`

Effective sampling rate expected by the predictor.

Example:

```python
model_fps = 10.0
```

A common configuration is:

```python
sensor_fps = 30.0
model_fps = 10.0
```

meaning the perception pipeline may operate at 30 Hz while the predictor uses trajectories sampled at 10 Hz.

### `obs_len`

Number of historical trajectory samples provided to the model.

```python
obs_len = 20
```

### `pred_len`

Number of future trajectory samples predicted by the model.

```python
pred_len = 30
```

For the provided checkpoints:

```python
obs_len = 20
pred_len = 30
```

The model therefore:

* observes 20 historical positions
* predicts 30 future positions

These values must match the configuration used during training.

---

## Checkpoint Compatibility

The provided checkpoints were trained with:

```python
preprocessor_params = {
    "obs_len": 20,
    "pred_len": 30,
}
```

If you use a custom checkpoint, ensure that:

```python
obs_len
pred_len
```

in `settings.py` match the checkpoint's training configuration.

Changing these values without retraining or using a compatible checkpoint may result in incorrect predictions or model loading errors.
