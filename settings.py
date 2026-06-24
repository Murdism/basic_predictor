from pathlib import Path


_PLUGIN_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = _PLUGIN_DIR / "chekpoint"


class ExtensionSettings:
    exclude = ["exclude", "filepath"]
    filepath: str = str(_PLUGIN_DIR / "config" / "e10_basic_predictor_plugin.yaml")

    sensor_fps: float = 30.0
    model_fps: float = 30.0
    obs_len: int = 20
    pred_len: int = 30
    device: str = "cuda:0"
    checkpoint: str = str(CHECKPOINT_DIR / "openv2v.pth")


PluginSettings = ExtensionSettings
