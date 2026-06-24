import os
import logging
from collections import deque

import numpy as np

from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
from avlite.c10_perception.c11_perception_model import PerceptionModel, PredictionMode
from .settings import ExtensionSettings

log = logging.getLogger(__name__)

_CHECKPOINT = ExtensionSettings.checkpoint


class HistoryBuffer:
    """Accumulates per-agent (x, y) positions subsampled from sensor rate to model rate.

    stride = sensor_fps / model_fps  (e.g. 30/10 = 3 → keep every 3rd frame)
    """

    def __init__(self, obs_len: int, sensor_fps: float, model_fps: float):
        self._obs_len = obs_len
        self._stride  = max(1, round(sensor_fps / model_fps))
        self._buf:    dict[int, deque] = {}
        self._frame = 0

    def update(self, agents) -> dict[int, np.ndarray]:
        """Push current sensor frame; return history per agent_id (min 2 points)."""
        self._frame += 1
        sample = (self._frame % self._stride == 0)

        # log.debug("[HistoryBuffer] frame=%d stride=%d sample=%s before=%s",
        #           self._frame, self._stride, sample,
        #           {aid: len(b) for aid, b in self._buf.items()})

        updated_ids = set()
        for agent in agents:
            aid = agent.agent_id
            updated_ids.add(aid)
            if aid not in self._buf:
                self._buf[aid] = deque(maxlen=self._obs_len)
            # if sample:
            self._buf[aid].append((agent.x, agent.y))

        for gone in set(self._buf) - updated_ids:
            del self._buf[gone]

        return {
            aid: np.array(list(buf), dtype=np.float32)
            for aid, buf in self._buf.items()
            if len(buf) >= 2
        }


class PluginBasicLSTMPredictor(PredictionStrategy):
    """Trajectory predictor using a pretrained LSTM.

    Torch and model weights are loaded lazily on first predict() call so that
    the class registers in PredictionStrategy.registry even when torch is not
    installed.
    """

    def __init__(
        self,
        checkpoint: str   = _CHECKPOINT,
        obs_len:    int   = ExtensionSettings.obs_len,
        pred_len:   int   = ExtensionSettings.pred_len,
        sensor_fps: float = ExtensionSettings.sensor_fps,
        model_fps:  float = ExtensionSettings.model_fps,
        device:     str   = ExtensionSettings.device,
    ):
        self._checkpoint = checkpoint
        self._pred_len   = pred_len
        self._device_str = device
        self._obs_len    = obs_len
        self._history      = HistoryBuffer(obs_len=obs_len, sensor_fps=sensor_fps, model_fps=model_fps)
        self._preprocessor = None
        self._model        = None
        self._load_failed  = False

    @property
    def requirements(self) -> set:
        return set()

    def _ensure_model(self):
        if self._load_failed:
            return

        try:
            import torch
            from .lstm import LSTMModel
            from .trajectory_preprocessor import TrajectoryPreprocessor
        except ImportError as e:
            log.warning("[LSTM] Import failed: %s", e)
            self._load_failed = True
            raise RuntimeError(
                "PluginBasicLSTMPredictor requires torch. Install it with: pip install torch"
            ) from e

        log.warning("[LSTM] torch imported OK, loading model...")
        self._torch = torch
        requested = torch.device(self._device_str)
        if requested.type == "cuda" and not torch.cuda.is_available():
            log.warning("[LSTM] CUDA requested but not available — falling back to CPU")
            self._device = torch.device("cpu")
        else:
            self._device = requested
        self._preprocessor = TrajectoryPreprocessor(obs_len=self._obs_len)
        log.info("[LSTM] TrajectoryPreprocessor ready (obs_len=%d)", self._obs_len)

        try:
            if os.path.isfile(self._checkpoint):
                checkpoint = torch.load(self._checkpoint, map_location="cpu")
                model_params = checkpoint.get("model_params", {})
                state_dict   = checkpoint.get("model_state_dict", checkpoint)
                log.info("[LSTM] Checkpoint keys: epoch=%s val_loss=%s model_params=%s",
                            checkpoint.get("epoch", "?"),
                            checkpoint.get("val_loss", "?"),
                            model_params)
                model = LSTMModel(**model_params)
                model.load_state_dict(state_dict)
            else:
                log.error("[LSTM] Checkpoint not found at %s", self._checkpoint)
                # model = LSTMModel(input_dim=2, output_type="point")
            model.to(self._device)
            model.eval()
            self._model = model
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(
                "[LSTM] Model ready | device=%s | epoch=%s | val_loss=%s\n"
                "  Architecture : input_dim=%s  embedding_dim=%s  hidden_dim=%s"
                "  num_layers=%s  dropout=%s  output_type=%s  output_dim=%s\n"
                "  Parameters   : total=%s  trainable=%s",
                self._device,
                checkpoint.get("epoch", "?"),
                checkpoint.get("val_loss", "?"),
                model.input_dim,
                model.embedding_dim,
                model.hidden_dim,
                model.num_layers,
                model.dropout,
                model.output_type,
                model.output_dim,
                f"{total_params:,}",
                f"{trainable_params:,}",
            )
        except Exception as e:
            log.warning("[LSTM] Model loading FAILED: %s", e, exc_info=True)
            self._load_failed = True
            raise

    def predict(self, perception_model: PerceptionModel) -> PerceptionModel:
        agents = perception_model.agent_vehicles
        if not agents:
            perception_model.trajectories = np.empty((0, self._pred_len, 2))
            return perception_model

        # 1. accumulate history at model FPS
        histories = self._history.update(agents)

        self._ensure_model()

        # 2. preprocess: pad + displacements + optional normalisation
        obs_batch, last_positions, valid_indices = [], [], []
        for i, agent in enumerate(agents):
            hist = histories.get(agent.agent_id)
            if hist is None:
                continue
            processed = self._preprocessor.process_observation(hist)
            if processed is None:
                continue
            obs_batch.append(processed["obs_disp"])
            last_positions.append(processed["last_position"])
            valid_indices.append(i)

        n_agents = len(agents)
        trajectories = np.zeros((n_agents, self._pred_len, 2), dtype=np.float32)

        if obs_batch:
            obs_tensor = self._torch.stack(obs_batch).to(self._device)  # [B, obs_len-1, 2]

            # 3. LSTM inference
            with self._torch.no_grad():
                pred_disp = self._model(obs_tensor, self._pred_len)      # [B, pred_len, 2]

            # 4. postprocess: cumsum displacements → absolute positions
            pred_disp = pred_disp.cpu().numpy()

            for batch_idx, agent_idx in enumerate(valid_indices):
                last_pos = last_positions[batch_idx].numpy()             # [2]
                trajectories[agent_idx] = last_pos + np.cumsum(pred_disp[batch_idx], axis=0)
        log.debug(f"histories {histories}")
        log.debug(f"input obs_batch {obs_batch}")
        log.debug(f" output trajectories: {trajectories}")
        perception_model.trajectories   = trajectories
        perception_model.prediction_mode = PredictionMode.TRAJECTORY
        log.debug("PluginBasicLSTMPredictor: %d/%d agents predicted, shape=%s",
                  len(valid_indices), n_agents, trajectories.shape)
        return perception_model
