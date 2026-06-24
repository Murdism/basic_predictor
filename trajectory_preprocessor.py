#!/usr/bin/env python3
import torch


class TrajectoryPreprocessor:
    """Converts raw (x, y) history into padded displacement tensors for the LSTM.

    Steps:
        1. Pad/trim trajectory to obs_len positions (right-aligned, zero-padded)
        2. Compute frame-to-frame displacements → [obs_len-1, 2]

    Args:
        obs_len: Number of observation steps the model expects (must match training).
    """

    def __init__(self, obs_len: int):
        self.obs_len = obs_len

    @staticmethod
    def compute_displacements(traj):
        """traj: [T, 2] → [T-1, 2]"""
        return traj[1:] - traj[:-1]

    def process_observation(self, obs):
        """
        Args:
            obs: np.ndarray or Tensor [T, 2], T <= obs_len

        Returns dict with:
            obs_disp      [obs_len-1, 2]  padded displacements  (LSTM input)
            obs_traj      [obs_len,   2]  padded raw positions
            obs_mask      [obs_len]       bool mask of valid positions
            last_position [2]             last valid (x,y) — anchor for postprocessing
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)

        if obs.dim() != 2 or obs.size(-1) != 2:
            raise ValueError("obs must have shape [T, 2].")

        obs_len = obs.shape[0]
        if obs_len < 2:
            return None

        obs_pad   = torch.zeros(self.obs_len, 2, dtype=torch.float32)
        obs_valid = torch.zeros(self.obs_len, dtype=torch.bool)

        if obs_len >= self.obs_len:
            obs_pad[:] = obs[-self.obs_len:]
            obs_valid[:] = True
        else:
            obs_pad[-obs_len:] = obs
            obs_valid[-obs_len:] = True

        valid_obs    = obs_pad[obs_valid]
        obs_disp     = self.compute_displacements(valid_obs)       # [T-1, 2]

        obs_disp_pad = torch.zeros(self.obs_len - 1, 2, dtype=torch.float32)
        obs_disp_pad[-obs_disp.shape[0]:] = obs_disp              # right-align

        return {
            "obs_disp":      obs_disp_pad,
            "obs_traj":      obs_pad,
            "obs_mask":      obs_valid,
            "last_position": valid_obs[-1],
        }
