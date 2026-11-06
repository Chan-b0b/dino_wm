"""LGES task-3 episodes (phys/collect.py, and real robot episodes converted to the same keys) as DINO-WM
trajectories.

An episode npz holds head frames (head_img jpg buffers + head_img_off offsets, head_t sim times) and fast rows
(t, cup / cup_base (x, y, z, yaw), pump, held, phase, wrench, q_live, base). A trajectory here is one frame
sequence between two chassis legs (drop_legs: the frames while the chassis drives are left out, so every action
is an arm action), its last frame dropped so every frame has the action that leads to the next one:

  action  (5)  cup tip delta to the next frame (dx, dy, dz, dyaw) in `frame` ("base" = base_link, "table"),
               + 1 if the pump was commanded on anywhere in that interval
  proprio (6)  cup tip (x, y, z, cos yaw, sin yaw) in `frame`, held (1 if the cup holds something)
  state   (9)  cup (x, y, z, yaw), held, base (x, y, yaw), Fz of the wrist (N)

Images are decoded only for the frames asked for (the upstream slicer decodes a whole trajectory per slice;
LGESSlicerDataset calls get_frames with the slice's frames instead).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision import transforms

from .traj_dset import TrajDataset

LEG_PREFIX = "drive to"


def squash_transform(img_size=224):
    """960x600 (or 480x300) -> img_size x img_size by squashing: the bins at both sides stay in the picture (the
    default Resize + CenterCrop would cut them off). DINO's patch features do not mind the aspect change."""
    return transforms.Compose([transforms.Resize((img_size, img_size), antialias=True),
                               transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),      # antialiasing overshoots a little
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _interp(tq, t, Y):
    """Per-column linear interpolation of Y (N, k) sampled at t, at times tq."""
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        return np.interp(tq, t, Y)
    return np.stack([np.interp(tq, t, Y[:, j]) for j in range(Y.shape[1])], 1)


def _nearest(tq, t):
    i = np.searchsorted(t, tq)
    i = np.clip(i, 1, len(t) - 1)
    return np.where(np.abs(t[i - 1] - tq) <= np.abs(t[i] - tq), i - 1, i)


class LGESDataset(TrajDataset):
    def __init__(
        self,
        files: Sequence[Path],
        transform: Optional[Callable] = None,
        frame: str = "base",
        drop_legs: bool = True,
        min_len: int = 8,
        normalize_action: bool = True,
        stats: Optional[dict] = None,
        with_force: bool = False,
        frame_lag: float = 1.0,
    ):
        """frame_lag: how many frame intervals the sim renderer's picture is behind the time it was read (measured
        2026-11-05: 1). An episode whose meta says collect.py already moved head_t back by `frame_lag` frames gets
        only the difference; real episodes (no sim meta) are taken as exact (lag 0)."""
        self.files = [Path(f) for f in files]
        self.transform, self.frame, self.drop_legs, self.with_force = transform, frame, drop_legs, with_force
        self.frame_lag = frame_lag
        self.eps = []                                     # per episode: the per-frame arrays (small)
        self.trajs = []                                   # (episode index, first frame, end frame)
        for e, f in enumerate(self.files):
            ep = self._load(f)
            if ep is None:
                continue
            self.eps.append(ep)
            for s, t in self._segments(ep, min_len):
                self.trajs.append((len(self.eps) - 1, s, t))
        if not self.trajs:
            raise RuntimeError(f"no usable trajectories in {len(self.files)} episodes")
        self.action_dim = 5
        self.proprio_dim = 6 + (1 if with_force else 0)
        self.state_dim = 9
        acts = np.concatenate([self._act(*tr) for tr in self.trajs])
        props = np.concatenate([self._proprio(*tr) for tr in self.trajs])
        states = np.concatenate([self._state(*tr) for tr in self.trajs])
        if stats is None:
            stats = dict(action_mean=acts.mean(0), action_std=acts.std(0) + 1e-6,
                         proprio_mean=props.mean(0), proprio_std=props.std(0) + 1e-6,
                         state_mean=states.mean(0), state_std=states.std(0) + 1e-6)
            stats["action_mean"][4], stats["action_std"][4] = 0.0, 1.0      # the pump bit stays 0 / 1
            stats["proprio_mean"][5], stats["proprio_std"][5] = 0.0, 1.0    # so does held
        self.stats = {k: np.asarray(v, np.float32) for k, v in stats.items()}
        for k in ("action", "proprio", "state"):
            mean = self.stats[f"{k}_mean"] if normalize_action else np.zeros_like(self.stats[f"{k}_mean"])
            std = self.stats[f"{k}_std"] if normalize_action else np.ones_like(self.stats[f"{k}_std"])
            setattr(self, f"{k}_mean", torch.tensor(mean))
            setattr(self, f"{k}_std", torch.tensor(std))
        print(f"LGES: {len(self.eps)} episodes, {len(self.trajs)} trajectories, "
              f"{sum(t - s for _, s, t in self.trajs)} frames (frame={frame}, drop_legs={drop_legs})")

    # ------------------------------------------------------------ loading
    def _load(self, f):
        z = np.load(f, allow_pickle=False)
        if "head_img" not in z.files or len(z["head_t"]) < 3:
            return None
        t, ht = z["t"], z["head_t"].astype(np.float64)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if meta.get("fps"):                                             # a sim episode: the renderer's lag
            ht = ht - (self.frame_lag - float(meta.get("frame_lag", 0.0))) / float(meta["fps"])
        cup = z["cup_base" if self.frame == "base" else "cup"]
        n = len(ht)
        near = _nearest(ht, t)
        cupf = _interp(ht, t, cup)                                      # (n, 4), yaw interpolated raw (no wrap)
        cupf[:, 3] = cup[near, 3]                                       # yaw from the nearest row: no wrap trouble
        pump = z["pump"].astype(bool)
        pump_iv = np.zeros(n, bool)                                     # pump on anywhere in [t_i, t_i+1)
        edges = np.searchsorted(t, ht)
        for i in range(n - 1):
            a, b = edges[i], max(edges[i + 1], edges[i] + 1)
            pump_iv[i] = pump[a:b].any() or pump[near[i]]
        pump_iv[-1] = pump[near[-1]]
        phases = json.loads(str(z["phases"])) if "phases" in z.files else []
        phase = z["phase"][near] if "phase" in z.files else np.zeros(n, int)
        leg = np.array([bool(phases) and phases[p].startswith(LEG_PREFIX) for p in phase])
        ep = dict(file=f, off=z["head_img_off"], n=n, t=ht, cup=cupf.astype(np.float32), pump=pump_iv,
                  held=(z["held"][near] >= 0).astype(np.float32), leg=leg,
                  base=_interp(ht, t, z["base"]).astype(np.float32) if "base" in z.files else np.zeros((n, 3), np.float32),
                  fz=_interp(ht, t, z["wrench"][:, 2]).astype(np.float32) if "wrench" in z.files else np.zeros(n, np.float32))
        return ep

    def _segments(self, ep, min_len):
        """Frame runs [s, t) with t - s >= min_len: between legs (drop_legs) or the whole episode; the last frame
        of each run is left out (its action would need the frame after it)."""
        n = ep["n"]
        if self.drop_legs:
            keep = ~ep["leg"]
            runs, s = [], None
            for i in range(n):
                if keep[i] and s is None:
                    s = i
                if (not keep[i] or i == n - 1) and s is not None:
                    e = i if not keep[i] else i + 1
                    runs.append((s, e))
                    s = None
        else:
            runs = [(0, n)]
        return [(s, e - 1) for s, e in runs if e - 1 - s >= min_len]

    # ------------------------------------------------------------ the arrays of one trajectory
    def _act(self, e, s, t):
        ep = self.eps[e]
        c0, c1 = ep["cup"][s:t], ep["cup"][s + 1:t + 1]
        d = c1 - c0
        d[:, 3] = _wrap(d[:, 3])
        return np.concatenate([d, ep["pump"][s:t, None].astype(np.float32)], 1).astype(np.float32)

    def _proprio(self, e, s, t):
        ep = self.eps[e]
        c = ep["cup"][s:t]
        cols = [c[:, :3], np.cos(c[:, 3:4]), np.sin(c[:, 3:4]), ep["held"][s:t, None]]
        if self.with_force:
            cols.append(ep["fz"][s:t, None])
        return np.concatenate(cols, 1).astype(np.float32)

    def _state(self, e, s, t):
        ep = self.eps[e]
        return np.concatenate([ep["cup"][s:t], ep["held"][s:t, None], ep["base"][s:t], ep["fz"][s:t, None]], 1).astype(np.float32)

    # ------------------------------------------------------------ TrajDataset
    def get_seq_length(self, idx):
        _, s, t = self.trajs[idx]
        return t - s

    def get_all_actions(self):
        return torch.cat([(torch.tensor(self._act(*tr)) - self.action_mean) / self.action_std for tr in self.trajs], 0)

    def decode(self, e, frames):
        """RGB uint8 (len(frames), H, W, 3) of episode e's absolute frames."""
        ep = self.eps[e]
        z = np.load(ep["file"], allow_pickle=False)
        buf, off = z["head_img"], ep["off"]
        out = []
        for i in frames:
            img = cv2.imdecode(buf[off[i]:off[i + 1]], cv2.IMREAD_COLOR)
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    def get_frames(self, idx, frames):
        """frames: indices into trajectory idx (0 = its first frame)."""
        e, s, t = self.trajs[idx]
        frames = np.asarray(list(frames), int)
        act = (torch.tensor(self._act(e, s, t)[frames]) - self.action_mean) / self.action_std
        proprio = (torch.tensor(self._proprio(e, s, t)[frames]) - self.proprio_mean) / self.proprio_std
        state = torch.tensor(self._state(e, s, t)[frames])
        image = torch.from_numpy(self.decode(e, s + frames)).float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        return {"visual": image, "proprio": proprio}, act, state, {"episode": str(self.eps[e]["file"].name), "start": int(s)}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.trajs)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        return rearrange(imgs, "b h w c -> b c h w") / 255.0


class LGESSlicerDataset(TrajDataset):
    """num_frames-frame windows (frameskip apart) over every trajectory, decoding only those frames; returns
    (obs, act, state) as datasets.traj_dset.TrajSlicerDataset does (actions of the skipped frames concatenated)."""

    def __init__(self, dataset: LGESDataset, num_frames: int, frameskip: int = 1):
        self.dataset, self.num_frames, self.frameskip = dataset, num_frames, frameskip
        self.slices = []
        for i in range(len(dataset)):
            T = dataset.get_seq_length(i)
            span = num_frames * frameskip
            if T - span + 1 > 0:
                self.slices += [(i, start, start + span) for start in range(T - span + 1)]
            else:
                print(f"Ignored short trajectory #{i}: len={T}, span={span}")
        self.slices = np.random.permutation(self.slices)
        self.proprio_dim = dataset.proprio_dim
        self.action_dim = dataset.action_dim * frameskip
        self.state_dim = dataset.state_dim

    def get_seq_length(self, idx):
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        e, s, t = self.dataset.trajs[i]
        obs, _, state, _ = self.dataset.get_frames(i, range(start, end, self.frameskip))
        act = (torch.tensor(self.dataset._act(e, s, t)[start:end]) - self.dataset.action_mean) / self.dataset.action_std
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
        return obs, act, state


def find_episodes(data_path):
    p = Path(data_path)
    files = sorted(p.rglob("ep_*.npz")) if p.is_dir() else sorted(Path().glob(str(p)))
    if not files:
        raise FileNotFoundError(f"no ep_*.npz under {data_path}")
    return files


def load_lges_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=1,
    frame="base",
    drop_legs=True,
    with_force=False,
    frame_lag=1.0,
    seed=42,
):
    files = find_episodes(data_path)
    if n_rollout:
        files = files[:n_rollout]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_train = max(1, int(round(split_ratio * len(files))))
    train_files = [files[i] for i in perm[:n_train]]
    val_files = [files[i] for i in perm[n_train:]] or train_files[-1:]
    num_frames = num_hist + num_pred
    min_len = max(num_frames * frameskip, 2)
    train_dset = LGESDataset(train_files, transform, frame, drop_legs, min_len, normalize_action, with_force=with_force,
                             frame_lag=frame_lag)
    val_dset = LGESDataset(val_files, transform, frame, drop_legs, min_len, normalize_action, stats=train_dset.stats,
                           with_force=with_force, frame_lag=frame_lag)
    datasets = {"train": LGESSlicerDataset(train_dset, num_frames, frameskip),
                "valid": LGESSlicerDataset(val_dset, num_frames, frameskip)}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
