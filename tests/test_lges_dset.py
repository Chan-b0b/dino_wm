"""Plumbing test for datasets.lges_dset without Isaac: writes two synthetic episodes in collect.py's layout (jpg
frames of a moving square, fast rows with a leg in the middle), then checks the trajectories, the action /
proprio shapes, the leg split, the slicer's windows and that a model-shaped batch comes out.

  .venv/bin/python tests/test_lges_dset.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.lges_dset import LGESDataset, LGESSlicerDataset, load_lges_slice_train_val, squash_transform  # noqa: E402


def synth_episode(path, seed, n_frames=40, fps=5.0, fast_hz=60.0, res=480):
    rng = np.random.default_rng(seed)
    T = n_frames / fps
    t = np.arange(0, T, 1 / fast_hz)
    m = len(t)
    cup = np.zeros((m, 4), np.float32)
    cup[:, 0] = 0.3 + 0.2 * np.sin(t / T * 2 * np.pi)
    cup[:, 1] = 0.4 + 0.1 * np.cos(t / T * 2 * np.pi)
    cup[:, 2] = 0.9 - 0.1 * np.abs(np.sin(t / T * 4 * np.pi))
    cup[:, 3] = -1.2 + 0.3 * np.sin(t / T * np.pi)
    phases = ["settle", "pick: down", "drive to the left stop", "place: down", "done"]
    phase = np.zeros(m, np.int16)
    phase[t > 0.2 * T] = 1
    phase[(t > 0.45 * T) & (t < 0.55 * T)] = 2                 # the leg: frames here are dropped
    phase[t >= 0.55 * T] = 3
    phase[t > 0.9 * T] = 4
    pump = ((t > 0.3 * T) & (t < 0.7 * T))
    held = np.where((t > 0.32 * T) & (t < 0.68 * T), 3, -1).astype(np.int8)
    base = np.tile([0.2, -0.47, 1.53], (m, 1)).astype(np.float32)
    base[phase >= 2] = [-0.4, -0.45, 1.59]
    wrench = rng.normal(0, 0.3, (m, 6)).astype(np.float32)
    head_t = np.arange(n_frames) / fps + 0.1
    bufs, off = [], [0]
    for i, ht in enumerate(head_t):
        img = np.full((300, res, 3), 40, np.uint8)
        k = int(np.interp(ht, t, np.arange(m)))
        x = int((cup[k, 0] - 0.1) / 0.4 * res)
        cv2.rectangle(img, (x, 100), (x + 40, 140), (0, 180, 255), -1)
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        bufs.append(b.reshape(-1))
        off.append(off[-1] + b.size)
    np.savez(path, side="left", t=t, q_live=np.zeros((m, 7), np.float32), q_other=np.zeros((m, 7), np.float32),
             wrench=wrench, cup=cup, cup_base=cup + np.float32([0.5, 0.0, 0.0, -1.5]), base=base, pump=pump, held=held,
             phase=phase, obj=np.zeros((m, 2, 4), np.float32), objects=json.dumps([]), phases=json.dumps(phases),
             events=json.dumps([]), steps=json.dumps([]), meta=json.dumps(dict(seed=seed)), outcome=json.dumps({}),
             head_img=np.concatenate(bufs), head_img_off=np.array(off, np.int64), head_t=head_t)


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for s in range(3):
            synth_episode(d / f"ep_{s:05d}.npz", s)
        ds = LGESDataset(sorted(d.glob("ep_*.npz")), squash_transform(224), frame="base", drop_legs=True, min_len=4)
        assert len(ds) == 6, f"3 episodes x 2 segments expected, got {len(ds)} trajectories"
        for i in range(len(ds)):
            e, s, t = ds.trajs[i]
            assert not ds.eps[e]["leg"][s:t].any(), "a leg frame inside a trajectory"
        obs, act, state, info = ds[0]
        T = ds.get_seq_length(0)
        assert obs["visual"].shape == (T, 3, 224, 224), obs["visual"].shape
        assert obs["proprio"].shape == (T, 6) and act.shape == (T, 5) and state.shape == (T, 9)
        a = ds._act(*ds.trajs[0])
        assert np.allclose(a[:, 4], a[:, 4].round()) and set(np.unique(a[:, 4])) <= {0.0, 1.0}, "pump bit not 0/1"
        assert np.abs(a[:, :3]).max() < 0.1, "a cup delta per 0.2 s frame should be centimetres, not metres"
        sl = LGESSlicerDataset(ds, num_frames=4, frameskip=2)
        o, ac, st = sl[0]
        assert o["visual"].shape == (4, 3, 224, 224) and ac.shape == (4, 10) and st.shape == (4, 9), (o["visual"].shape, ac.shape, st.shape)
        assert o["visual"].min() >= -1.0 and o["visual"].max() <= 1.0
        dsets, trajs = load_lges_slice_train_val(squash_transform(224), str(d), split_ratio=0.67, num_hist=3, num_pred=1, frameskip=1)
        batch = torch.utils.data.DataLoader(dsets["train"], batch_size=8, shuffle=True, num_workers=0)
        o, ac, st = next(iter(batch))
        assert o["visual"].shape == (8, 4, 3, 224, 224) and ac.shape == (8, 4, 5), (o["visual"].shape, ac.shape)
        assert torch.equal(trajs["valid"].action_mean, trajs["train"].action_mean), "val must use the train stats"
        print("OK:", len(ds), "trajectories;", len(sl), "slices; batch", tuple(o["visual"].shape), "act", tuple(ac.shape),
              "proprio", tuple(o["proprio"].shape), "action mean", np.round(trajs["train"].stats["action_mean"], 4).tolist())


if __name__ == "__main__":
    main()
