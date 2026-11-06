"""Side-by-side video of what the world model predicts against what the camera saw.

    .venv/bin/python rollout_video.py --run outputs/2026-11-05/16-33-42            # 3 validation trajectories
    .venv/bin/python rollout_video.py --run outputs/2026-11-05/16-33-42 --n 6 --split train --horizon 40
    .venv/bin/python rollout_video.py --run ... --device cpu --n 1 --horizon 12    # no GPU (slow)

Each video (out/rollout_video/<run>_<split>_<traj>.mp4, 5 fps = real time) has three panels per frame:
  ground truth | one-step prediction (model sees the true last num_hist frames and the actions, predicts this frame)
              | open-loop prediction (model sees only the first num_hist frames, then rolls the recorded actions forward)
The header carries the frame index / time and the latent error of each prediction (MSE of the DINO patch features
against the encoded true frame, the quantity the training loss uses). Predictions are the decoder's picture of the
predicted latent, so the decoder's own blur is part of what you see; the first num_hist frames of both prediction
panels are the decoder's reconstruction of the true frames (the context), labeled as such.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
os.environ.setdefault("DATASET_DIR", str(Path.home() / "Humanoid/LGES_Sim/data"))
os.environ.setdefault("TORCH_HOME", str(HERE / ".torch_home"))

import cv2
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

MODEL_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")


def load_model(run: Path, ckpt: str, device):
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    payload = torch.load(run / "checkpoints" / ckpt, map_location=device, weights_only=False)
    parts = {k: payload[k].to(device) for k in MODEL_KEYS if k in payload}
    if "encoder" not in parts:                       # the frozen DINO encoder is not saved
        parts["encoder"] = hydra.utils.instantiate(cfg.encoder).to(device)
    if "decoder" not in parts:
        raise SystemExit("this checkpoint has no decoder (has_decoder=False): nothing to draw")
    model = hydra.utils.instantiate(
        cfg.model, encoder=parts["encoder"], proprio_encoder=parts["proprio_encoder"],
        action_encoder=parts["action_encoder"], predictor=parts["predictor"], decoder=parts["decoder"],
        proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim, concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
    ).to(device)
    model.eval()          # VWorldModel.train() returns None, so this cannot be chained
    return model, cfg, int(payload["epoch"])


def load_trajs(cfg, split):
    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    return traj[split]


def to_uint8(img):
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 RGB."""
    img = (img.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5) * 255
    return rearrange(img, "c h w -> h w c").numpy().round().astype(np.uint8)


@torch.no_grad()
def decode_frames(model, z_obs, chunk=8):
    """z_obs: dict of (1, T, ...) latents -> (T, 3, H, W) decoded pictures, decoded a few frames at a time."""
    T = z_obs["visual"].shape[1]
    out = []
    for a in range(0, T, chunk):
        part = {k: v[:, a:a + chunk] for k, v in z_obs.items()}
        out.append(model.decode_obs(part)[0]["visual"][0])
    return torch.cat(out, 0)


@torch.no_grad()
def predict_traj(model, obs, act, num_hist, frameskip, device):
    """obs: dict of (T, ...) true frames, act: (T, A) actions (frame i -> i+1).
    Returns dicts (T, ...) of: true latents, one-step predicted latents, open-loop predicted latents, their pictures.
    Frames < num_hist of the prediction outputs are the encoded/reconstructed true frames (the context)."""
    T = obs["visual"].shape[0]
    obs_b = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}
    act_b = act.unsqueeze(0).to(device)
    z = model.encode(obs_b, act_b)                                   # (1, T, P, D) true latents with the actions
    z_true_obs, _ = model.separate_emb(z)

    # one step: windows of num_hist true frames -> the next frame
    windows = torch.stack([z[0, t - num_hist:t] for t in range(num_hist, T)], 0)   # (T-h, h, P, D)
    preds = []
    for a in range(0, len(windows), 16):
        preds.append(model.predict(windows[a:a + 16])[:, -1])        # (n, P, D)
    z_one = torch.cat([z[0, :num_hist], torch.cat(preds, 0)], 0).unsqueeze(0)      # (1, T, P, D)
    z_one_obs, _ = model.separate_emb(z_one)

    # open loop: first num_hist frames, then the recorded actions
    obs_0 = {k: v[:, :num_hist] for k, v in obs_b.items()}
    z_open_obs, _ = model.rollout(obs_0, act_b[:, :T - 1])           # (1, T, P, D)

    err = {}
    for name, zo in (("one", z_one_obs), ("open", z_open_obs)):
        e = ((zo["visual"][0] - z_true_obs["visual"][0]) ** 2).mean(dim=(1, 2))   # (T,)
        e[:num_hist] = float("nan")
        err[name] = e.cpu().numpy()
    pics = {"recon": decode_frames(model, z_true_obs), "one": decode_frames(model, z_one_obs),
            "open": decode_frames(model, z_open_obs)}
    return err, pics


def label(panel, text, color=(255, 255, 255)):
    cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def compose(gt, one, opn, t, T, num_hist, err_one, err_open, title, scale):
    panels = []
    for img, head in ((gt, "camera"),
                      (one, "context (recon)" if t < num_hist else f"1-step   z err {err_one:.4f}"),
                      (opn, "context (recon)" if t < num_hist else f"open-loop   z err {err_open:.4f}")):
        p = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
        label(p, head, (255, 255, 255) if head.startswith(("camera", "context")) else (120, 255, 120))
        panels.append(p)
    row = np.concatenate(panels, 1)
    bar = np.zeros((32, row.shape[1], 3), np.uint8)
    label(bar, f"{title}   frame {t:3d}/{T - 1}   t = {t * 0.2:5.1f} s")
    frame = np.concatenate([bar, row], 0)
    return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]  # even sizes for h264


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="hydra output dir with .hydra/config.yaml and checkpoints/")
    ap.add_argument("--ckpt", default="model_latest.pth")
    ap.add_argument("--split", default="valid", choices=("train", "valid"))
    ap.add_argument("--traj", type=int, nargs="*", help="trajectory indices in the split (default: the n longest)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=0, help="frames per video (0: whole trajectory)")
    ap.add_argument("--fps", type=float, default=5.0, help="playback rate; the data is 5 fps")
    ap.add_argument("--scale", type=float, default=2.0, help="panel upscaling (224 px -> 448 px)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(HERE.parent.parent / "out" / "rollout_video"))
    args = ap.parse_args()

    run = Path(args.run).resolve()
    device = torch.device(args.device)
    model, cfg, epoch = load_model(run, args.ckpt, device)
    num_hist, frameskip = int(cfg.num_hist), int(cfg.frameskip)
    if frameskip != 1:
        raise SystemExit("frameskip != 1 is not handled here yet")
    dset = load_trajs(cfg, args.split)
    lens = np.array([dset.get_seq_length(i) for i in range(len(dset))])
    trajs = args.traj if args.traj else list(np.argsort(-lens)[: args.n])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{run.parent.name}_{run.name}_e{epoch}"
    print(f"[video] {run.name} epoch {epoch} on {device}, {args.split} split: {len(dset)} trajectories, "
          f"len {lens.min()}..{lens.max()} frames")

    summary = []
    for i in trajs:
        obs, act, state, info = dset[int(i)]
        T = obs["visual"].shape[0] if args.horizon <= 0 else min(args.horizon, obs["visual"].shape[0])
        obs = {k: v[:T] for k, v in obs.items()}
        act = act[:T]
        err, pics = predict_traj(model, obs, act, num_hist, frameskip, device)
        title = f"{info['episode']} +{info['start']}  ({args.split} #{i}, epoch {epoch})"
        path = out / f"{tag}_{args.split}_{int(i):03d}.mp4"
        with imageio.get_writer(str(path), fps=args.fps, codec="libx264", quality=8, macro_block_size=None) as w:
            for t in range(T):
                w.append_data(compose(to_uint8(obs["visual"][t]), to_uint8(pics["one"][t]), to_uint8(pics["open"][t]),
                                      t, T, num_hist, err["one"][t], err["open"][t], title, args.scale))
        one, opn = np.nanmean(err["one"]), np.nanmean(err["open"])
        held = int(state[:, 4].sum()) if state.shape[1] > 4 else -1
        summary.append((int(i), T, one, opn, err["open"][-1], path.name))
        print(f"[video] traj {i:3d}: {T:3d} frames, held frames {held:3d}, z err 1-step {one:.4f}  "
              f"open-loop mean {opn:.4f} last {err['open'][-1]:.4f}  -> {path}")
    print("\n  traj  frames  1-step   open mean  open last   file")
    for s in summary:
        print(f"  {s[0]:4d}  {s[1]:6d}  {s[2]:.4f}   {s[3]:.4f}     {s[4]:.4f}   {s[5]}")


if __name__ == "__main__":
    main()
