"""Video quality and rate metrics: PSNR, MS-SSIM, LPIPS, tLPIPS, NIQE, FID, KID, PVCS and bpp."""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pyiqa
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from actdiff.paths import CHECKPOINT_DIR

_PVCS_MODEL_CACHE: Dict[str, torch.nn.Module] = {}

# Per-frame metrics where a lower value is better (used for plot labels).
LOWER_IS_BETTER = {"lpips", "niqe", "tlpips"}


def Calculate_Psnr(frame1, frame2) -> float:
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)
    mse = np.mean(diff * diff, dtype=np.float64)
    if mse <= 1e-12:
        return 100.0
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def Calculate_Ms_Ssim(
    frame1: np.ndarray,
    frame2: np.ndarray,
    device,
    win_size: int = 11,
) -> float:
    # frame*: HxWx3 uint8 RGB
    x = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    y = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        score = ms_ssim(
            x,
            y,
            data_range=1.0,
            size_average=True,
            win_size=win_size,
        )  # returns mean over batch
    return float(score.item())


def Calculate_Lpips(frame1: np.ndarray, frame2: np.ndarray, lpips_model, device) -> float:
    """Computes the LPIPS between two frames."""

    def To_Lpips_Tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
        # img: HxWx3 uint8 RGB
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # 1x3xHxW
        t = t / 255.0 * 2.0 - 1.0  # [0,1] -> [-1,1]
        return t.to(device)

    t1 = To_Lpips_Tensor(frame1, device)
    t2 = To_Lpips_Tensor(frame2, device)
    with torch.no_grad():
        score = lpips_model(t1, t2)
    return float(score.item())


def Calculate_Tlpips(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    lpips_model,
    device: torch.device,
) -> Tuple[List[float], float]:
    """
    Temporal LPIPS (tLPIPS) per Chu et al.:
    average over t of | LPIPS(g_{t-1}, g_t) - LPIPS(xhat_{t-1}, xhat_t) |.
    Returns (per_pair_values_over_t, average). Lower is better.
    """
    assert (
        len(real_frames) == len(fake_frames) and len(real_frames) >= 2
    ), "Need same-length real/fake and at least 2 frames"

    diffs: List[float] = []
    for t in range(1, len(real_frames)):
        lp_real = Calculate_Lpips(real_frames[t - 1], real_frames[t], lpips_model, device)
        lp_fake = Calculate_Lpips(fake_frames[t - 1], fake_frames[t], lpips_model, device)
        diffs.append(abs(lp_real - lp_fake))
    return diffs, float(np.mean(diffs))


def Calculate_Niqe(frame: np.ndarray, niqe_metric, device) -> float:
    """
    NIQE on a single HxWx3 uint8 RGB frame. pyiqa expects NCHW float in [0,1].
    """
    x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        score = niqe_metric(x)
    return float(score.item())


def Stack_Frames(frames_np_list: List[np.ndarray]) -> torch.Tensor:
    """Stack list of HxWx3 uint8 RGB frames into a uint8 tensor (N,3,H,W)."""
    arr = np.stack(frames_np_list, axis=0)  # (N,H,W,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
    return t


def Calculate_Fid(
    all_real: torch.Tensor,
    all_fake: torch.Tensor,
    device: torch.device,
    normalize: bool = False,
    batch_size: int = 64,
) -> float:
    """Compute Frechet Inception Distance for two image sets.
    all_real/all_fake: uint8 tensors (N,3,H,W) in [0,255], or float [0,1] if normalize=True.
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=normalize).to(device).eval()
    with torch.no_grad():
        for start in range(0, all_real.shape[0], batch_size):
            fid.update(all_real[start : start + batch_size].to(device), real=True)
        for start in range(0, all_fake.shape[0], batch_size):
            fid.update(all_fake[start : start + batch_size].to(device), real=False)
        score = fid.compute()
    return float(score.item())


def Calculate_Kid(
    all_real: torch.Tensor,
    all_fake: torch.Tensor,
    device: torch.device,
    subsets: int = 50,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Computes Kernel Inception Distance (KID) between two image sets.

    all_real/all_fake: uint8 tensors (N,3,H,W) in [0,255], or float [0,1] if normalize=True.
    Returns: (kid_mean, kid_std)
    """
    if seed is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    num_samples = int(min(all_real.shape[0], all_fake.shape[0], 1000))
    kid = KernelInceptionDistance(subsets=subsets, subset_size=max(num_samples, 1), normalize=False).to(
        device
    )
    with torch.no_grad():
        kid.update(all_real.to(device), real=True)
        kid.update(all_fake.to(device), real=False)
        kid_mean, kid_std = kid.compute()
    return float(kid_mean.item()), float(kid_std.item())


def Make_Clips(
    frames: List[np.ndarray], clip_len: int = 16, stride: Optional[int] = None, pad_tail: bool = True
) -> torch.Tensor:
    """
    Turn a list of HxWx3 uint8 RGB frames into (N, 3, T, H, W) uint8 clips.
    - Non-overlapping by default because stride defaults to clip_len.
    - If pad_tail=True, the last short window is padded by repeating the last frame.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")
    if stride is None:
        stride = clip_len
    N = len(frames)
    clips: List[np.ndarray] = []
    for start in range(0, max(1, N - clip_len + 1), stride):
        end = start + clip_len
        if end <= N:
            seq = frames[start:end]
        else:
            if not pad_tail:
                break
            seq = frames[start:N] + [frames[-1]] * (end - N)
        arr = np.stack(seq, axis=0)  # (T, H, W, 3) uint8
        clips.append(arr)
    if not clips:
        seq = frames + [frames[-1]] * (clip_len - len(frames))
        clips = [np.stack(seq, axis=0)]
    clips_np = np.stack(clips, axis=0)  # (N, T, H, W, 3)
    clips_t = torch.from_numpy(clips_np).permute(0, 4, 1, 2, 3).contiguous()  # (N, 3, T, H, W)
    return clips_t


def Load_I3d(device: torch.device, weights_path: Optional[str] = None):
    """
    Lazily create (and cache) an I3D (Inception-style) backbone used for PVCS.

    Uses a vendored Inception-I3D implementation (from piergiaj/pytorch-i3d),
    and loads pretrained weights from `checkpoints/i3d/i3d_pretrained_400.pt` by default.
    """
    if weights_path is None:
        weights_path = str(CHECKPOINT_DIR / "i3d" / "i3d_pretrained_400.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"PVCS requires pretrained Inception-I3D weights at {weights_path} "
            "(run `pixi run download-i3d-weights`)."
        )

    key = f"{device}|{weights_path}"
    if key in _PVCS_MODEL_CACHE:
        return _PVCS_MODEL_CACHE[key]

    from actdiff.metrics.i3d import InceptionI3d

    model = InceptionI3d(num_classes=400, in_channels=3).to(device).eval()
    state = torch.load(weights_path, map_location=device)
    # Handle common checkpoint formats.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    model = model.to(device).eval()
    _PVCS_MODEL_CACHE[key] = model
    return model


@torch.no_grad()
def Calculate_Pvcs(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    device: torch.device,
    clip_len: int = 10,
    stride: int = 1,
    resize_hw: Tuple[int, int] = (224, 224),  # output crop size; protocol uses 224
    layer_names: Tuple[str, str, str, str] = ("Conv3d_2c_3x3", "Mixed_3c", "Mixed_4f", "Mixed_5c"),
    batch_size: int = 4,
    max_frames: Optional[int] = None,
    weights_path: Optional[str] = None,
) -> float:
    """
    PVCS (Perceptual Video Clip Similarity) using an I3D (Inception-style) backbone and LPIPS-style
    "spatial feature distance" on spatiotemporal activations.

    Protocol (common reference pattern):
    - split into 10-frame clips in a sliding window (clip_len=10, stride=1)
    - run pretrained I3D (Inception) backbone
    - extract activations from Conv3d_2c_3x3, Mixed_3c, Mixed_4f, Mixed_5c
    - compute LPIPS-like distance:
        normalize features across channels,
        squared difference,
        sum over channels,
        mean over space-time,
        average over layers and clips
    """
    assert len(real_frames) > 0 and len(fake_frames) > 0, "Empty frame lists"

    # Align (and optionally truncate if max_frames is provided).
    T = min(len(real_frames), len(fake_frames))
    if max_frames is not None:
        T = min(T, int(max_frames))
    real_frames = real_frames[:T]
    fake_frames = fake_frames[:T]

    # Build aligned uint8 clips (N, 3, T, H, W).
    # Reference PVCS uses 10-frame sliding windows (stride=1). We don't pad the tail; we only use full clips.
    real_u8 = Make_Clips(real_frames, clip_len=int(clip_len), stride=int(stride), pad_tail=False)
    fake_u8 = Make_Clips(fake_frames, clip_len=int(clip_len), stride=int(stride), pad_tail=False)
    n = min(real_u8.shape[0], fake_u8.shape[0])
    real_u8 = real_u8[:n]
    fake_u8 = fake_u8[:n]
    if n == 0:
        # If the video is shorter than clip_len, fall back to a single padded clip.
        real_u8 = Make_Clips(real_frames, clip_len=int(clip_len), stride=int(clip_len), pad_tail=True)
        fake_u8 = Make_Clips(fake_frames, clip_len=int(clip_len), stride=int(clip_len), pad_tail=True)
        n = min(real_u8.shape[0], fake_u8.shape[0])
        real_u8 = real_u8[:n]
        fake_u8 = fake_u8[:n]

    def Preprocess(clips_u8: torch.Tensor, resolution: int = 224) -> torch.Tensor:
        """
        Match common I3D preprocessing used in reference FVD/PVCS code:
        - input uint8 (N,3,T,H,W)
        - scale shorter side to `resolution`
        - center crop to resolution x resolution
        - map [0,1] -> [-1,1]
        """
        x = clips_u8.float() / 255.0  # (N,3,T,H,W) in [0,1]
        N, C, Tt, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(N * Tt, C, H, W)  # (N*T,3,H,W)

        # scale shorter side to resolution
        scale = float(resolution) / float(min(H, W))
        if H < W:
            new_h, new_w = resolution, int(np.ceil(W * scale))
        else:
            new_h, new_w = int(np.ceil(H * scale)), resolution
        frames = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        # center crop
        _, _, h2, w2 = frames.shape
        w_start = (w2 - resolution) // 2
        h_start = (h2 - resolution) // 2
        frames = frames[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

        frames = (frames - 0.5) * 2.0  # [-1,1]
        out = frames.reshape(N, Tt, C, resolution, resolution).permute(0, 2, 1, 3, 4).contiguous()
        return out.to(device)

    real_x = Preprocess(real_u8, resolution=int(resize_hw[0]))
    fake_x = Preprocess(fake_u8, resolution=int(resize_hw[0]))

    model = Load_I3d(device, weights_path=weights_path)

    # Prepare feature hooks once per forward.
    def Find_Module_Key(named: Dict[str, torch.nn.Module], target: str) -> str:
        if target in named:
            return target
        # Common: modules are nested; try suffix match, then contains.
        for k in named.keys():
            if k.endswith(target):
                return k
        for k in named.keys():
            if target in k:
                return k
        raise KeyError(target)

    def Extract_Features(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}

        def Hook(name: str):
            def Store_Output(_m, _inp, out):
                feats[name] = out

            return Store_Output

        handles = []
        named = dict(model.named_modules())
        resolved: Dict[str, str] = {}
        for nm in layer_names:
            try:
                resolved[nm] = Find_Module_Key(named, nm)
            except KeyError as e:
                # Provide a helpful message with a small sample of module keys.
                sample = list(named.keys())[:50]
                raise RuntimeError(
                    f"PVCS could not find target layer '{nm}' in the I3D model. "
                    f"Sample module keys: {sample}"
                ) from e

        for nm, key in resolved.items():
            handles.append(named[key].register_forward_hook(Hook(nm)))

        _ = model(x)
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        return feats

    def Feature_Distance(fr: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        """
        LPIPS-style spatial feature distance generalized to (B,C,T,H,W) activations:
        - channel L2-normalize
        - squared difference
        - sum over channels
        - mean over (T,H,W)
        Returns: (B,)
        """
        fr = fr / (fr.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
        ff = ff / (ff.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
        diff = (fr - ff).pow(2).sum(dim=1, keepdim=True)  # (B,1,T,H,W)
        return diff.mean(dim=(2, 3, 4)).squeeze(1)

    # Compute per-clip PVCS distance, averaged across layers.
    dists: List[float] = []
    n = int(real_x.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        rp = real_x[start:end]
        fg = fake_x[start:end]
        feats_r = Extract_Features(rp)
        feats_f = Extract_Features(fg)

        per_layer = []
        for nm in layer_names:
            fr = feats_r[nm]
            ff = feats_f[nm]

            per_layer.append(Feature_Distance(fr, ff))

        dist_b = torch.stack(per_layer, dim=0).mean(dim=0)  # [B]
        dists.extend(dist_b.detach().float().cpu().tolist())

    return float(np.mean(dists)) if dists else float("nan")


def Bpp_From_Bitstream(bitstream_dir: str, total_px: int):
    """Bits per pixel of all keyframe and trajectory files in `bitstream_dir`."""
    if total_px <= 0:
        return float('nan'), 0, 0, 0
    key_bytes = 0
    traj_bytes = 0
    for fname in os.listdir(bitstream_dir):
        if not fname.endswith('.bin'):
            continue
        fsize = os.path.getsize(os.path.join(bitstream_dir, fname))
        if fname.startswith('keyframe_'):
            key_bytes += fsize
        elif fname.startswith('chunk_') and 'trajectories' in fname:
            traj_bytes += fsize
    total_bits = (key_bytes + traj_bytes) * 8.0
    return float(total_bits / float(total_px)), total_bits, key_bytes * 8.0, traj_bytes * 8.0


def Compute_Video_Metrics(
    original_frames: List[np.ndarray],
    generated_frames: List[np.ndarray],
    output_dir: str,
    args: argparse.Namespace,
    save_plots: bool = False,
) -> Dict:
    """Computes all metrics; writes metrics.json and per_frame_metrics.csv to `output_dir`."""
    assert len(original_frames) == len(generated_frames), "Videos must have same number of frames"

    lpips_model = lpips.LPIPS(net='alex').to(args.device)
    lpips_model.eval()
    niqe_metric = pyiqa.create_metric('niqe').to(args.device).eval()

    # Per-frame metrics
    metrics = {
        'psnr': {'per_frame': [], 'average': 0.0},
        'ms_ssim': {'per_frame': [], 'average': 0.0},
        'lpips': {'per_frame': [], 'average': 0.0},
        'niqe': {'per_frame': [], 'average': 0.0},
        'tlpips': {'per_frame': [], 'average': 0.0},
    }

    for orig, gen in zip(original_frames, generated_frames):
        metrics['psnr']['per_frame'].append(Calculate_Psnr(orig, gen))
        metrics['ms_ssim']['per_frame'].append(Calculate_Ms_Ssim(orig, gen, args.device))
        metrics['lpips']['per_frame'].append(Calculate_Lpips(orig, gen, lpips_model, args.device))
        metrics['niqe']['per_frame'].append(Calculate_Niqe(gen, niqe_metric, args.device))

    # Temporal LPIPS (over adjacent frame pairs)
    metrics['tlpips']['per_frame'], _ = Calculate_Tlpips(
        original_frames, generated_frames, lpips_model, args.device
    )

    for metric in metrics:
        metrics[metric]['average'] = float(np.mean(metrics[metric]['per_frame']))

    # Distribution metrics over all frames of the video
    real_tensor = Stack_Frames(original_frames)
    fake_tensor = Stack_Frames(generated_frames)
    kid_mean, kid_std = Calculate_Kid(real_tensor, fake_tensor, args.device, subsets=50, seed=42)
    fid_value = Calculate_Fid(real_tensor, fake_tensor, args.device, normalize=False, batch_size=64)
    pvcs_value = Calculate_Pvcs(original_frames, generated_frames, device=args.device)

    # Bit rate: total stream bits over all pixels at the codec (diffusion) resolution.
    H_res, W_res = generated_frames[0].shape[:2]
    total_pixels = int(len(generated_frames) * H_res * W_res)
    bpp_value, total_bits, key_bits, traj_bits = Bpp_From_Bitstream(args.bitstream_dir, total_pixels)

    summary = {k: v['average'] for k, v in metrics.items()}
    summary.update(
        {
            'fid': fid_value,
            'kid': {'mean': kid_mean, 'std': kid_std},
            'pvcs': pvcs_value,
            'bpp': bpp_value,
            'bit_breakdown': {
                'keyframes_pct': float(key_bits / total_bits) if total_bits else float('nan'),
                'trajectories_pct': float(traj_bits / total_bits) if total_bits else float('nan'),
                'total_bits': float(total_bits),
                'keyframes_bits': float(key_bits),
                'trajectories_bits': float(traj_bits),
            },
            'bpp_meta': {'bpp_reference_hw': [H_res, W_res]},
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nMetrics summary (averages):")
    print(json.dumps(summary, indent=4))

    # Per-frame values (tLPIPS is defined on frame pairs, so it is only in metrics.json)
    with open(os.path.join(output_dir, "per_frame_metrics.csv"), 'w') as f:
        f.write("frame,psnr,ms_ssim,lpips,niqe\n")
        for i in range(len(original_frames)):
            f.write(
                f"{i},{metrics['psnr']['per_frame'][i]:.4f},{metrics['ms_ssim']['per_frame'][i]:.4f},"
                f"{metrics['lpips']['per_frame'][i]:.4f},{metrics['niqe']['per_frame'][i]:.4f}\n"
            )

    if save_plots:
        plot_dir = os.path.join(output_dir, "intermediate", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        Plot_Metrics(metrics, plot_dir)
    return summary


def Plot_Metrics(metrics: Dict, output_dir: str):
    """Plots each per-frame metric over time, one figure per metric plus a combined figure."""

    def Labels(metric_name: str):
        direction = "lower is better" if metric_name in LOWER_IS_BETTER else "higher is better"
        unit = "dB, " if metric_name == "psnr" else ""
        return direction, f'{metric_name.upper()} ({unit}{direction})'

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    fig.suptitle('Quality Metrics Over Time', fontsize=16)
    for ax, (metric_name, metric_data) in zip(np.atleast_1d(axes), metrics.items()):
        direction, ylabel = Labels(metric_name)
        ax.plot(metric_data['per_frame'], label=f'Per-frame {metric_name.upper()} ({direction})')
        ax.axhline(
            y=metric_data['average'],
            color='r',
            linestyle='--',
            label=f"Average: {metric_data['average']:.4f}",
        )
        ax.set_title(f'{metric_name.upper()} over frames ({direction})')
        ax.set_xlabel('Frame number')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    for metric_name, metric_data in metrics.items():
        direction, ylabel = Labels(metric_name)
        plt.figure(figsize=(10, 6))
        plt.plot(metric_data['per_frame'], label=f'Per-frame {metric_name.upper()} ({direction})')
        plt.axhline(
            y=metric_data['average'],
            color='r',
            linestyle='--',
            label=f"Average: {metric_data['average']:.4f}",
        )
        plt.title(f'{metric_name.upper()} over frames ({direction})')
        plt.xlabel('Frame number')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
