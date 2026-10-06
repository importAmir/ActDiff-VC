"""ActDiff-VC encoder: dense tracking, adaptive GOP, sparse trajectory selection and bitstream writing."""
import os
import time

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from alltracker.nets.alltracker import Net as AllTrackerNet
from alltracker.utils.basic import gridcloud2d
from alltracker.utils.saveload import load as alltracker_load

from actdiff.codecs.image_codec import KeyframeCodec
from actdiff.codecs.trajectory_codec import Compress_Trajectories
from actdiff.config import DAS_FRAME_HW
from actdiff.point_selection.selector import PointSelector


def Motion_Blur(frame, kernel_size, angle):
    """Linear motion blur of an HxWx3 uint8 frame (robustness test)."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(center, angle, 1.0), (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-8)
    return cv2.filter2D(frame, -1, kernel)


def Add_Trajectory_Drift(tracks, sigma_pixels, rho, seed, image_hw, reference_max_dim=720.0):
    """AR(1) random drift on [T, N, 2] tracks (robustness test). `sigma_pixels` is at 480x720 scale."""
    rng = np.random.default_rng(seed)
    perturbed = tracks.copy().astype(np.float32)
    num_frames, num_points, _ = perturbed.shape
    if num_frames <= 1 or num_points == 0:
        return perturbed

    h, w = int(image_hw[0]), int(image_hw[1])
    sigma = float(sigma_pixels) * max(h, w) / float(reference_max_dim)
    drift = np.zeros((num_points, 2), dtype=np.float32)
    for t in range(1, num_frames):
        drift = float(rho) * drift + rng.normal(0.0, sigma, size=(num_points, 2)).astype(np.float32)
        perturbed[t] += drift
    perturbed[..., 0] = np.clip(perturbed[..., 0], 0, w - 1)
    perturbed[..., 1] = np.clip(perturbed[..., 1], 0, h - 1)
    return perturbed


def Splat_Keyframe(keyframe, trajectory, depth, visibility, confidence, threshold):
    """Z-buffered forward warp of a [3, H, W] keyframe to per-pixel target positions [2, H, W].

    Only pixels whose visibility and confidence pass `threshold` are warped.
    Returns the warped image and the occupancy mask O_t.
    """
    C, H, W = keyframe.shape
    device = keyframe.device
    N = H * W
    valid = ((visibility >= threshold) & (confidence >= threshold)).view(-1)
    target_x = torch.clamp(torch.round(trajectory[0]), 0, W - 1).to(torch.long)
    target_y = torch.clamp(torch.round(trajectory[1]), 0, H - 1).to(torch.long)
    target = (target_y * W + target_x).view(-1).to(device=device, dtype=torch.long)

    depth_flat = depth.view(-1)
    depth_sel = torch.where(valid, depth_flat, torch.full_like(depth_flat, float('inf'))).to(
        device, torch.float32
    )
    min_depth = torch.full((N,), float('inf'), device=device, dtype=torch.float32)
    min_depth = min_depth.scatter_reduce_(0, target, depth_sel, reduce='amin', include_self=True)

    # Among sources tied at the nearest depth of a target pixel, keep the lowest source index.
    source = torch.arange(N, device=device, dtype=torch.long)
    winner = torch.where((depth_sel == min_depth[target]) & valid, source, torch.full_like(source, N * 2))
    first = torch.full((N,), N * 2, device=device, dtype=torch.long)
    first = first.scatter_reduce_(0, target, winner, reduce='amin', include_self=True)
    occupied = first < N * 2

    warped = torch.zeros((C, N), device=device, dtype=torch.float32)
    if occupied.any():
        warped[:, occupied] = keyframe.view(C, -1).index_select(1, first[occupied])
    return warped.view(C, H, W).clamp(0.0, 1.0), occupied.view(H, W)


class Encoder:
    def __init__(self, args, depth_estimator):
        self.args = args
        self.device = args.device
        self.depth_estimator = depth_estimator
        self.keyframe_codec = KeyframeCodec(args)
        self.point_selector = PointSelector(args)

        self.tracker = AllTrackerNet(args.tracker_window)
        alltracker_load(
            None,
            args.alltracker_checkpoint,
            self.tracker,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        self.tracker.to(device=self.device).eval()
        for p in self.tracker.parameters():
            p.requires_grad = False

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(DAS_FRAME_HW)

    @torch.no_grad()
    def Estimate_Depth(self, image):
        """[3, H, W] in [0, 1] -> relative depth [1, H, W]."""
        image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return self.to_tensor(self.depth_estimator(image)[0])

    @torch.no_grad()
    def Track(self, video):
        """Dense tracking from the first frame of `video` [T, 3, H, W] in [0, 1].

        Returns absolute pixel positions [1, T, 2, H, W] and visibility / confidence [1, T, 2, H, W].
        """
        args = self.args
        rgbs = video.unsqueeze(0)
        _, T, _, H, W = rgbs.shape

        blur_kernel = int(args.tracker_blur_kernel or 0)
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                raise ValueError("--tracker_blur_kernel must be odd, e.g. 9 or 17")
            frames_u8 = (rgbs[0].float().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round()
            blurred = [
                Motion_Blur(f.astype(np.uint8), blur_kernel, float(args.tracker_blur_angle))
                for f in frames_u8
            ]
            rgbs = torch.stack(
                [torch.from_numpy(b).permute(2, 0, 1).float() / 255.0 for b in blurred]
            ).unsqueeze(0)

        rgbs = (rgbs * 255.0).to(dtype=torch.float32, device=self.device)
        grid_xy = gridcloud2d(1, H, W, norm=False, device=self.device).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

        start = time.time()
        flows, visconf, _, _ = self.tracker.forward_sliding(
            rgbs, iters=args.tracker_iters, sw=None, is_training=False
        )
        flows = flows.to(device=grid_xy.device, dtype=grid_xy.dtype)
        if flows.dim() == 4:  # single-frame output has no time axis
            flows = flows.unsqueeze(1)
        flows += grid_xy
        if args.verbose:
            print(f"Tracked {T} frames in {time.time() - start:.2f} s")
        return flows, visconf.to(device=grid_xy.device, dtype=torch.float32)

    @torch.no_grad()
    def Gop_Length(self, video, trajectories, visconf, lpips_model):
        """Adaptive GOP (paper Sec. 4.3): the GOP ends at the first frame of the earliest run of
        `gop_patience` frames whose warped-keyframe coverage or masked LPIPS fails. At least 2."""
        args = self.args
        T, _, H, W = video.shape
        keyframe = video[0].to(device=self.device, dtype=torch.float32)
        depth = self.Estimate_Depth(keyframe).to(device=self.device)[0]

        fail_count, first_fail = 0, None
        for t in range(1, T):
            warped, occupied = Splat_Keyframe(
                keyframe,
                trajectories[0, t],
                depth,
                visconf[0, t, 0],
                visconf[0, t, 1],
                float(args.gop_confidence_threshold),
            )
            coverage = float(occupied.sum().item()) / float(H * W)

            # LPIPS on the covered region only: both images are zeroed where nothing was warped.
            mask = occupied.to(dtype=torch.float32).unsqueeze(0)
            frame = video[t].to(device=self.device, dtype=torch.float32)
            distance = lpips_model(
                (warped * mask).unsqueeze(0) * 2.0 - 1.0, (frame * mask).unsqueeze(0) * 2.0 - 1.0
            )[0, 0]
            if distance.shape[-2:] != (H, W):
                distance = F.interpolate(
                    distance[None, None], size=(H, W), mode='bilinear', align_corners=False
                )[0, 0]
            perceptual = float(distance.mean().item())

            if coverage < float(args.gop_coverage_threshold) or perceptual > float(args.gop_lpips_threshold):
                if fail_count == 0:
                    first_fail = t
                fail_count += 1
                if fail_count >= int(args.gop_patience):
                    # End before the first violating frame, so a scene cut never becomes the last
                    # keyframe of the previous scene; the next GOP then starts at the cut.
                    if args.verbose:
                        print(f"GOP length {first_fail}: coverage {coverage:.3f}, LPIPS {perceptual:.3f}")
                    return max(2, int(first_fail))
            else:
                fail_count = 0
        return max(2, T)

    @torch.no_grad()
    def Compress_Chunk(self, video, chunk_idx, trajectories=None, visconf=None):
        """Write the keyframe(s) and sparse trajectories of one GOP [T, 3, H, W] to the stream folder."""
        args = self.args
        if chunk_idx == 0:
            self.keyframe_codec.Compress(
                video[0:1].to(self.device, torch.float32), args.bitstream_dir, "keyframe_0.bin"
            )
        self.keyframe_codec.Compress(
            video[-1:].to(self.device, torch.float32), args.bitstream_dir, f"keyframe_{chunk_idx + 1}.bin"
        )

        trajectory_path = os.path.join(args.bitstream_dir, f"chunk_{chunk_idx}_trajectories.bin")
        T = int(video.shape[0])
        if T <= 2:  # keyframes only: store just the length
            Compress_Trajectories(np.zeros((T, 0, 2), dtype=np.int16), trajectory_path)
            return

        if trajectories is None:
            trajectories, visconf = self.Track(video)

        sigma_index = None
        if args.uniform_trajectories:
            tracks = self.point_selector.Uniform_Selection(trajectories, args.num_trajectories)
        elif args.high_motion_trajectories:
            tracks = self.point_selector.High_Motion_Selection(
                trajectories, args.refine_grid_cells, args.num_trajectories
            )
        else:
            sketch = self.point_selector.Get_Sketch(video[0].to(self.device, torch.float32))
            tracks, sigma_index = self.point_selector.Sparse_Selection(
                sketch,
                trajectories,
                args.num_anchor_cells,
                args.refine_grid_cells,
                args.anchor_threshold,
                args.num_trajectories,
                points_per_step=args.points_per_step,
                use_uniform_weights=args.no_sketch_weights,
            )

        if args.trajectory_drift_sigma > 0:
            tracks = Add_Trajectory_Drift(
                tracks,
                args.trajectory_drift_sigma,
                args.trajectory_drift_rho,
                args.trajectory_drift_seed + chunk_idx,
                video.shape[2:],
            )

        num_bytes = Compress_Trajectories(tracks, trajectory_path, sigma_index=sigma_index)
        if args.verbose:
            print(f"Chunk {chunk_idx}: {T} frames, {tracks.shape[1]} trajectories, {num_bytes} bytes")

    @torch.no_grad()
    def Compress_Video(self, frames):
        """Encode a list of RGB PIL frames. Consecutive GOPs share one keyframe.

        Returns the frames at codec resolution as HxWx3 uint8 arrays, the reference for evaluation.
        """
        args = self.args
        frames = [self.resize(f) for f in frames]
        total = len(frames)
        lpips_model = None if args.fixed_gop else lpips.LPIPS(net='alex').to(self.device).eval()

        start, chunk_idx = 0, 0
        while True:
            video = torch.stack(
                [self.to_tensor(f) for f in frames[start : min(total, start + args.max_gop_length)]]
            )
            if video.shape[0] <= 2:
                length = video.shape[0]
                self.Compress_Chunk(video, chunk_idx)
            else:
                trajectories, visconf = self.Track(video)
                length = (
                    video.shape[0]
                    if lpips_model is None
                    else self.Gop_Length(video, trajectories, visconf, lpips_model)
                )
                self.Compress_Chunk(
                    video[:length],
                    chunk_idx,
                    trajectories[:, :length].contiguous(),
                    visconf[:, :length].contiguous(),
                )
            if start + length >= total:
                break
            start += length - 1
            chunk_idx += 1

        return [
            (self.to_tensor(f).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8) for f in frames
        ]
