"""Compress a video with ActDiff-VC, decode it, and evaluate the reconstruction."""
import gc
import json
import os
import shutil

import cv2
import numpy as np
import torch
from image_gen_aux import DepthPreprocessor

from actdiff.config import Build_Parser, Parse_Arguments
from actdiff.decoder import Decoder
from actdiff.encoder import Encoder
from actdiff.metrics.video_metrics import Compute_Video_Metrics
from actdiff.utils.video_io import Load_Video, Write_Video


def Run_Name(args):
    """`<keyframe codec>_gop<max_gop_length>`, plus a tag for every other setting that differs from its
    default (config.json in the run folder lists all of them)."""
    if args.run_name:
        return args.run_name
    default = Build_Parser().get_default
    parts = [args.keyframe_codec, f"gop{args.max_gop_length}"]
    if args.num_trajectories != default("num_trajectories"):
        parts.append(f"traj{args.num_trajectories}")
    if (args.gop_coverage_threshold, args.gop_lpips_threshold) != (
        default("gop_coverage_threshold"),
        default("gop_lpips_threshold"),
    ):
        parts.append(f"cov{args.gop_coverage_threshold:g}-lpips{args.gop_lpips_threshold:g}")
    if args.diffusion_steps != default("diffusion_steps"):
        parts.append(f"steps{args.diffusion_steps}")
    flags = {
        "fixedgop": args.fixed_gop,
        "uniform": args.uniform_trajectories,
        "highmotion": args.high_motion_trajectories,
        "nosketch": args.no_sketch_weights,
        "single": args.first_keyframe_only,
    }
    parts += [tag for tag, on in flags.items() if on]
    if args.fast_motion_factor > 0:
        parts.append(f"fast{args.fast_motion_factor}")
    if args.tracker_blur_kernel > 1:
        parts.append(f"blur{args.tracker_blur_kernel}-{args.tracker_blur_angle:g}")
    if args.trajectory_drift_sigma > 0:
        parts.append(f"drift{args.trajectory_drift_sigma:g}-{args.trajectory_drift_rho:g}")
    return "_".join(parts)


def Reset_Directory(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def Side_By_Side(original, generated):
    """Original | decoded, with a caption bar."""
    frame = np.concatenate([original, generated], axis=1)
    banner = np.zeros((max(40, frame.shape[0] // 20), frame.shape[1], 3), dtype=np.uint8)
    y = int(banner.shape[0] * 0.7)
    for text, x in (("Original", 10), ("Generated", original.shape[1] + 10)):
        cv2.putText(banner, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([banner, frame], axis=0)


def Main():
    args = Parse_Arguments()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)  # third-party code (e.g. AllTracker) calls `.cuda()`
    if args.verbose:
        for key, value in vars(args).items():
            print(f"{key:32}: {value}")

    step = max(1, args.fast_motion_factor)
    frames, fps = Load_Video(args, num_frames=None if args.num_frames is None else args.num_frames * step)
    frames = frames[::step][: args.num_frames]
    args.fps = fps // step if args.fast_motion_factor > 0 else fps

    name = Run_Name(args)
    args.output_dir = os.path.join(args.output_dir, name)
    args.bitstream_dir = os.path.join(args.bitstream_dir, name)
    Reset_Directory(args.output_dir)
    Reset_Directory(args.bitstream_dir)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Outputs: {args.output_dir}")

    depth_estimator = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti").to(args.device)

    # Encoder and decoder are loaded one after the other to keep peak GPU memory low.
    encoder = Encoder(args, depth_estimator)
    original = encoder.Compress_Video(frames)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    decoder = Decoder(args, depth_estimator)
    generated = decoder.Decode_Video()
    del decoder
    gc.collect()
    torch.cuda.empty_cache()

    if args.fast_motion_factor > 0:  # evaluate every setting on the same frames (every 4th original frame)
        original = original[:: 4 // args.fast_motion_factor]
        generated = generated[:: 4 // args.fast_motion_factor]
    assert len(original) == len(generated), (len(original), len(generated))

    Write_Video(generated, args.fps, os.path.join(args.output_dir, "reconstruction.mp4"))
    Write_Video(
        [Side_By_Side(o, g) for o, g in zip(original, generated)],
        args.fps,
        os.path.join(args.output_dir, "comparison.mp4"),
    )
    Compute_Video_Metrics(original, generated, args.output_dir, args, save_plots=args.save_intermediate)

    for name in os.listdir(args.output_dir):  # HiFiC writes a log file per model load
        if name.startswith("logs_"):
            os.remove(os.path.join(args.output_dir, name))
    if not args.keep_bitstream:  # the bit rate is already in metrics.json
        shutil.rmtree(args.bitstream_dir, ignore_errors=True)
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    Main()
