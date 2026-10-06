"""Command-line arguments."""
import argparse

from actdiff.paths import CHECKPOINT_DIR

# Diffusion-as-Shader (CogVideoX) always generates 49-frame clips at 480x720.
# A GOP therefore spans at most 49 frames, and every video is processed at this resolution.
DAS_NUM_FRAMES = 49
DAS_FRAME_HW = (480, 720)

KEYFRAME_CODECS = ("hific-low", "hific-med", "hific-high", "mlic")


def Build_Parser():
    parser = argparse.ArgumentParser(description="Compress, decode and evaluate a video with ActDiff-VC.")
    Add_Run_Arguments(parser)
    Add_Method_Arguments(parser)
    Add_Yuv_Arguments(parser)
    Add_Ablation_Arguments(parser)
    Add_Checkpoint_Arguments(parser)
    return parser


def Parse_Arguments(argv=None):
    parser = Build_Parser()
    args = parser.parse_args(argv)
    if not 2 <= args.max_gop_length <= DAS_NUM_FRAMES:
        parser.error(
            f"--max_gop_length must be in [2, {DAS_NUM_FRAMES}] (the Diffusion-as-Shader clip length)"
        )
    args.num_anchor_cells = int(args.num_trajectories * args.anchor_ratio)
    return args


def Add_Run_Arguments(parser):
    group = parser.add_argument_group("run")
    group.add_argument(
        "--video_path", type=str, required=True, help="Input video (.mp4/.avi/.mov) or raw .yuv file"
    )
    group.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Compress only the first N frames (default: the whole video)",
    )
    group.add_argument(
        "--output_dir", type=str, default="outputs", help="Folder for decoded videos and metrics"
    )
    group.add_argument(
        "--bitstream_dir", type=str, default="bitstream", help="Folder for the compressed files"
    )
    group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Sub-folder name of this run (default: from the main settings)",
    )
    group.add_argument(
        "--keep_bitstream",
        action="store_true",
        help="Keep the compressed files after the bit rate is measured (deleted by default)",
    )
    group.add_argument("--device", type=str, default="cuda:0", help="GPU to run on")
    group.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Also save keyframes, tracking videos, per-GOP videos and metric plots",
    )
    group.add_argument("--verbose", action="store_true", help="Print arguments and per-step progress")


def Add_Method_Arguments(parser):
    group = parser.add_argument_group("method")
    group.add_argument(
        "--keyframe_codec",
        type=str,
        default="hific-low",
        choices=KEYFRAME_CODECS,
        help="Image codec for the keyframes",
    )
    group.add_argument(
        "--max_gop_length",
        type=int,
        default=DAS_NUM_FRAMES,
        help=f"Longest GOP in frames (at most {DAS_NUM_FRAMES}); shorter GOPs mean more keyframes "
        "and a higher bit rate",
    )
    group.add_argument(
        "--gop_coverage_threshold",
        type=float,
        default=0.7,
        help="Adaptive GOP: a frame fails if the warped keyframe covers less than this fraction of it",
    )
    group.add_argument(
        "--gop_lpips_threshold",
        type=float,
        default=0.3,
        help="Adaptive GOP: a frame fails if its masked LPIPS to the warped keyframe is above this",
    )
    group.add_argument(
        "--gop_patience",
        type=int,
        default=3,
        help="Adaptive GOP: consecutive failing frames before a new keyframe is placed",
    )
    group.add_argument(
        "--gop_confidence_threshold",
        type=float,
        default=0.3,
        help="Adaptive GOP: tracker visibility / confidence needed for a pixel to be warped",
    )
    group.add_argument("--num_trajectories", type=int, default=300, help="Trajectory budget per GOP")
    group.add_argument(
        "--anchor_ratio",
        type=float,
        default=0.33,
        help="Fraction of the trajectory budget used as anchor-grid cells",
    )
    group.add_argument(
        "--anchor_threshold", type=float, default=0.5, help="Minimum sketch weight of an anchor point"
    )
    group.add_argument(
        "--refine_grid_cells", type=int, default=600, help="Grid cells searched when adding trajectories"
    )
    group.add_argument(
        "--points_per_step", type=int, default=10, help="Trajectories added per refinement step"
    )
    group.add_argument("--tracker_window", type=int, default=16, help="AllTracker sliding-window length")
    group.add_argument("--tracker_iters", type=int, default=4, help="AllTracker refinement iterations")
    group.add_argument(
        "--diffusion_steps",
        type=int,
        default=20,
        help="Diffusion denoising steps (50 gives lower FID / KID, about 2.5x slower)",
    )


def Add_Yuv_Arguments(parser):
    group = parser.add_argument_group("raw YUV input (size and frame rate are required for .yuv files)")
    group.add_argument("--yuv_width", type=int, default=None, help="Frame width")
    group.add_argument("--yuv_height", type=int, default=None, help="Frame height")
    group.add_argument("--yuv_fps", type=float, default=None, help="Frame rate")
    group.add_argument(
        "--yuv_subsampling", type=str, default="420", choices=("420", "422", "444"), help="Chroma subsampling"
    )
    group.add_argument("--yuv_bit_depth", type=int, default=8, help="Bits per sample")
    group.add_argument("--yuv_order", type=str, default="YUV", choices=("YUV", "YVU"), help="Plane order")
    group.add_argument(
        "--yuv_color_matrix", type=str, default="bt709", choices=("bt709", "bt601"), help="YUV to RGB matrix"
    )
    group.add_argument(
        "--yuv_color_range", type=str, default="limited", choices=("limited", "full"), help="Sample range"
    )


def Add_Ablation_Arguments(parser):
    group = parser.add_argument_group("ablations and robustness tests")
    group.add_argument(
        "--fixed_gop",
        action="store_true",
        help="GOPs of exactly --max_gop_length frames instead of adaptive keyframe selection",
    )
    group.add_argument("--uniform_trajectories", action="store_true", help="Trajectories on a uniform grid")
    group.add_argument(
        "--high_motion_trajectories",
        action="store_true",
        help="The largest-motion trajectory of each grid cell",
    )
    group.add_argument(
        "--no_sketch_weights", action="store_true", help="Trajectory selection without HED sketch weighting"
    )
    group.add_argument(
        "--first_keyframe_only",
        action="store_true",
        help="Condition the diffusion model on the first keyframe only",
    )
    group.add_argument(
        "--fast_motion_factor",
        type=int,
        default=0,
        choices=[0, 1, 2, 4],
        help="Simulate faster motion by keeping every k-th frame (0 disables)",
    )
    group.add_argument(
        "--tracker_blur_kernel",
        type=int,
        default=0,
        help="Odd size of a motion blur applied to the tracker input only (0 disables)",
    )
    group.add_argument("--tracker_blur_angle", type=float, default=0.0, help="Motion blur angle in degrees")
    group.add_argument(
        "--trajectory_drift_sigma",
        type=float,
        default=0.0,
        help="Std (pixels at 480x720) of smooth drift added to the sent trajectories (0 disables)",
    )
    group.add_argument(
        "--trajectory_drift_rho", type=float, default=0.95, help="Temporal correlation of the drift"
    )
    group.add_argument("--trajectory_drift_seed", type=int, default=0, help="Random seed of the drift")


def Add_Checkpoint_Arguments(parser):
    group = parser.add_argument_group("checkpoints")
    group.add_argument("--das_checkpoint", type=str, default=str(CHECKPOINT_DIR / "Diffusion-As-Shader"))
    group.add_argument(
        "--alltracker_checkpoint", type=str, default=str(CHECKPOINT_DIR / "alltracker" / "alltracker.pth")
    )
    group.add_argument(
        "--hed_checkpoint", type=str, default=str(CHECKPOINT_DIR / "hed" / "network-bsds500.pth")
    )
    group.add_argument(
        "--mlic_checkpoint", type=str, default=str(CHECKPOINT_DIR / "mlic" / "mlicpp_mse_q5_2960000.pth.tar")
    )
    group.add_argument(
        "--hific_low_checkpoint", type=str, default=str(CHECKPOINT_DIR / "hific" / "hific_low.pt")
    )
    group.add_argument(
        "--hific_med_checkpoint", type=str, default=str(CHECKPOINT_DIR / "hific" / "hific_med.pt")
    )
    group.add_argument(
        "--hific_high_checkpoint", type=str, default=str(CHECKPOINT_DIR / "hific" / "hific_hi.pt")
    )
