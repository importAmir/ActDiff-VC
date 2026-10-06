"""Video reading (container formats and raw planar YUV) and MP4 writing."""
import os

import numpy as np
from diffusers.utils import load_video
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image

# (Kr, Kb) per matrix; (y_scale, y_offset, c_scale, c_offset) per range.
YUV_MATRICES = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}
YUV_RANGES = {"limited": (255 / 219, 16, 255 / 224, 128), "full": (1.0, 0, 1.0, 128)}


def Write_Video(frames, fps, path):
    """Write RGB frames (PIL images or HxWx3 uint8 arrays) to an H.264 MP4."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    ImageSequenceClip(frames, fps=fps).write_videofile(path, codec="libx264", fps=fps, logger=None)


def Read_Yuv(
    path,
    width,
    height,
    subsampling="420",
    bit_depth=8,
    num_frames=None,
    yuv_order="YUV",
    color_matrix="bt709",
    color_range="limited",
):
    """Read a planar raw YUV file into a list of RGB PIL images (chroma upsampled by repetition)."""
    cw, ch = {
        "420": ((width + 1) // 2, (height + 1) // 2),
        "422": ((width + 1) // 2, height),
        "444": (width, height),
    }[subsampling]
    dtype = np.uint8 if bit_depth <= 8 else np.uint16  # >8-bit samples stored in 16-bit words
    y_size, c_size = width * height, cw * ch
    frame_size = y_size + 2 * c_size

    available = os.path.getsize(path) // (frame_size * np.dtype(dtype).itemsize)
    num_frames = available if num_frames is None else min(num_frames, available)
    data = np.fromfile(path, dtype=dtype, count=num_frames * frame_size).reshape(num_frames, frame_size)

    Kr, Kb = YUV_MATRICES[color_matrix]
    Kg = 1 - Kr - Kb
    y_scale, y_off, c_scale, c_off = YUV_RANGES[color_range]

    images = []
    for frame in data:
        Y = frame[:y_size].reshape(height, width)
        U = frame[y_size : y_size + c_size].reshape(ch, cw)
        V = frame[y_size + c_size :].reshape(ch, cw)
        if yuv_order == "YVU":
            U, V = V, U

        if bit_depth > 8:
            max_value = (1 << bit_depth) - 1
            Y, U, V = (p.astype(np.float32) * 255.0 / max_value for p in (Y, U, V))
        else:
            Y, U, V = (p.astype(np.float32) for p in (Y, U, V))

        if subsampling == "420":
            U = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)[:height, :width]
            V = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)[:height, :width]
        elif subsampling == "422":
            U = np.repeat(U, 2, axis=1)[:, :width]
            V = np.repeat(V, 2, axis=1)[:, :width]

        Yf = (Y - y_off) * y_scale
        Cb = (U - c_off) * c_scale
        Cr = (V - c_off) * c_scale
        r = Yf + (2 - 2 * Kr) * Cr
        b = Yf + (2 - 2 * Kb) * Cb
        g = Yf - (Kb * (2 - 2 * Kb) / Kg) * Cb - (Kr * (2 - 2 * Kr) / Kg) * Cr
        rgb = np.stack([r, g, b], axis=-1)
        np.clip(rgb, 0, 255, out=rgb)
        images.append(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
    return images


def Load_Video(args, num_frames=None):
    """Returns (list of RGB PIL frames, fps); reads only the first `num_frames` frames when given."""
    ext = os.path.splitext(args.video_path)[1].lower()
    if ext == ".yuv":
        if None in (args.yuv_width, args.yuv_height, args.yuv_fps):
            raise ValueError("Raw .yuv input needs --yuv_width, --yuv_height and --yuv_fps")
        frames = Read_Yuv(
            args.video_path,
            args.yuv_width,
            args.yuv_height,
            args.yuv_subsampling,
            args.yuv_bit_depth,
            num_frames,
            args.yuv_order,
            args.yuv_color_matrix,
            args.yuv_color_range,
        )
        fps = args.yuv_fps
    elif ext in (".mp4", ".avi", ".mov"):
        with VideoFileClip(args.video_path) as clip:
            fps = clip.fps
        frames = load_video(args.video_path)[:num_frames]
    else:
        raise ValueError(f"Unsupported video format: {ext}")
    return frames, fps
