"""Renders sparse 3D tracks as the coloured-square "tracking video" that conditions Diffusion-as-Shader."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

RECT_SIZE = 480 // 70  # half-width of each square, as in DaS


def Render_Tracking_Video(tracks, height, width, fps):
    """tracks: [T, N, 3] (x, y, depth). Returns float [T', 3, H, W] in [0, 1].

    Each point is a filled square whose colour encodes its first-frame (x, y, 1/depth); nearer points
    are drawn last. Points at x == 0 or y == 0 are skipped, as in the DaS renderer.
    """
    tracks = tracks.detach().cpu().numpy() if torch.is_tensor(tracks) else np.asarray(tracks)
    T, N, _ = tracks.shape

    norm_x = plt.Normalize(tracks[0, :, 0].min(), tracks[0, :, 0].max())
    norm_y = plt.Normalize(tracks[0, :, 1].min(), tracks[0, :, 1].max())
    norm_z = plt.Normalize(*np.percentile(1 / tracks[0, :, 2], [2, 98]))
    colors = np.zeros((T, N, 3))
    for n in range(N):
        color = (
            np.array([norm_x(tracks[0, n, 0]), norm_y(tracks[0, n, 1]), norm_z(1 / tracks[0, n, 2])])[None]
            * 255
        )
        colors[:, n] = np.repeat(color, T, axis=0)

    frames = []
    for t in range(T):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        points = [
            (i, tracks[t, i, 0], tracks[t, i, 1], tracks[t, i, 2])
            for i in range(N)
            if tracks[t, i, 0] != 0 and tracks[t, i, 1] != 0
        ]
        points.sort(key=lambda p: p[3], reverse=True)
        for i, x, y, _ in points:
            top_left = (int(x - RECT_SIZE), int(y - RECT_SIZE / 1.5))
            bottom_right = (int(x + RECT_SIZE), int(y + RECT_SIZE / 1.5))
            cv2.rectangle(frame, top_left, bottom_right, colors[t, i].tolist(), thickness=-1)
        frames.append(frame)

    # The paper's results passed the frames through moviepy, which repeats the last frame once
    # (49 -> 50 frames at every frame rate we use). Kept for exact reproducibility.
    frames = np.array(list(ImageSequenceClip(frames, fps=fps).iter_frames())) / 255.0
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float()
