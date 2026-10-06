"""ActDiff-VC decoder: keyframe decoding and trajectory-conditioned diffusion between keyframes."""
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from actdiff.codecs.image_codec import KeyframeCodec
from actdiff.codecs.trajectory_codec import Decompress_Trajectories, Read_Trajectory_Length
from actdiff.config import DAS_NUM_FRAMES
from actdiff.diffusion.pipeline import DiffusionDecoder
from actdiff.diffusion.tracking_video import Render_Tracking_Video
from actdiff.utils.video_io import Write_Video


def To_Image(frame):
    """[3, H, W] tensor in [0, 1] -> RGB PIL image."""
    return Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))


class Decoder:
    def __init__(self, args, depth_estimator):
        self.args = args
        self.depth_estimator = depth_estimator
        self.diffusion = DiffusionDecoder(
            args.das_checkpoint, args.device, dual_conditioning=not args.first_keyframe_only
        )
        self.keyframe_codec = KeyframeCodec(args)
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def Estimate_Depth(self, image):
        """[3, H, W] in [0, 1] -> relative depth [1, H, W]."""
        image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return self.to_tensor(self.depth_estimator(image)[0])

    @torch.no_grad()
    def Tracking_Video(self, tracks_2d, keyframe):
        """Lift decoded 2D tracks [T, N, 2] to 3D with the keyframe's depth and render the DaS condition."""
        H, W = keyframe.shape[1], keyframe.shape[2]
        depth = self.Estimate_Depth(keyframe).squeeze(0).numpy()
        xy0 = np.rint(tracks_2d[0]).astype(int)
        xy0[:, 0] = np.clip(xy0[:, 0], 0, W - 1)
        xy0[:, 1] = np.clip(xy0[:, 1], 0, H - 1)
        z0 = depth[xy0[:, 1], xy0[:, 0]]
        z0[z0 <= 1e-3] = 1e-3  # the renderer uses 1 / depth

        T = tracks_2d.shape[0]
        z = torch.from_numpy(z0.astype(np.float32))[None, :].repeat(T, 1)
        tracks = torch.cat([torch.from_numpy(tracks_2d).float(), z.unsqueeze(-1)], dim=-1)
        return Render_Tracking_Video(tracks, H, W, self.args.fps)

    @torch.no_grad()
    def Decode_Chunk(self, chunk_idx):
        """Returns the frames of GOP `chunk_idx` (both keyframes included) as RGB PIL images."""
        args = self.args
        first = self.keyframe_codec.Decompress(args.bitstream_dir, f"keyframe_{chunk_idx}.bin")
        last = self.keyframe_codec.Decompress(args.bitstream_dir, f"keyframe_{chunk_idx + 1}.bin")
        first_image, last_image = To_Image(first), To_Image(last)
        if args.save_intermediate:
            keyframe_dir = os.path.join(args.output_dir, "intermediate", "keyframes")
            os.makedirs(keyframe_dir, exist_ok=True)
            if chunk_idx == 0:
                first_image.save(os.path.join(keyframe_dir, "keyframe_000.png"))
            last_image.save(os.path.join(keyframe_dir, f"keyframe_{chunk_idx + 1:03d}.png"))

        trajectory_path = os.path.join(args.bitstream_dir, f"chunk_{chunk_idx}_trajectories.bin")
        length = Read_Trajectory_Length(trajectory_path)
        if length == 2:
            return [first_image, last_image]

        # Trajectories are resampled to the generator's fixed length; the output is resampled back.
        tracks_2d = Decompress_Trajectories(trajectory_path, target_T=DAS_NUM_FRAMES)
        if tracks_2d.shape[1] == 0 or np.allclose(tracks_2d, 0.0):  # no motion: hold the first keyframe
            frames = [first_image.copy() for _ in range(max(2, length))]
            frames[-1] = last_image
        else:
            tracking_video = self.Tracking_Video(tracks_2d, first)
            frames = self.diffusion.Generate(tracking_video, first.cpu(), last.cpu(), args.diffusion_steps)
            frames[0], frames[-1] = first_image, last_image
            if length != len(frames):
                idx = np.clip(
                    np.linspace(0, len(frames) - 1, num=length).round().astype(int), 0, len(frames) - 1
                )
                frames = [frames[i] for i in idx]
            if args.save_intermediate:
                Write_Video(
                    tracking_video[:DAS_NUM_FRAMES].permute(0, 2, 3, 1).mul(255).round().byte().numpy(),
                    args.fps,
                    os.path.join(args.output_dir, "intermediate", "tracking", f"gop_{chunk_idx:03d}.mp4"),
                )

        if args.save_intermediate:
            Write_Video(
                frames,
                args.fps,
                os.path.join(args.output_dir, "intermediate", "gops", f"gop_{chunk_idx:03d}.mp4"),
            )
        return frames

    @torch.no_grad()
    def Decode_Video(self):
        """Decode every GOP in the stream folder; consecutive GOPs share their boundary keyframe."""
        chunk_ids = sorted(
            int(name[len("chunk_") : -len("_trajectories.bin")])
            for name in os.listdir(self.args.bitstream_dir)
            if name.startswith("chunk_") and name.endswith("_trajectories.bin")
        )
        video = []
        for chunk_idx in chunk_ids:
            frames = [np.array(f) for f in self.Decode_Chunk(chunk_idx)]
            video.extend(frames if not video else frames[1:])
        return video
