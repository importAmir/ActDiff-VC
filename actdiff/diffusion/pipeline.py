"""Diffusion-as-Shader generator conditioned on keyframes and a rendered tracking video."""
import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from DiffusionAsShader.models.cogvideox_tracking import (
    CogVideoXImageToVideoPipelineTracking,
    CogVideoXTransformer3DModelTracking,
)
from actdiff.config import DAS_NUM_FRAMES
from actdiff.diffusion.cogvideox_tracking_dual import CogVideoXImageToVideoPipelineTrackingDual

NEGATIVE_PROMPT = (
    "The video is not of a high quality, it has a low resolution. Watermark present in each frame. "
    "The background is solid. Strange body and strange trajectory. Distortion."
)


def To_Pil(image):
    """[3, H, W] tensor in [0, 1] -> RGB PIL image."""
    array = image.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8))


class DiffusionDecoder:
    def __init__(self, model_path, device, dual_conditioning=True, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.dual_conditioning = dual_conditioning

        pipeline_cls = (
            CogVideoXImageToVideoPipelineTrackingDual
            if dual_conditioning
            else CogVideoXImageToVideoPipelineTracking
        )
        self.pipe = pipeline_cls(
            vae=AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae"),
            text_encoder=T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder"),
            tokenizer=T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer"),
            transformer=CogVideoXTransformer3DModelTracking.from_pretrained(
                model_path, subfolder="transformer"
            ),
            scheduler=CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler"),
        )
        self.pipe.transformer.eval()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.to(device, dtype=dtype)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.transformer.gradient_checkpointing = False

    @torch.no_grad()
    def Generate(
        self, tracking_video, first_frame, last_frame, num_inference_steps, guidance_scale=6.0, seed=42
    ):
        """tracking_video: [T, 3, H, W]; keyframes: [3, H, W] in [0, 1]. Returns 49 PIL frames."""
        tracking = tracking_video.float().to(device=self.device, dtype=self.dtype)
        height, width = tracking.shape[2], tracking.shape[3]
        # Seeded so that decoding is reproducible (the generation below is seeded too).
        tracking_generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.pipe.vae.encode(tracking.unsqueeze(0).permute(0, 2, 1, 3, 4)).latent_dist
        latents = latents.sample(generator=tracking_generator)
        tracking_maps = (latents * self.pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4)

        kwargs = dict(
            prompt="",
            negative_prompt=NEGATIVE_PROMPT,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=DAS_NUM_FRAMES,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            tracking_maps=tracking_maps,
            height=height,
            width=width,
        )
        if not self.dual_conditioning:
            return self.pipe(image=To_Pil(first_frame), tracking_image=tracking[0:1], **kwargs).frames[0]

        return self.pipe(
            first_frame=To_Pil(first_frame),
            last_frame=To_Pil(last_frame),
            tracking_image_first=tracking[0:1],
            tracking_image_last=tracking[-1:],
            **kwargs,
        ).frames[0]
