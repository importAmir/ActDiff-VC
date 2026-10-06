"""CogVideoX tracking pipeline with bidirectional (first + last keyframe) conditioning (paper Sec. 4.2)."""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput, retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    CogVideoXImageToVideoPipeline,
    retrieve_latents,
)
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from DiffusionAsShader.models.cogvideox_tracking import CogVideoXTransformer3DModelTracking


class CogVideoXImageToVideoPipelineTrackingDual(CogVideoXImageToVideoPipeline):
    """DaS tracking pipeline whose image latent is [first keyframe, zeros, ..., last keyframe]."""

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

        if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking):
            raise ValueError(
                "The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking"
            )
        self.transformer = torch.compile(self.transformer)

    def Prepare_Latents_Dual(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 49,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """Like CogVideoXImageToVideoPipeline.prepare_latents, but the image latent holds both keyframes."""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Got {len(generator)} generators for an effective batch size of {batch_size}.")

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # CogVideoX 1.5 pads the latent length to the temporal patch size (unused by DaS)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        # Encode both first and last frames
        first_frame = first_frame.unsqueeze(2)  # [B, C, F, H, W]
        last_frame = last_frame.unsqueeze(2)  # [B, C, F, H, W]

        if isinstance(generator, list):
            first_frame_latents = [
                retrieve_latents(self.vae.encode(first_frame[i].unsqueeze(0)), generator[i])
                for i in range(batch_size)
            ]
            last_frame_latents = [
                retrieve_latents(self.vae.encode(last_frame[i].unsqueeze(0)), generator[i])
                for i in range(batch_size)
            ]
        else:
            first_frame_latents = [
                retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in first_frame
            ]
            last_frame_latents = [
                retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in last_frame
            ]

        first_frame_latents = (
            torch.cat(first_frame_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)
        )  # [B, F, C, H, W]
        last_frame_latents = (
            torch.cat(last_frame_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)
        )  # [B, F, C, H, W]

        if not self.vae.config.invert_scale_latents:
            first_frame_latents = self.vae_scaling_factor_image * first_frame_latents
            last_frame_latents = self.vae_scaling_factor_image * last_frame_latents
        else:
            # CogVideoX was trained without the scaling factor on image latents
            first_frame_latents = 1 / self.vae_scaling_factor_image * first_frame_latents
            last_frame_latents = 1 / self.vae_scaling_factor_image * last_frame_latents

        middle_padding_shape = (
            batch_size,
            num_frames - 2,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        middle_padding = torch.zeros(middle_padding_shape, device=device, dtype=dtype)

        # [first keyframe, zeros, ..., zeros, last keyframe]
        image_latents = torch.cat([first_frame_latents, middle_padding, last_frame_latents], dim=1)

        # Handle patch size requirements for CogVideoX1.5
        if self.transformer.config.patch_size_t is not None:
            current_length = image_latents.size(1)
            if current_length % self.transformer.config.patch_size_t != 0:
                padding_needed = self.transformer.config.patch_size_t - (
                    current_length % self.transformer.config.patch_size_t
                )
                extra_padding = torch.zeros(
                    (
                        batch_size,
                        padding_needed,
                        num_channels_latents,
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                    ),
                    device=device,
                    dtype=dtype,
                )
                image_latents = torch.cat([image_latents, extra_padding], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    @torch.no_grad()
    def __call__(
        self,
        first_frame: Union[torch.Tensor, Image.Image],
        last_frame: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
        tracking_image_first: Optional[torch.Tensor] = None,
        tracking_image_last: Optional[torch.Tensor] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """Same as the DaS tracking pipeline call, with `first_frame` / `last_frame` keyframes."""
        # 1. Check inputs and set default values
        self.check_inputs(
            first_frame,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        if timesteps is not None:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = int(timesteps.numel())
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents with dual-frame conditioning
        first_frame_processed = self.video_processor.preprocess(first_frame, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        last_frame_processed = self.video_processor.preprocess(last_frame, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels

        latents, image_latents = self.Prepare_Latents_Dual(
            first_frame_processed,
            last_frame_processed,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        del first_frame_processed, last_frame_processed

        # Handle tracking maps if provided
        if tracking_maps is not None and tracking_image_first is not None:
            tracking_image_first_processed = self.video_processor.preprocess(
                tracking_image_first, height=height, width=width
            ).to(device, dtype=prompt_embeds.dtype)
            tracking_image_last_processed = self.video_processor.preprocess(
                tracking_image_last, height=height, width=width
            ).to(device, dtype=prompt_embeds.dtype)

            _, tracking_image_latents = self.Prepare_Latents_Dual(
                tracking_image_first_processed,
                tracking_image_last_processed,
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None,
            )
            del tracking_image_first_processed, tracking_image_last_processed
        else:
            tracking_image_latents = None

        # 6. Extra scheduler-step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        effective_steps = int(len(timesteps))
        num_warmup_steps = max(effective_steps - effective_steps * self.scheduler.order, 0)

        with self.progress_bar(total=effective_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = (
                    torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                )

                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                del latent_image_input

                # Handle tracking maps
                if tracking_maps is not None and tracking_image_latents is not None:
                    latents_tracking_image = (
                        torch.cat([tracking_image_latents] * 2)
                        if do_classifier_free_guidance
                        else tracking_image_latents
                    )

                    tracking_maps_input = (
                        torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                    )
                    tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
                    del latents_tracking_image
                else:
                    tracking_maps_input = None

                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                self.transformer.to(dtype=latent_model_input.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=tracking_maps_input,
                    return_dict=False,
                )[0]
                del latent_model_input
                if tracking_maps_input is not None:
                    del tracking_maps_input
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi * ((effective_steps - t.item()) / max(len(timesteps), 1)) ** 5.0
                            )
                        )
                        / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
