# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from models.worldstereo import WorldStereoRefSModel
from src.vae_utils import keyframe_vae_encode, keyframe_vae_decode

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class RefKFDMDGeneratorPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: UMT5EncoderModel,
            image_encoder: CLIPVisionModel,
            image_processor: CLIPImageProcessor,
            transformer: WorldStereoRefSModel,
            vae: AutoencoderKLWan,
            scheduler,
            device,
            vae_compile: bool = False,
            vae_compile_mode: str = "max-autotune"
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer=transformer
        )
        # self.transformer = transformer
        self.device_ = device
        
        # VAE compile settings
        self.vae_compile = vae_compile
        self.vae_compile_mode = vae_compile_mode

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _execution_device(self):
        return self.device_

    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_videos_per_prompt: int = 1,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device()
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
            self,
            image: PipelineImageInput,
            device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device()
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            num_videos_per_prompt: int = 1,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device()

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
            self,
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
                not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
            self,
            image: PipelineImageInput,
            batch_size: int,
            num_channels_latents: int = 16,
            height: int = 480,
            width: int = 832,
            num_frames: int = 81,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            latent_cond_mode="full_vae"
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 5.1 Calculate latent dimensions
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 5.2 Generate random noise latents
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # 5.3 Prepare video_condition tensor (optimized: 1 real frame + 8 zero frames = 9 frames -> 3 latents)
        image = image.unsqueeze(2)  # (b, c, 1, h, w)

        # 5.4 Prepare normalization params
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if latent_cond_mode == "full_vae":
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
            # Explicitly cast to VAE weight dtype to avoid input/bias dtype mismatch under autocast
            video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

            if isinstance(generator, list):
                latent_condition = [
                    retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
                latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latent_condition = (latent_condition - latents_mean) * latents_std
        elif latent_cond_mode == "first_frame_only":
            # Only encode first frame, then concatenate zeros in latent space
            # Explicitly cast to VAE weight dtype to avoid input/bias dtype mismatch under autocast
            first_frame = image.to(device=device, dtype=self.vae.dtype)  # [b, c, 1, h, w]
            first_frame_latent = retrieve_latents(self.vae.encode(first_frame), sample_mode="argmax")
            first_frame_latent = first_frame_latent.repeat(batch_size, 1, 1, 1, 1)
            first_frame_latent = (first_frame_latent - latents_mean) * latents_std
            # Concatenate zeros in latent space for remaining frames
            zero_image = image.new_zeros(image.shape[0], image.shape[1], 1, height, width).to(device=device, dtype=self.vae.dtype)
            zero_latents = retrieve_latents(self.vae.encode(zero_image), sample_mode="argmax")
            zero_latents = zero_latents.repeat(batch_size, 1, num_latent_frames - 1, 1, 1)
            zero_latents = (zero_latents - latents_mean) * latents_std
            latent_condition = torch.cat([first_frame_latent, zero_latents], dim=2)
        else:
            raise NotImplementedError(f"latent_cond_mode {latent_cond_mode} not implemented")

        # 5.8 Prepare mask
        vae_t = self.vae_scale_factor_temporal
        mask_lat_size = torch.zeros(
            batch_size, vae_t, num_latent_frames, latent_height, latent_width,
            device=latent_condition.device, dtype=latent_condition.dtype
        )
        # Only the first frame (across all temporal positions) should be 1
        mask_lat_size[:, :, 0, :, :] = 1.0

        # 5.9 Concatenate mask and latent_condition
        result = torch.concat([mask_lat_size, latent_condition], dim=1)

        return latents, result

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    # @torch.no_grad()
    # we need to update generator through the full generator inference
    # NOTE: generator does not need CFG
    def __call__(
            self,
            image: PipelineImageInput,
            render_video: torch.Tensor,
            render_mask: torch.Tensor,
            camera_embedding: torch.Tensor = None,
            extrinsics: torch.Tensor = None,
            intrinsics: torch.Tensor = None,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            # new params for reference
            reference_video=None,
            ref_index=None,
            camera_qt=None,
            camera_qt_ref=None,
            latent_cond_mode="full_vae",
            # new params end
            mode: str = "train",
            height: int = 480,
            width: int = 768,
            num_frames: int = 81,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            image_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            **kwargs
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct 0%
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = 1.0
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device()

        # 2. Define call parameters 0%
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt 2-3% 0.3-0.4s
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            with torch.no_grad():
                image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps 0%
        if mode == "train":  # random selection during training, return all steps during inference
            timesteps = self.scheduler.gen_train_timesteps()
        else:
            timesteps = self.scheduler.gen_test_timesteps()

        # 5. Prepare latent variables 31% 4.5-4.8s -> 0.5s -> 0.1s (next version)
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            latents, condition = self.prepare_latents(
                image,
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
                latent_cond_mode
            )

        ### 5.5 Prepare keyframe render_latent ###
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # Explicitly cast to VAE weight dtype to avoid input/bias dtype mismatch under autocast
            render_video = render_video.to(dtype=self.vae.dtype)
            if render_video.shape[2] == num_frames // 4 + 1:  # process with divided 21 frames
                render_latent = keyframe_vae_encode(self.vae, render_video, rescale=True,
                                                    use_compile=self.vae_compile, compile_mode=self.vae_compile_mode)
            else:
                render_latent = retrieve_latents(self.vae.encode(render_video), sample_mode="argmax")
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(render_latent.device, render_latent.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                    render_latent.device, render_latent.dtype
                )
                render_latent = (render_latent - latents_mean) * latents_std

        # Prepare reference_latent
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            if reference_video is not None:
                reference_latent = keyframe_vae_encode(self.vae, reference_video, rescale=True,
                                                       use_compile=self.vae_compile, compile_mode=self.vae_compile_mode)
                reference_latent_mask = torch.ones_like(reference_latent[:, :4]).to(reference_latent.dtype).to(reference_latent.device)
                reference_latent = torch.cat([reference_latent, reference_latent_mask, reference_latent], dim=1)
            else:
                reference_latent = None

        # 6. Denoising loop 55-60% 8-9s
        num_warmup_steps = 0
        self._num_timesteps = len(timesteps)

        # Pre-convert condition to transformer dtype (avoid repeated conversion in loop)
        condition = condition.to(transformer_dtype)
        
        # Pre-expand all timesteps (avoid repeated expand in loop)
        expanded_timesteps = [t.expand(latents.shape[0]).to(device) for t in timesteps]

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # Use pre-converted condition and concatenate with latents
                latent_model_input = torch.cat([latents.to(transformer_dtype), condition], dim=1)
                timestep = expanded_timesteps[i]

                # Only allow gradient backprop on the last step in train mode
                if mode != "train" or (mode == "train" and i < len(timesteps) - 1):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            render_latent=render_latent,
                            point_map_latent=None,
                            point_map_ref_latent=None,
                            reference_latent=reference_latent,
                            render_mask=render_mask,
                            camera_embedding=camera_embedding,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            ref_index=ref_index,
                            camera_qt=camera_qt,
                            camera_qt_ref=camera_qt_ref,
                        )[0]
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            render_latent=render_latent,
                            point_map_latent=None,
                            point_map_ref_latent=None,
                            reference_latent=reference_latent,
                            render_mask=render_mask,
                            camera_embedding=camera_embedding,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            ref_index=ref_index,
                            camera_qt=camera_qt,
                            camera_qt_ref=camera_qt_ref,
                        )[0]

                # compute the previous noisy sample x_t -> x_t-1, if t is the last step of generator, pred x_0
                latents = self.scheduler.step(noise_pred, latents, timesteps, i)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps:
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7. decode latent 2.5% 0.38s
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                video = keyframe_vae_decode(self.vae, latents, rescale=True)  # (b, c, f, h, w)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    # only used for training
    def fast_infer(
            self,
            random_noise_latent,
            condition_latent,
            render_latent,
            render_mask,
            prompt_embeds,
            image_embeds,
            camera_embedding=None,
            # new params for reference
            reference_latent=None,
            camera_qt=None,
            camera_qt_ref=None,
            ref_index=None
    ):

        timesteps = self.scheduler.gen_train_timesteps()
        latents = random_noise_latent
        device = self._execution_device()

        # print("[DEBUG] FAST INFER Timesteps:", timesteps)

        pred_latents = None

        for i, t in enumerate(timesteps):

            latent_model_input = torch.cat([latents, condition_latent], dim=1)
            timestep = t.expand(latents.shape[0]).to(device)

            # Only the last step allows gradient backpropagation
            if i < len(timesteps) - 1:
                with torch.no_grad():
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        render_latent=render_latent,
                        point_map_latent=None,
                        point_map_ref_latent=None,
                        reference_latent=reference_latent,
                        render_mask=render_mask,
                        camera_embedding=camera_embedding,
                        camera_qt=camera_qt,
                        camera_qt_ref=camera_qt_ref,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        return_dict=False,
                        ref_index=ref_index,
                    )[0]
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, latents, timesteps, i)
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    render_latent=render_latent,
                    point_map_latent=None,
                    point_map_ref_latent=None,
                    reference_latent=reference_latent,
                    render_mask=render_mask,
                    camera_embedding=camera_embedding,
                    camera_qt=camera_qt,
                    camera_qt_ref=camera_qt_ref,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                    ref_index=ref_index,
                )[0]
                # compute the previous noisy sample x_t -> x_t-1, if t is the last step of generator, pred x_0
                pred_latents = self.scheduler.step(noise_pred, latents, timesteps, i)

        return pred_latents