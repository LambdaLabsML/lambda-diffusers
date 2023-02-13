import inspect
from typing import List, Optional, Union

import numpy as np

from transformers import CLIPModel, CLIPFeatureExtractor, CLIPTokenizer, CLIPVisionModel

from diffusers.onnx_utils import OnnxRuntimeModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import torch
import PIL
import warnings


class StableDiffusionImageEmbedOnnxPipeline(DiffusionPipeline):
    vae_decoder: OnnxRuntimeModel
    unet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: StableDiffusionSafetyChecker

    def __init__(
        self,
        vae_decoder: OnnxRuntimeModel,
        unet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        scheduler = scheduler.set_format("np")
        self.register_modules(
            vae_decoder=vae_decoder,
            unet=unet,
            scheduler=scheduler,
        )
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    def __call__(
        self,
        input_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if isinstance(input_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(input_image, list):
            batch_size = len(input_image)
        else:
            raise ValueError(f"`input_image` has to be of type `str` or `list` but is {type(input_image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if not isinstance(input_image, torch.FloatTensor):
            input_image = self.feature_extractor(images=input_image, return_tensors="pt").to(self.device)

        image_encoder_output = self.image_encoder.vision_model(input_image["pixel_values"])[1]
        image_embeddings = self.image_encoder.visual_projection(image_encoder_output)
        image_embeddings = image_embeddings.unsqueeze(1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        
        image_embeddings = image_embeddings.cpu().detach().numpy()
        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = np.random.randn(*latents_shape).astype(np.float32)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")


        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                sample=latent_model_input, timestep=np.array([t]), encoder_hidden_states=image_embeddings
            )
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latent_sample=latents)[0]

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        # run safety checker
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
