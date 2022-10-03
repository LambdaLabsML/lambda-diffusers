import os
from platform import mac_ver
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
prompt = "a photo of an astronaut riding a horse on mars"
output_folder = "_".join(prompt.split(" "))
os.makedirs(output_folder, exist_ok=True)

num_images = 1
width = 512
height = 512
min_inference_steps = 10
max_inference_steps = 100

list_ssim = []
list_nmse = []
list_psnr = []

# Create piplines for single and half-precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    use_auth_token=True,
    torch_dtype=torch.float32)
pipe = pipe.to(device)

pipe_half = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True)
pipe_half = pipe_half.to(device)

# Generate latent vectors
generator = torch.Generator(device=device)
latents = None
seeds = []
for _ in range(num_images):
    # Get a new random seed, store it and use it as the generator state
    seed = generator.seed()
    seeds.append(seed)
    generator = generator.manual_seed(seed)
    
    image_latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )
    latents = image_latents if latents is None else torch.cat((latents, image_latents))

for num_inference_steps in range(min_inference_steps, max_inference_steps, 5):
    # Inference with single and half-precision

    torch.cuda.empty_cache()
    images = pipe(
            [prompt] * num_images,
            guidance_scale=7.5,
            latents = latents,
            num_inference_steps = num_inference_steps
    )["sample"]

    torch.cuda.empty_cache()
    with torch.autocast(device):
        images_half = pipe_half(
                [prompt] * num_images,
                guidance_scale=7.5,
                latents = latents,
                num_inference_steps = num_inference_steps
        )["sample"]

    m_ssim = []
    m_nmse = []
    m_psnr = []

    for idx, (image, image_half) in enumerate(zip(images, images_half)):
        # Need to convert to float because uint8 can't store negative value
        np_image = np.float32(np.asarray(image)) / 255.0
        np_image_half = np.float32(np.asarray(image_half)) / 255.0
        np_image_diff = np.absolute(np.float32(np.asarray(image)) - np.float32(np.asarray(image_half)))

        # Compute quantitative metrics
        m_ssim.append(ssim(np_image, np_image_half, channel_axis=2))
        m_nmse.append(nmse(np_image, np_image_half))
        m_psnr.append(psnr(np_image, np_image_half))
        im_diff = Image.fromarray(np.uint8(np_image_diff))

        # Compose results in a single output image
        dst = Image.new('RGB', (image.width + image_half.width + im_diff.width, image.height))
        dst.paste(image, (0, 0))
        dst.paste(image_half, (image.width, 0))
        dst.paste(im_diff, (image.width + image_half.width, 0))
        I1 = ImageDraw.Draw(dst)
        font = ImageFont.truetype('../docs/pictures/FreeMono.ttf', 16)
        I1.text((32, 32), "Single Precision", font=font, fill=(255, 255, 255))
        I1.text((image.width + 32, 32), "Half Precision", font=font, fill=(255, 255, 255))
        I1.text((image.width + image_half.width + 32, 32), "Step " + str(num_inference_steps), font=font, fill=(255, 255, 255))
        dst.save(output_folder + "/" + str(idx) + "_" + str(num_inference_steps) + ".png")

    list_ssim.append(sum(m_ssim) / len(m_ssim))
    list_nmse.append(sum(m_nmse) / len(m_nmse))
    list_psnr.append(sum(m_psnr) / len(m_psnr))

print("ssim: ")
print(list_ssim)

print("nmse: ")
print(list_nmse)

print("psnr: ")
print(list_psnr)
