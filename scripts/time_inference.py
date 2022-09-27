

from diffusers import StableDiffusionPipeline
import torch


def setup_inference_pipeline(dtype=torch.float16):
    '''
    returns HuggingFace diffuser pipeline
    cf https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    note: could not download from CompVis/stable-diffusion-v1-4 (access restricted)
    '''
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "lambdalabs/sd-pokemon-diffusers",
        torch_dtype=dtype
    )
    pipe = pipe.to("cuda")
    # Disable safety
    disable_safety = True
    if disable_safety:
        def null_safety(images, **kwargs):
            return images, False
        pipe.safety_checker = null_safety
    return pipe


def do_inference(pipe):
    prompt = "a photo of an astronaut riding a horse on mars"
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
        # n_samples=1, guidance_scale=3.0
        # images = pipe(n_samples*[prompt], guidance_scale=guidance_scale).images # -> batches ?
        # image.save("astronaut_rides_horse.png")
    return image


def get_inference_timing(n_repeats=2):
    from torch.utils.benchmark import Timer
    pipe = setup_inference_pipeline()
    timer = Timer(
        stmt="do_inference(pipe)",
        setup="from __main__ import do_inference",
        globals={"pipe": pipe},
    )
    profile_result = timer.timeit(n_repeats) # benchmark.Timer seems to performs 2 iterations for warmup
    return f"Latency: {profile_result.mean * 1000:.5f} ms"



if __name__ == "__main__":

    r = get_inference_timing()
    print(r)
