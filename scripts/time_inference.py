from diffusers import StableDiffusionPipeline
import torch

def setup_inference_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-pokemon-diffusers", torch_dtype=torch.float16)  
    pipe = pipe.to("cuda")

    # Disable safety
    disable_safety = True
    if disable_safety:
        def null_safety(images, **kwargs):
            return images, False
        pipe.safety_checker = null_safety
    
    return pipe


def do_inference(pipe, n_samples=1, guidance_scale=3.0):
    prompt = "Lincoln"
    with torch.autocast("cuda"):
        images = pipe(n_samples*[prompt], guidance_scale=guidance_scale).images
    return images


def get_inference_timing():
    import timeit
    setup="from __main__ import do_inference,setup_inference_pipeline; pipe=setup_inference_pipeline()"
    time_loop="do_inference(pipe)"
    min_loop_runtime = min(timeit.repeat(time_loop, setup=setup, number=1, repeat=3))
    return min_loop_runtime


if __name__ == "__main__":

    print('Best time:', get_inference_timing())