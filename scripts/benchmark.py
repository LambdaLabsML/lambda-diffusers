import torch
from diffusers import StableDiffusionPipeline

def get_inference_pipeline(dtype=torch.float16):
    '''
    returns HuggingFace diffuser pipeline
    cf https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    note: could not download from CompVis/stable-diffusion-v1-4 (access restricted)
    '''
    pipe = StableDiffusionPipeline.from_pretrained(
        "lambdalabs/sd-pokemon-diffusers",
        # TODO: get vanilla stable diffusion pretrained model (?)
        #"CompVis/stable-diffusion-v1-4",
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

        # TODO: check if batch size can be implemented in benchmark
        # n_samples=1, guidance_scale=3.0
        # images = pipe(n_samples*[prompt], guidance_scale=guidance_scale).images # -> batches ?
        # image.save("astronaut_rides_horse.png")
    return image

def get_inference_time(pipe=get_inference_pipeline(), n_repeats=2):
    from torch.utils.benchmark import Timer
    timer = Timer(
        stmt="do_inference(pipe)",
        setup="from __main__ import do_inference",
        globals={"pipe": pipe},
    )
    profile_result = timer.timeit(n_repeats) # benchmark.Timer seems to performs 2 iterations for warmup
    return f"{profile_result.mean * 1000:.5f} ms"

def get_inference_memory(pipe=get_inference_pipeline()):
    if not torch.cuda.is_available():
        return 0
    prompt = "a photo of an astronaut riding a horse on mars"
    torch.cuda.empty_cache()
    with torch.autocast("cuda"):
        _ = pipe(prompt).images[0]
        mem = torch.cuda.memory_reserved()
    return '%.3gG' % (mem / 1E9)  # (GB)

def run_benchmark(pipe=get_inference_pipeline(), n_repeats=3):
    logs = {
        'memory usage' : get_inference_memory(pipe), # reserved memory is constant, no need for n_repeats
        'latency' : get_inference_time(pipe, n_repeats) 
    }
    print(logs)
    return logs


if __name__ == "__main__":

    run_benchmark()