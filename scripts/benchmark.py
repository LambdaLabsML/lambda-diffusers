import os
import csv
from click import command
import torch
from diffusers import StableDiffusionPipeline


def get_inference_pipeline(precision):
    """
    returns HuggingFace diffuser pipeline
    cf https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    note: could not download from CompVis/stable-diffusion-v1-4 (access restricted)
    """

    assert precision in ("half", "single"), "precision in ['half', 'single']"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=os.environ['ACCESS_TOKEN'],
        torch_dtype=torch.float32 if precision == "single" else torch.float16,
    )
    pipe = pipe.to("cuda")
    # Disable safety
    disable_safety = True
    if disable_safety:

        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety
    return pipe


def do_inference(pipe, n_samples):
    prompt = "a photo of an astronaut riding a horse on mars"
    with torch.autocast("cuda"):
        images = pipe(prompt=[prompt] * n_samples).images
    return images


def get_inference_time(pipe, n_samples, n_repeats):
    from torch.utils.benchmark import Timer

    timer = Timer(
        stmt="do_inference(pipe, n_samples)",
        setup="from __main__ import do_inference",
        globals={"pipe": pipe, "n_samples": n_samples},
    )
    profile_result = timer.timeit(
        n_repeats
    )  # benchmark.Timer performs 2 iterations for warmup
    return f"{profile_result.mean * 1000:.5f} ms"


def get_inference_memory(pipe, n_samples):
    if not torch.cuda.is_available():
        return 0
    prompt = "a photo of an astronaut riding a horse on mars"
    torch.cuda.empty_cache()
    with torch.autocast("cuda"):
        with torch.autocast("cuda"):
            images = pipe(prompt=[prompt] * n_samples).images
        mem = torch.cuda.memory_reserved()
    return "%.3gG" % (mem / 1e9)  # (GB)


def run_benchmark(n_repeats, n_samples, precision):
    """
    * n_repeats: nb datapoints for inference latency benchmark
    * n_samples: number of samples to generate (~ batch size)
    * precision: 'half' or 'single' (use fp16 or fp32 tensors)

    returns:
    dict like {'memory usage': '17.7G', 'latency': '8671.23817 ms'}
    """

    pipe = get_inference_pipeline(precision)

    logs = {
        "memory": get_inference_memory(pipe, n_samples),
        "latency": get_inference_time(pipe, n_samples, n_repeats),
    }
    print(f'n_samples: {n_samples}\tprecision: {precision}')
    print(logs,'\n')
    return logs


def get_device_description():
    """
    returns descriptor of cuda device such as
    'NVIDIA RTX A6000'
    """

    n_devices = torch.cuda.device_count()
    if n_devices < 1:
        return "CPU"
    else:
        return torch.cuda.get_device_name()


def run_benchmark_grid(grid, n_repeats, csv_fpath):
    """
    * grid : dict like
        {
            "n_samples": (1, 2),
            "precision": ("single", "half"),
        }
    * n_repeats: nb datapoints for inference latency benchmark
    * csv_path : location of benchmark output csv file
    """

    device = get_device_description()
    header = ["device", "precision", "n_samples", "latency", "memory"]

    with open(csv_fpath, "a") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for n_samples in grid["n_samples"]:
            for precision in grid["precision"]:
                new_log = run_benchmark(
                    n_repeats=n_repeats, n_samples=n_samples, precision=precision
                )
                latency = new_log["latency"]
                memory = new_log["memory"]
                new_row = [device, precision, n_samples, latency, memory]
                writer.writerow(new_row)


if __name__ == "__main__":

    grid = {"n_samples": (1, 2), "precision": ("single", "half")}

    run_benchmark_grid(
        grid,
        n_repeats=3,
        csv_fpath="/home/eole/Workspaces/lambda-diffusers/benchmark.csv",
    )
