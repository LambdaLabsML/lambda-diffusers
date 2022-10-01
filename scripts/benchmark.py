import os
import subprocess
import multiprocessing
import pathlib
import csv
import torch
from diffusers import StableDiffusionPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_inference_steps = 50
n_repeats = 3

def get_inference_pipeline(precision):
    """
    returns HuggingFace diffuser pipeline
    cf https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    note: could not download from CompVis/stable-diffusion-v1-4 (access restricted)
    """

    assert precision in ("half", "single"), "precision in ['half', 'single']"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=os.environ["ACCESS_TOKEN"],
        torch_dtype=torch.float32 if precision == "single" else torch.float16,
    )

    pipe = pipe.to(device)
    # Disable safety
    disable_safety = True
    if disable_safety:

        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety
    return pipe


def do_inference(pipe, n_samples, precision):
    prompt = "a photo of an astronaut riding a horse on mars"
    if precision == "half":
        with torch.autocast(device.type):
            images = pipe(prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps).images
    else:
        images = pipe(prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps).images
    return images


def get_inference_time(pipe, n_samples, n_repeats, precision):
    from torch.utils.benchmark import Timer
    timer = Timer(
        stmt="do_inference(pipe, n_samples, precision)",
        setup="from __main__ import do_inference",
        globals={"pipe": pipe, "n_samples": n_samples, "precision": precision},
        num_threads=multiprocessing.cpu_count()
    )
    profile_result = timer.timeit(
        n_repeats
    )  # benchmark.Timer performs 2 iterations for warmup
    return f"{profile_result.mean * 1000:.5f} ms"


def get_inference_memory(pipe, n_samples, precision):
    if not torch.cuda.is_available():
        return 0
    prompt = "a photo of an astronaut riding a horse on mars"
    
    torch.cuda.empty_cache()
    if precision == "half":
        with torch.autocast(device.type):
            images = pipe(prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps).images
    else:
        images = pipe(prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps).images
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
        "memory": "0.0G" if device.type=="cpu" else get_inference_memory(pipe, n_samples, precision),
        "latency": get_inference_time(pipe, n_samples, n_repeats, precision),
    }
    print(f"n_samples: {n_samples}\tprecision: {precision}")
    print(logs, "\n")
    return logs


def get_device_description():
    """
    returns descriptor of cuda device such as
    'NVIDIA RTX A6000'
    """
    if device.type == "cpu":
        name = subprocess.check_output(
            "grep -m 1 'model name' /proc/cpuinfo", 
            shell=True
        ).decode("utf-8") 
        name = " ".join(name.split(" ")[2:]).strip()
        return name
    else:
        return torch.cuda.get_device_name()


def run_benchmark_grid(grid, n_repeats):
    """
    * grid : dict like
        {
            "n_samples": (1, 2),
            "precision": ("single", "half"),
        }
    * n_repeats: nb datapoints for inference latency benchmark
    """

    csv_fpath = pathlib.Path(__file__).parent.parent / "benchmark.csv"
    # create benchmark.csv if not exists
    if not os.path.isfile(csv_fpath):
        header = ["device", "precision", "n_samples", "latency", "memory"]
        with open(csv_fpath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # append new benchmark results to it if benchmark.csv already exists
    with open(csv_fpath, "a") as f:
        writer = csv.writer(f)
        device_desc = get_device_description()
        for n_samples in grid["n_samples"]:
            for precision in grid["precision"]:
                new_log = run_benchmark(
                    n_repeats=n_repeats, n_samples=n_samples, precision=precision
                )
                latency = new_log["latency"]
                memory = new_log["memory"]
                new_row = [device_desc, precision, n_samples, latency, memory]
                writer.writerow(new_row)


if __name__ == "__main__":
    # Only use single precision for cpu because "LayerNormKernelImpl" not implemented for 'Half' on cpu, 
    # Remove autocast won't help. Ref:
    # https://github.com/CompVis/stable-diffusion/issues/307
    # https://github.com/CompVis/stable-diffusion/issues/307
    grid = {"n_samples": (1, 2), "precision": ("single", "half") if device.type != "cpu" else ("single",)}
    run_benchmark_grid(grid, n_repeats=n_repeats)
