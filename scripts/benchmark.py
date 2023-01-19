import os
import subprocess
import multiprocessing
import argparse
import pathlib
import csv
from contextlib import nullcontext
import itertools
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionOnnxPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prompt = "a photo of an astronaut riding a horse on mars"

def make_bool(yes_or_no):
    if yes_or_no.lower() == "yes":
        return True
    elif yes_or_no.lower() == "no":
        return False
    else:
        raise ValueError(f"unrecognised input {yes_or_no}")

def get_inference_pipeline(precision, backend):
    """
    returns HuggingFace diffuser pipeline
    cf https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    """

    assert precision in ("half", "single"), "precision in ['half', 'single']"
    assert backend in ("pytorch", "onnx"), "backend in ['pytorch', 'onnx']"

    if backend == "pytorch":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="main" if precision == "single" else "fp16",
            torch_dtype=torch.float32 if precision == "single" else torch.float16,
        )
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionOnnxPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=os.environ["ACCESS_TOKEN"],
            revision="onnx",
            provider="CPUExecutionProvider"
            if device.type == "cpu"
            else "CUDAExecutionProvider",
            torch_dtype=torch.float32 if precision == "single" else torch.float16,
        )

    # Disable safety
    disable_safety = True
    if disable_safety:

        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety
    return pipe


def do_inference(pipe, n_samples, use_autocast, num_inference_steps):
    torch.cuda.empty_cache()
    context = (
        autocast if (device.type == "cuda" and use_autocast) else nullcontext
    )
    with context("cuda"):
        images = pipe(
            prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps
        ).images

    return images


def get_inference_time(
    pipe, n_samples, n_repeats, use_autocast, num_inference_steps
):
    from torch.utils.benchmark import Timer

    timer = Timer(
        stmt="do_inference(pipe, n_samples, use_autocast, num_inference_steps)",
        setup="from __main__ import do_inference",
        globals={
            "pipe": pipe,
            "n_samples": n_samples,
            "use_autocast": use_autocast,
            "num_inference_steps": num_inference_steps,
        },
        num_threads=multiprocessing.cpu_count(),
    )
    profile_result = timer.timeit(
        n_repeats
    )  # benchmark.Timer performs 2 iterations for warmup
    return round(profile_result.mean, 2)


def get_inference_memory(pipe, n_samples, use_autocast, num_inference_steps):
    if not torch.cuda.is_available():
        return 0

    torch.cuda.empty_cache()
    context = autocast if (device.type == "cuda" and use_autocast) else nullcontext
    with context("cuda"):
        images = pipe(
            prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps
        ).images

    mem = torch.cuda.memory_reserved()
    return round(mem / 1e9, 2)

@torch.inference_mode()
def run_benchmark(
    n_repeats, n_samples, precision, use_autocast, xformers, backend, num_inference_steps
):
    """
    * n_repeats: nb datapoints for inference latency benchmark
    * n_samples: number of samples to generate (~ batch size)
    * precision: 'half' or 'single' (use fp16 or fp32 tensors)

    returns:
    dict like {'memory usage': 17.70, 'latency': 86.71'}
    """

    print(f"n_samples: {n_samples}\tprecision: {precision}\tautocast: {use_autocast}\txformers: {xformers}\tbackend: {backend}")

    pipe = get_inference_pipeline(precision, backend)
    if xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if n_samples>16:
        pipe.enable_vae_slicing()

    logs = {
        "memory": 0.00
        if device.type == "cpu"
        else get_inference_memory(
            pipe, n_samples, use_autocast, num_inference_steps
        ),
        "latency": get_inference_time(
            pipe, n_samples, n_repeats, use_autocast, num_inference_steps
        ),
    }
    print(logs, "\n")
    print("============================")
    return logs


def get_device_description():
    """
    returns descriptor of cuda device such as
    'NVIDIA RTX A6000'
    """
    if device.type == "cpu":
        name = subprocess.check_output(
            "grep -m 1 'model name' /proc/cpuinfo", shell=True
        ).decode("utf-8")
        name = " ".join(name.split(" ")[2:]).strip()
        return name
    else:
        return torch.cuda.get_device_name()


def run_benchmark_grid(grid, n_repeats, num_inference_steps, csv_fpath):
    """
    * grid : dict like
        {
            "n_samples": (1, 2),
            "precision": ("single", "half"),
            "autocast" : ("yes", "no")
        }
    * n_repeats: nb datapoints for inference latency benchmark
    """

    # create benchmark.csv if not exists
    if not os.path.isfile(csv_fpath):
        header = [
            "device",
            "precision",
            "autocast",
            "xformers"
            "runtime",
            "n_samples",
            "latency",
            "memory",
        ]
        with open(csv_fpath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # append new benchmark results to it if benchmark_tmp.csv already exists
    with open(csv_fpath, "a") as f:
        writer = csv.writer(f)
        device_desc = get_device_description()
        for trial in itertools.product(*grid.values()):

            n_samples, precision, use_autocast, xformers, backend = trial
            use_autocast = make_bool(use_autocast)
            xformers = make_bool(xformers)

            try:
                new_log = run_benchmark(
                    n_repeats=n_repeats,
                    n_samples=n_samples,
                    precision=precision,
                    use_autocast=use_autocast,
                    xformers=xformers,
                    backend=backend,
                    num_inference_steps=num_inference_steps,
                )
            except Exception as e:
                if "CUDA out of memory" in str(
                    e
                ) or "Failed to allocate memory" in str(e):
                    print(str(e))
                    torch.cuda.empty_cache()
                    new_log = {"latency": -1.00, "memory": -1.00}
                else:
                    raise e

            latency = new_log["latency"]
            memory = new_log["memory"]
            new_row = [
                device_desc,
                precision,
                use_autocast,
                xformers,
                backend,
                n_samples,
                latency,
                memory,
            ]
            writer.writerow(new_row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        default="1",
        type=str,
        help="Comma sepearated list of batch sizes (number of samples)",
    )

    parser.add_argument(
        "--steps", default=50, type=int, help="Number of diffusion steps."
    )

    parser.add_argument(
        "--repeats",
        default=3,
        type=int,
        help="Number of repeats.",
    )

    parser.add_argument(
        "--autocast",
        default="no",
        type=str,
        help="If 'yes', will perform additional runs with autocast activated for half precision inferences",
    )

    parser.add_argument(
        "--xformers",
        default="yes",
        type=str,
        help="If 'yes', will use xformers flash attention",
    )

    parser.add_argument(
        "--output_file",
        default="results.py",
        type=str,
        help="Path to output csv file to write",
    )

    args = parser.parse_args()

    grid = {
        "n_samples": tuple(map(int, args.samples.split(","))),
        # Only use single-precision for cpu because "LayerNormKernelImpl" not implemented for 'Half' on cpu,
        # Remove autocast won't help. Ref:
        # https://github.com/CompVis/stable-diffusion/issues/307
        "precision": ("single",) if device.type == "cpu" else ("single", "half"),
        "autocast": args.autocast.split(","),
        "xformers": args.xformers.split(","),
        # Only use onnx for cpu, until issues are fixed by upstreams. Ref:
        # https://github.com/huggingface/diffusers/issues/489#issuecomment-1261577250
        # https://github.com/huggingface/diffusers/pull/440
        "backend": ("pytorch", "onnx") if device.type == "cpu" else ("pytorch",),
    }
    run_benchmark_grid(grid, n_repeats=args.repeats, num_inference_steps=args.steps, csv_fpath=args.output_file)
