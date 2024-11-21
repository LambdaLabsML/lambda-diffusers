# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
from pathlib import Path

import torch
from torch.onnx import export

from lambda_diffusers import StableDiffusionImageEmbedOnnxPipeline
from diffusers.onnx_utils import OnnxRuntimeModel
from packaging import version

from lambda_diffusers import StableDiffusionImageEmbedPipeline

from PIL import Image
import requests

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int):
    pipeline = StableDiffusionImageEmbedPipeline.from_pretrained(model_path)
    output_path = Path(output_path)


    # UNET
    onnx_export(
        pipeline.unet,
        model_args=(torch.randn(2, 4, 64, 64), torch.LongTensor([0, 1]), torch.randn(2, 77, 768), False),
        output_path=output_path / "unet" / "model.onnx",
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )

    # VAE ENCODER
    vae_encoder = pipeline.vae
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(torch.randn(1, 3, 512, 512), False),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(torch.randn(1, 4, 64, 64), False),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )


    onnx_pipeline = StableDiffusionImageEmbedOnnxPipeline(
        vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
    )

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    _ = StableDiffusionImageEmbedOnnxPipeline.from_pretrained(output_path, provider="CPUExecutionProvider")
    print("ONNX pipeline is loadable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=14,
        type=str,
        help="The version of the ONNX operator set to use.",
    )

    args = parser.parse_args()

    convert_models(args.model_path, args.output_path, args.opset)