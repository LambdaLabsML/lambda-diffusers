# Lambda Diffusers

_Additional models and pipelines for 🤗 Diffusers created by [Lambda Labs](https://lambdalabs.com/)_

- [Stable Diffusion Image Variations](#stable-diffusion-image-variations)
- [Pokemon text to image](#pokemon-text-to-image)

## Installation

```bash
git clone https://github.com/LambdaLabsML/lambda-diffusers.git
cd lambda-diffusers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Stable Diffusion Image Variations

![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

A fine-tuned version of Stable Diffusion conditioned on CLIP image embeddings to enabel Image Variations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JqNbI_kDq_Gth2MIYdsphgNgyGIJxBgB?usp=sharing)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-orange)](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations)
[![Open in Replicate](https://img.shields.io/badge/%F0%9F%9A%80-Open%20in%20Replicate-%23fff891)](https://replicate.com/lambdal/stable-diffusion-image-variation)

- Download the weights ported to 🤗 Diffusers [here](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
- See the original training repo [here](https://github.com/justinpinkney/stable-diffusion).

### Usage

```python
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open("path/to/image.jpg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device)

out = sd_pipe(inp, guidance_scale=3)
out["images"][0].save("result.jpg")

```

## Pokemon text to image

__Stable Diffusion fine tuned on Pokémon by [Lambda Labs](https://lambdalabs.com/).__

[![Open in Replicate](https://img.shields.io/badge/%F0%9F%9A%80-Open%20in%20Replicate-%23fff891)](https://replicate.com/lambdal/text-to-pokemon)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LambdaLabsML/lambda-diffusers/blob/main/notebooks/pokemon_demo.ipynb)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-orange)](https://huggingface.co/spaces/lambdalabs/text-to-pokemon)

Put in a text prompt and generate your own Pokémon character, no "prompt engineering" required!

If you want to find out how to train your own Stable Diffusion variants, see this [example](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning) from Lambda Labs.

![](https://raw.githubusercontent.com/LambdaLabsML/examples/main/stable-diffusion-finetuning/README_files/montage.jpg)

> Girl with a pearl earring, Cute Obama creature, Donald Trump, Boris Johnson, Totoro, Hello Kitty

## Model description

Trained on [BLIP captioned Pokémon images](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) using 2xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud) for around 15,000 step (about 6 hours, at a cost of about $10).

## Usage


```python
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-pokemon-diffusers", torch_dtype=torch.float16)  
pipe = pipe.to("cuda")

prompt = "Yoda"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

for idx, im in enumerate(images):
  im.save(f"{idx:06}.png")
```

## Benchmarking inference

Detailed benchmark documentation can be found [here](./docs/benchmark.md).

### Setup

Before running the benchmark, make sure you have completed the repository [installation steps](#installation).

You will then need to set the huggingface access token:
1. Create a user account on HuggingFace and generate an access token.
2. Set your huggingface access token as the `ACCESS_TOKEN` environment variable:
```
export ACCESS_TOKEN=<hf_...>
```

### Usage

Launch the benchmark script to append benchmark results to the existing [benchmark.csv](./benchmark.csv) results file:
```
python ./scripts/benchmark.py
```

### Results

<img src="./docs/pictures/pretty_benchmark_sd_txt2img_latency.png" alt="Stable Diffusion Text2Image Latency (seconds)" width="850"/>

## Links

- [Captioned Pokémon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
- [Model weights in Diffusers format](https://huggingface.co/lambdalabs/sd-pokemon-diffusers)
- [Original model weights](https://huggingface.co/justinpinkney/pokemon-stable-diffusion)
- [Training code](https://github.com/justinpinkney/stable-diffusion)

Trained by [Justin Pinkney](justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda Labs](https://lambdalabs.com/).
