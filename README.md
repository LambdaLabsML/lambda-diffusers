# Lambda Diffusers

_Additional models and pipelines for 🤗 Diffusers created by [Lambda Labs](https://lambdalabs.com/)_

![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

Currently supports a fine-tuned version of Stable Diffusion conditioned on CLIP image embeddings to enabel Image Variations.

[![Open Demo](https://img.shields.io/badge/%CE%BB-Open%20Demo-blueviolet)](https://47725.gradio.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JqNbI_kDq_Gth2MIYdsphgNgyGIJxBgB?usp=sharing)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-orange)]()

- Download the weights ported to 🤗 Diffusers [here](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
- See the original training repo [here](https://github.com/justinpinkney/stable-diffusion).

## Usage

### Installation
 
```bash
git clone https://github.com/LambdaLabsML/lambda-diffusers.git
cd lambda-diffusers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from pathlib import Path
from lambda_diffusers import StableDiffusionImageEmbedPipeline
from PIL import Image
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageEmbedPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers")
pipe = pipe.to(device)
im = Image.open("your/input/image/here.jpg")
num_samples = 4
image = pipe(num_samples*[im], guidance_scale=3.0)
image = image["sample"]
base_path = Path("outputs/im2im")
base_path.mkdir(exist_ok=True, parents=True)
for idx, im in enumerate(image):
    im.save(base_path/f"{idx:06}.jpg")
```
