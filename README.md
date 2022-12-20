# Stable Animations
Generates interpolations between close instances in the latent space. 
The purpose is to generate a moving image that does not differ that much from the original prompt.

This repo is different from the great [stable-diffusion-videos](https://github.com/nateraw/stable-diffusion-videos) because
we interpolate only the image embedding and not the text embedding.

## Examples

Prompt: "a steampunk city"

![Alt Text](https://github.com/fspinna/stable_animations/blob/main/notebooks/example.gif)

Check out the [Jupyter Notebook](https://github.com/fspinna/stable_animations/blob/main/notebooks/example.ipynb)!

## How to install
Standard
```bash
pip install git+https://github.com/fspinna/stable_animations.git
```
With cuda (recommended)
```bash
pip install git+https://github.com/fspinna/stable_animations.git --extra-index-url https://download.pytorch.org/whl/cu116
```

## Future work
- add more latent sampling functions
- make animation smoother (maybe some way to reorder frames)

## Credits
I took a lot of inspiration (and code) from these great articles:
- https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8
- https://towardsdatascience.com/stable-diffusion-using-hugging-face-variations-of-stable-diffusion-56fd2ab7a265

Thanks to Valentina for the latent sampling technique